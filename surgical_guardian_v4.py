"""
╔══════════════════════════════════════════════════════════════════════╗
║             SURGICAL GUARDIAN  v4  —  Safety-First Edition          ║
║        Real-time laparoscopic surgical safety monitoring system      ║
║                                                                      ║
║  PRIMARY MISSION: Prevent accidental damage to blood vessels         ║
║  during laparoscopic surgery through real-time AI detection,         ║
║  proximity alerting, motion analysis, and event logging.             ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
    # Live camera (DroidCam)
    python surgical_guardian_v4.py

    # Video file (for demo / expo)
    python surgical_guardian_v4.py --source path/to/video.mp4

    # Webcam
    python surgical_guardian_v4.py --source 0

    # Custom model / thresholds
    python surgical_guardian_v4.py --model best.pt --conf 0.25

KEYBOARD CONTROLS (during run):
    Q / ESC  — quit and save report
    P        — pause / resume
    S        — save screenshot now
    R        — reset session stats
    +/-      — increase / decrease confidence threshold live

NEW IN v4:
    ✦ CLI arguments  (--source, --model, --conf, --no-record, --no-log)
    ✦ Tool velocity & approach-direction analysis
    ✦ Per-tool danger scoring  (hook > scissors > bipolar > others)
    ✦ Confidence-weighted detections  (low-conf = phantom-safe hold)
    ✦ Temporal smoothing  (bbox IOU tracking, no flicker)
    ✦ Freeze-frame on CRITICAL  (0.25s freeze so alert registers)
    ✦ Auto post-session PDF-style text report
    ✦ Screenshot hotkey
    ✦ Pause / resume
    ✦ Live conf threshold adjustment
    ✦ Graceful reconnect on camera drop
    ✦ Organ overlap warning (tool inside organ bbox)
    ✦ Annotated output video with all overlays baked in

INSTALL:
    pip install ultralytics opencv-python numpy
"""

import argparse
import csv
import math
import os
import platform
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Surgical Guardian v4 — real-time laparoscopic safety monitor"
    )
    p.add_argument("--source",    default="http://100.91.39.55:4747/video",
                   help="Video source: URL, file path, or webcam index (default: DroidCam URL)")
    p.add_argument("--model",     default="best.pt",
                   help="Path to YOLO .pt model file (default: best.pt)")
    p.add_argument("--conf",      type=float, default=0.30,
                   help="Initial confidence threshold 0-1 (default: 0.30)")
    p.add_argument("--no-record", action="store_true",
                   help="Disable session video recording")
    p.add_argument("--no-log",    action="store_true",
                   help="Disable CSV alert logging")
    p.add_argument("--width",     type=int, default=640)
    p.add_argument("--height",    type=int, default=480)
    return p.parse_args()

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "bipolar", "clipper", "grasper", "hook",
    "irrigator", "scissors", "specimen_bag",      # 0-6   Tools
    "liver", "gallbladder", "abdominal_wall",
    "fat", "GI_tract", "connective_tissue",
    "liver_ligament",                             # 7-13  Organs
    "CYSTIC ARTERY", "CYSTIC DUCT",              # 14-15 Vessels
]

TOOLS   = set(range(0, 7))
ORGANS  = set(range(7, 14))
VESSELS = {14, 15}

# Danger weight per tool — higher = riskier instrument
TOOL_DANGER_WEIGHT = {
    0: 0.8,   # bipolar    — cuts & coagulates
    1: 0.9,   # clipper    — clips vessels, high risk if misplaced
    2: 0.4,   # grasper    — blunt, lower risk
    3: 1.0,   # hook       — highest risk (dissects near vessels)
    4: 0.2,   # irrigator  — fluid, low risk
    5: 0.85,  # scissors   — cuts, high risk
    6: 0.1,   # specimen_bag — retrieval, minimal risk
}

# Proximity thresholds (pixels at 640×480)
CAUTION_DIST  = 150
WARNING_DIST  = 100
CRITICAL_DIST = 60

# Colors  BGR
C_TOOL     = (0, 220, 255)
C_ORGAN    = (0, 140, 255)
C_VESSEL   = (0,   0, 255)
C_CAUTION  = (0, 200, 255)
C_WARNING  = (0, 100, 255)
C_CRITICAL = (0,   0, 255)
C_OK       = (0, 200,  80)
C_HUD      = (0, 255, 180)
C_APPROACH = (0,  50, 255)   # approaching arrow overlay

ALERT_TIERS = [
    # (threshold_px, label, color, tier_int, beep_hz)
    (CRITICAL_DIST, "!! CRITICAL — STOP !!",     C_CRITICAL, 3, 1100),
    (WARNING_DIST,  "!  WARNING — Too Close",     C_WARNING,  2,  800),
    (CAUTION_DIST,  "   CAUTION — Approaching",  C_CAUTION,  1,  600),
]

TRAIL_LEN       = 25
ALERT_COOLDOWN  = 1.2    # seconds between beeps
FREEZE_FRAMES   = 7      # frames to freeze display on CRITICAL
IOU_MATCH_THRESH = 0.25  # min IOU to consider two boxes the same object

# ══════════════════════════════════════════════════════════════════════
# AUDIO
# ══════════════════════════════════════════════════════════════════════
_last_beep = 0.0

def play_beep(freq=1000, duration_ms=280):
    global _last_beep
    now = time.time()
    if now - _last_beep < ALERT_COOLDOWN:
        return
    _last_beep = now

    def _do():
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(int(freq), duration_ms)
            else:
                dur = duration_ms / 1000
                os.system(
                    f"play -nq -t alsa synth {dur:.2f} sine {int(freq)} 2>/dev/null || "
                    f"afplay /System/Library/Sounds/Sosumi.aiff 2>/dev/null || "
                    f"echo -e '\\a'"
                )
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════
# LATENCY-FREE THREADED CAPTURE
# ══════════════════════════════════════════════════════════════════════
class VideoCapture:
    """
    Works for:
      - Live camera URLs  (DroidCam, RTSP)
      - Local video files (mp4, avi …)
      - Webcam indices    (0, 1 …)
    For files: reads as fast as inference allows (no artificial sleep).
    For cameras: drops stale frames, always delivers the newest.
    """
    def __init__(self, source, w, h):
        self.source  = source
        self.w, self.h = w, h
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = False
        self._eof    = False

        # Detect if source is a file
        src = str(source)
        self.is_file = (src.isdigit() is False and
                        not src.startswith("http") and
                        not src.startswith("rtsp") and
                        os.path.isfile(src))

        self._open()

    def _open(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        self.cap = cv2.VideoCapture(src)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

    def start(self):
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    self._eof = True
                    self.running = False
                    break
                # Camera dropped — try reconnect
                time.sleep(1.0)
                self._open()
                continue
            with self.lock:
                self.frame = frame
            if self.is_file:
                time.sleep(0.001)   # yield slightly for file sources

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    @property
    def eof(self):
        return self._eof

    def stop(self):
        self.running = False
        self.cap.release()

# ══════════════════════════════════════════════════════════════════════
# IMAGE ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """CLAHE on L-channel — removes laparoscopic smoke/fog haze."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# ══════════════════════════════════════════════════════════════════════
# IOU-BASED TEMPORAL TRACKER  (prevents bbox flicker)
# ══════════════════════════════════════════════════════════════════════
def iou(a, b):
    xi1 = max(a["x1"], b["x1"]); yi1 = max(a["y1"], b["y1"])
    xi2 = min(a["x2"], b["x2"]); yi2 = min(a["y2"], b["y2"])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    ua = (a["x2"]-a["x1"])*(a["y2"]-a["y1"])
    ub = (b["x2"]-b["x1"])*(b["y2"]-b["y1"])
    return inter / (ua + ub - inter)

def smooth_detections(prev: list, curr: list, alpha=0.5) -> list:
    """Exponential smoothing of bbox coordinates using IOU matching."""
    if not prev:
        return curr
    out = []
    for c in curr:
        best_iou, best_p = 0.0, None
        for p in prev:
            if p["cls"] == c["cls"]:
                sc = iou(c, p)
                if sc > best_iou:
                    best_iou, best_p = sc, p
        if best_p and best_iou > IOU_MATCH_THRESH:
            # Smooth coordinates
            c = c.copy()
            for k in ("x1","y1","x2","y2","cx","cy"):
                c[k] = int(alpha * c[k] + (1 - alpha) * best_p[k])
        out.append(c)
    return out

# ══════════════════════════════════════════════════════════════════════
# VELOCITY & APPROACH ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def compute_velocity(trail: deque):
    """Returns (vx, vy, speed) in px/frame from last two trail points."""
    pts = list(trail)
    if len(pts) < 2:
        return 0.0, 0.0, 0.0
    vx = pts[-1][0] - pts[-2][0]
    vy = pts[-1][1] - pts[-2][1]
    return vx, vy, math.hypot(vx, vy)

def approach_rate(trail: deque, vessel_cx: int, vessel_cy: int):
    """
    Returns rate of change of distance to vessel (px/frame).
    Negative = approaching, Positive = retreating.
    """
    pts = list(trail)
    if len(pts) < 3:
        return 0.0
    d_now  = math.hypot(pts[-1][0] - vessel_cx, pts[-1][1] - vessel_cy)
    d_prev = math.hypot(pts[-3][0] - vessel_cx, pts[-3][1] - vessel_cy)
    return (d_now - d_prev) / 2.0  # avg over 2 frames

def is_inside_bbox(point_cx, point_cy, det):
    """Check if a center point is inside a detection's bounding box."""
    return (det["x1"] <= point_cx <= det["x2"] and
            det["y1"] <= point_cy <= det["y2"])

# ══════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════
def draw_label(frame, text, x1, y1, color, font_scale=0.48, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y_top = max(y1 - th - 8, 0)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def draw_trails(frame, tool_trails):
    for pts_dq in tool_trails.values():
        pts = list(pts_dq)
        for i in range(1, len(pts)):
            alpha = int(220 * i / len(pts))
            thick = max(1, i // 6)
            cv2.line(frame, pts[i-1], pts[i], (0, alpha, 255), thick, cv2.LINE_AA)

def draw_velocity_arrow(frame, cx, cy, vx, vy, speed):
    """Draw a short arrow showing tool movement direction."""
    if speed < 1.5:
        return
    scale = min(speed * 3, 40)
    ex = int(cx + vx / max(speed, 1e-6) * scale)
    ey = int(cy + vy / max(speed, 1e-6) * scale)
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), C_APPROACH, 2,
                    tipLength=0.4, line_type=cv2.LINE_AA)

def draw_hud(frame, tools, vessels, organs, fps, stats, conf_thresh, paused):
    h, w = frame.shape[:2]
    panel_w = 195
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    def txt(text, x, y, color=C_HUD, scale=0.48, bold=False):
        t = 2 if bold else 1
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)

    def row(label, val, y, val_color=C_HUD):
        txt(label, 10, y, (140, 140, 140))
        txt(val, panel_w - 10 - len(val)*8, y, val_color)

    # Title
    txt("SURGICAL", 10, 26, C_HUD, 0.60, True)
    txt("GUARDIAN v4", 10, 46, C_HUD, 0.50)
    cv2.line(frame, (10, 54), (panel_w - 10, 54), C_HUD, 1)

    # Status
    status_color = (0, 80, 255) if paused else C_OK
    status_label = "PAUSED" if paused else "LIVE"
    txt(status_label, panel_w // 2 - 20, 72, status_color, 0.50, True)

    row("FPS",      f"{int(fps):3d}",          90,  C_HUD)
    row("CONF",     f"{conf_thresh:.2f}",      108,  (180, 180, 180))
    row("TOOLS",    str(len(tools)),           130,  C_TOOL)
    row("ORGANS",   str(len(organs)),          148,  C_ORGAN)
    row("VESSELS",  str(len(vessels)),         166,  C_VESSEL)

    cv2.line(frame, (10, 178), (panel_w - 10, 178), (50, 50, 50), 1)

    row("ALERTS",   str(stats["total"]),       196,  C_WARNING)
    row("CRITICAL", str(stats["critical"]),    214,  C_CRITICAL)
    row("WARNING",  str(stats["warning"]),     232,  C_WARNING)
    row("CAUTION",  str(stats["caution"]),     250,  C_CAUTION)

    cv2.line(frame, (10, 262), (panel_w - 10, 262), (50, 50, 50), 1)
    row("SESSION",  stats["elapsed"],          280,  C_HUD)
    row("FRAMES",   str(stats["frames"]),      298,  (140, 140, 140))

    # Min dist readout
    cv2.line(frame, (10, 312), (panel_w - 10, 312), (50, 50, 50), 1)
    txt("MIN DIST", 10, 332, (120, 120, 120), 0.42)
    min_d = stats["min_dist"]
    d_color = (C_CRITICAL if min_d < CRITICAL_DIST else
               C_WARNING  if min_d < WARNING_DIST  else
               C_CAUTION  if min_d < CAUTION_DIST  else C_OK)
    dist_str = f"{int(min_d)}px" if min_d < 9999 else "---"
    txt(dist_str, 10, 360, d_color, 0.72, True)

    # Approaching indicator
    if stats.get("approaching"):
        txt(">> APPROACHING", 6, 390, C_CRITICAL, 0.40, True)

    # Controls hint
    cv2.line(frame, (10, h - 80), (panel_w - 10, h - 80), (40, 40, 40), 1)
    for i, hint in enumerate(["P:pause  S:screenshot",
                               "+/-:conf  R:reset",
                               "Q/ESC: quit+report"]):
        txt(hint, 6, h - 62 + i * 16, (80, 80, 80), 0.36)

    # REC dot
    if stats.get("recording"):
        cv2.circle(frame, (panel_w - 15, 18), 6, (0, 0, 200), -1)
        txt("REC", panel_w - 50, 22, (0, 0, 200), 0.38)

def draw_organ_overlap_warning(frame, tools, organs, w):
    """Warn if a tool center is inside an organ bounding box."""
    for t in tools:
        for o in organs:
            if is_inside_bbox(t["cx"], t["cy"], o):
                organ_name = CLASS_NAMES[o["cls"]]
                msg = f"TOOL INSIDE {organ_name.upper()}"
                (mw, mh), _ = cv2.getTextSize(
                    msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                mx = w // 2 - mw // 2
                cv2.rectangle(frame, (mx - 8, 108), (mx + mw + 8, 138),
                              (0, 0, 0), -1)
                cv2.putText(frame, msg, (mx, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            C_WARNING, 2, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════
# SESSION REPORT
# ══════════════════════════════════════════════════════════════════════
def write_report(stats, args, output_dir):
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    path  = os.path.join(output_dir, f"session_report_{ts}.txt")
    lines = [
        "=" * 62,
        "  SURGICAL GUARDIAN v4 — SESSION SAFETY REPORT",
        "=" * 62,
        f"  Date/Time   : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        f"  Source      : {args.source}",
        f"  Model       : {args.model}",
        f"  Confidence  : {args.conf}",
        f"  Duration    : {stats['elapsed']}",
        f"  Frames proc : {stats['frames']}",
        "-" * 62,
        "  PROXIMITY EVENTS",
        f"  Total alerts    : {stats['total']}",
        f"  CRITICAL (<{CRITICAL_DIST}px) : {stats['critical']}",
        f"  WARNING  (<{WARNING_DIST}px) : {stats['warning']}",
        f"  CAUTION  (<{CAUTION_DIST}px)  : {stats['caution']}",
        f"  Closest approach: {int(stats['closest_ever'])} px",
        "-" * 62,
        "  SAFETY ASSESSMENT",
    ]
    score = 100
    score -= stats["critical"] * 15
    score -= stats["warning"]  * 5
    score -= stats["caution"]  * 1
    score  = max(0, score)
    assessment = ("EXCELLENT" if score >= 90 else
                  "GOOD"      if score >= 75 else
                  "MODERATE"  if score >= 50 else
                  "HIGH RISK — REVIEW FOOTAGE")
    lines += [
        f"  Safety score    : {score}/100",
        f"  Assessment      : {assessment}",
        "=" * 62,
        "  All events logged to alert_log.csv",
        "=" * 62,
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n  Report saved → {path}")

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    OUTPUT_DIR = "sg_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────
    print("=" * 62)
    print("  SURGICAL GUARDIAN v4")
    print("=" * 62)
    print(f"  Source : {args.source}")
    print(f"  Model  : {args.model}")
    print(f"  Conf   : {args.conf}")
    print()
    print("[1/3] Loading YOLO model...")
    if not os.path.exists(args.model):
        print(f"  ERROR: Model file '{args.model}' not found.")
        sys.exit(1)
    model = YOLO(args.model)

    # ── Camera / video ───────────────────────────────────────────────
    print("[2/3] Opening video source...")
    cap = VideoCapture(args.source, args.width, args.height).start()
    if not cap.is_file:
        time.sleep(0.8)
    print("[3/3] Running. Controls: P=pause  S=screenshot  +/-=conf  Q=quit\n")

    # ── Output files ─────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    writer = None
    if not args.no_record:
        vpath  = os.path.join(OUTPUT_DIR, f"session_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vpath, fourcc, 20.0, (args.width, args.height))
        print(f"  Recording → {vpath}")

    csv_file = csv_writer = None
    if not args.no_log:
        lpath      = os.path.join(OUTPUT_DIR, f"alert_log_{ts}.csv")
        csv_file   = open(lpath, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "frame", "level",
                             "tool", "tool_conf", "vessel",
                             "dist_px", "approach_rate",
                             "tool_danger_weight"])
        print(f"  Logging  → {lpath}")

    # ── State ────────────────────────────────────────────────────────
    conf_thresh  = args.conf
    tool_trails: dict = {}        # cls_id → deque[(cx,cy)]
    prev_dets:   list = []
    last_dets:   list = []
    freeze_left        = 0
    paused             = False
    session_t0         = time.time()
    prev_time          = time.time()

    stats = {
        "total": 0, "critical": 0, "warning": 0, "caution": 0,
        "frames": 0, "min_dist": 9999.0, "closest_ever": 9999.0,
        "elapsed": "00:00", "approaching": False, "recording": writer is not None,
    }

    # ── Main loop ────────────────────────────────────────────────────
    while True:
        # EOF for video files
        if cap.eof:
            print("\n  Video file ended.")
            break

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):       # Q or ESC
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            sname = os.path.join(OUTPUT_DIR,
                f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg")
            # Screenshot saved at end of loop after drawing
            save_screenshot = True
        elif key == ord("r"):
            stats.update({"total":0,"critical":0,"warning":0,
                          "caution":0,"frames":0,"min_dist":9999.0,
                          "closest_ever":9999.0})
            tool_trails.clear()
        elif key == ord("+") or key == ord("="):
            conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
        elif key == ord("-"):
            conf_thresh = max(0.05, round(conf_thresh - 0.05, 2))
        else:
            save_screenshot = False

        if paused:
            time.sleep(0.03)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        if frame.shape[:2] != (args.height, args.width):
            frame = cv2.resize(frame, (args.width, args.height))

        h, w = frame.shape[:2]
        frame = enhance_frame(frame)
        stats["frames"] += 1

        # ── YOLO inference ───────────────────────────────────────────
        results    = model(frame, conf=conf_thresh, imgsz=416,
                           verbose=False, stream=True)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id      = int(box.cls[0])
                conf        = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                detections.append({
                    "cls":   cls_id,
                    "conf":  conf,
                    "group": ("tool" if cls_id in TOOLS else
                              "vessel" if cls_id in VESSELS else "organ"),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1+x2)//2, "cy": (y1+y2)//2,
                })

        # Smooth with previous frame to remove flicker
        detections = smooth_detections(prev_dets, detections, alpha=0.55)
        prev_dets  = detections

        draw_dets  = detections if detections else last_dets
        last_dets  = detections if detections else last_dets

        tools   = [d for d in draw_dets if d["group"] == "tool"]
        organs  = [d for d in draw_dets if d["group"] == "organ"]
        vessels = [d for d in draw_dets if d["group"] == "vessel"]

        # ── Update motion trails ──────────────────────────────────────
        seen = set()
        for t in tools:
            tid = t["cls"]
            seen.add(tid)
            tool_trails.setdefault(tid, deque(maxlen=TRAIL_LEN))
            tool_trails[tid].append((t["cx"], t["cy"]))
        for tid in list(tool_trails):
            if tid not in seen:
                del tool_trails[tid]

        # ── FREEZE frame on recent critical ─────────────────────────
        if freeze_left > 0:
            freeze_left -= 1
            # Skip drawing, just show last frame
            cv2.imshow("Surgical Guardian v4", frame)
            continue

        # ── Draw motion trails ────────────────────────────────────────
        draw_trails(frame, tool_trails)

        # ── Draw velocity arrows ──────────────────────────────────────
        for t in tools:
            trail = tool_trails.get(t["cls"])
            if trail:
                vx, vy, speed = compute_velocity(trail)
                draw_velocity_arrow(frame, t["cx"], t["cy"], vx, vy, speed)

        # ── Draw all bounding boxes ───────────────────────────────────
        for d in draw_dets:
            color = (C_TOOL   if d["group"] == "tool"   else
                     C_VESSEL if d["group"] == "vessel" else C_ORGAN)
            cv2.rectangle(frame,
                          (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
            label = f"{CLASS_NAMES[d['cls']]} {d['conf']:.2f}"
            draw_label(frame, label, d["x1"], d["y1"], color)

        # ── Draw vessel danger rings ──────────────────────────────────
        for v in vessels:
            for radius, color in [(CAUTION_DIST, C_CAUTION),
                                  (WARNING_DIST, C_WARNING),
                                  (CRITICAL_DIST, C_CRITICAL)]:
                cv2.circle(frame, (v["cx"], v["cy"]), radius, color, 1)
            cv2.putText(frame, f"! {CLASS_NAMES[v['cls']]}",
                        (v["cx"] - 50, v["cy"] - CAUTION_DIST - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_VESSEL, 1, cv2.LINE_AA)

        # ── Organ overlap warning ─────────────────────────────────────
        draw_organ_overlap_warning(frame, tools, organs, w)

        # ── Proximity analysis ────────────────────────────────────────
        frame_min_dist  = 9999.0
        frame_alert_lvl = 0
        alert_color     = C_OK
        frame_approaching = False

        for t in tools:
            danger_w = TOOL_DANGER_WEIGHT.get(t["cls"], 0.5)
            trail    = tool_trails.get(t["cls"])

            for v in vessels:
                dist = math.hypot(t["cx"] - v["cx"], t["cy"] - v["cy"])

                # Effective distance — dangerous tools feel "closer"
                eff_dist = dist / danger_w

                if dist < frame_min_dist:
                    frame_min_dist = dist

                # Approach rate (is tool moving toward vessel?)
                app_rate = 0.0
                if trail:
                    app_rate = approach_rate(trail, v["cx"], v["cy"])
                    if app_rate < -1.0:   # negative = approaching
                        frame_approaching = True

                # Determine alert tier using effective distance
                for threshold, msg, color, tier, freq in ALERT_TIERS:
                    if eff_dist < threshold:
                        # Connection line
                        cv2.line(frame, (t["cx"], t["cy"]),
                                 (v["cx"], v["cy"]), color, 2, cv2.LINE_AA)

                        # Banner
                        full_msg = f"{msg}  [{CLASS_NAMES[t['cls']]} → {CLASS_NAMES[v['cls']]}]"
                        (bw, bh), _ = cv2.getTextSize(
                            full_msg, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)
                        bx = w // 2 - bw // 2
                        cv2.rectangle(frame, (bx - 10, 58),
                                      (bx + bw + 10, 98), (0, 0, 0), -1)
                        cv2.putText(frame, full_msg, (bx, 88),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.72,
                                    color, 2, cv2.LINE_AA)

                        # Distance label on line midpoint
                        mx = (t["cx"] + v["cx"]) // 2
                        my = (t["cy"] + v["cy"]) // 2
                        cv2.putText(frame, f"{int(dist)}px",
                                    (mx + 4, my - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    color, 1, cv2.LINE_AA)

                        if tier > frame_alert_lvl:
                            frame_alert_lvl = tier
                            alert_color     = color

                        # Log event
                        if csv_writer:
                            level_name = ["", "CAUTION", "WARNING", "CRITICAL"][tier]
                            csv_writer.writerow([
                                datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                stats["frames"], level_name,
                                CLASS_NAMES[t["cls"]], f"{t['conf']:.2f}",
                                CLASS_NAMES[v["cls"]], f"{dist:.1f}",
                                f"{app_rate:.2f}", f"{danger_w:.2f}",
                            ])
                        break   # highest tier only per pair

        # Update stats
        stats["min_dist"]      = frame_min_dist
        stats["approaching"]   = frame_approaching
        if frame_min_dist < stats["closest_ever"]:
            stats["closest_ever"] = frame_min_dist

        if frame_alert_lvl == 3:
            stats["total"]    += 1
            stats["critical"] += 1
            play_beep(freq=1100, duration_ms=350)
            freeze_left = FREEZE_FRAMES
        elif frame_alert_lvl == 2:
            stats["total"]   += 1
            stats["warning"] += 1
            play_beep(freq=800, duration_ms=220)
        elif frame_alert_lvl == 1:
            stats["total"]   += 1
            stats["caution"] += 1
            play_beep(freq=600, duration_ms=160)

        # Approaching accent (even without threshold breach)
        if frame_approaching and frame_alert_lvl == 0 and vessels:
            cv2.putText(frame, "Approaching vessel...",
                        (w // 2 - 100, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_CAUTION, 1, cv2.LINE_AA)

        # Border flash
        if frame_alert_lvl:
            thickness = 12 if frame_alert_lvl == 3 else 6
            cv2.rectangle(frame, (0, 0), (w, h), alert_color, thickness)

        # ── HUD ──────────────────────────────────────────────────────
        elapsed = int(time.time() - session_t0)
        stats["elapsed"] = f"{elapsed//60:02d}:{elapsed%60:02d}"
        fps = 1.0 / max(1e-6, time.time() - prev_time)
        prev_time = time.time()
        draw_hud(frame, tools, vessels, organs, fps, stats, conf_thresh, paused)

        # ── Write / show ─────────────────────────────────────────────
        if writer:
            writer.write(frame)

        if save_screenshot:
            sname = os.path.join(OUTPUT_DIR,
                f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(sname, frame)
            print(f"  Screenshot → {sname}")

        cv2.imshow("Surgical Guardian v4", frame)

    # ── Cleanup ───────────────────────────────────────────────────────
    print("\nShutting down...")
    cap.stop()
    if writer:
        writer.release()
        print(f"  Video saved.")
    if csv_file:
        csv_file.close()
        print(f"  Log saved.")

    write_report(stats, args, OUTPUT_DIR)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
