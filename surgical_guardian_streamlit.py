"""
╔══════════════════════════════════════════════════════════════════════╗
║         SURGICAL GUARDIAN v4 — Streamlit Web Edition                ║
║      Real-time laparoscopic surgical safety monitoring system        ║
║                                                                      ║
║  HOW TO RUN:                                                         ║
║    1. pip install streamlit ultralytics opencv-python-headless numpy ║
║    2. Place best.pt in the same folder as this file                  ║
║    3. streamlit run surgical_guardian_streamlit.py                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import csv
import math
import os
import tempfile
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Surgical Guardian v4",
    page_icon="🔬",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "bipolar", "clipper", "grasper", "hook",
    "irrigator", "scissors", "specimen_bag",
    "liver", "gallbladder", "abdominal_wall",
    "fat", "GI_tract", "connective_tissue",
    "liver_ligament",
    "CYSTIC ARTERY", "CYSTIC DUCT",
]

TOOLS   = set(range(0, 7))
ORGANS  = set(range(7, 14))
VESSELS = {14, 15}

TOOL_DANGER_WEIGHT = {
    0: 0.8,   # bipolar
    1: 0.9,   # clipper
    2: 0.4,   # grasper
    3: 1.0,   # hook
    4: 0.2,   # irrigator
    5: 0.85,  # scissors
    6: 0.1,   # specimen_bag
}

CAUTION_DIST  = 150
WARNING_DIST  = 100
CRITICAL_DIST = 60

# BGR colors
C_TOOL     = (0, 220, 255)
C_ORGAN    = (0, 140, 255)
C_VESSEL   = (0,   0, 255)
C_CAUTION  = (0, 200, 255)
C_WARNING  = (0, 100, 255)
C_CRITICAL = (0,   0, 255)
C_OK       = (0, 200,  80)
C_HUD      = (0, 255, 180)
C_APPROACH = (0,  50, 255)

ALERT_TIERS = [
    (CRITICAL_DIST, "!! CRITICAL — STOP !!",    C_CRITICAL, 3),
    (WARNING_DIST,  "!  WARNING — Too Close",    C_WARNING,  2),
    (CAUTION_DIST,  "   CAUTION — Approaching",  C_CAUTION,  1),
]

TRAIL_LEN        = 25
IOU_MATCH_THRESH = 0.25

# ══════════════════════════════════════════════════════════════════════
# HELPERS — same logic as original, no cv2.imshow / keyboard / beep
# ══════════════════════════════════════════════════════════════════════
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def iou(a, b):
    xi1 = max(a["x1"], b["x1"]); yi1 = max(a["y1"], b["y1"])
    xi2 = min(a["x2"], b["x2"]); yi2 = min(a["y2"], b["y2"])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    ua = (a["x2"]-a["x1"]) * (a["y2"]-a["y1"])
    ub = (b["x2"]-b["x1"]) * (b["y2"]-b["y1"])
    return inter / (ua + ub - inter)

def smooth_detections(prev, curr, alpha=0.55):
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
            c = c.copy()
            for k in ("x1", "y1", "x2", "y2", "cx", "cy"):
                c[k] = int(alpha * c[k] + (1 - alpha) * best_p[k])
        out.append(c)
    return out

def compute_velocity(trail):
    pts = list(trail)
    if len(pts) < 2:
        return 0.0, 0.0, 0.0
    vx = pts[-1][0] - pts[-2][0]
    vy = pts[-1][1] - pts[-2][1]
    return vx, vy, math.hypot(vx, vy)

def approach_rate(trail, vessel_cx, vessel_cy):
    pts = list(trail)
    if len(pts) < 3:
        return 0.0
    d_now  = math.hypot(pts[-1][0] - vessel_cx, pts[-1][1] - vessel_cy)
    d_prev = math.hypot(pts[-3][0] - vessel_cx, pts[-3][1] - vessel_cy)
    return (d_now - d_prev) / 2.0

def is_inside_bbox(px, py, det):
    return det["x1"] <= px <= det["x2"] and det["y1"] <= py <= det["y2"]

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
    if speed < 1.5:
        return
    scale = min(speed * 3, 40)
    ex = int(cx + vx / max(speed, 1e-6) * scale)
    ey = int(cy + vy / max(speed, 1e-6) * scale)
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), C_APPROACH, 2,
                    tipLength=0.4, line_type=cv2.LINE_AA)

def draw_organ_overlap_warning(frame, tools, organs, w):
    for t in tools:
        for o in organs:
            if is_inside_bbox(t["cx"], t["cy"], o):
                organ_name = CLASS_NAMES[o["cls"]]
                msg = f"TOOL INSIDE {organ_name.upper()}"
                (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                mx = w // 2 - mw // 2
                cv2.rectangle(frame, (mx - 8, 108), (mx + mw + 8, 138), (0, 0, 0), -1)
                cv2.putText(frame, msg, (mx, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WARNING, 2, cv2.LINE_AA)

def draw_hud(frame, tools, vessels, organs, fps, stats, conf_thresh):
    h, w = frame.shape[:2]
    panel_w = 195
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    def txt(text, x, y, color=C_HUD, scale=0.48, bold=False):
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)

    def row(label, val, y, val_color=C_HUD):
        txt(label, 10, y, (140, 140, 140))
        txt(val, panel_w - 10 - len(val) * 8, y, val_color)

    txt("SURGICAL",     10, 26, C_HUD, 0.60, True)
    txt("GUARDIAN v4",  10, 46, C_HUD, 0.50)
    cv2.line(frame, (10, 54), (panel_w - 10, 54), C_HUD, 1)

    txt("PROCESSING", panel_w // 2 - 36, 72, C_OK, 0.45, True)

    row("FPS",     f"{int(fps):3d}",        90,  C_HUD)
    row("CONF",    f"{conf_thresh:.2f}",   108,  (180, 180, 180))
    row("TOOLS",   str(len(tools)),        130,  C_TOOL)
    row("ORGANS",  str(len(organs)),       148,  C_ORGAN)
    row("VESSELS", str(len(vessels)),      166,  C_VESSEL)

    cv2.line(frame, (10, 178), (panel_w - 10, 178), (50, 50, 50), 1)

    row("ALERTS",   str(stats["total"]),    196, C_WARNING)
    row("CRITICAL", str(stats["critical"]), 214, C_CRITICAL)
    row("WARNING",  str(stats["warning"]),  232, C_WARNING)
    row("CAUTION",  str(stats["caution"]),  250, C_CAUTION)

    cv2.line(frame, (10, 262), (panel_w - 10, 262), (50, 50, 50), 1)
    row("FRAMES",   str(stats["frames"]),   280, (140, 140, 140))

    cv2.line(frame, (10, 292), (panel_w - 10, 292), (50, 50, 50), 1)
    txt("MIN DIST", 10, 312, (120, 120, 120), 0.42)
    min_d = stats["min_dist"]
    d_color = (C_CRITICAL if min_d < CRITICAL_DIST else
               C_WARNING  if min_d < WARNING_DIST  else
               C_CAUTION  if min_d < CAUTION_DIST  else C_OK)
    dist_str = f"{int(min_d)}px" if min_d < 9999 else "---"
    txt(dist_str, 10, 340, d_color, 0.72, True)

    if stats.get("approaching"):
        txt(">> APPROACHING", 6, 365, C_CRITICAL, 0.40, True)

def compute_safety_score(stats):
    score = 100
    score -= stats["critical"] * 15
    score -= stats["warning"]  * 5
    score -= stats["caution"]  * 1
    return max(0, score)

# ══════════════════════════════════════════════════════════════════════
# CORE PROCESSING — process a single frame, return annotated frame
# ══════════════════════════════════════════════════════════════════════
def process_frame(frame, model, conf_thresh, tool_trails, prev_dets, stats):
    """
    Runs YOLO inference + all drawing logic on one frame.
    Returns (annotated_frame, updated_prev_dets, alert_level 0-3).
    """
    h, w = frame.shape[:2]
    frame = enhance_frame(frame)
    stats["frames"] += 1

    results    = model(frame, conf=conf_thresh, imgsz=416, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id       = int(box.cls[0])
            conf         = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "cls":   cls_id,
                "conf":  conf,
                "group": ("tool"   if cls_id in TOOLS   else
                          "vessel" if cls_id in VESSELS else "organ"),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1+x2)//2, "cy": (y1+y2)//2,
            })

    detections = smooth_detections(prev_dets, detections, alpha=0.55)

    tools   = [d for d in detections if d["group"] == "tool"]
    organs  = [d for d in detections if d["group"] == "organ"]
    vessels = [d for d in detections if d["group"] == "vessel"]

    # Update motion trails
    seen = set()
    for t in tools:
        tid = t["cls"]
        seen.add(tid)
        tool_trails.setdefault(tid, deque(maxlen=TRAIL_LEN))
        tool_trails[tid].append((t["cx"], t["cy"]))
    for tid in list(tool_trails):
        if tid not in seen:
            del tool_trails[tid]

    # Draw
    draw_trails(frame, tool_trails)

    for t in tools:
        trail = tool_trails.get(t["cls"])
        if trail:
            vx, vy, speed = compute_velocity(trail)
            draw_velocity_arrow(frame, t["cx"], t["cy"], vx, vy, speed)

    for d in detections:
        color = (C_TOOL   if d["group"] == "tool"   else
                 C_VESSEL if d["group"] == "vessel" else C_ORGAN)
        cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
        draw_label(frame, f"{CLASS_NAMES[d['cls']]} {d['conf']:.2f}",
                   d["x1"], d["y1"], color)

    for v in vessels:
        for radius, color in [(CAUTION_DIST, C_CAUTION),
                              (WARNING_DIST,  C_WARNING),
                              (CRITICAL_DIST, C_CRITICAL)]:
            cv2.circle(frame, (v["cx"], v["cy"]), radius, color, 1)
        cv2.putText(frame, f"! {CLASS_NAMES[v['cls']]}",
                    (v["cx"] - 50, v["cy"] - CAUTION_DIST - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_VESSEL, 1, cv2.LINE_AA)

    draw_organ_overlap_warning(frame, tools, organs, w)

    # Proximity analysis
    frame_min_dist    = 9999.0
    frame_alert_lvl   = 0
    alert_color       = C_OK
    frame_approaching = False
    alert_events      = []

    for t in tools:
        danger_w = TOOL_DANGER_WEIGHT.get(t["cls"], 0.5)
        trail    = tool_trails.get(t["cls"])
        for v in vessels:
            dist     = math.hypot(t["cx"] - v["cx"], t["cy"] - v["cy"])
            eff_dist = dist / danger_w
            if dist < frame_min_dist:
                frame_min_dist = dist

            app_rate = 0.0
            if trail:
                app_rate = approach_rate(trail, v["cx"], v["cy"])
                if app_rate < -1.0:
                    frame_approaching = True

            for threshold, msg, color, tier in ALERT_TIERS:
                if eff_dist < threshold:
                    cv2.line(frame, (t["cx"], t["cy"]),
                             (v["cx"], v["cy"]), color, 2, cv2.LINE_AA)
                    full_msg = f"{msg}  [{CLASS_NAMES[t['cls']]} → {CLASS_NAMES[v['cls']]}]"
                    (bw, _), _ = cv2.getTextSize(
                        full_msg, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)
                    bx = w // 2 - bw // 2
                    cv2.rectangle(frame, (bx - 10, 58), (bx + bw + 10, 98), (0,0,0), -1)
                    cv2.putText(frame, full_msg, (bx, 88),
                                cv2.FONT_HERSHEY_DUPLEX, 0.72, color, 2, cv2.LINE_AA)
                    mx = (t["cx"] + v["cx"]) // 2
                    my = (t["cy"] + v["cy"]) // 2
                    cv2.putText(frame, f"{int(dist)}px", (mx + 4, my - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
                    if tier > frame_alert_lvl:
                        frame_alert_lvl = tier
                        alert_color     = color

                    level_name = ["", "CAUTION", "WARNING", "CRITICAL"][tier]
                    alert_events.append({
                        "time":     datetime.now().strftime("%H:%M:%S"),
                        "frame":    stats["frames"],
                        "level":    level_name,
                        "tool":     CLASS_NAMES[t["cls"]],
                        "vessel":   CLASS_NAMES[v["cls"]],
                        "dist_px":  round(dist, 1),
                    })
                    break

    # Update stats
    stats["min_dist"]    = frame_min_dist
    stats["approaching"] = frame_approaching
    if frame_min_dist < stats["closest_ever"]:
        stats["closest_ever"] = frame_min_dist

    if frame_alert_lvl == 3:
        stats["total"]    += 1
        stats["critical"] += 1
    elif frame_alert_lvl == 2:
        stats["total"]   += 1
        stats["warning"] += 1
    elif frame_alert_lvl == 1:
        stats["total"]   += 1
        stats["caution"] += 1

    if frame_approaching and frame_alert_lvl == 0 and vessels:
        cv2.putText(frame, "Approaching vessel...",
                    (w // 2 - 100, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_CAUTION, 1, cv2.LINE_AA)

    if frame_alert_lvl:
        thickness = 12 if frame_alert_lvl == 3 else 6
        cv2.rectangle(frame, (0, 0), (w, h), alert_color, thickness)

    draw_hud(frame, tools, vessels, organs, 0, stats, conf_thresh)

    return frame, detections, frame_alert_lvl, alert_events

# ══════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════
def main():
    # ── Custom CSS ───────────────────────────────────────────────────
    st.markdown("""
    <style>
        .main { background-color: #0d0d0d; color: #e0e0e0; }
        .stApp { background-color: #0d0d0d; }
        h1, h2, h3 { color: #00ffb4 !important; }
        .metric-card {
            background: #1a1a2e;
            border: 1px solid #00ffb4;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .alert-critical { color: #ff3333; font-weight: bold; font-size: 18px; }
        .alert-warning  { color: #ff8800; font-weight: bold; }
        .alert-caution  { color: #ffee00; }
        .alert-ok       { color: #00c853; }
    </style>
    """, unsafe_allow_html=True)

    # ── Title ────────────────────────────────────────────────────────
    st.markdown("# 🔬 Surgical Guardian v4")
    st.markdown("**Real-time Laparoscopic Safety Monitoring System**")
    st.markdown("---")

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        model_file = st.file_uploader(
            "Upload YOLO Model (.pt)", type=["pt"],
            help="Upload your best.pt model file"
        )

        conf_thresh = st.slider(
            "Confidence Threshold", 0.05, 0.95, 0.30, 0.05,
            help="Minimum detection confidence"
        )

        st.markdown("---")
        st.markdown("## 📁 Input Source")
        source_type = st.radio(
            "Video Source",
            ["Upload Video File", "Webcam (Live)"],
            index=0
        )

        skip_frames = st.slider(
            "Process every Nth frame", 1, 5, 1,
            help="Skip frames to speed up processing"
        )

        st.markdown("---")
        st.markdown("## ℹ️ Detection Classes")
        st.markdown("""
        🔧 **Tools:** bipolar, clipper, grasper, hook, irrigator, scissors, specimen_bag  
        🫀 **Vessels:** CYSTIC ARTERY, CYSTIC DUCT  
        🫁 **Organs:** liver, gallbladder, abdominal_wall, fat, GI_tract, connective_tissue, liver_ligament
        """)

        st.markdown("---")
        st.markdown("## 🚨 Alert Thresholds")
        st.markdown(f"""
        - 🔴 **CRITICAL** < {CRITICAL_DIST}px  
        - 🟠 **WARNING** < {WARNING_DIST}px  
        - 🟡 **CAUTION** < {CAUTION_DIST}px
        """)

    # ── Load model ───────────────────────────────────────────────────
    model = None
    if model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(model_file.read())
            tmp_model_path = tmp.name
        try:
            with st.spinner("Loading YOLO model..."):
                model = YOLO(tmp_model_path)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
    elif os.path.exists("best.pt"):
        try:
            with st.spinner("Loading default best.pt model..."):
                model = YOLO("best.pt")
            st.info("✅ Loaded default best.pt")
        except Exception as e:
            st.error(f"❌ Failed to load best.pt: {e}")
    else:
        st.warning("⚠️ Please upload your **best.pt** model file in the sidebar to begin.")

    # ── Main Area ────────────────────────────────────────────────────
    if model is None:
        st.info("👆 Upload your YOLO model (.pt) in the sidebar to start analysis.")
        return

    # ── Video File Mode ──────────────────────────────────────────────
    if source_type == "Upload Video File":
        video_file = st.file_uploader(
            "Upload Surgical Video", type=["mp4", "avi", "mov", "mkv"],
            help="Upload a laparoscopic surgery video for analysis"
        )

        if video_file is None:
            st.info("📹 Upload a video file to begin analysis.")
            return

        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(video_file.read())
            tmp_video_path = tmp_vid.name

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📺 Live Analysis Feed")
            frame_placeholder = st.empty()
            status_placeholder = st.empty()

        with col2:
            st.markdown("### 📊 Session Statistics")
            total_placeholder    = st.empty()
            critical_placeholder = st.empty()
            warning_placeholder  = st.empty()
            caution_placeholder  = st.empty()
            dist_placeholder     = st.empty()
            score_placeholder    = st.empty()

        st.markdown("---")
        st.markdown("### 🚨 Alert Event Log")
        log_placeholder = st.empty()

        # Controls
        col_a, col_b = st.columns(2)
        with col_a:
            run_btn  = st.button("▶️ Start Analysis", type="primary", use_container_width=True)
        with col_b:
            stop_btn = st.button("⏹️ Stop", type="secondary", use_container_width=True)

        if "running" not in st.session_state:
            st.session_state.running = False
        if run_btn:
            st.session_state.running = True
        if stop_btn:
            st.session_state.running = False

        if st.session_state.running:
            cap = cv2.VideoCapture(tmp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            stats = {
                "total": 0, "critical": 0, "warning": 0, "caution": 0,
                "frames": 0, "min_dist": 9999.0, "closest_ever": 9999.0,
                "elapsed": "00:00", "approaching": False,
            }

            tool_trails = {}
            prev_dets   = []
            all_events  = []
            frame_count = 0
            session_t0  = time.time()

            progress_bar = st.progress(0, text="Analysing video...")

            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue

                frame = cv2.resize(frame, (640, 480))

                annotated, prev_dets, alert_lvl, events = process_frame(
                    frame, model, conf_thresh, tool_trails, prev_dets, stats
                )

                all_events.extend(events)

                # Update elapsed
                elapsed = int(time.time() - session_t0)
                stats["elapsed"] = f"{elapsed//60:02d}:{elapsed%60:02d}"

                # Show frame (convert BGR → RGB for Streamlit)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                # Alert status banner
                if alert_lvl == 3:
                    status_placeholder.markdown(
                        '<p class="alert-critical">🔴 !! CRITICAL — STOP !!</p>',
                        unsafe_allow_html=True)
                elif alert_lvl == 2:
                    status_placeholder.markdown(
                        '<p class="alert-warning">🟠 ! WARNING — Too Close</p>',
                        unsafe_allow_html=True)
                elif alert_lvl == 1:
                    status_placeholder.markdown(
                        '<p class="alert-caution">🟡 CAUTION — Approaching</p>',
                        unsafe_allow_html=True)
                else:
                    status_placeholder.markdown(
                        '<p class="alert-ok">🟢 SAFE</p>',
                        unsafe_allow_html=True)

                # Stats panel
                safety_score = compute_safety_score(stats)
                score_color  = ("🟢" if safety_score >= 90 else
                                "🟡" if safety_score >= 75 else
                                "🟠" if safety_score >= 50 else "🔴")

                total_placeholder.metric("⚠️ Total Alerts",   stats["total"])
                critical_placeholder.metric("🔴 Critical",    stats["critical"])
                warning_placeholder.metric("🟠 Warning",      stats["warning"])
                caution_placeholder.metric("🟡 Caution",      stats["caution"])
                min_d = stats["min_dist"]
                dist_placeholder.metric(
                    "📏 Min Distance",
                    f"{int(min_d)}px" if min_d < 9999 else "N/A"
                )
                score_placeholder.metric(
                    f"{score_color} Safety Score",
                    f"{safety_score}/100"
                )

                # Event log (last 10)
                if all_events:
                    recent = all_events[-10:][::-1]
                    log_md = "| Time | Frame | Level | Tool | Vessel | Distance |\n"
                    log_md += "|------|-------|-------|------|--------|----------|\n"
                    for ev in recent:
                        lvl_icon = {"CRITICAL": "🔴", "WARNING": "🟠", "CAUTION": "🟡"}.get(ev["level"], "")
                        log_md += f"| {ev['time']} | {ev['frame']} | {lvl_icon} {ev['level']} | {ev['tool']} | {ev['vessel']} | {ev['dist_px']}px |\n"
                    log_placeholder.markdown(log_md)

                # Progress
                if total_frames > 0:
                    pct = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(pct, text=f"Frame {frame_count}/{total_frames} — {stats['elapsed']}")

            cap.release()
            progress_bar.progress(1.0, text="✅ Analysis complete!")
            st.session_state.running = False

            # ── Final Report ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📋 Session Safety Report")

            safety_score = compute_safety_score(stats)
            assessment = ("✅ EXCELLENT" if safety_score >= 90 else
                          "👍 GOOD"      if safety_score >= 75 else
                          "⚠️ MODERATE"  if safety_score >= 50 else
                          "🚨 HIGH RISK — REVIEW FOOTAGE")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Safety Score",    f"{safety_score}/100")
            c2.metric("Total Alerts",    stats["total"])
            c3.metric("Critical Events", stats["critical"])
            c4.metric("Closest Approach",
                      f"{int(stats['closest_ever'])}px" if stats["closest_ever"] < 9999 else "N/A")

            st.markdown(f"### Assessment: {assessment}")

            # Download CSV log
            if all_events:
                import io
                output = io.StringIO()
                writer = csv.DictWriter(output,
                    fieldnames=["time", "frame", "level", "tool", "vessel", "dist_px"])
                writer.writeheader()
                writer.writerows(all_events)
                st.download_button(
                    label="📥 Download Alert Log (CSV)",
                    data=output.getvalue(),
                    file_name=f"alert_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

    # ── Webcam Mode ──────────────────────────────────────────────────
    else:
        st.markdown("### 📷 Webcam / Live Camera Mode")
        st.info("""
        **Note:** Streamlit Cloud does not support direct webcam access.  
        To use webcam, run this app **locally** with:
        ```
        streamlit run surgical_guardian_streamlit.py
        ```
        Then select 'Webcam (Live)' and click Start.
        """)

        col1, col2 = st.columns([2, 1])
        with col1:
            frame_placeholder  = st.empty()
            status_placeholder = st.empty()
        with col2:
            st.markdown("### 📊 Live Stats")
            total_ph    = st.empty()
            critical_ph = st.empty()
            warning_ph  = st.empty()
            caution_ph  = st.empty()
            dist_ph     = st.empty()

        col_a, col_b = st.columns(2)
        with col_a:
            run_btn  = st.button("▶️ Start Webcam", type="primary", use_container_width=True)
        with col_b:
            stop_btn = st.button("⏹️ Stop Webcam",  type="secondary", use_container_width=True)

        if "cam_running" not in st.session_state:
            st.session_state.cam_running = False
        if run_btn:
            st.session_state.cam_running = True
        if stop_btn:
            st.session_state.cam_running = False

        if st.session_state.cam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot access webcam. Make sure it is connected and not in use.")
                st.session_state.cam_running = False
            else:
                stats = {
                    "total": 0, "critical": 0, "warning": 0, "caution": 0,
                    "frames": 0, "min_dist": 9999.0, "closest_ever": 9999.0,
                    "elapsed": "00:00", "approaching": False,
                }
                tool_trails = {}
                prev_dets   = []
                session_t0  = time.time()

                while st.session_state.cam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Cannot read from webcam.")
                        break

                    frame = cv2.resize(frame, (640, 480))
                    annotated, prev_dets, alert_lvl, _ = process_frame(
                        frame, model, conf_thresh, tool_trails, prev_dets, stats
                    )

                    elapsed = int(time.time() - session_t0)
                    stats["elapsed"] = f"{elapsed//60:02d}:{elapsed%60:02d}"

                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                    if alert_lvl == 3:
                        status_placeholder.markdown(
                            '<p class="alert-critical">🔴 !! CRITICAL — STOP !!</p>',
                            unsafe_allow_html=True)
                    elif alert_lvl == 2:
                        status_placeholder.markdown(
                            '<p class="alert-warning">🟠 ! WARNING — Too Close</p>',
                            unsafe_allow_html=True)
                    elif alert_lvl == 1:
                        status_placeholder.markdown(
                            '<p class="alert-caution">🟡 CAUTION — Approaching</p>',
                            unsafe_allow_html=True)
                    else:
                        status_placeholder.markdown(
                            '<p class="alert-ok">🟢 SAFE</p>',
                            unsafe_allow_html=True)

                    total_ph.metric("⚠️ Total Alerts",    stats["total"])
                    critical_ph.metric("🔴 Critical",     stats["critical"])
                    warning_ph.metric("🟠 Warning",       stats["warning"])
                    caution_ph.metric("🟡 Caution",       stats["caution"])
                    min_d = stats["min_dist"]
                    dist_ph.metric("📏 Min Distance",
                                   f"{int(min_d)}px" if min_d < 9999 else "N/A")

                cap.release()


if __name__ == "__main__":
    main()
