"""
ROADGUARD AI - Streamlit dashboard.

    streamlit run app.py

Every number on screen comes from (a) real YOLO inference, (b) files written
by the dataset / training scripts, or (c) a clearly labelled derived
calculation (severity engine). Nothing is hard-coded.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import dataset_manager as dm
from src.detector import RoadDamageDetector, available_weights, decode_image, ensure_pretrained_weights
from src.report_generator import (append_history, build_pdf, build_report, clear_history, load_history,
                                  save_outputs)
from src.severity import DISCLAIMER, SeverityThresholds, assess_severity
from src.utils import PROJECT_ROOT, class_names, get_path, hardware_info, list_images, load_config, read_json

st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️", layout="wide", initial_sidebar_state="expanded")
CFG = load_config()

# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------
LIME, AMBER, RED = "#C6F24E", "#F2B84E", "#F2654E"
INK, SURFACE, LINE, MUTED, PAPER = "#121416", "#1A1D20", "#2A2E33", "#8E959C", "#ECEAE3"
SEV_COLORS = {"GOOD": LIME, "MODERATE": AMBER, "SEVERE": RED, "UNAVAILABLE": MUTED}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
section[data-testid="stSidebar"] {{ background: #0E1011; border-right: 1px solid {LINE}; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.rg-brand {{ font-weight: 700; font-size: 1.25rem; letter-spacing: 0.14em; color: {PAPER}; }}
.rg-brand span {{ color: {LIME}; }}
.rg-tag {{ color: {MUTED}; font-size: 0.78rem; margin-top: -0.2rem; }}
.rg-title {{ font-size: 1.9rem; font-weight: 700; margin: 0; color: {PAPER}; }}
.rg-sub {{ color: {MUTED}; margin: 0.15rem 0 1.2rem 0; }}
.rg-kpi {{ border: 1px solid {LINE}; border-left: 3px solid {LIME}; background: {SURFACE};
           padding: 0.75rem 0.9rem; border-radius: 4px; height: 100%; min-width: 0; margin-bottom: 0.6rem; }}
.rg-kpi .l {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }}
.rg-kpi .v {{ font-family: 'JetBrains Mono', monospace; font-size: clamp(1.05rem, 1.6vw, 1.45rem); font-weight: 600; color: {PAPER};
              margin-top: 0.15rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.rg-kpi .s {{ color: {MUTED}; font-size: 0.74rem; margin-top: 0.1rem; }}
.rg-sev {{ display: inline-block; padding: 0.35rem 0.9rem; border-radius: 3px; font-weight: 700;
           letter-spacing: 0.12em; font-family: 'JetBrains Mono', monospace; }}
.rg-section {{ color: {PAPER}; font-weight: 600; font-size: 1.02rem; border-bottom: 1px solid {LINE};
               padding-bottom: 0.35rem; margin: 1.2rem 0 0.7rem 0; }}
.rg-note {{ color: {MUTED}; font-size: 0.8rem; }}
.rg-badge {{ display:inline-block; border:1px solid {LINE}; padding: 0.1rem 0.5rem; border-radius: 3px;
             font-size: 0.72rem; color: {MUTED}; font-family: 'JetBrains Mono', monospace; }}
.rg-badge.ok {{ color: {LIME}; border-color: {LIME}55; }}
.rg-badge.warn {{ color: {AMBER}; border-color: {AMBER}55; }}
.rg-empty {{ border: 1px dashed {LINE}; padding: 1.4rem; color: {MUTED}; text-align: center; border-radius: 4px; }}
div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; }}
</style>
""", unsafe_allow_html=True)


def kpi(col, label: str, value, sub: str = "") -> None:
    col.markdown(f"<div class='rg-kpi'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"<div class='rg-section'>{title}</div>", unsafe_allow_html=True)


def header(title: str, sub: str) -> None:
    st.markdown(f"<p class='rg-title'>{title}</p><p class='rg-sub'>{sub}</p>", unsafe_allow_html=True)


def empty(msg: str) -> None:
    st.markdown(f"<div class='rg-empty'>{msg}</div>", unsafe_allow_html=True)


def sev_badge(sev: str) -> str:
    c = SEV_COLORS.get(sev, MUTED)
    return f"<span class='rg-sev' style='color:{INK};background:{c}'>{sev}</span>"


def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=80), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=PAPER, family="Inter"),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, title_text=""),
                      title_font=dict(size=13))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig


def project_path(p: str) -> Path:
    """Paths in the model registry are stored relative to the project root (portable across machines)."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def get_detector(weights: str) -> RoadDamageDetector:
    return RoadDamageDetector(weights)


@st.cache_data(ttl=30)
def get_registry() -> dict:
    return read_json(get_path("trained_dir") / "model_registry.json", {}) or {}


@st.cache_data(ttl=30)
def get_summary(mode: str):
    return dm.load_summary(mode)


@st.cache_data(ttl=60)
def get_hw() -> dict:
    return hardware_info()


def weight_options() -> list[tuple[str, Path]]:
    opts = available_weights()
    return [(n, p) for n, p in opts if p.exists() or p.name == "yolov8n.pt"]


def model_selector(key: str) -> RoadDamageDetector | None:
    opts = weight_options()
    labels = [n for n, _ in opts]
    choice = st.selectbox("Model weights", labels, index=0, key=key,
                          help="Fine-tuned weights are produced by scripts/train.py. The COCO-pretrained model "
                               "has no pothole class and is included only to compare against transfer learning.")
    path = dict(opts)[choice]
    try:
        if not path.exists():
            with st.spinner("Downloading pretrained yolov8n.pt ..."):
                ensure_pretrained_weights()
        return get_detector(str(path))
    except Exception as exc:
        st.error(f"Could not load model weights `{path.name}`: {exc}")
        return None


def model_banner(det: RoadDamageDetector) -> None:
    if det.info.kind == "fine-tuned":
        reg = det.info.registry
        m = reg.get("test_metrics", {})
        extra = f" · test mAP50 {m['mAP50']:.3f}" if m else ""
        st.markdown(f"<span class='rg-badge ok'>FINE-TUNED</span> &nbsp;<span class='rg-note'>"
                    f"{det.info.label} · {det.info.weights.name} · device {det.info.device}{extra}</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='rg-badge warn'>PRETRAINED · COCO</span> &nbsp;<span class='rg-note'>"
                    f"This model was trained on 80 everyday object classes and has <b>no pothole class</b>. "
                    f"Detections are COCO objects; road severity is unavailable. Run scripts/train.py to fine-tune."
                    f"</span>", unsafe_allow_html=True)


def analyse(det: RoadDamageDetector, img: np.ndarray, conf: float, iou: float) -> tuple:
    res = det.predict(img, conf=conf, iou=iou)
    if det.info.detects_damage:
        # severity uses exactly the boxes the user sees (same confidence threshold as the slider)
        th = SeverityThresholds.from_config()
        th.min_confidence = conf
        sev = assess_severity(res.detections, res.width, res.height, th)
    else:  # never invent a road condition from a model that cannot see potholes
        sev = assess_severity([], res.width, res.height)
        sev.update(severity="UNAVAILABLE", detections=res.detections,
                   reasons=["model has no road-damage classes (COCO pretrained)"])
    return res, sev


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = ["Overview", "Detect Potholes", "Live Detection", "Dataset Explorer", "Model Performance", "About Project"]
with st.sidebar:
    st.markdown(f"<div class='rg-brand'>ROAD<span>GUARD</span> AI</div>"
                f"<div class='rg-tag'>{CFG['project']['tagline']}</div>", unsafe_allow_html=True)
    st.write("")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    reg_side = get_registry()
    if "pothole" in reg_side or "multiclass" in reg_side:
        st.markdown("<span class='rg-badge ok'>MODEL FINE-TUNED</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='rg-badge warn'>INFERENCE-ONLY · PRETRAINED</span>", unsafe_allow_html=True)
    hw = get_hw()
    st.caption(f"Python {hw['python']} · torch {hw['torch']} · "
               f"{'GPU ' + hw['gpu_name'] if hw['cuda_available'] else 'CPU inference'}")


# ===========================================================================
# 1. OVERVIEW
# ===========================================================================
def page_overview() -> None:
    header("Overview", "Dataset, model and detection activity at a glance.")
    summ = get_summary("pothole")
    multi = get_summary("multiclass")
    reg = get_registry()
    hist = load_history()

    c = st.columns(3) + st.columns(3)
    if summ:
        total = sum(v["images"] for v in summ["splits"].values())
        annotated = sum(v["with_objects"] for v in summ["splits"].values())
        raw_total = summ.get("raw_totals", {}).get("images_downloaded", "—")
        kpi(c[0], "Dataset images", f"{total:,}", f"{raw_total} downloaded raw")
        kpi(c[1], "Annotated (pothole)", f"{annotated:,}", f"{total - annotated} background")
    else:
        kpi(c[0], "Dataset images", "—", "run prepare_dataset.py")
        kpi(c[1], "Annotated", "—", "")
    kpi(c[2], "Model", "YOLOv8n", "Ultralytics · 3.0M params")
    if "pothole" in reg:
        r = reg["pothole"]
        kpi(c[3], "Training status", "Fine-tuned", f"{r['epochs_completed']} epochs · {r['device']}")
    else:
        kpi(c[3], "Training status", "Pretrained", "not fine-tuned yet")
    n_cls = len(multi["classes"]) if multi else len(class_names("multiclass"))
    kpi(c[4], "Damage classes", n_cls, "pothole + 4 crack/patch types")
    if hist:
        last = hist[-1]
        kpi(c[5], "Latest result", last["severity"], f"{last['potholes']} pothole(s) · {last['timestamp'][11:16]}")
    else:
        kpi(c[5], "Latest result", "—", "no detections yet")

    left, mid, right = st.columns([1, 1.3, 1])
    with left:
        section("Road condition distribution")
        if hist:
            cnt = Counter(h["severity"] for h in hist)
            fig = go.Figure(go.Pie(labels=list(cnt), values=list(cnt.values()), hole=0.62,
                                   marker=dict(colors=[SEV_COLORS.get(k, MUTED) for k in cnt]),
                                   textinfo="label+value", sort=False))
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 280), width="stretch")
        else:
            empty("No analyses yet. Run a detection on the <b>Detect Potholes</b> page.")
    with mid:
        section("Detection activity")
        if hist:
            df = pd.DataFrame(hist)
            df["#"] = range(1, len(df) + 1)
            fig = px.bar(df.tail(40), x="#", y="potholes", color="severity", color_discrete_map=SEV_COLORS,
                         hover_data=["image", "avg_conf", "damaged_area_pct", "timestamp"])
            fig.update_layout(xaxis_title="analysis #", yaxis_title="potholes detected", bargap=0.25)
            st.plotly_chart(style_fig(fig, 280), width="stretch")
            st.caption(f"{len(hist)} analyses · {int(df['potholes'].sum())} potholes detected in total · "
                       f"mean inference {df['inference_ms'].mean():.0f} ms")
        else:
            empty("Activity appears here after the first analysis.")
    with right:
        section("Model performance (test split)")
        if "pothole" in reg:
            m = reg["pothole"]["test_metrics"]
            vals = {"Precision": m["precision"], "Recall": m["recall"], "mAP@50": m["mAP50"], "mAP@50-95": m["mAP50-95"]}
            fig = go.Figure(go.Bar(x=list(vals.values()), y=list(vals), orientation="h", marker_color=LIME,
                                   text=[f"{v:.3f}" for v in vals.values()], textposition="outside"))
            fig.update_layout(xaxis=dict(range=[0, 1.05]))
            st.plotly_chart(style_fig(fig, 280), width="stretch")
        else:
            empty("No evaluation metrics yet - the model has not been fine-tuned.<br>"
                  "Run <code>python scripts/train.py</code>.")

    section("Recent detection history")
    if hist:
        df = pd.DataFrame(hist[::-1][:15])[["timestamp", "image", "source", "model", "severity", "potholes",
                                            "other_damage", "avg_conf", "damaged_area_pct", "inference_ms"]]
        st.dataframe(df, width="stretch", hide_index=True)
        if st.button("Clear history"):
            clear_history()
            st.rerun()
    else:
        empty("History is empty.")


# ===========================================================================
# 2. DETECT POTHOLES
# ===========================================================================
def render_results(det: RoadDamageDetector, img: np.ndarray, name: str, source: str, conf: float, iou: float,
                   gt_count: int | None = None) -> None:
    key = hashlib.md5(img.tobytes()[:2_000_000]).hexdigest() + f"|{det.info.weights}|{conf}|{iou}"
    with st.spinner("Running YOLOv8 inference..."):
        try:
            res, sev = analyse(det, img, conf, iou)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return
    report = build_report(name, res, sev, conf, iou, source=source)
    if st.session_state.get("last_saved_key") != key:  # persist once per unique (image, model, thresholds)
        save_outputs(report, res.annotated)
        append_history(report)
        st.session_state["last_saved_key"] = key
        st.session_state["last_report"] = report
    else:
        report = st.session_state.get("last_report", report)

    a, b = st.columns(2)
    a.markdown("**Original image**")
    a.image(to_rgb(img), width="stretch")
    b.markdown("**Detections**")
    b.image(to_rgb(res.annotated), width="stretch")

    s = sev["severity"]
    c = st.columns(3) + st.columns(3)
    c[0].markdown(f"<div class='rg-kpi'><div class='l'>Road condition</div><div style='margin-top:0.35rem'>"
                  f"{sev_badge(s)}</div><div class='s'>damage index {sev['damage_index']}/100 (derived)</div></div>",
                  unsafe_allow_html=True)
    kpi(c[1], "Potholes", sev["pothole_count"], f"ground truth: {gt_count}" if gt_count is not None else "detected")
    kpi(c[2], "Other damage", sev["other_damage_count"], "cracks / patches")
    kpi(c[3], "Avg confidence", f"{sev['average_confidence']:.2f}", f"max {sev['maximum_confidence']:.2f}")
    kpi(c[4], "Damaged area", f"{sev['damaged_area_percentage']:.1f}%", "union of boxes / image")
    kpi(c[5], "Inference", f"{res.inference_ms:.0f} ms", f"{res.width}×{res.height} px · {res.model.device}")

    st.write("")
    st.markdown(f"<span class='rg-note'><b>Why:</b> {'; '.join(sev['reasons'])}.<br>{DISCLAIMER}</span>",
                unsafe_allow_html=True)

    section("Individual detections")
    if res.detections:
        rows = [{"#": i, "class": d["class_name"], "confidence": d["confidence"],
                 "x1": d["bbox"][0], "y1": d["bbox"][1], "x2": d["bbox"][2], "y2": d["bbox"][3],
                 "area %": d["area_percentage"]} for i, d in enumerate(res.detections, 1)]
        t1, t2 = st.columns([1.4, 1])
        t1.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                  format="%.3f")})
        fig = go.Figure(go.Bar(x=[f"#{r['#']}" for r in rows], y=[r["confidence"] for r in rows],
                               marker_color=[LIME if r["class"] == "pothole" else AMBER for r in rows]))
        fig.add_hline(y=conf, line_dash="dot", line_color=MUTED, annotation_text="threshold")
        fig.update_layout(yaxis=dict(range=[0, 1]), title="confidence per detection")
        t2.plotly_chart(style_fig(fig, 260), width="stretch")
    else:
        empty(f"No objects above the confidence threshold ({conf:.2f}). Lower the threshold to see weaker detections.")

    section("Export")
    d1, d2, d3, _ = st.columns([1, 1, 1, 2])
    ok, png = cv2.imencode(".png", res.annotated)
    stem = Path(name).stem
    d1.download_button("Annotated image (PNG)", png.tobytes() if ok else b"", f"{stem}_roadguard.png", "image/png",
                       width="stretch")
    d2.download_button("JSON report", json.dumps(report, indent=2), f"{stem}_report.json", "application/json",
                       width="stretch")
    pdf = build_pdf(report, res.annotated)
    if pdf:
        d3.download_button("PDF report", pdf, f"{stem}_report.pdf", "application/pdf", width="stretch")
    with st.expander("Raw JSON report"):
        st.json(report)


def page_detect() -> None:
    header("Detect Potholes", "Upload a road image - YOLOv8 localises potholes and the severity engine grades the road.")
    ctl, _ = st.columns([3, 1])
    with st.container():
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            det = model_selector("det_model")
        conf = c2.slider("Confidence threshold", 0.05, 0.95, float(CFG["inference"]["conf"]), 0.05,
                         help="Minimum class confidence for a box to be kept.")
        iou = c3.slider("NMS IoU threshold", 0.10, 0.90, float(CFG["inference"]["iou"]), 0.05,
                        help="Boxes overlapping more than this IoU are merged by Non-Maximum Suppression.")
    if det is None:
        return
    model_banner(det)
    st.write("")

    src = st.segmented_control("Input", ["Upload image", "Sample images"], default="Upload image")
    if src == "Sample images":
        samples = list_images(get_path("samples_dir"))
        if not samples:
            empty("No sample images. Run <code>python scripts/prepare_dataset.py</code> to copy held-out test images.")
            return
        st.caption("Samples are taken from the held-out TEST split - the model never saw them during training.")
        thumbs = st.columns(min(6, len(samples)))
        for i, p in enumerate(samples[:6]):
            thumbs[i].image(to_rgb(cv2.imread(str(p))), caption=p.stem.replace("sample_", ""), width="stretch")
        pick = st.selectbox("Choose a sample", [p.name for p in samples])
        path = get_path("samples_dir") / pick
        img = cv2.imread(str(path))
        gt_file = path.with_suffix(".txt")
        gt = None
        if gt_file.exists():
            gt = sum(1 for ln in gt_file.read_text().splitlines() if ln.strip() and ln.split()[0] == "0")
        render_results(det, img, pick, "sample", conf, iou, gt_count=gt)
    else:
        up = st.file_uploader("Road image (JPG, JPEG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if up is None:
            empty("Drop a road photo above to start. Tip: switch to <b>Sample images</b> for held-out test photos.")
            return
        try:
            img = decode_image(up.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return
        render_results(det, img, up.name, "upload", conf, iou)


# ===========================================================================
# 3. LIVE DETECTION
# ===========================================================================
def page_live() -> None:
    header("Live Detection", "Real-time pothole detection on a webcam feed (OpenCV on the machine running this app).")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        det = model_selector("live_model")
    conf = c2.slider("Confidence", 0.05, 0.95, float(CFG["inference"]["conf"]), 0.05, key="live_conf")
    cam_idx = c3.number_input("Camera index", 0, 5, 0, help="0 = default webcam")
    imgsz = c4.select_slider("Input size", [320, 416, 512, 640], value=416, help="smaller = faster on CPU")
    if det is None:
        return
    model_banner(det)

    mode = st.segmented_control("Mode", ["OpenCV live stream", "Browser snapshot"], default="OpenCV live stream")
    if mode == "Browser snapshot":
        shot = st.camera_input("Take a photo of the road")
        if shot is not None:
            try:
                render_results(det, decode_image(shot.getvalue()), "camera_snapshot.jpg", "camera", conf,
                               CFG["inference"]["iou"])
            except ValueError as exc:
                st.error(str(exc))
        return

    st.session_state.setdefault("live_running", False)
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Start camera", type="primary", width="stretch"):
        st.session_state.live_running = True
    if b2.button("Stop camera", width="stretch"):
        st.session_state.live_running = False

    frame_slot = st.empty()
    stats_slot = st.empty()
    if not st.session_state.live_running:
        empty("Camera is stopped. Press <b>Start camera</b>. The webcam is opened by OpenCV on this computer; "
              "use <b>Browser snapshot</b> if the app runs remotely.")
        return

    cap = None
    try:
        cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(int(cam_idx))
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(cam_idx))
        if not cap.isOpened():
            st.session_state.live_running = False
            st.error(f"No webcam available at index {cam_idx}. Check that a camera is connected and not used by "
                     f"another application, or use Browser snapshot mode.")
            return
        frames, t_start, fails = 0, time.time(), 0
        while st.session_state.live_running:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 30:
                    st.warning("Camera stopped delivering frames.")
                    break
                continue
            res = det.predict(frame, conf=conf, iou=CFG["inference"]["iou"], imgsz=int(imgsz))
            th = SeverityThresholds.from_config()
            th.min_confidence = conf
            sev = assess_severity(res.detections, res.width, res.height, th) if det.info.detects_damage else None
            frames += 1
            fps = frames / max(1e-6, time.time() - t_start)
            frame_slot.image(to_rgb(res.annotated), width="stretch")
            if sev:
                stats_slot.markdown(
                    f"{sev_badge(sev['severity'])} &nbsp; <span class='rg-note'>potholes in frame: "
                    f"<b>{sev['pothole_count']}</b> · damaged area {sev['damaged_area_percentage']:.1f}% · "
                    f"{res.inference_ms:.0f} ms/frame · {fps:.1f} FPS</span>", unsafe_allow_html=True)
            else:
                stats_slot.markdown(f"<span class='rg-note'>{len(res.detections)} COCO objects · {fps:.1f} FPS</span>",
                                    unsafe_allow_html=True)
    except Exception as exc:  # never let the webcam crash the rest of the app
        st.session_state.live_running = False
        st.error(f"Live detection stopped: {exc}")
    finally:
        if cap is not None:
            cap.release()


# ===========================================================================
# 4. DATASET EXPLORER
# ===========================================================================
def page_dataset() -> None:
    header("Dataset Explorer", "Real annotated road images used to train and evaluate the model.")
    mode = st.segmented_control("Dataset", ["pothole", "multiclass"], default="pothole",
                                format_func=lambda m: "Pothole-only (1 class)" if m == "pothole" else "Multi-class (5 classes)")
    mode = mode or "pothole"
    summ = get_summary(mode)
    manifest = dm.load_raw_manifest()
    if not summ:
        empty("Dataset not prepared yet.<br><code>python scripts/download_dataset.py</code> then "
              "<code>python scripts/prepare_dataset.py</code>")
        return
    sp = summ["splits"]
    c = st.columns(5)
    raw_tot = (manifest or {}).get("totals", {})
    kpi(c[0], "Downloaded images", f"{raw_tot.get('images_downloaded', 0):,}", "all sources, before cleaning")
    kpi(c[1], "Train", f"{sp['train']['images']:,}", f"{sp['train']['background']} background")
    kpi(c[2], "Validation", f"{sp['val']['images']:,}", f"{sp['val']['background']} background")
    kpi(c[3], "Test", f"{sp['test']['images']:,}", f"{sp['test']['background']} background")
    inst = sum(sum(v.values()) for v in summ["class_instances"].values())
    kpi(c[4], "Labelled objects", f"{inst:,}", f"{len(summ['classes'])} class(es)")

    l, r = st.columns(2)
    with l:
        section("Class distribution (instances)")
        rows = [{"split": s, "class": k, "instances": v} for s, d in summ["class_instances"].items() for k, v in d.items()]
        if rows:
            fig = px.bar(pd.DataFrame(rows), x="class", y="instances", color="split", barmode="group",
                         color_discrete_sequence=[LIME, AMBER, "#7FB3F2"])
            st.plotly_chart(style_fig(fig, 300), width="stretch")
    with r:
        section("Images per source")
        rows = [{"split": s, "source": k, "images": v} for s, d in summ["sources"].items() for k, v in d.items()]
        if rows:
            fig = px.bar(pd.DataFrame(rows), y="source", x="images", color="split", orientation="h",
                         color_discrete_sequence=[LIME, AMBER, "#7FB3F2"])
            st.plotly_chart(style_fig(fig, 300), width="stretch")

    section("Browse annotated images (ground truth)")
    f1, f2, f3 = st.columns([1, 2, 1])
    split = f1.selectbox("Split", list(dm.SPLITS))
    imgs = dm.list_images(mode, split)
    sources = sorted({dm.source_of(p) for p in imgs})
    chosen = f2.multiselect("Sources", sources, default=sources)
    imgs = [p for p in imgs if dm.source_of(p) in chosen]
    # deterministic shuffle so every page mixes sources/countries instead of alphabetical blocks
    imgs = sorted(imgs, key=lambda p: hashlib.md5(p.name.encode()).hexdigest())
    per_page = 12
    pages = max(1, (len(imgs) + per_page - 1) // per_page)
    pg = f3.number_input(f"Page (1-{pages})", 1, pages, 1)
    cols = st.columns(4)
    for i, p in enumerate(imgs[(pg - 1) * per_page: pg * per_page]):
        boxes = dm.read_labels(mode, split, p)
        col = cols[i % 4]
        if boxes is None:
            col.image(to_rgb(cv2.imread(str(p))), width="stretch")
            col.caption(f"⚠ UNANNOTATED · {p.name[:40]}")
            continue
        col.image(to_rgb(dm.draw_ground_truth(p, boxes, mode)), width="stretch")
        names = class_names(mode)
        cnt = Counter(names.get(b[0], "?") for b in boxes)
        desc = ", ".join(f"{v}× {k}" for k, v in cnt.items()) if boxes else "background (no damage labelled)"
        col.caption(f"{dm.source_of(p)} · {desc}")
    st.caption(f"{len(imgs)} images in this view. Boxes are the human annotations from the source dataset.")

    section("Pre-processing log")
    pp = summ["preprocessing"]
    pcols = st.columns(6)
    for col, (k, lab) in zip(pcols, [("images_found", "raw images found"), ("missing_annotations", "unannotated (excluded)"),
                                     ("corrupt_images", "corrupt"), ("exact_duplicates_removed", "exact duplicates"),
                                     ("invalid_boxes_removed", "invalid boxes removed"),
                                     ("unmapped_boxes_removed", "out-of-schema boxes")]):
        kpi(col, lab, f"{pp.get(k, 0):,}")
    v = summ["verification"]
    st.markdown(f"<br><span class='rg-badge {'ok' if v['ok'] else 'warn'}'>"
                f"{'IMAGE-LABEL CORRESPONDENCE VERIFIED' if v['ok'] else 'VERIFICATION ISSUES'}</span> "
                f"<span class='rg-note'>split ratio {summ['split_ratio']} · images resized to max {summ['max_side']} px · "
                f"near-duplicates grouped into the same split · generated {summ['generated']}</span>",
                unsafe_allow_html=True)

    section("Sources & licences")
    for s in (manifest or {}).get("sources", []):
        status = {"ok": "ok", "skipped": "warn", "failed": "warn"}.get(s["status"], "")
        with st.expander(f"{s['name']}  —  {s['status'].upper()}  ·  {s['images_downloaded']} images"):
            st.markdown(f"<span class='rg-badge {status}'>{s['status'].upper()}</span>", unsafe_allow_html=True)
            st.markdown(f"**URL:** {s['url']}  \n**Licence:** {s['license']}  \n**Format:** {s['format']}  \n"
                        f"**Citation:** {s['citation']}")
            st.markdown(f"images {s['images_downloaded']} · annotated {s['images_annotated']} · "
                        f"background {s['background_images']} · instances {s['label_instances']} · "
                        f"duplicates skipped {s['duplicates_skipped']} · unannotated skipped {s['unannotated_skipped']}")
            if s.get("class_counts"):
                st.caption("raw class counts: " + ", ".join(f"{k}: {v}" for k, v in s["class_counts"].items()))
            for n in s.get("notes", []):
                st.caption("· " + n)


# ===========================================================================
# 5. MODEL PERFORMANCE
# ===========================================================================
def page_performance() -> None:
    header("Model Performance", "Metrics read from the actual Ultralytics training and evaluation outputs.")
    reg = get_registry()
    modes = [m for m in ("pothole", "multiclass") if m in reg]
    if not modes:
        st.warning("The model has not been fine-tuned yet, so there are no accuracy metrics to show. "
                   "The app currently runs pretrained-model inference only.")
        st.code("python scripts/train.py --epochs 15", language="bash")
        return
    mode = st.segmented_control("Model", modes, default=modes[0]) or modes[0]
    r = reg[mode]

    section("Model information")
    info = {
        "Architecture": f"{r['architecture']} (CSPDarknet backbone · PAN-FPN neck · anchor-free decoupled head)",
        "Base weights": f"{r['base_weights']} (COCO-pretrained, transfer learning)",
        "Classes": ", ".join(f"{k}: {v}" for k, v in r["classes"].items()),
        "Input size": f"{r['imgsz']} × {r['imgsz']} (letterboxed)",
        "Weight file": r["weights"],
        "Training set": f"{r['dataset']['train']} train · {r['dataset']['val']} val · {r['dataset']['test']} test images",
        "Epochs": f"{r['epochs_completed']} completed of {r['epochs_requested']} requested (early-stopping patience "
                  f"{CFG['training']['patience']})",
        "Hardware / time": f"{r['device']} · batch {r['batch']} · {r['train_minutes']} min · {r['trained_at']}",
        "Status": r["status"],
    }
    st.dataframe(pd.DataFrame(info.items(), columns=["property", "value"]), hide_index=True, width="stretch")

    for split_name, key in (("Validation split", "val_metrics"), ("Held-out test split", "test_metrics")):
        section(split_name)
        m = r[key]
        c = st.columns(4)
        kpi(c[0], "Precision", f"{m['precision']:.3f}", "TP / (TP + FP)")
        kpi(c[1], "Recall", f"{m['recall']:.3f}", "TP / (TP + FN)")
        kpi(c[2], "mAP@50", f"{m['mAP50']:.3f}", "IoU ≥ 0.50")
        kpi(c[3], "mAP@50-95", f"{m['mAP50-95']:.3f}", "IoU 0.50 : 0.95")
        if len(m.get("per_class", {})) > 1:
            st.dataframe(pd.DataFrame(m["per_class"]).T, width="stretch")

    run_dir = project_path(r["run_dir"])
    csv = run_dir / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        section("Training curves (per epoch)")
        a, b = st.columns(2)
        loss_cols = [c for c in df.columns if "loss" in c]
        long = df.melt(id_vars="epoch", value_vars=loss_cols, var_name="series", value_name="loss")
        fig = px.line(long, x="epoch", y="loss", color="series", markers=True,
                      color_discrete_sequence=[LIME, "#9BD13A", "#6E9124", AMBER, "#D9973A", "#9C6A26"])
        fig.update_layout(title="train vs validation losses (box / cls / dfl)")
        a.plotly_chart(style_fig(fig, 340), width="stretch")
        met_cols = [c for c in df.columns if c.startswith("metrics/")]
        long = df.melt(id_vars="epoch", value_vars=met_cols, var_name="metric", value_name="value")
        long["metric"] = long["metric"].str.replace("metrics/", "").str.replace("(B)", "")
        fig = px.line(long, x="epoch", y="value", color="metric", markers=True,
                      color_discrete_sequence=[LIME, AMBER, "#7FB3F2", PAPER])
        fig.update_layout(title="validation metrics", yaxis=dict(range=[0, 1]))
        b.plotly_chart(style_fig(fig, 340), width="stretch")
    else:
        st.info("results.csv not found - training curves unavailable.")

    section("Evaluation plots (held-out test split)")
    test_dir = project_path(r["test_metrics"]["save_dir"])
    plots = [("confusion_matrix_normalized.png", "Confusion matrix (normalised)"), ("BoxPR_curve.png", "Precision-recall curve"),
             ("BoxF1_curve.png", "F1-confidence curve"), ("BoxP_curve.png", "Precision-confidence curve")]
    found = [(test_dir / f, t) for f, t in plots if (test_dir / f).exists()]
    if found:
        cols = st.columns(2)
        for i, (p, t) in enumerate(found):
            cols[i % 2].image(str(p), caption=t, width="stretch")
    else:
        st.info("No evaluation plots found in " + str(test_dir))
    pred_imgs = sorted(test_dir.glob("val_batch*_pred.jpg"))[:2]
    if pred_imgs:
        section("Test batch: predictions vs. labels")
        for p in pred_imgs:
            lab = p.with_name(p.name.replace("_pred", "_labels"))
            c1, c2 = st.columns(2)
            if lab.exists():
                c1.image(str(lab), caption="ground truth labels", width="stretch")
            c2.image(str(p), caption="model predictions", width="stretch")


# ===========================================================================
# 6. ABOUT
# ===========================================================================
def page_about() -> None:
    header("About Project", CFG["project"]["tagline"])
    st.markdown("""
**RoadGuard AI** is a deep-learning mini-project for automatic pothole detection and indicative
road-condition assessment from images.

#### Pipeline
1. **Dataset acquisition** – real annotated images from RDD2022 (CC BY 4.0) and the RoadEye unified pothole
   dataset (Hugging Face); Roboflow Universe projects when an API key is supplied.
2. **Pre-processing** – Pascal VOC → YOLO conversion, class normalisation, box validation, MD5 de-duplication,
   perceptual-hash grouping to prevent leakage, 80/10/10 split, resizing, `data.yaml` generation.
3. **Transfer learning** – COCO-pretrained **YOLOv8n** fine-tuned with Ultralytics / PyTorch.
4. **Inference** – letterbox → CNN forward pass → confidence filter → Non-Maximum Suppression.
5. **Severity engine** – rule layer on pothole count, box-area ratio (union), and confidence → GOOD / MODERATE / SEVERE.
6. **Reporting** – annotated image, JSON and PDF reports, detection history dashboard.

#### Technology
Python · PyTorch · Ultralytics YOLOv8 · OpenCV · NumPy · Pillow · Streamlit · Plotly · ReportLab
""")
    section("Severity thresholds (config.yaml)")
    st.json(CFG["severity"])
    st.caption(DISCLAIMER)
    section("Documentation")
    for doc in ("project_report.md", "methodology.md", "viva_questions.md"):
        p = PROJECT_ROOT / "docs" / doc
        if p.exists():
            with st.expander(doc):
                st.markdown(p.read_text(encoding="utf-8"))


ROUTES = {"Overview": page_overview, "Detect Potholes": page_detect, "Live Detection": page_live,
          "Dataset Explorer": page_dataset, "Model Performance": page_performance, "About Project": page_about}
ROUTES[page]()
