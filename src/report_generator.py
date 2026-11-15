"""JSON / PDF analysis reports and the persistent detection history."""
from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils import get_path, read_json, relpath, write_json

HISTORY_LIMIT = 500


def build_report(image_name: str, result, severity: dict[str, Any], conf: float, iou: float,
                 source: str = "upload") -> dict[str, Any]:
    """Assemble the per-image analysis report from REAL inference output."""
    dets = severity["detections"]
    return {
        "report_id": uuid.uuid4().hex[:12],
        "image_filename": image_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_source": source,
        "image_size": {"width": result.width, "height": result.height},
        "model": {
            "architecture": "YOLOv8n",
            "type": result.model.kind,
            "mode": result.model.mode,
            "weights": relpath(result.model.weights),
            "classes": result.model.class_names if len(result.model.class_names) <= 10 else "COCO-80",
            "device": result.model.device,
        },
        "inference": {"confidence_threshold": conf, "iou_threshold": iou,
                      "inference_time_ms": round(result.inference_ms, 1)},
        "detection_count": len(dets),
        "detection_classes": sorted({d["class_name"] for d in dets}),
        "confidence_values": [d["confidence"] for d in dets],
        "detections": dets,
        "assessment": {k: v for k, v in severity.items() if k != "detections"},
    }


def save_outputs(report: dict[str, Any], annotated_bgr: np.ndarray) -> tuple[Path, Path]:
    """Write the annotated image and JSON report under outputs/."""
    stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{report['report_id']}"
    img_path = get_path("detections_dir") / f"{stem}.jpg"
    json_path = get_path("reports_dir") / f"{stem}.json"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), annotated_bgr)
    report["annotated_image"] = relpath(img_path)
    write_json(json_path, report)
    return img_path, json_path


def append_history(report: dict[str, Any]) -> None:
    path = get_path("history_file")
    hist = read_json(path, []) or []
    a = report["assessment"]
    hist.append({
        "report_id": report["report_id"],
        "timestamp": report["timestamp"],
        "image": report["image_filename"],
        "source": report["input_source"],
        "model": f"{report['model']['type']} ({report['model']['mode']})",
        "severity": a["severity"],
        "potholes": a["pothole_count"],
        "other_damage": a["other_damage_count"],
        "avg_conf": a["average_confidence"],
        "damaged_area_pct": a["damaged_area_percentage"],
        "inference_ms": report["inference"]["inference_time_ms"],
        "annotated_image": report.get("annotated_image"),
    })
    write_json(path, hist[-HISTORY_LIMIT:])


def load_history() -> list[dict[str, Any]]:
    return read_json(get_path("history_file"), []) or []


def clear_history() -> None:
    write_json(get_path("history_file"), [])


def build_pdf(report: dict[str, Any], annotated_bgr: np.ndarray) -> bytes | None:
    """Render a one-page PDF report. Returns None if reportlab is not installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError:
        return None

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    a = report["assessment"]
    c.setFillColor(colors.HexColor("#1b1d1f"))
    c.rect(0, H - 28 * mm, W, 28 * mm, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#c6f24e"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(15 * mm, H - 15 * mm, "ROADGUARD AI")
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 9)
    c.drawString(15 * mm, H - 21 * mm, "Road Condition Analysis Report  |  AI-generated, indicative only")

    y = H - 38 * mm
    c.setFillColor(colors.black)
    rows = [
        ("Image", report["image_filename"]),
        ("Timestamp", report["timestamp"]),
        ("Model", f"{report['model']['architecture']} {report['model']['type']} ({report['model']['mode']})"),
        ("Weights", report["model"]["weights"]),
        ("Thresholds", f"conf={report['inference']['confidence_threshold']}  IoU={report['inference']['iou_threshold']}"),
        ("Severity", a["severity"]),
        ("Potholes detected", str(a["pothole_count"])),
        ("Other damage", str(a["other_damage_count"])),
        ("Average / max confidence", f"{a['average_confidence']:.2f} / {a['maximum_confidence']:.2f}"),
        ("Damaged area", f"{a['damaged_area_percentage']:.2f} % of image"),
        ("Reasons", "; ".join(a["reasons"])),
    ]
    for k, v in rows:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(15 * mm, y, k)
        c.setFont("Helvetica", 9)
        c.drawString(60 * mm, y, str(v)[:95])
        y -= 5.5 * mm

    rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if ok:
        img = ImageReader(io.BytesIO(enc.tobytes()))
        ih, iw = annotated_bgr.shape[:2]
        max_w, max_h = W - 30 * mm, 105 * mm
        s = min(max_w / iw, max_h / ih)
        c.drawImage(img, 15 * mm, y - ih * s - 3 * mm, iw * s, ih * s)
        y = y - ih * s - 10 * mm

    c.setFont("Helvetica-Bold", 9)
    c.drawString(15 * mm, y, "#   Class                 Confidence   Box [x1, y1, x2, y2] (px)       Area %")
    c.setFont("Helvetica", 8.5)
    for i, d in enumerate(report["detections"][:18], 1):
        y -= 4.6 * mm
        c.drawString(15 * mm, y, f"{i:<3d} {d['class_name']:<22s}{d['confidence']:<13.3f}"
                                 f"{str([int(v) for v in d['bbox']]):<32s}{d['area_percentage']:.2f}")
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(15 * mm, 12 * mm, a["disclaimer"][:150])
    c.showPage()
    c.save()
    return buf.getvalue()
