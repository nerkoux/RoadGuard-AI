"""
YOLOv8 inference wrapper used by the dashboard, the webcam page and scripts.

Weight resolution order:
  1. models/trained/roadguard_<mode>_best.pt   -> "fine-tuned" (trained by scripts/train.py)
  2. models/pretrained/yolov8n.pt               -> "pretrained" (COCO, 80 classes, NO pothole class)

The app always shows which of the two is in use, so pretrained COCO inference
is never presented as pothole detection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils import get_path, load_config, read_json, setup_logging

log = setup_logging("detector")

# BGR colours per class for drawing (lime accent for potholes)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "pothole": (60, 230, 170),          # lime green (BGR)
    "longitudinal_crack": (255, 190, 80),
    "transverse_crack": (240, 120, 200),
    "alligator_crack": (80, 160, 255),
    "road_patch": (200, 200, 200),
}
DEFAULT_COLOR = (180, 180, 180)
DAMAGE_CLASSES = set(CLASS_COLORS)


@dataclass
class ModelInfo:
    weights: Path
    kind: str                 # "fine-tuned" | "pretrained"
    mode: str                 # "pothole" | "multiclass" | "coco"
    class_names: dict[int, str]
    device: str
    registry: dict[str, Any] = field(default_factory=dict)

    @property
    def detects_damage(self) -> bool:
        return any(n in DAMAGE_CLASSES for n in self.class_names.values())

    @property
    def label(self) -> str:
        return f"YOLOv8n {self.kind} ({self.mode})"


@dataclass
class DetectionResult:
    detections: list[dict[str, Any]]
    annotated: np.ndarray          # BGR image with boxes drawn
    width: int
    height: int
    inference_ms: float
    model: ModelInfo


def available_weights() -> list[tuple[str, Path]]:
    """List selectable weight files: (display name, path)."""
    out = []
    trained = get_path("trained_dir")
    for mode in ("pothole", "multiclass"):
        p = trained / f"roadguard_{mode}_best.pt"
        if p.exists():
            out.append((f"Fine-tuned YOLOv8n - {mode}", p))
    pre = get_path("pretrained_dir") / "yolov8n.pt"
    out.append(("Pretrained YOLOv8n - COCO (no pothole class)", pre))
    return out


def ensure_pretrained_weights() -> Path:
    """Download the official COCO-pretrained yolov8n.pt into models/pretrained if missing."""
    dst = get_path("pretrained_dir") / "yolov8n.pt"
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    log.info("Downloading pretrained YOLOv8n weights -> %s", dst)
    got = Path(attempt_download_asset(str(dst)))
    if got.resolve() != dst.resolve() and got.exists():
        got.replace(dst)
    if not dst.exists():
        raise FileNotFoundError("Could not download yolov8n.pt - check your internet connection.")
    return dst


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class RoadDamageDetector:
    """Loads a YOLO model once and runs inference on BGR numpy images."""

    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        from ultralytics import YOLO  # imported lazily: heavy import

        if weights is None:
            weights = next((p for _, p in available_weights() if p.exists()), None) or ensure_pretrained_weights()
        weights = Path(weights)
        if not weights.exists():
            if weights.name == "yolov8n.pt":
                weights = ensure_pretrained_weights()
            else:
                raise FileNotFoundError(f"Model weights not found: {weights}. Train with scripts/train.py "
                                        f"or use the pretrained model.")
        self.model = YOLO(str(weights))
        names = {int(k): v for k, v in self.model.names.items()}
        trained_dir = get_path("trained_dir")
        # A model is "fine-tuned" if its output classes are our road-damage classes
        # (the COCO checkpoint has 80 everyday classes and no pothole class).
        is_trained = bool(names) and set(names.values()) <= DAMAGE_CLASSES
        mode = "coco"
        if is_trained:
            mode = "multiclass" if len(names) > 1 else "pothole"
        in_registry = trained_dir in weights.resolve().parents
        registry = (read_json(trained_dir / "model_registry.json", {}) or {}).get(mode, {}) if in_registry else {}
        self.info = ModelInfo(weights=weights, kind="fine-tuned" if is_trained else "pretrained",
                              mode=mode, class_names=names, device=pick_device(device), registry=registry)
        log.info("Loaded %s from %s on %s", self.info.label, weights, self.info.device)

    # ------------------------------------------------------------------
    def predict(self, image_bgr: np.ndarray, conf: float | None = None, iou: float | None = None,
                imgsz: int | None = None) -> DetectionResult:
        if image_bgr is None or image_bgr.ndim != 3:
            raise ValueError("Invalid image: expected an HxWx3 colour image.")
        icfg = load_config()["inference"]
        conf = icfg["conf"] if conf is None else conf
        iou = icfg["iou"] if iou is None else iou
        imgsz = imgsz or icfg["imgsz"]
        h, w = image_bgr.shape[:2]

        t0 = time.perf_counter()
        try:
            # Ultralytics performs letterbox resize -> CNN forward pass -> confidence
            # filtering -> Non-Maximum Suppression (IoU threshold) internally.
            res = self.model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, device=self.info.device,
                                     max_det=icfg["max_det"], verbose=False)[0]
        except RuntimeError as exc:  # CUDA OOM / driver errors -> retry on CPU
            if self.info.device != "cpu" and ("CUDA" in str(exc) or "out of memory" in str(exc).lower()):
                log.warning("GPU inference failed (%s) - falling back to CPU", exc)
                self.info.device = "cpu"
                res = self.model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, device="cpu",
                                         max_det=icfg["max_det"], verbose=False)[0]
            else:
                raise
        ms = (time.perf_counter() - t0) * 1000

        detections: list[dict[str, Any]] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                detections.append({
                    "class_id": int(k),
                    "class_name": self.info.class_names.get(int(k), str(k)),
                    "confidence": round(float(c), 4),
                    "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                    "bbox_normalized": [round(float(x1) / w, 5), round(float(y1) / h, 5),
                                        round(float(x2) / w, 5), round(float(y2) / h, 5)],
                    "area_percentage": round(float((x2 - x1) * (y2 - y1)) / (w * h) * 100, 3),
                })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        annotated = draw_detections(image_bgr, detections)
        return DetectionResult(detections, annotated, w, h, ms, self.info)


def draw_detections(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and 'class conf' labels on a copy of the image."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    fscale = max(0.45, min(h, w) / 1100)
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = CLASS_COLORS.get(d["class_name"], DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        text = f"#{i} {d['class_name']} {d['confidence']:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
        ty = y1 - 4 if y1 - th - 8 > 0 else y1 + th + 8
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + base - 2), color, -1)
        cv2.putText(out, text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes (JPG/PNG/WEBP/BMP) to a BGR array; raise on corrupt data."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # Pillow handles some formats/edge cases OpenCV cannot (e.g. CMYK JPEG, some WEBP)
        from io import BytesIO
        from PIL import Image, UnidentifiedImageError
        try:
            pil = Image.open(BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The file is not a valid or supported image (corrupt or unknown format).") from exc
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img
