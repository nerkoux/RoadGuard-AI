"""Read-only access to the prepared datasets for the dashboard (no fabricated numbers)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.detector import CLASS_COLORS, DEFAULT_COLOR
from src.utils import class_names, get_path, read_json

SPLITS = ("train", "val", "test")


def dataset_dir(mode: str = "pothole") -> Path:
    return get_path("dataset_dir" if mode == "pothole" else "multiclass_dataset_dir")


def load_summary(mode: str = "pothole") -> dict[str, Any] | None:
    """Summary written by scripts/prepare_dataset.py, or None if not prepared yet."""
    return read_json(dataset_dir(mode) / "dataset_summary.json", None)


def load_raw_manifest() -> dict[str, Any] | None:
    return read_json(get_path("raw_dir") / "sources.json", None)


def count_split_files(mode: str = "pothole") -> dict[str, int]:
    """Count images on disk per split (works even without a summary file)."""
    d = dataset_dir(mode)
    return {s: len(list((d / "images" / s).glob("*.jpg"))) for s in SPLITS}


def list_images(mode: str, split: str) -> list[Path]:
    return sorted((dataset_dir(mode) / "images" / split).glob("*.jpg"))


def read_labels(mode: str, split: str, image: Path) -> list[tuple[int, float, float, float, float]] | None:
    """Return YOLO boxes, [] for a background image, or None if the label file is missing."""
    lbl = dataset_dir(mode) / "labels" / split / f"{image.stem}.txt"
    if not lbl.exists():
        return None
    boxes = []
    for line in lbl.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) == 5:
            boxes.append((int(v[0]), *map(float, v[1:])))
    return boxes


def draw_ground_truth(image: Path, boxes: list[tuple], mode: str) -> np.ndarray:
    """Draw human-annotated (ground-truth) boxes - dashed look via thinner lines."""
    img = cv2.imread(str(image))
    if img is None:
        raise ValueError(f"Cannot read {image}")
    names = class_names(mode)
    h, w = img.shape[:2]
    for cid, xc, yc, bw, bh in boxes:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        name = names.get(cid, str(cid))
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"GT {name}", (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def source_of(image: Path) -> str:
    return image.name.split("__")[0]
