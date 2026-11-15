"""
Road condition severity engine.

Turns raw YOLO detections into an INDICATIVE road-condition class
(GOOD / MODERATE / SEVERE). It is a transparent rule-based layer on top of
the neural network - not a certified engineering assessment.

Inputs considered (all computed from the actual detections):
  * number of potholes above the confidence floor
  * area of every bounding box relative to the image
  * combined damaged-area ratio (union of boxes, so overlaps are not double counted)
  * detection confidence (low-confidence boxes are ignored; the damage index is
    confidence-weighted)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

DISCLAIMER = ("AI-generated indicative classification based on visible surface damage in a single image. "
              "It is not a certified engineering or structural assessment.")

POTHOLE_CLASS = "pothole"


@dataclass
class SeverityThresholds:
    min_confidence: float = 0.25
    moderate_min_potholes: int = 1
    severe_min_potholes: int = 3
    moderate_area_pct: float = 1.0
    severe_area_pct: float = 8.0
    severe_single_pothole_pct: float = 5.0
    high_confidence: float = 0.60

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> "SeverityThresholds":
        if cfg is None:
            from src.utils import load_config
            cfg = load_config().get("severity", {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


def union_area_fraction(boxes: Sequence[Sequence[float]], width: int, height: int, grid: int = 400) -> float:
    """
    Fraction of the image covered by the UNION of boxes.
    Boxes are rasterised on a coarse grid so overlapping boxes are counted once.
    """
    if not boxes or width <= 0 or height <= 0:
        return 0.0
    gw = grid
    gh = max(1, round(grid * height / width))
    mask = np.zeros((gh, gw), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        c1, c2 = int(np.floor(x1 / width * gw)), int(np.ceil(x2 / width * gw))
        r1, r2 = int(np.floor(y1 / height * gh)), int(np.ceil(y2 / height * gh))
        mask[max(0, r1):min(gh, r2), max(0, c1):min(gw, c2)] = True
    return float(mask.mean())


def assess_severity(detections: list[dict[str, Any]], image_width: int, image_height: int,
                    thresholds: SeverityThresholds | None = None) -> dict[str, Any]:
    """
    Parameters
    ----------
    detections : list of dicts with keys class_name, confidence, bbox ([x1, y1, x2, y2] in pixels)
    image_width, image_height : size of the analysed image in pixels

    Returns the severity report dict (see README / project brief for the schema).
    """
    t = thresholds or SeverityThresholds.from_config()
    img_area = float(image_width * image_height) or 1.0

    valid = [d for d in detections if d["confidence"] >= t.min_confidence]
    potholes = [d for d in valid if d["class_name"] == POTHOLE_CLASS]
    others = [d for d in valid if d["class_name"] != POTHOLE_CLASS]

    def box_pct(d: dict) -> float:
        x1, y1, x2, y2 = d["bbox"]
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / img_area * 100

    confs = [d["confidence"] for d in potholes]
    damaged_pct = union_area_fraction([d["bbox"] for d in valid], image_width, image_height) * 100
    pothole_pct = union_area_fraction([d["bbox"] for d in potholes], image_width, image_height) * 100
    largest_pothole_pct = max((box_pct(d) for d in potholes), default=0.0)
    n = len(potholes)

    # ---- rule-based classification (thresholds live in config.yaml) -----------
    reasons: list[str] = []
    severity = "GOOD"
    if n >= t.severe_min_potholes:
        severity = "SEVERE"
        reasons.append(f"{n} potholes detected (>= {t.severe_min_potholes})")
    if damaged_pct >= t.severe_area_pct:
        severity = "SEVERE"
        reasons.append(f"combined damaged area {damaged_pct:.1f}% >= {t.severe_area_pct}%")
    big = [d for d in potholes if box_pct(d) >= t.severe_single_pothole_pct and d["confidence"] >= t.high_confidence]
    if big:
        severity = "SEVERE"
        reasons.append(f"large high-confidence pothole covering {max(box_pct(d) for d in big):.1f}% of the frame")
    if severity != "SEVERE":
        if n >= t.moderate_min_potholes:
            severity = "MODERATE"
            reasons.append(f"{n} pothole(s) detected")
        if damaged_pct >= t.moderate_area_pct:
            severity = "MODERATE"
            reasons.append(f"damaged area {damaged_pct:.1f}% >= {t.moderate_area_pct}%")
        elif others and severity == "GOOD":
            reasons.append(f"{len(others)} minor surface defect(s) below the area threshold")
    if severity == "GOOD" and not reasons:
        reasons.append("no potholes or significant damage detected above the confidence threshold")

    # Damage index (0-100): a derived, confidence-weighted summary for charts.
    #   (100 / severe_min_potholes) points per confidence-weighted pothole
    #   + 4 points per % of damaged area, capped at 100.
    damage_index = min(100.0, 100.0 / max(1, t.severe_min_potholes) * sum(confs) + 4.0 * damaged_pct)

    return {
        "severity": severity,
        "pothole_count": n,
        "other_damage_count": len(others),
        "average_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
        "maximum_confidence": round(float(np.max(confs)), 4) if confs else 0.0,
        "damaged_area_percentage": round(damaged_pct, 2),
        "pothole_area_percentage": round(pothole_pct, 2),
        "largest_pothole_percentage": round(largest_pothole_pct, 2),
        "damage_index": round(damage_index, 1),
        "ignored_low_confidence": len(detections) - len(valid),
        "reasons": reasons,
        "thresholds": asdict(t),
        "disclaimer": DISCLAIMER,
        "detections": valid,
    }
