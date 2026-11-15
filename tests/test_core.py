"""Unit tests for the deterministic parts of the pipeline.

    python -m pytest tests -q        (or)        python tests/test_core.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import Stats, _parse_yolo_lines, clean_boxes, map_class_name  # noqa: E402
from src.severity import SeverityThresholds, assess_severity, union_area_fraction  # noqa: E402

T = SeverityThresholds(min_confidence=0.25, moderate_min_potholes=1, severe_min_potholes=3,
                       moderate_area_pct=1.0, severe_area_pct=8.0, severe_single_pothole_pct=5.0,
                       high_confidence=0.6)


def det(x1, y1, x2, y2, conf=0.8, cls="pothole"):
    return {"class_name": cls, "confidence": conf, "bbox": [x1, y1, x2, y2]}


def test_good_when_empty():
    r = assess_severity([], 1000, 1000, T)
    assert r["severity"] == "GOOD" and r["pothole_count"] == 0 and r["damaged_area_percentage"] == 0


def test_moderate_single_small_pothole():
    r = assess_severity([det(0, 0, 100, 100, 0.9)], 1000, 1000, T)  # 1 % area
    assert r["severity"] == "MODERATE"
    assert r["pothole_count"] == 1
    assert abs(r["damaged_area_percentage"] - 1.0) < 0.2
    assert r["average_confidence"] == 0.9 and r["maximum_confidence"] == 0.9


def test_severe_by_count():
    d = [det(i * 50, 0, i * 50 + 20, 20, 0.7) for i in range(3)]
    assert assess_severity(d, 1000, 1000, T)["severity"] == "SEVERE"


def test_severe_by_area():
    assert assess_severity([det(0, 0, 300, 300, 0.4)], 1000, 1000, T)["severity"] == "SEVERE"  # 9 %


def test_low_confidence_ignored():
    r = assess_severity([det(0, 0, 100, 100, 0.1)], 1000, 1000, T)
    assert r["severity"] == "GOOD" and r["ignored_low_confidence"] == 1


def test_union_area_not_double_counted():
    a = union_area_fraction([[0, 0, 100, 100], [0, 0, 100, 100]], 1000, 1000)
    assert abs(a - 0.01) < 0.002


def test_cracks_do_not_count_as_potholes():
    r = assess_severity([det(0, 0, 50, 50, 0.9, "longitudinal_crack")], 1000, 1000, T)
    assert r["pothole_count"] == 0 and r["other_damage_count"] == 1 and r["severity"] == "GOOD"


def test_class_mapping():
    assert map_class_name("D40") == 0 and map_class_name("D00") == 1 and map_class_name("D20") == 3
    assert map_class_name("Pothole") == 0 and map_class_name("Repair") == 4 and map_class_name("D43") is None


def test_polygon_to_box_and_validation():
    s = Stats()
    boxes = _parse_yolo_lines("0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n0 0.5 0.5\n5 0.5 0.5 0.1 0.1", {0: 0}, s)
    assert len(boxes) == 1 and np.allclose(boxes[0][1:], (0.2, 0.25, 0.2, 0.3))
    assert s.invalid_boxes_removed == 1 and s.unmapped_boxes_removed == 1
    cleaned = clean_boxes([(0, 0.99, 0.5, 0.2, 0.2), (0, 0.5, 0.5, 0.0001, 0.2)], 0.004, s)
    assert len(cleaned) == 1 and cleaned[0][1] + cleaned[0][3] / 2 <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("passed", f.__name__)
    print(f"{len(fns)} tests passed")
