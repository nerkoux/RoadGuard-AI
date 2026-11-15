"""
Dataset pre-processing: normalise every raw source into one YOLO schema,
validate annotations, remove duplicates, split without leakage and write
the final Ultralytics dataset (images/, labels/, data.yaml).

Unified multi-class schema (config.yaml -> classes.multiclass):
    0 pothole | 1 longitudinal_crack | 2 transverse_crack | 3 alligator_crack | 4 road_patch
The pothole-only dataset keeps class 0 and drops the rest.
"""
from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from src.utils import IMAGE_EXTS, class_names, get_path, load_config, md5_file, setup_logging, write_json

log = setup_logging("prepare")

# RDD2022 damage codes -> unified class id (others such as D43/D44/D50 are not road damage we model)
RDD_MAP = {"D40": 0, "D00": 1, "D10": 2, "D20": 3, "Repair": 4}


def map_class_name(name: str) -> int | None:
    """Map a free-text class name from any source to the unified schema."""
    n = name.lower().replace("-", "_").replace(" ", "_")
    if name in RDD_MAP:
        return RDD_MAP[name]
    if "pothole" in n or "pot_hole" in n:
        return 0
    if "longitudinal" in n:
        return 1
    if "transverse" in n or "lateral" in n:
        return 2
    if "alligator" in n:
        return 3
    if "patch" in n or "repair" in n:
        return 4
    return None


@dataclass
class Record:
    image: Path
    source: str
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)  # cls, xc, yc, w, h
    md5: str = ""
    dhash: str = ""


@dataclass
class Stats:
    images_found: int = 0
    corrupt_images: int = 0
    missing_annotations: int = 0
    invalid_boxes_removed: int = 0
    unmapped_boxes_removed: int = 0
    exact_duplicates_removed: int = 0
    near_duplicate_groups: int = 0


# ---------------------------------------------------------------------------
# Readers for each raw source
# ---------------------------------------------------------------------------
def _parse_yolo_lines(text: str, id_map: dict[int, int | None], stats: Stats) -> list[tuple]:
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            stats.invalid_boxes_removed += 1
            continue
        if len(coords) == 4:
            xc, yc, w, h = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:  # polygon (segmentation) -> bounding box
            xs, ys = coords[0::2], coords[1::2]
            xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            stats.invalid_boxes_removed += 1
            continue
        mapped = id_map.get(cid)
        if mapped is None:
            stats.unmapped_boxes_removed += 1
            continue
        boxes.append((mapped, xc, yc, w, h))
    return boxes


def read_roadeye(root: Path, stats: Stats) -> list[Record]:
    recs = []
    for img in sorted((root / "images").glob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        stats.images_found += 1
        lbl = root / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            stats.missing_annotations += 1  # unannotated -> never used for training
            continue
        recs.append(Record(img, "roadeye", _parse_yolo_lines(lbl.read_text(encoding="utf-8"), {0: 0}, stats)))
    return recs


def read_rdd2022(root: Path, stats: Stats) -> list[Record]:
    recs = []
    for country_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        for img in sorted((country_dir / "images").glob("*.jpg")):
            stats.images_found += 1
            xml = country_dir / "xmls" / f"{img.stem}.xml"
            if not xml.exists():
                stats.missing_annotations += 1
                continue
            try:
                tree = ET.parse(xml).getroot()
            except ET.ParseError:
                stats.missing_annotations += 1
                continue
            w = float(tree.findtext("size/width") or 0)
            h = float(tree.findtext("size/height") or 0)
            if w <= 0 or h <= 0:  # fall back to reading the image size
                im = cv2.imread(str(img))
                if im is None:
                    stats.corrupt_images += 1
                    continue
                h, w = im.shape[:2]
            boxes = []
            for obj in tree.iter("object"):
                cid = map_class_name((obj.findtext("name") or "").strip())
                if cid is None:
                    stats.unmapped_boxes_removed += 1
                    continue
                try:
                    x1 = float(obj.findtext("bndbox/xmin")); y1 = float(obj.findtext("bndbox/ymin"))
                    x2 = float(obj.findtext("bndbox/xmax")); y2 = float(obj.findtext("bndbox/ymax"))
                except (TypeError, ValueError):
                    stats.invalid_boxes_removed += 1
                    continue
                # Pascal VOC (absolute corner coords) -> YOLO (normalised centre/size)
                boxes.append((cid, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
            recs.append(Record(img, f"rdd2022_{country_dir.name}", boxes))
    return recs


def read_roboflow(root: Path, stats: Stats) -> list[Record]:
    recs = []
    for proj in sorted(p for p in root.glob("*") if p.is_dir()):
        yml = next(iter(proj.rglob("data.yaml")), None)
        if yml is None:
            continue
        names = yaml.safe_load(yml.read_text(encoding="utf-8")).get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        id_map = {i: map_class_name(str(n)) for i, n in enumerate(names)}
        for img in sorted(proj.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXTS or img.parent.name != "images":
                continue
            stats.images_found += 1
            lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
            if not lbl.exists():
                stats.missing_annotations += 1
                continue
            recs.append(Record(img, f"roboflow_{proj.name}", _parse_yolo_lines(lbl.read_text(encoding="utf-8"), id_map, stats)))
    return recs


# ---------------------------------------------------------------------------
# Validation / de-duplication helpers
# ---------------------------------------------------------------------------
def clean_boxes(boxes: Iterable[tuple], min_norm: float, stats: Stats) -> list[tuple]:
    """Clip boxes into the image and drop degenerate / out-of-range ones."""
    out = []
    for cid, xc, yc, w, h in boxes:
        x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
        x2, y2 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
        nw, nh = x2 - x1, y2 - y1
        if not all(np.isfinite([xc, yc, w, h])) or nw < min_norm or nh < min_norm:
            stats.invalid_boxes_removed += 1
            continue
        out.append((cid, (x1 + x2) / 2, (y1 + y2) / 2, nw, nh))
    return out


def dhash(img: np.ndarray, size: int = 8) -> str:
    """Perceptual difference hash - near-identical photos share the same hash."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def load_and_resize(path: Path, max_side: int) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:  # only downscale; normalised YOLO coords are unaffected by uniform resizing
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def collect_records() -> tuple[list[Record], Stats]:
    raw = get_path("raw_dir")
    stats = Stats()
    recs: list[Record] = []
    if (raw / "rdd2022").exists():
        recs += read_rdd2022(raw / "rdd2022", stats)
    if (raw / "roadeye").exists():
        recs += read_roadeye(raw / "roadeye", stats)
    if (raw / "roboflow").exists():
        recs += read_roboflow(raw / "roboflow", stats)
    return recs, stats


def build_processed(recs: list[Record], stats: Stats) -> list[dict]:
    """Validate, resize and store every record in data/processed (multiclass schema)."""
    cfg = load_config()["dataset"]
    out = get_path("processed_dir")
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)

    seen_md5: set[str] = set()
    items: list[dict] = []
    for i, r in enumerate(recs):
        r.md5 = md5_file(r.image)
        if r.md5 in seen_md5:
            stats.exact_duplicates_removed += 1
            continue
        img = load_and_resize(r.image, cfg["max_side"])
        if img is None:
            stats.corrupt_images += 1
            continue
        seen_md5.add(r.md5)
        boxes = clean_boxes(r.boxes, cfg["min_box_norm"], stats)
        name = f"{r.source}__{r.image.stem}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / f"{name}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes), encoding="utf-8"
        )
        items.append({"name": name, "source": r.source, "boxes": boxes, "dhash": dhash(img),
                      "h": img.shape[0], "w": img.shape[1]})
        if (i + 1) % 250 == 0:
            log.info("processed %d/%d", i + 1, len(recs))
    return items


def split_items(items: list[dict], seed: int, ratios: list[float]) -> dict[str, list[dict]]:
    """
    Leakage-aware split: near-duplicate images (same dHash) are grouped and a
    group always lands in ONE split. Splitting is done per source so every
    split contains a similar mix of datasets / countries.
    """
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    by_source: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_source[it["source"]][it["dhash"]].append(it)
    for src in sorted(by_source):
        groups = list(by_source[src].values())
        rng.shuffle(groups)
        n = sum(len(g) for g in groups)
        n_train, n_val = ratios[0] * n, ratios[1] * n
        acc = 0
        for g in groups:
            target = "train" if acc < n_train else "val" if acc < n_train + n_val else "test"
            splits[target] += g
            acc += len(g)
    return splits


def write_yolo_dataset(splits: dict[str, list[dict]], mode: str, out_dir: Path, bg_fraction: float,
                       seed: int) -> dict:
    """Write images/labels/{split} + data.yaml for 'pothole' or 'multiclass'."""
    names = class_names(mode)
    keep = set(names)
    processed = get_path("processed_dir")
    if out_dir.exists():
        for sub in ("images", "labels"):
            shutil.rmtree(out_dir / sub, ignore_errors=True)
    rng = random.Random(seed)
    summary = {"mode": mode, "classes": names, "splits": {}, "class_instances": {}, "sources": {}}
    class_counter: dict[str, Counter] = {}

    for split, its in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        positives = [it for it in its if any(b[0] in keep for b in it["boxes"])]
        negatives = [it for it in its if not any(b[0] in keep for b in it["boxes"])]
        rng.shuffle(negatives)
        # Background images (no target object) teach the model what is NOT a pothole,
        # but too many dilute training, so their share is capped.
        max_bg = int(bg_fraction * len(positives) / max(1e-6, 1 - bg_fraction))
        selected = positives + negatives[:max_bg]
        cc = Counter()
        src_counter = Counter()
        for it in selected:
            shutil.copy2(processed / "images" / f"{it['name']}.jpg", out_dir / "images" / split / f"{it['name']}.jpg")
            lines = [b for b in it["boxes"] if b[0] in keep]
            (out_dir / "labels" / split / f"{it['name']}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in lines), encoding="utf-8"
            )
            for b in lines:
                cc[names[b[0]]] += 1
            src_counter[it["source"]] += 1
        summary["splits"][split] = {"images": len(selected), "with_objects": len(positives),
                                    "background": min(len(negatives), max_bg)}
        class_counter[split] = cc
        summary["sources"][split] = dict(src_counter)
    summary["class_instances"] = {s: dict(c) for s, c in class_counter.items()}

    # `path` is omitted on purpose: Ultralytics then resolves paths relative to
    # this yaml file, so the project folder can be moved/copied freely.
    data_yaml = {"train": "images/train", "val": "images/val", "test": "images/test",
                 "nc": len(names), "names": {int(k): v for k, v in names.items()}}
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False)
    return summary


def verify_dataset(out_dir: Path) -> dict:
    """Check image<->label correspondence and label syntax for every split."""
    problems: list[str] = []
    counts = {}
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (out_dir / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (out_dir / "labels" / split).glob("*.txt")}
        counts[split] = len(imgs)
        problems += [f"{split}: image without label {s}" for s in sorted(imgs - lbls)[:5]]
        problems += [f"{split}: label without image {s}" for s in sorted(lbls - imgs)[:5]]
        for lp in (out_dir / "labels" / split).glob("*.txt"):
            for ln in lp.read_text(encoding="utf-8").splitlines():
                v = ln.split()
                if len(v) != 5 or not all(0.0 <= float(x) <= 1.0 for x in v[1:]):
                    problems.append(f"{split}: bad line in {lp.name}: {ln}")
    # leakage check: no file name appears in two splits
    names = {s: {p.stem for p in (out_dir / "images" / s).glob("*.jpg")} for s in counts}
    overlap = (names["train"] & names["val"]) | (names["train"] & names["test"]) | (names["val"] & names["test"])
    if overlap:
        problems.append(f"{len(overlap)} images appear in more than one split")
    return {"counts": counts, "problems": problems, "ok": not problems}


def copy_samples(dataset_dir: Path, n: int) -> list[str]:
    """Copy held-out TEST images (never seen in training) to data/samples for demos."""
    samples = get_path("samples_dir")
    samples.mkdir(parents=True, exist_ok=True)
    for old in samples.glob("*"):
        if old.is_file():
            old.unlink()
    test_imgs = sorted((dataset_dir / "images" / "test").glob("*.jpg"))
    rng = random.Random(7)
    def largest_box(p: Path) -> float:
        areas = [float(v[3]) * float(v[4]) for v in (ln.split() for ln in
                 (dataset_dir / "labels" / "test" / f"{p.stem}.txt").read_text().splitlines()) if len(v) == 5]
        return max(areas, default=0.0)

    # Demo samples: test images whose largest ANNOTATED pothole covers >= 1 % of the frame
    # (clearly visible to a viewer). Selection uses ground truth only, never model output.
    with_obj = [p for p in test_imgs if largest_box(p) >= 0.01]
    without = [p for p in test_imgs if largest_box(p) == 0.0]  # true background images
    rng.shuffle(with_obj)
    rng.shuffle(without)
    # prefer a mix of sources and pothole counts for a varied demo
    by_src: dict[str, list[Path]] = defaultdict(list)
    for p in with_obj:
        by_src[p.name.split("__")[0]].append(p)
    picked: list[Path] = []
    while len(picked) < n - 2 and any(by_src.values()):
        for src in sorted(by_src):
            if by_src[src] and len(picked) < n - 2:
                picked.append(by_src[src].pop())
    picked += without[:2]
    out = []
    for i, p in enumerate(picked, 1):
        dst = samples / f"sample_{i:02d}_{p.name.split('__')[0]}.jpg"
        shutil.copy2(p, dst)
        shutil.copy2(dataset_dir / "labels" / "test" / f"{p.stem}.txt", dst.with_suffix(".txt"))
        out.append(dst.name)
    return out
