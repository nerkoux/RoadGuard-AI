"""
Evaluate a YOLO model on a dataset split and print REAL metrics.

    python scripts/evaluate.py                       # fine-tuned pothole model on test split
    python scripts/evaluate.py --split val
    python scripts/evaluate.py --weights models/pretrained/yolov8n.pt   # baseline (COCO, expected ~0)
    python scripts/evaluate.py --mode multiclass

Also runs the severity engine over the test images and reports the
distribution of predicted road conditions vs. the condition implied by
ground-truth boxes (a sanity check of the rule layer).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_path, read_json, relpath, setup_logging, write_json  # noqa: E402

log = setup_logging("evaluate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pothole", "multiclass"], default="pothole")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--severity-check", action="store_true", default=True)
    args = ap.parse_args()

    from scripts.train import metrics_dict
    from src.detector import RoadDamageDetector, pick_device
    from src.severity import assess_severity

    data_dir = get_path("dataset_dir" if args.mode == "pothole" else "multiclass_dataset_dir")
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        log.error("Dataset missing (%s). Run prepare_dataset.py first.", data_yaml)
        return 1
    weights = Path(args.weights) if args.weights else get_path("trained_dir") / f"roadguard_{args.mode}_best.pt"
    if not weights.exists():
        log.error("Weights not found: %s - train first (python scripts/train.py) or pass --weights.", weights)
        return 1
    registry = read_json(get_path("trained_dir") / "model_registry.json", {}) or {}
    imgsz = args.imgsz or registry.get(args.mode, {}).get("imgsz", 640)
    device = pick_device(args.device)

    det = RoadDamageDetector(weights, device=device)
    if not det.info.detects_damage:
        log.warning("These weights have no road-damage classes (COCO pretrained). mAP for 'pothole' "
                    "cannot be computed meaningfully; skipping box metrics.")
    else:
        m = det.model.val(data=str(data_yaml), split=args.split, imgsz=imgsz, batch=8, device=device,
                          project=str(get_path("runs_dir")), name=f"{args.mode}_eval_{args.split}",
                          exist_ok=True, plots=True, verbose=False)
        md = metrics_dict(m)
        print(f"\n[{weights.name} on {args.split}]  P={md['precision']:.3f}  R={md['recall']:.3f}  "
              f"mAP50={md['mAP50']:.3f}  mAP50-95={md['mAP50-95']:.3f}")
        for cname, cm in md["per_class"].items():
            print(f"   {cname:20s} P={cm['precision']:.3f} R={cm['recall']:.3f} mAP50={cm['mAP50']:.3f}")
        write_json(get_path("runs_dir") / f"{args.mode}_eval_{args.split}" / "metrics.json", md)

    # --- severity sanity check over the split ------------------------------
    import cv2
    from src.dataset_manager import read_labels
    names = det.info.class_names
    pred_c, gt_c, agree, total = Counter(), Counter(), 0, 0
    for img_path in sorted((data_dir / "images" / args.split).glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = det.predict(img, imgsz=imgsz)
        pred = assess_severity(res.detections, w, h)["severity"]
        gt_boxes = read_labels(args.mode, args.split, img_path) or []
        gt_dets = [{"class_name": names.get(c, str(c)) if det.info.detects_damage else "pothole",
                    "confidence": 1.0,
                    "bbox": [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]}
                   for c, x, y, bw, bh in gt_boxes]
        gt = assess_severity(gt_dets, w, h)["severity"]
        pred_c[pred] += 1
        gt_c[gt] += 1
        agree += pred == gt
        total += 1
    if total:
        print(f"\nSeverity on {total} {args.split} images (rule layer applied to predictions vs ground truth):")
        print(f"   predicted   : {dict(pred_c)}")
        print(f"   ground truth: {dict(gt_c)}")
        print(f"   agreement   : {agree / total * 100:.1f}%")
        write_json(get_path("runs_dir") / f"{args.mode}_eval_{args.split}" / "severity_check.json",
                   {"predicted": dict(pred_c), "ground_truth": dict(gt_c), "agreement": round(agree / total, 4),
                    "images": total, "weights": relpath(weights)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
