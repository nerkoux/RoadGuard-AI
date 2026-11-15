"""
Fine-tune YOLOv8n on the prepared road-damage dataset (transfer learning).

    python scripts/train.py                          # pothole-only, config defaults
    python scripts/train.py --epochs 30 --batch 16
    python scripts/train.py --mode multiclass        # 5 road-damage classes
    python scripts/train.py --device cpu --imgsz 416 # faster CPU run

What happens:
  1. COCO-pretrained yolov8n.pt is downloaded (backbone already knows generic
     visual features: edges, textures, shapes).
  2. The detection head is re-initialised for our classes and the whole
     network is fine-tuned on road images (transfer learning).
  3. Early stopping watches validation fitness (0.1*mAP50 + 0.9*mAP50-95).
  4. best.pt is copied to models/trained/roadguard_<mode>_best.pt and real
     validation + test metrics are written to models/trained/model_registry.json.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detector import ensure_pretrained_weights  # noqa: E402
from src.utils import get_path, hardware_info, load_config, read_json, relpath, setup_logging, write_json  # noqa: E402

log = setup_logging("train")


def auto_batch(hw: dict, imgsz: int) -> int:
    """Pick a batch size: Ultralytics AutoBatch (-1) on GPU, fixed small batch on CPU."""
    if hw["cuda_available"]:
        return -1  # AutoBatch: chooses the largest batch using ~60 % of GPU memory
    return 16 if imgsz <= 512 else 8


def auto_workers(hw: dict) -> int:
    return max(0, min(8, (hw["cpu_count"] or 2) // 2))


def pick_cache(n_images: int, imgsz: int) -> bool | str:
    """Cache decoded images in RAM only when there is clearly enough free memory."""
    try:
        import psutil
        need = n_images * imgsz * imgsz * 3 * 1.3
        return "ram" if psutil.virtual_memory().available > 2.5 * need else False
    except ImportError:
        return False


def read_results_csv(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "results.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = []
        for r in csv.DictReader(fh):
            rows.append({k.strip(): float(v) for k, v in r.items() if v not in (None, "")})
        return rows


def metrics_dict(m) -> dict:
    """Extract real metrics from an Ultralytics DetMetrics object."""
    names = m.names if hasattr(m, "names") else {}
    per_class = {}
    try:
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            per_class[names.get(int(c), str(c))] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                                    "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}
    except Exception:
        pass
    return {
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "fitness": round(float(m.fitness), 4),
        "per_class": per_class,
        "speed_ms": {k: round(float(v), 2) for k, v in (m.speed or {}).items()},
        "save_dir": relpath(m.save_dir),
    }


def main() -> int:
    cfg = load_config()
    tcfg = cfg["training"]
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8n for road damage detection")
    ap.add_argument("--mode", choices=["pothole", "multiclass"], default="pothole")
    ap.add_argument("--epochs", type=int, default=tcfg["epochs"])
    ap.add_argument("--batch", type=int, default=None, help="default: auto")
    ap.add_argument("--imgsz", type=int, default=None, help="default: 640 GPU / cpu_imgsz CPU")
    ap.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' to force CPU")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=tcfg["patience"])
    ap.add_argument("--fraction", type=float, default=None, help="fraction of train images to use")
    ap.add_argument("--model", default=tcfg["model"], help="base weights (default yolov8n.pt)")
    ap.add_argument("--finalize", action="store_true",
                    help="do not train: register + evaluate the best checkpoint of an existing (stopped) run")
    args = ap.parse_args()

    hw = hardware_info()
    log.info("Hardware: %s", hw)
    data_dir = get_path("dataset_dir" if args.mode == "pothole" else "multiclass_dataset_dir")
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        log.error("Dataset not found: %s\nRun: python scripts/download_dataset.py && python scripts/prepare_dataset.py",
                  data_yaml)
        return 1
    n_train = len(list((data_dir / "images" / "train").glob("*.jpg")))
    n_val = len(list((data_dir / "images" / "val").glob("*.jpg")))
    n_test = len(list((data_dir / "images" / "test").glob("*.jpg")))
    if n_train < 50 or n_val < 5:
        log.error("Dataset too small to train (%d train / %d val). Use inference-only mode "
                  "(streamlit run app.py) with pretrained weights instead.", n_train, n_val)
        return 1

    gpu = hw["cuda_available"] and args.device != "cpu"
    device = args.device or ("0" if gpu else "cpu")
    imgsz = args.imgsz or (tcfg["imgsz"] if gpu else tcfg["cpu_imgsz"])
    batch = args.batch or auto_batch(hw if gpu else {**hw, "cuda_available": False}, imgsz)
    workers = args.workers if args.workers is not None else auto_workers(hw)
    fraction = args.fraction or (1.0 if gpu else tcfg["cpu_fraction"])
    cache = pick_cache(int(n_train * fraction), imgsz)
    if not gpu:
        log.warning("No CUDA GPU in use - training on CPU with imgsz=%d, batch=%d. This is slower; "
                    "reduce --epochs or --imgsz if needed.", imgsz, batch)

    base = ensure_pretrained_weights() if args.model == "yolov8n.pt" else Path(args.model)
    from ultralytics import YOLO

    model = YOLO(str(base))  # COCO-pretrained backbone + neck + head
    runs = get_path("runs_dir")
    run_name = f"{args.mode}_train"
    log.info("Training %s on %s: epochs=%d imgsz=%d batch=%s device=%s workers=%d cache=%s fraction=%.2f",
             base.name, data_yaml, args.epochs, imgsz, batch, device, workers, cache, fraction)

    t0 = time.time()
    status = "completed"
    if args.finalize:
        # Training was stopped externally; reuse the epochs that actually ran.
        status = "stopped early (shortened schedule)"
        rows = read_results_csv(runs / run_name)
        if not rows:
            log.error("No results.csv in %s - nothing to finalize.", runs / run_name)
            return 1
        t0 = time.time() - rows[-1].get("time", 0)
        log.info("Finalizing existing run %s (%d epochs completed)", runs / run_name, len(rows))
    try:
        if args.finalize:
            raise StopIteration
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=imgsz, batch=batch, device=device,
            workers=workers, patience=args.patience, cache=cache, fraction=fraction,
            project=str(runs), name=run_name, exist_ok=True, seed=cfg["dataset"]["seed"],
            pretrained=True, optimizer="auto", amp=gpu, plots=True, verbose=True,
        )
    except StopIteration:
        pass
    except KeyboardInterrupt:
        status = "interrupted"
        log.warning("Training interrupted by user - keeping the best checkpoint saved so far.")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            log.error("GPU out of memory. Retry with a smaller batch, e.g. --batch 4, or --device cpu.")
        raise
    minutes = (time.time() - t0) / 60

    run_dir = runs / run_name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        log.error("No checkpoint produced - training did not complete an epoch. Nothing was saved.")
        return 1
    trained = get_path("trained_dir")
    trained.mkdir(parents=True, exist_ok=True)
    final = trained / f"roadguard_{args.mode}_best.pt"
    shutil.copy2(best, final)
    log.info("Saved fine-tuned weights -> %s", final)

    # Real evaluation of the BEST checkpoint on validation and held-out test splits.
    best_model = YOLO(str(final))
    val_m = best_model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=8, device=device,
                           project=str(runs), name=f"{args.mode}_val", exist_ok=True, plots=True, verbose=False)
    test_m = best_model.val(data=str(data_yaml), split="test", imgsz=imgsz, batch=8, device=device,
                            project=str(runs), name=f"{args.mode}_test", exist_ok=True, plots=True, verbose=False)
    history = read_results_csv(run_dir)

    registry_path = trained / "model_registry.json"
    registry = read_json(registry_path, {}) or {}
    registry[args.mode] = {
        "architecture": "YOLOv8n",
        "base_weights": base.name,
        "weights": relpath(final),
        "status": status,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_minutes": round(minutes, 1),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "imgsz": imgsz,
        "batch": batch,
        "device": "cuda:" + device if device != "cpu" else "cpu",
        "fraction": fraction,
        "classes": {int(k): v for k, v in best_model.names.items()},
        "dataset": {"train": n_train, "val": n_val, "test": n_test, "data_yaml": relpath(data_yaml)},
        "run_dir": relpath(run_dir),
        "val_metrics": metrics_dict(val_m),
        "test_metrics": metrics_dict(test_m),
    }
    write_json(registry_path, registry)

    print("\n" + "=" * 72 + f"\nTRAINING {status.upper()}  ({minutes:.1f} min, {len(history)} epochs)\n" + "=" * 72)
    for split, m in (("validation", registry[args.mode]["val_metrics"]), ("test", registry[args.mode]["test_metrics"])):
        print(f"  {split:10s}: P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"mAP50={m['mAP50']:.3f}  mAP50-95={m['mAP50-95']:.3f}")
    print(f"  weights   : {final}\n  registry  : {registry_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
