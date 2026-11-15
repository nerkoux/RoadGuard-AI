"""
Prepare the YOLO datasets from data/raw.

    python scripts/prepare_dataset.py

Outputs
    data/processed/                      unified multi-class images + labels
    data/potholes/{images,labels}/{train,val,test} + data.yaml   (class 0 = pothole)
    data/road_damage_multiclass/...      5-class variant
    data/potholes/dataset_summary.json   statistics shown in the dashboard
    data/samples/                        held-out test images for demonstrations
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import (build_processed, collect_records, copy_samples, split_items,  # noqa: E402
                               verify_dataset, write_yolo_dataset)
from src.utils import get_path, load_config, read_json, setup_logging, write_json  # noqa: E402

log = setup_logging("prepare")


def main() -> int:
    cfg = load_config()["dataset"]
    t0 = time.time()
    recs, stats = collect_records()
    if not recs:
        log.error("No annotated raw images found in data/raw. Run: python scripts/download_dataset.py")
        return 1
    log.info("Collected %d annotated raw records", len(recs))

    items = build_processed(recs, stats)
    groups = {it["dhash"] for it in items}
    stats.near_duplicate_groups = len(items) - len(groups)
    splits = split_items(items, cfg["seed"], cfg["split"])
    log.info("Split sizes: %s", {k: len(v) for k, v in splits.items()})

    manifest = read_json(get_path("raw_dir") / "sources.json", {}) or {}
    results = {}
    for mode, key in (("pothole", "dataset_dir"), ("multiclass", "multiclass_dataset_dir")):
        out_dir = get_path(key)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = write_yolo_dataset(splits, mode, out_dir, cfg["background_fraction"], cfg["seed"])
        check = verify_dataset(out_dir)
        summary.update({
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "preprocessing": asdict(stats),
            "processed_images": len(items),
            "verification": check,
            "raw_sources": manifest.get("sources", []),
            "raw_totals": manifest.get("totals", {}),
            "max_side": cfg["max_side"],
            "split_ratio": cfg["split"],
        })
        write_json(out_dir / "dataset_summary.json", summary)
        results[mode] = summary
        status = "OK" if check["ok"] else f"{len(check['problems'])} problems"
        log.info("[%s] %s -> %s  verification: %s", mode, out_dir, summary["splits"], status)
        for p in check["problems"][:10]:
            log.warning("  %s", p)

    samples = copy_samples(get_path("dataset_dir"), cfg["samples_per_split"])
    log.info("Copied %d held-out test images to data/samples", len(samples))

    print("\n" + "=" * 72 + "\nDATASET PREPARATION SUMMARY\n" + "=" * 72)
    s = asdict(stats)
    for k, v in s.items():
        print(f"  {k:28s}: {v}")
    print(f"  processed (unique, valid)   : {len(items)}")
    for mode, summ in results.items():
        print(f"\n  [{mode}]")
        for split, d in summ["splits"].items():
            print(f"    {split:5s}: {d['images']:5d} images ({d['with_objects']} with objects, "
                  f"{d['background']} background)  instances={summ['class_instances'].get(split, {})}")
    print(f"\nDone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
