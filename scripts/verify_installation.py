"""
Check that the environment can run RoadGuard AI.

    python scripts/verify_installation.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
problems = 0


def check(label: str, fn) -> None:
    global problems
    try:
        status, msg = fn()
    except Exception as exc:
        status, msg = FAIL, f"{type(exc).__name__}: {exc}"
    if status == FAIL:
        problems += 1
    print(f"{status} {label:28s} {msg}")


def python_version():
    v = sys.version_info
    return (OK if v >= (3, 9) else FAIL), f"{sys.version.split()[0]} ({sys.executable})"


def module(name: str, attr: str = "__version__"):
    def _f():
        m = importlib.import_module(name)
        return OK, str(getattr(m, attr, "installed"))
    return _f


def torch_device():
    import torch
    if torch.cuda.is_available():
        return OK, f"CUDA {torch.version.cuda} · {torch.cuda.get_device_name(0)}"
    hint = ""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            hint = f" - NVIDIA GPU present ({out.splitlines()[0]}) but torch is a CPU build; see README to enable CUDA"
    except Exception:
        pass
    return WARN, "CPU only" + hint


def weights():
    from src.detector import ensure_pretrained_weights
    from src.utils import get_path
    p = ensure_pretrained_weights()
    trained = sorted(get_path("trained_dir").glob("roadguard_*_best.pt"))
    msg = f"pretrained {p.name} ({p.stat().st_size / 1e6:.1f} MB)"
    msg += f" · fine-tuned: {', '.join(t.name for t in trained)}" if trained else " · no fine-tuned weights yet"
    return (OK if trained else WARN), msg


def dataset():
    from src.dataset_manager import count_split_files, load_summary
    c = count_split_files("pothole")
    if not sum(c.values()):
        return WARN, "not prepared - run download_dataset.py and prepare_dataset.py"
    s = load_summary("pothole")
    ok = s and s["verification"]["ok"]
    return (OK if ok else WARN), f"{c} (verification {'passed' if ok else 'has issues'})"


def inference():
    import numpy as np
    from src.detector import RoadDamageDetector
    from src.severity import assess_severity
    from src.utils import get_path, list_images
    import cv2
    det = RoadDamageDetector()
    samples = list_images(get_path("samples_dir"))
    img = cv2.imread(str(samples[0])) if samples else np.full((480, 640, 3), 120, np.uint8)
    res = det.predict(img)
    sev = assess_severity(res.detections, res.width, res.height)
    return OK, f"{det.info.label}: {len(res.detections)} detections in {res.inference_ms:.0f} ms -> {sev['severity']}"


def disk():
    import shutil
    free = shutil.disk_usage(Path(__file__).resolve().parent.parent).free / 1e9
    return (OK if free > 2 else WARN), f"{free:.1f} GB free"


if __name__ == "__main__":
    print("RoadGuard AI - installation check\n" + "-" * 60)
    check("Python", python_version)
    for name in ("torch", "torchvision", "ultralytics", "cv2", "numpy", "PIL", "streamlit", "plotly", "pandas", "yaml"):
        check(name, module(name))
    check("reportlab (PDF, optional)", module("reportlab", "Version"))
    check("Compute device", torch_device)
    check("Disk space", disk)
    check("Model weights", weights)
    check("Dataset", dataset)
    check("End-to-end inference", inference)
    print("-" * 60)
    print("All required checks passed." if not problems else f"{problems} check(s) failed.")
    sys.exit(1 if problems else 0)
