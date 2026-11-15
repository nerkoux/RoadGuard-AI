"""Shared helpers: configuration, paths, logging, hashing and hardware info."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load config.yaml (cached)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_path(key: str) -> Path:
    """Resolve a `paths.<key>` entry from config.yaml to an absolute path."""
    rel = load_config()["paths"][key]
    return (PROJECT_ROOT / rel).resolve()


def ensure_dirs() -> None:
    """Create every directory listed in config.paths (except files)."""
    for key, rel in load_config()["paths"].items():
        p = PROJECT_ROOT / rel
        (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)


def class_names(mode: str = "pothole") -> dict[int, str]:
    """Return {id: name} for 'pothole' or 'multiclass' schemas."""
    raw = load_config()["classes"][mode]
    return {int(k): str(v) for k, v in raw.items()}


def setup_logging(name: str = "roadguard", level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        logging.getLogger("roadguard").warning("Corrupt JSON in %s: %s", path, exc)
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, indent=2, default=str)
    tmp.write_text(payload, encoding="utf-8")
    # Atomic rename; on Windows the target can be briefly locked (OneDrive sync, antivirus,
    # another process reading it), so retry and finally fall back to a direct write.
    import time
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    path.write_text(payload, encoding="utf-8")
    try:
        tmp.unlink()
    except OSError:
        pass


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def hardware_info() -> dict[str, Any]:
    """Describe Python / torch / CUDA so scripts can pick sensible defaults."""
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "torch": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "disk_free_gb": round(shutil.disk_usage(PROJECT_ROOT).free / 1e9, 1),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception as exc:  # torch missing or broken CUDA install
        info["torch_error"] = str(exc)
    return info


def relpath(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)
