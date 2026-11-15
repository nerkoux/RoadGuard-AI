"""
RoadGuard AI - dataset acquisition.

Downloads REAL, ANNOTATED road-damage images from public sources:

  A. RDD2022 (sekilab / CRDDC 2022, CC BY 4.0) - Pascal VOC XML, multi-class.
     The official release is a single 13 GB archive on Figshare. Instead of
     storing all 13 GB we read the zip's central directory over HTTP range
     requests and fetch only the selected images + XML annotations.
  B/C. Roboflow Universe pothole / road-damage projects (YOLOv8 export).
     Requires a free API key in the ROBOFLOW_API_KEY environment variable;
     skipped with an explanation if absent. No auth is bypassed.
  D. Hugging Face "AkumaDachi/RoadEye_Yolo_Aug" - YOLO format, pothole only.
     Only original images are taken (files ending in "_aug" are augmented
     copies of other images and would cause train/test leakage).

Every image is kept ONLY together with its annotation file. Exact duplicate
images (MD5) are skipped. A manifest of sources, licences and counts is
written to data/raw/sources.json.

Usage:
    python scripts/download_dataset.py                # all sources
    python scripts/download_dataset.py --sources roadeye
    python scripts/download_dataset.py --roadeye-max 500 --rdd-scale 0.5
"""
from __future__ import annotations

import argparse
import io
import os
import random
import re
import struct
import sys
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_path, load_config, md5_bytes, md5_file, setup_logging, write_json  # noqa: E402

log = setup_logging("download")
USER_AGENT = "RoadGuardAI-dataset-downloader/1.0 (academic project)"

RDD_FIGSHARE_URL = "https://ndownloader.figshare.com/files/38030910"
HF_REPO = "AkumaDachi/RoadEye_Yolo_Aug"
HF_SUBDIR = "RoadEyeUnified_SingleClass_Aug"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
@dataclass
class SourceReport:
    key: str
    name: str
    url: str
    license: str
    citation: str
    format: str
    status: str = "pending"
    images_downloaded: int = 0
    images_annotated: int = 0
    background_images: int = 0
    label_instances: int = 0
    unannotated_skipped: int = 0
    duplicates_skipped: int = 0
    failed: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    return s


def get_with_retry(session: requests.Session, url: str, retries: int, timeout: int, **kw) -> requests.Response:
    """GET with exponential backoff. Raises the last error if all attempts fail."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout, **kw)
            if r.status_code == 429:  # respect rate limiting
                wait = int(r.headers.get("Retry-After", 2 ** (attempt + 2)))
                log.warning("Rate limited, waiting %ss", wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            return r
        except (requests.RequestException, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} attempts: {url} ({last})")


class DedupIndex:
    """Thread-safe MD5 index of every image already stored in data/raw."""

    def __init__(self, raw_dir: Path):
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        for p in raw_dir.rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and p.is_file():
                self._seen.add(md5_file(p))

    def add(self, digest: str) -> bool:
        """Return True if new, False if an exact duplicate."""
        with self._lock:
            if digest in self._seen:
                return False
            self._seen.add(digest)
            return True


# ---------------------------------------------------------------------------
# D. Hugging Face RoadEye
# ---------------------------------------------------------------------------
def hf_list_files(session: requests.Session, path: str, timeout: int) -> list[str]:
    """List files in a Hugging Face dataset folder (handles pagination)."""
    url: str | None = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/{path}?limit=1000"
    files: list[str] = []
    while url:
        r = get_with_retry(session, url, 3, timeout)
        if r.status_code != 200:
            raise RuntimeError(f"HF listing failed ({r.status_code}) for {path}")
        payload = r.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected HF listing for {path}: {str(payload)[:200]}")
        files += [it["path"] for it in payload if it.get("type") == "file"]
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url = m.group(1) if m else None
    return files


def download_roadeye(cfg: dict, out_root: Path, dedup: DedupIndex, max_images: int) -> SourceReport:
    rep = SourceReport(
        key="roadeye",
        name="RoadEye Unified Pothole Dataset (Hugging Face)",
        url=f"https://huggingface.co/datasets/{HF_REPO}",
        license="Mixed / 'unknown' on HF card: aggregates Roboflow, Mendeley, Kaggle and CC-BY-4.0 sources "
        "(see individual source licences). Used here for non-commercial academic purposes only.",
        citation="AkumaDachi, RoadEye_Yolo_Aug, Hugging Face Datasets, 2026.",
        format="YOLO txt (class 0 = pothole)",
    )
    dl = cfg["download"]
    session = make_session()
    img_dir, lbl_dir = out_root / "images", out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1) Enumerate ORIGINAL images (drop *_aug files to avoid leakage).
    candidates: list[tuple[str, str]] = []  # (split, filename)
    for split in ("train", "val", "test"):
        try:
            names = hf_list_files(session, f"{HF_SUBDIR}/images/{split}", dl["timeout_s"])
        except RuntimeError as exc:
            rep.notes.append(f"Could not list split '{split}': {exc}")
            log.warning("RoadEye: %s", exc)
            continue
        for full in names:
            fname = full.rsplit("/", 1)[-1]
            if "_aug" in Path(fname).stem:
                continue
            candidates.append((split, fname))
    log.info("RoadEye: %d original images listed", len(candidates))
    if not candidates:
        rep.status = "failed"
        return rep

    # 2) Stratified sample over sub-source prefixes (kg_ / mwpd_ / rf_).
    rng = random.Random(cfg["dataset"]["seed"])
    by_prefix: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for c in candidates:
        by_prefix[c[1].split("_")[0]].append(c)
    chosen: list[tuple[str, str]] = []
    total = len(candidates)
    for prefix, items in sorted(by_prefix.items()):
        rng.shuffle(items)
        quota = max(1, round(max_images * len(items) / total))
        chosen += items[:quota]
        rep.notes.append(f"sub-source '{prefix}': {len(items)} available, {min(quota, len(items))} selected")
    chosen = chosen[:max_images]

    base = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_SUBDIR}"
    lock = threading.Lock()

    def fetch(item: tuple[str, str]) -> None:
        split, fname = item
        stem = Path(fname).stem
        dst_img, dst_lbl = img_dir / fname, lbl_dir / f"{stem}.txt"
        if dst_img.exists() and dst_lbl.exists():  # resume support
            _count_label(dst_lbl.read_text(encoding="utf-8"), rep, lock, {0: "pothole"})
            return
        try:
            lr = get_with_retry(session, f"{base}/labels/{split}/{stem}.txt", dl["retries"], dl["timeout_s"])
            if lr.status_code == 404:
                with lock:
                    rep.unannotated_skipped += 1  # never train on an image without labels
                return
            lr.raise_for_status()
            ir = get_with_retry(session, f"{base}/images/{split}/{fname}", dl["retries"], dl["timeout_s"])
            ir.raise_for_status()
            if not dedup.add(md5_bytes(ir.content)):
                with lock:
                    rep.duplicates_skipped += 1
                return
            dst_img.write_bytes(ir.content)
            dst_lbl.write_text(lr.text, encoding="utf-8")
            _count_label(lr.text, rep, lock, {0: "pothole"})
        except Exception as exc:  # keep going on individual failures
            with lock:
                rep.failed += 1
            log.debug("RoadEye fetch failed for %s: %s", fname, exc)

    _run_parallel(fetch, chosen, dl["workers"], "RoadEye")
    rep.status = "ok" if rep.images_downloaded else "failed"
    return rep


def _count_label(text: str, rep: SourceReport, lock: threading.Lock, names: dict[int, str]) -> None:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    with lock:
        rep.images_downloaded += 1
        if lines:
            rep.images_annotated += 1
        else:
            rep.background_images += 1
        rep.label_instances += len(lines)
        for ln in lines:
            try:
                cname = names.get(int(float(ln.split()[0])), "other")
            except (ValueError, IndexError):
                cname = "invalid"
            rep.class_counts[cname] = rep.class_counts.get(cname, 0) + 1


def _run_parallel(fn, items: list, workers: int, label: str) -> None:
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 100 == 0 or done == len(items):
                log.info("%s: %d/%d processed", label, done, len(items))


# ---------------------------------------------------------------------------
# A. RDD2022 via HTTP range requests on the official archive
# ---------------------------------------------------------------------------
class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file object backed by HTTP Range requests."""

    def __init__(self, url: str, session: requests.Session, base: int = 0, size: int | None = None):
        self.session = session
        r = session.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
        if r.status_code != 206:
            raise RuntimeError(f"Server does not support range requests (HTTP {r.status_code})")
        self.url = r.url  # resolved (signed) URL
        self.total = int(r.headers["Content-Range"].split("/")[1])
        self.base = base
        self.size = size if size is not None else self.total
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def fetch(self, start: int, length: int) -> bytes:
        """Fetch absolute bytes [base+start, base+start+length)."""
        a = self.base + start
        for attempt in range(4):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={a}-{a + length - 1}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == length:
                    return r.content
                if r.status_code in (401, 403):  # signed URL expired -> refresh it
                    self.url = self.session.get(RDD_FIGSHARE_URL, headers={"Range": "bytes=0-0"}, timeout=60).url
                raise RuntimeError(f"HTTP {r.status_code}, got {len(r.content)}/{length} bytes")
            except Exception as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Range fetch failed at {a}: {last}")

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.fetch(self.pos, n)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _local_data_offset(f: HttpRangeFile, header_offset: int) -> int:
    hdr = f.fetch(header_offset, 30)
    sig, *_ = struct.unpack("<I", hdr[:4])
    if sig != 0x04034B50:
        raise RuntimeError("Bad local zip header")
    n_len, e_len = struct.unpack("<HH", hdr[26:30])
    return header_offset + 30 + n_len + e_len


def _extract_members(f: HttpRangeFile, infos: list[zipfile.ZipInfo], workers: int,
                     max_chunk: int = 8 << 20, max_gap: int = 512 << 10) -> dict[str, bytes]:
    """Fetch many zip members with few HTTP requests by grouping nearby members."""
    infos = sorted(infos, key=lambda i: i.header_offset)
    chunks: list[list[zipfile.ZipInfo]] = []
    for info in infos:
        if chunks:
            first, last = chunks[-1][0], chunks[-1][-1]
            end = last.header_offset + 30 + len(last.filename.encode()) + 1024 + last.compress_size
            if info.header_offset - end < max_gap and info.header_offset + info.compress_size - first.header_offset < max_chunk:
                chunks[-1].append(info)
                continue
        chunks.append([info])

    out: dict[str, bytes] = {}
    lock = threading.Lock()

    def get_chunk(chunk: list[zipfile.ZipInfo]) -> None:
        start = chunk[0].header_offset
        last = chunk[-1]
        end = last.header_offset + 30 + len(last.filename.encode()) + 2048 + last.compress_size
        end = min(end, f.size)
        blob = f.fetch(start, end - start)
        for info in chunk:
            rel = info.header_offset - start
            n_len, e_len = struct.unpack("<HH", blob[rel + 26: rel + 30])
            d0 = rel + 30 + n_len + e_len
            raw = blob[d0: d0 + info.compress_size]
            if info.compress_type == zipfile.ZIP_DEFLATED:
                data = zlib.decompressobj(-15).decompress(raw)
            elif info.compress_type == zipfile.ZIP_STORED:
                data = raw
            else:
                continue
            if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
                log.debug("CRC mismatch for %s", info.filename)
                continue
            with lock:
                out[info.filename] = data

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(get_chunk, c) for c in chunks]):
            try:
                fut.result()
            except Exception as exc:
                log.warning("RDD chunk failed: %s", exc)
    return out


def parse_voc(xml_bytes: bytes) -> list[str]:
    """Return list of object class names in a Pascal VOC annotation."""
    root = ET.fromstring(xml_bytes)
    return [(o.findtext("name") or "").strip() for o in root.iter("object")]


def download_rdd2022(cfg: dict, out_root: Path, dedup: DedupIndex, scale: float) -> SourceReport:
    rep = SourceReport(
        key="rdd2022",
        name="RDD2022 - Multi-national Road Damage Dataset (CRDDC 2022)",
        url="https://github.com/sekilab/RoadDamageDetector  |  https://doi.org/10.6084/m9.figshare.21431547",
        license="CC BY 4.0",
        citation="Arya D. et al., 'RDD2022: A multi-national image dataset for automatic Road Damage "
        "Detection', Geoscience Data Journal, 2024. arXiv:2209.08538",
        format="Pascal VOC XML (D00 longitudinal, D10 transverse, D20 alligator, D40 pothole, Repair)",
    )
    dl = cfg["download"]
    session = make_session()
    rng = random.Random(cfg["dataset"]["seed"])
    try:
        outer_f = HttpRangeFile(RDD_FIGSHARE_URL, session)
        outer = zipfile.ZipFile(io.BufferedReader(outer_f, buffer_size=256 << 10))
    except Exception as exc:
        rep.status = "failed"
        rep.notes.append(f"Could not open the Figshare archive: {exc}")
        log.error("RDD2022 unavailable: %s", exc)
        return rep
    rep.notes.append(f"Official archive size: {outer_f.total / 1e9:.1f} GB - only selected files fetched")
    country_infos = {Path(i.filename).stem: i for i in outer.infolist() if i.filename.endswith(".zip")}

    for country, cap in cfg["download"]["rdd2022"]["countries"].items():
        cap = max(1, int(cap * scale))
        info = country_infos.get(country)
        if info is None or info.compress_type != zipfile.ZIP_STORED:
            rep.notes.append(f"{country}: not found / not seekable in archive")
            continue
        try:
            start = _local_data_offset(outer_f, info.header_offset)
            inner_f = HttpRangeFile(RDD_FIGSHARE_URL, session, base=start, size=info.file_size)
            inner = zipfile.ZipFile(io.BufferedReader(inner_f, buffer_size=512 << 10))
            members = {i.filename: i for i in inner.infolist()}
            xml_infos = [i for n, i in members.items() if "/train/annotations/xmls/" in n and n.endswith(".xml")]
            log.info("RDD2022 %s: %d train annotations, fetching XMLs...", country, len(xml_infos))
            xmls = _extract_members(inner_f, xml_infos, dl["workers"])

            # Index images by damage classes so potholes are prioritised.
            img_index = {Path(n).stem: n for n in members if "/train/images/" in n and n.lower().endswith(".jpg")}
            with_pothole, with_damage, background = [], [], []
            parsed: dict[str, list[str]] = {}
            xml_by_stem = {Path(n).stem: n for n in xmls}
            for xml_name, data in xmls.items():
                stem = Path(xml_name).stem
                if stem not in img_index:
                    continue  # annotation without image
                try:
                    objs = parse_voc(data)
                except ET.ParseError:
                    continue
                parsed[stem] = objs
                if "D40" in objs:
                    with_pothole.append(stem)
                elif any(o in ("D00", "D10", "D20", "Repair") for o in objs):
                    with_damage.append(stem)
                else:
                    background.append(stem)
            for lst in (with_pothole, with_damage, background):
                rng.shuffle(lst)
            n_bg = max(0, int(cap * 0.05))
            chosen = with_pothole[:cap]
            chosen += with_damage[: max(0, cap - len(chosen) - n_bg)]
            chosen += background[: max(0, cap - len(chosen))]
            rep.notes.append(
                f"{country}: {len(parsed)} annotated train images ({len(with_pothole)} with potholes) -> {len(chosen)} selected"
            )

            cdir = out_root / country
            (cdir / "images").mkdir(parents=True, exist_ok=True)
            (cdir / "xmls").mkdir(parents=True, exist_ok=True)
            todo = [s for s in chosen if not (cdir / "images" / f"{s}.jpg").exists()]
            imgs = _extract_members(inner_f, [members[img_index[s]] for s in todo], dl["workers"]) if todo else {}
            for stem in chosen:
                dst = cdir / "images" / f"{stem}.jpg"
                if not dst.exists():
                    data = imgs.get(img_index[stem])
                    if data is None:
                        rep.failed += 1
                        continue
                    if not dedup.add(md5_bytes(data)):
                        rep.duplicates_skipped += 1
                        continue
                    dst.write_bytes(data)
                    (cdir / "xmls" / f"{stem}.xml").write_bytes(xmls[xml_by_stem[stem]])
                objs = parsed[stem]
                rep.images_downloaded += 1
                rep.images_annotated += 1 if objs else 0
                rep.background_images += 0 if objs else 1
                rep.label_instances += len(objs)
                for o in objs:
                    rep.class_counts[o] = rep.class_counts.get(o, 0) + 1
            log.info("RDD2022 %s: stored %d images", country, len(chosen))
        except Exception as exc:
            rep.notes.append(f"{country}: failed - {exc}")
            log.error("RDD2022 %s failed: %s", country, exc)
    rep.status = "ok" if rep.images_downloaded else "failed"
    return rep


# ---------------------------------------------------------------------------
# B / C. Roboflow Universe (API key required)
# ---------------------------------------------------------------------------
def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract a zip while preventing path traversal ('zip slip')."""
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Unsafe path in archive: {member.filename}")
    zf.extractall(dest)


def download_roboflow(cfg: dict, out_root: Path) -> list[SourceReport]:
    reports = []
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    session = make_session()
    for proj in cfg["download"]["roboflow"]["projects"]:
        ws, name, ver = proj["workspace"], proj["project"], proj.get("version")
        rep = SourceReport(
            key=f"roboflow_{name}",
            name=f"Roboflow Universe: {ws}/{name}",
            url=f"https://universe.roboflow.com/{ws}/{name}",
            license="See project page (Roboflow Universe projects are typically CC BY 4.0)",
            citation=f"{ws}, '{name}' dataset, Roboflow Universe.",
            format="YOLOv8 export",
        )
        if not key:
            rep.status = "skipped"
            rep.notes.append(
                "Roboflow requires a free API key for programmatic downloads. Create one at "
                "https://app.roboflow.com/settings/api, then run:  set ROBOFLOW_API_KEY=<key>  (PowerShell: "
                "$env:ROBOFLOW_API_KEY='<key>') and re-run this script."
            )
            log.warning("Roboflow %s/%s skipped: ROBOFLOW_API_KEY not set", ws, name)
            reports.append(rep)
            continue
        try:
            if ver is None:
                meta = session.get(f"https://api.roboflow.com/{ws}/{name}", params={"api_key": key}, timeout=60).json()
                versions = meta.get("versions") or []
                if not versions:
                    raise RuntimeError("no public versions")
                ver = str(versions[0]["id"]).rsplit("/", 1)[-1]
            link = None
            for _ in range(20):  # export may need a few seconds to be generated
                r = session.get(f"https://api.roboflow.com/{ws}/{name}/{ver}/yolov8", params={"api_key": key}, timeout=60)
                r.raise_for_status()
                link = (r.json().get("export") or {}).get("link")
                if link:
                    break
                time.sleep(3)
            if not link:
                raise RuntimeError("export link not ready")
            zr = session.get(link, timeout=600)
            zr.raise_for_status()
            dest = out_root / name
            dest.mkdir(parents=True, exist_ok=True)
            _safe_extract(zipfile.ZipFile(io.BytesIO(zr.content)), dest)
            names = list(dest.rglob("*.jpg")) + list(dest.rglob("*.png"))
            rep.images_downloaded = len(names)
            rep.images_annotated = sum(1 for p in names if (p.parent.parent / "labels" / f"{p.stem}.txt").exists())
            rep.status = "ok"
        except Exception as exc:
            rep.status = "failed"
            rep.notes.append(str(exc))
            log.error("Roboflow %s/%s failed: %s", ws, name, exc)
        reports.append(rep)
    return reports


# ---------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Download public annotated road-damage datasets")
    ap.add_argument("--sources", nargs="+", default=["rdd2022", "roadeye", "roboflow"],
                    choices=["rdd2022", "roadeye", "roboflow"])
    ap.add_argument("--roadeye-max", type=int, default=cfg["download"]["roadeye"]["max_images"])
    ap.add_argument("--rdd-scale", type=float, default=1.0, help="Multiply RDD per-country caps")
    args = ap.parse_args(list(argv) if argv is not None else None)

    raw = get_path("raw_dir")
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "sources.json"
    previous = {s["key"]: s for s in (json_load(manifest_path) or {}).get("sources", [])}

    log.info("Indexing existing raw images for de-duplication...")
    dedup = DedupIndex(raw)
    reports: list[SourceReport] = []

    if "rdd2022" in args.sources and cfg["download"]["rdd2022"]["enabled"]:
        log.info("=== Source A: RDD2022 ===")
        reports.append(download_rdd2022(cfg, raw / "rdd2022", dedup, args.rdd_scale))
    if "roboflow" in args.sources and cfg["download"]["roboflow"]["enabled"]:
        log.info("=== Sources B/C: Roboflow Universe ===")
        reports += download_roboflow(cfg, raw / "roboflow")
    if "roadeye" in args.sources and cfg["download"]["roadeye"]["enabled"]:
        log.info("=== Source D: Hugging Face RoadEye ===")
        reports.append(download_roadeye(cfg, raw / "roadeye", dedup, args.roadeye_max))

    for r in reports:
        previous[r.key] = r.__dict__
    totals = Counter()
    for s in previous.values():
        if s["status"] == "ok":
            for k in ("images_downloaded", "images_annotated", "background_images", "label_instances"):
                totals[k] += s[k]
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totals": dict(totals),
        "sources": list(previous.values()),
    }
    write_json(manifest_path, manifest)

    print("\n" + "=" * 72)
    print("DATASET DOWNLOAD SUMMARY")
    print("=" * 72)
    for s in previous.values():
        print(f"- {s['name']}\n    status={s['status']}  images={s['images_downloaded']}  "
              f"annotated={s['images_annotated']}  background={s['background_images']}  "
              f"instances={s['label_instances']}  dup_skipped={s['duplicates_skipped']}  "
              f"unannotated_skipped={s['unannotated_skipped']}  failed={s['failed']}")
        for n in s["notes"]:
            print(f"    · {n}")
    print(f"\nTOTAL images: {totals['images_downloaded']}  annotated: {totals['images_annotated']}  "
          f"label instances: {totals['label_instances']}")
    print(f"Manifest: {manifest_path}")
    return 0 if totals["images_downloaded"] else 1


def json_load(p: Path) -> Any:
    from src.utils import read_json
    return read_json(p, None)


if __name__ == "__main__":
    sys.exit(main())
