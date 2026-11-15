# ROADGUARD AI
### *Making Every Road Safer Through Artificial Intelligence.*

RoadGuard AI is a pothole detection and road condition assessment system built on a fine-tuned **YOLOv8n** model (Ultralytics / PyTorch), with a **Streamlit** dashboard on top.

You upload a road image. The app draws boxes around the potholes it finds and shows the confidence for each one. It also counts the potholes, estimates how much of the road surface is damaged, and grades the road **GOOD / MODERATE / SEVERE**. You can download the result as a JSON or PDF report. Live webcam detection is supported as well.

> Nothing in this project is a mock. Every box, confidence and metric comes from real YOLO inference or from the Ultralytics training and evaluation output files. When the model has not been trained, the app says so; it does not show placeholder accuracy.

---

## 1. Quick start (Windows PowerShell)

```powershell
cd roadguard-AI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/verify_installation.py   # checks Python, torch, GPU, weights, dataset
python scripts/download_dataset.py      # ~5-8 min, ~420 MB of real annotated images
python scripts/prepare_dataset.py       # ~1-2 min, YOLO dataset + data.yaml
python scripts/train.py                 # fine-tune YOLOv8n (see timing below)
python scripts/evaluate.py              # real test-split metrics + severity check
streamlit run app.py                    # http://localhost:8501
```

**Inference-only quick start.** This mode does not wait for training. It uses the fine-tuned weights already shipped in `models/trained/` if they exist, and otherwise falls back to the COCO-pretrained YOLOv8n, which the app clearly labels:

```powershell
.venv\Scripts\activate
streamlit run app.py
```
or double-click **`run_app.bat`**.

> PowerShell blocks `activate` with an execution-policy error? Run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe -m streamlit run app.py` directly.

> Low on disk and PyTorch already installed system-wide? Create the venv with
> `python -m venv --system-site-packages .venv`. pip will then reuse the existing torch install.

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_dataset.py && python scripts/prepare_dataset.py
python scripts/train.py
streamlit run app.py
```

### Using an NVIDIA GPU
`pip install torch` gives you the CPU build on Windows. For CUDA (needs about 5 GB of disk):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python scripts/verify_installation.py   # should now show "CUDA ... GeForce ..."
python scripts/train.py                 # auto: imgsz 640, AutoBatch, AMP
```

---

## 2. Training options

| Command | Effect |
|---|---|
| `python scripts/train.py` | pothole-only model, 10 epochs, auto batch/device (~8 min/epoch on CPU) |
| `python scripts/train.py --epochs 30 --batch 16` | custom schedule |
| `python scripts/train.py --mode multiclass` | 5 classes: pothole, longitudinal / transverse / alligator crack, road patch |
| `python scripts/train.py --device cpu --imgsz 416` | faster CPU run |
| `python scripts/train.py --fraction 0.5` | train on half the training images |

Outputs:
* `models/trained/roadguard_<mode>_best.pt`: the best checkpoint, selected on validation fitness.
* `models/trained/model_registry.json`: real validation and test metrics, epochs, hardware and training time.
* `outputs/runs/<mode>_train/`: Ultralytics `results.csv`, loss curves, confusion matrix and PR curves.

Early stopping (patience 5) is on. If you interrupt training with Ctrl+C, the best checkpoint so far is still saved and the registry records `status: interrupted`.

---

## 3. Datasets

`scripts/download_dataset.py` downloads **real, annotated** data and logs the source, licence and counts for each dataset in `data/raw/sources.json`.

| Source | Access | Format | Licence |
|---|---|---|---|
| **RDD2022** (sekilab / CRDDC 2022): India, Japan, Czech Republic, USA | public. The official release is one 13 GB Figshare zip, so the script uses **HTTP range requests** to fetch only the selected images and XML files without storing the whole archive | Pascal VOC XML: D00, D10, D20, D40, Repair | CC BY 4.0 |
| **RoadEye** (`AkumaDachi/RoadEye_Yolo_Aug`, Hugging Face) | public | YOLO, pothole class | mixed; see dataset card |
| **Roboflow Universe** pothole / road-damage projects | requires a **free API key**: set `ROBOFLOW_API_KEY` | YOLOv8 export | per project, usually CC BY 4.0 |

Rules the pipeline enforces:
* An image is used only if a matching annotation file exists. Unannotated images are counted and excluded.
* RoadEye's augmented `_aug` copies are skipped, so the same photo cannot end up in both train and test.
* Exact duplicates are removed by MD5. Near-identical images (perceptual dHash) are always kept in the same split.
* Every class name is mapped to one schema: `0 pothole, 1 longitudinal_crack, 2 transverse_crack, 3 alligator_crack, 4 road_patch`.
* Split is 80/10/10, per source. Images are resized to a maximum side of 640 px. Background images (no damage) are capped at 10 %.

Roboflow example:
```powershell
$env:ROBOFLOW_API_KEY = "your_key"
python scripts/download_dataset.py --sources roboflow
python scripts/prepare_dataset.py
```

---

## 4. Dashboard pages

| Page | What it shows |
|---|---|
| **Overview** | dataset size, annotated count, model and training status, class count, latest result, condition distribution, activity chart, test metrics, recent history |
| **Detect Potholes** | upload JPG/PNG/WEBP or choose a held-out sample, confidence and IoU sliders, original and annotated images side by side, per-box confidences, severity, damaged area, downloads (PNG, JSON, PDF) |
| **Live Detection** | OpenCV webcam stream with start/stop, live count and severity, FPS; a browser snapshot fallback |
| **Dataset Explorer** | split and class distribution, sources, browsable ground-truth boxes, pre-processing log, licences |
| **Model Performance** | P / R / mAP@50 / mAP@50-95 for validation and test, training and validation losses, confusion matrix, PR and F1 curves |
| **About Project** | pipeline, technology, severity thresholds, documentation |

---

## 5. Severity engine (`src/severity.py`)

The thresholds are configurable in `config.yaml → severity`. The engine reads the real detections and grades the road as follows:

* **SEVERE**: at least 3 potholes, *or* a combined damaged area of at least 8 %, *or* one high-confidence pothole covering at least 5 % of the image.
* **MODERATE**: at least 1 pothole, *or* a damaged area of at least 1 %.
* **GOOD**: anything else.

Damaged area is the **union** of the boxes, so overlapping boxes are not counted twice. Output schema:

```json
{"severity": "MODERATE", "pothole_count": 2, "average_confidence": 0.87, "maximum_confidence": 0.93,
 "damaged_area_percentage": 4.5, "detections": [ ... ], "reasons": [ ... ], "disclaimer": "..."}
```
*The severity grade is an indicative AI result, not a certified engineering assessment.*

---

## 6. Project structure

```
roadguard-AI/
├── app.py                     Streamlit dashboard (6 pages)
├── config.yaml                paths, classes, download caps, training & severity thresholds
├── requirements.txt  run_app.bat  README.md
├── src/
│   ├── detector.py            YOLO loading (cached), inference, drawing, CPU fallback
│   ├── severity.py            rule-based condition engine
│   ├── preprocessing.py       VOC→YOLO, validation, dedup, leakage-aware split
│   ├── dataset_manager.py     dataset statistics / browsing for the UI
│   ├── report_generator.py    JSON + PDF reports, detection history
│   └── utils.py               config, paths, hashing, hardware info
├── scripts/
│   ├── download_dataset.py    RDD2022 (range requests), RoadEye (HF), Roboflow (API key)
│   ├── prepare_dataset.py     builds data/potholes and data/road_damage_multiclass
│   ├── train.py               YOLOv8n fine-tuning + metric registry
│   ├── evaluate.py            test-split metrics + severity agreement check
│   └── verify_installation.py
├── tests/test_core.py         unit tests (severity, label parsing, class mapping)
├── data/ raw/ processed/ potholes/ road_damage_multiclass/ samples/
├── models/ pretrained/ trained/
├── outputs/ detections/ reports/ runs/
└── docs/ project_report.md  methodology.md  viva_questions.md
```

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Model weights not found` | run `python scripts/train.py`, or pick the pretrained model in the dropdown |
| `Dataset not prepared` | `python scripts/download_dataset.py` then `python scripts/prepare_dataset.py` |
| Hugging Face "Rate limited" | the script waits and retries automatically; re-running resumes where it stopped |
| CUDA out of memory | `python scripts/train.py --batch 4` or `--device cpu` |
| No webcam | use *Browser snapshot* mode on the Live Detection page |
| Port in use | `streamlit run app.py --server.port 8502` |

Run the unit tests with `python tests/test_core.py`.
