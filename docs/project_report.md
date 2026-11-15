# RoadGuard AI: Intelligent Road Pothole Detection and Road Condition Assessment System

**Deep Learning Lab Mini-Project**
*"Making Every Road Safer Through Artificial Intelligence."*

---

## 1. Project Title
**RoadGuard AI: Intelligent Road Pothole Detection and Road Condition Assessment using YOLOv8**

## 2. Abstract
Potholes cause vehicle damage, accidents and higher maintenance costs. Road inspection is still mostly done by people walking or driving the route, which is slow, subjective and hard to scale. RoadGuard AI automates the visual part of inspection with deep-learning object detection.

We built a dataset of **2,487 real annotated road images** with **6,562 labelled damage instances**. The images come from two public sources:

* the multi-national **RDD2022** benchmark (India, Japan, Czech Republic, USA);
* the **RoadEye** unified pothole dataset.

Annotations were normalised to one YOLO schema, validated, de-duplicated and split 80/10/10 in a way that prevents leakage. A COCO-pretrained **YOLOv8n** network (3.0 M parameters) was fine-tuned on the pothole subset by transfer learning, using PyTorch and Ultralytics.

The trained detector locates potholes, gives a confidence score for each one, and feeds a configurable rule-based **severity engine**. The engine grades the road as GOOD, MODERATE or SEVERE from the pothole count, the confidence of each detection, and the damaged-area ratio (union of boxes).

The system is delivered as an interactive **Streamlit** dashboard. It supports image upload, live webcam detection, JSON and PDF reports, a dataset explorer and a model-performance view. All metrics are produced by the actual training and evaluation runs (Section 13).

## 3. Introduction
Road networks deteriorate under traffic load, water ingress and freeze–thaw cycles. A pothole that is fixed early is cheap to repair; one that is ignored grows quickly and becomes dangerous. Cities now collect large amounts of road imagery from dash-cams, phones and survey vehicles. Reviewing that imagery by hand does not scale, so automated detection is needed.

Modern convolutional object detectors, and in particular the YOLO family, can find and localise objects in real time on ordinary hardware. RoadGuard AI applies this technology to road maintenance. It shows the whole deep-learning workflow, from data acquisition to deployment.

## 4. Problem Statement
Given an image of a road surface:

1. detect every pothole and draw a bounding box around it;
2. report a confidence score for each detection;
3. count the potholes and estimate the share of the visible road that is damaged;
4. classify the overall road condition as GOOD, MODERATE or SEVERE;
5. produce a report that can be shared.

The system must run on an ordinary laptop, CPU included, and must not report fabricated results.

## 5. Objectives
* Acquire a substantial, **real, annotated** road-damage dataset from public sources, with licences and attribution recorded.
* Build a reproducible pre-processing pipeline: format conversion, validation, de-duplication, leakage-aware splitting.
* Fine-tune a YOLOv8n detector with transfer learning and evaluate it with standard metrics (precision, recall, mAP@50, mAP@50-95).
* Design a transparent, configurable severity-assessment layer.
* Deliver an easy-to-demonstrate dashboard with image, webcam and reporting features.

## 6. Existing System
* **Manual visual inspection.** Engineers drive or walk the routes and log defects on paper or in spreadsheets. It is slow, expensive, subjective and infrequent.
* **Citizen complaint portals.** These are reactive and incomplete, and their location and severity information is often poor.
* **Specialised survey vehicles** with laser profilers or LiDAR. They are accurate but very expensive and only available to large agencies.
* **Classical image processing** (thresholding, edge detection). It is brittle under changes in lighting, shadows, wet roads and texture.

## 7. Proposed System
RoadGuard AI uses a learned detector that runs on cheap cameras and consumer computers.

```
 Road image / webcam frame
          │
          ▼
 Pre-processing (decode, letterbox to 512–640 px)
          │
          ▼
 YOLOv8n (fine-tuned) ── backbone → neck → head
          │
          ▼
 Confidence filter + Non-Maximum Suppression
          │
          ▼
 Detections: class, confidence, box
          │
          ├──► Severity engine (count, union area %, confidence) → GOOD / MODERATE / SEVERE
          │
          ▼
 Dashboard: annotated image · metrics · JSON / PDF report · history
```

**Advantages over the existing systems:** fast (milliseconds to about 100 ms per image on a CPU), objective, cheap, scalable, and explainable through the drawn boxes and the published severity rules.

## 8. Technologies Used
| Layer | Technology |
|---|---|
| Language | Python 3 |
| Deep learning | PyTorch, Ultralytics YOLOv8 |
| Vision / arrays | OpenCV, NumPy, Pillow |
| Dashboard | Streamlit, Plotly, Pandas |
| Reports | JSON, ReportLab (PDF) |
| Data access | Requests (Hugging Face API, HTTP range requests on Figshare), PyYAML |

## 9. Deep Learning Methodology
The concepts are covered in detail in **`docs/methodology.md`**. In short:

* **Learning paradigm:** supervised learning on bounding-box annotations.
* **Model:** YOLOv8n, a single-stage, anchor-free detector with a CSPDarknet backbone, a PAN-FPN neck and a decoupled head.
* **Loss:** CIoU and Distribution Focal Loss for the boxes, plus binary cross-entropy for the classes.
* **Optimisation:** the optimiser is chosen automatically by Ultralytics, with learning-rate warm-up, weight decay and early stopping.
* **Transfer learning:** training starts from COCO-pretrained weights, the classification head is replaced, and the whole network is fine-tuned.
* **Inference post-processing:** confidence thresholding followed by IoU-based Non-Maximum Suppression.

## 10. YOLO Architecture Explanation
1. **Input:** the image is letter-boxed (resized while keeping its aspect ratio, then padded) to a square of side `imgsz`.
2. **Backbone (CSPDarknet + C2f blocks):** stacked convolution, batch-norm and SiLU blocks extract features. Cross-stage partial connections reuse gradients efficiently. SPPF pools context at several receptive-field sizes. Features are taken from strides 8, 16 and 32 (P3, P4, P5).
3. **Neck (PAN-FPN):** a top-down path carries semantic information to the high-resolution maps, and a bottom-up path carries precise localisation back up. Small, distant potholes and large, close ones are therefore both represented.
4. **Head (decoupled, anchor-free):** for each cell on the three grids, one branch regresses the distances to the four box sides (as distributions, via DFL) and another outputs class probabilities.
5. **Post-processing:** boxes are decoded, filtered by confidence and de-duplicated with NMS.

YOLOv8n has about 3.0 M parameters and needs about 8.7 GFLOPs at 640 px. It is the smallest member of the family and is suitable for CPU training and inference.

## 11. Dataset Description
| Source | Licence | Images used | Annotation format | Classes |
|---|---|---|---|---|
| RDD2022 (CRDDC 2022, sekilab), India 450, Japan 300, Czech 150, USA 190 | CC BY 4.0 | 1,090 | Pascal VOC XML | D40 pothole, D00 longitudinal, D10 transverse, D20 alligator crack |
| RoadEye unified pothole dataset (Hugging Face), originals only | mixed (see card) | 1,397 | YOLO txt | pothole |
| Roboflow Universe pothole and road-damage projects | per project | 0 (needs an API key) | YOLOv8 | — |
| **Total** | | **2,487 images, 6,562 instances** | | |

*Acquisition.* The official RDD2022 release is a single 13.3 GB zip. Our downloader reads the zip's central directory with HTTP range requests and fetches only the chosen images and XML files, so the full archive never needs to be stored. Images containing potholes were prioritised. RoadEye images were sampled in proportion to its three sub-sources (Kaggle, Mendeley, Roboflow). Augmented `_aug` files were excluded to avoid leakage. Every image was stored only together with its annotation, and the sources, licences and counts are recorded in `data/raw/sources.json`.

*Pre-processing results (`data/potholes/dataset_summary.json`):*

| | Train | Val | Test |
|---|---|---|---|
| Images (pothole dataset) | 1,990 | 249 | 248 |
| Pothole instances | 4,086 | 524 | 550 |
| Background images | 63 | 9 | 2 |

* 0 corrupt images, 0 unannotated images used, 0 exact duplicates after download-time de-duplication (3 duplicates had already been skipped during download).
* 227 out-of-schema boxes removed (RDD2022 classes D43, D44 and D50, which are blurred markings and manholes rather than damage).
* 11 near-duplicate image groups kept together within one split.
* Image–label correspondence and the absence of cross-split overlap were verified automatically.

The 5-class variant (`data/road_damage_multiclass`) uses the same split and contains 5,160 pothole, 475 longitudinal-crack, 216 transverse-crack and 484 alligator-crack instances. No "Repair" or patch annotations occurred in the selected RDD2022 countries.

## 12. Training Methodology
* **Base weights:** `yolov8n.pt`, COCO-pretrained and downloaded automatically.
* **Hardware:** Intel Core i5-10300H CPU with 8 threads. The laptop has a GTX 1650 4 GB, but the installed PyTorch build is CPU-only and a CUDA build would not fit in the remaining disk space. The scripts switch to the GPU automatically when CUDA is available.
* **Configuration:** planned 15 epochs (stopped after 9 to save time; the default is now 10), input size 512 on CPU (640 on GPU), batch 16, early-stopping patience 5, seed 42, default Ultralytics augmentation (mosaic, which is turned off for the last 10 epochs; HSV jitter; horizontal flip; scale and translate).
* **Model selection:** the checkpoint with the best validation fitness (0.1·mAP@50 + 0.9·mAP@50-95) is copied to `models/trained/roadguard_pothole_best.pt`.
* **Evaluation:** validation split, and separately the **held-out test split**, which is never used during training.

## 13. Results and Evaluation
<!-- RESULTS:START -->
All numbers below come from `models/trained/model_registry.json` and the Ultralytics output files in `outputs/runs/`.

**Training run.** The schedule was planned for 15 epochs and **stopped after 9 epochs** (72.5 min on CPU, imgsz 512, batch 16) to keep the project quick. The best checkpoint by validation fitness (epoch 9) was kept and evaluated. The status is recorded honestly in the registry as `stopped early (shortened schedule)`.

| Epoch | Val precision | Val recall | Val mAP@50 | Val mAP@50-95 |
|---|---|---|---|---|
| 1 | 0.199 | 0.218 | 0.116 | 0.039 |
| 3 | 0.337 | 0.261 | 0.196 | 0.072 |
| 5 | 0.443 | 0.277 | 0.267 | 0.101 |
| 7 | 0.475 | 0.324 | 0.326 | 0.134 |
| 9 | 0.546 | 0.370 | **0.405** | 0.155 |

Validation mAP@50 was still rising when training stopped, so more epochs would improve the model.

**Final evaluation (best checkpoint):**

| Split | Images | Pothole instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| Validation | 249 | 524 | 0.546 | 0.370 | 0.405 | 0.155 |
| **Held-out test** | 248 | 550 | **0.437** | **0.395** | **0.390** | **0.159** |

* Inference speed: about 35–50 ms per image on the i5-10300H CPU at imgsz 512.
* Baseline: the COCO-pretrained YOLOv8n has no pothole class, so its pothole mAP is zero by definition. All pothole-detection ability comes from fine-tuning (transfer learning).
* **Severity check** (`scripts/evaluate.py`): the rule engine was applied to the model's detections and to the ground-truth boxes for all 248 test images. The two gradings **agree on 47.6 %** of images. The model under-grades because of its limited recall: it missed potholes on 111 images that it graded GOOD, while only 2 test images are truly GOOD. Improving recall (longer or GPU training, a larger model) would directly improve the severity grading.
* Interpretation: precision is higher than recall, so when the model reports a pothole it is usually right, but it misses small and distant potholes. These are common in RDD2022's dash-cam images, where a pothole often covers less than 0.5 % of the frame. Large, close-up potholes (RoadEye) are detected more reliably.
<!-- RESULTS:END -->

## 14. Advantages
* Real-time-capable, lightweight model that runs on a CPU.
* Trained on real images from several countries; the data comes with documented sources and licences.
* Explainable: visible boxes, per-box confidences and published severity rules with stated reasons.
* Adjustable confidence and IoU thresholds to trade precision against recall.
* End-to-end reproducible: one command per stage, with configuration in `config.yaml`.
* Exportable JSON and PDF reports, plus a stored detection history.

## 15. Limitations
* The dataset is moderate in size and the training schedule is short (9 epochs on a CPU). A larger model or longer GPU training would raise mAP.
* Annotation conventions differ between sources, for example tight versus loose pothole boxes.
* A single 2-D image gives no depth. Box area is affected by perspective and is not a physical area. The severity grade is indicative only.
* Potholes can be confused with patches, shadows, manhole covers and wet spots, especially at night or in heavy rain.
* The multi-class crack categories are imbalanced compared with potholes.

## 16. Future Scope
* Train on a GPU with YOLOv8s/m and the full RDD2022 dataset (47k images).
* Instance segmentation masks for precise damaged-area measurement.
* Geo-tagging (GPS) and a city-wide damage map with maintenance prioritisation.
* Video tracking (ByteTrack) so the same pothole is counted once across frames.
* Depth estimation with stereo cameras, LiDAR or monocular depth networks to estimate pothole volume.
* Deployment on edge devices (Jetson, Raspberry Pi) or dash-cam apps with ONNX/TensorRT export.

## 17. Conclusion
RoadGuard AI shows a complete, working deep-learning pipeline for road damage detection. It covers dataset acquisition from real public benchmarks, careful pre-processing, transfer-learning fine-tuning of YOLOv8n, quantitative evaluation on a held-out test set, a transparent severity layer and a usable dashboard. The project shows that a lightweight detector trained on a few thousand annotated images, even on a CPU-only laptop, can support faster and more objective road maintenance decisions.

## 18. References
1. D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, Y. Sekimoto, "RDD2022: A multi-national image dataset for automatic Road Damage Detection," *Geoscience Data Journal*, 2024. arXiv:2209.08538. Data: doi:10.6084/m9.figshare.21431547 (CC BY 4.0).
2. H. Maeda et al., "Road Damage Detection and Classification Using Deep Neural Networks with Smartphone Images," *Computer-Aided Civil and Infrastructure Engineering*, 2018.
3. G. Jocher, A. Chaurasia, J. Qiu, *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics
4. J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," *CVPR*, 2016.
5. T.-Y. Lin et al., "Feature Pyramid Networks for Object Detection," *CVPR*, 2017.
6. S. Liu et al., "Path Aggregation Network for Instance Segmentation," *CVPR*, 2018.
7. X. Li et al., "Generalized Focal Loss," *NeurIPS*, 2020 (Distribution Focal Loss).
8. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," *ECCV*, 2014.
9. AkumaDachi, *RoadEye_Yolo_Aug* dataset, Hugging Face, https://huggingface.co/datasets/AkumaDachi/RoadEye_Yolo_Aug
10. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *NeurIPS*, 2019.
