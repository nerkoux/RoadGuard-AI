# RoadGuard AI — Methodology

This document explains the deep-learning concepts behind RoadGuard AI, in the order they show up in the pipeline.

---

## 1. What is object detection?

Computer vision has three common levels of image understanding:

| Task | Question answered | Output |
|---|---|---|
| Image classification | *Is there a pothole in this picture?* | one label per image |
| **Object detection** | *Where is every pothole, and how sure are we?* | a list of **bounding boxes + class + confidence** |
| Segmentation | *Which exact pixels belong to the pothole?* | a mask per object |

RoadGuard AI uses **object detection**. Counting potholes and measuring how much of the road they cover needs each pothole's location, so a single yes/no label per image is not enough.

---

## 2. How CNNs extract visual features

A **Convolutional Neural Network (CNN)** learns filters (small weight matrices such as 3×3) that slide across the image. Each filter responds to a pattern:

* **Early layers:** edges, colour changes and simple textures (asphalt grain, shadow borders).
* **Middle layers:** combinations of edges, such as curved rims, cracks and broken patches.
* **Deep layers:** whole-object patterns, such as a dark, irregular, rimmed depression in the road.

Key operations:

* **Convolution:** a weighted sum over a local window that produces a *feature map*.
* **Batch normalisation:** keeps activations in a stable range and speeds up training.
* **Activation (SiLU in YOLOv8):** a non-linearity that lets the network model complex shapes.
* **Down-sampling (stride-2 convolutions):** shrinks the spatial size and widens the receptive field, so deeper layers "see" more of the image.

Weights are learned by **back-propagation**. The loss is differentiated with respect to every weight, and an optimiser (SGD or AdamW) nudges each weight in the direction that lowers the loss.

---

## 3. How YOLO works

**YOLO ("You Only Look Once")** is a *single-stage* detector. One forward pass of the network predicts every box at once. Two-stage detectors such as Faster R-CNN first propose regions and then classify each one; YOLO skips that step, which makes it fast enough for real-time video.

### YOLOv8 architecture (used here: the *nano* variant, ~3.0 M parameters, 8.7 GFLOPs)

```
Input 640×640×3
   │
   ▼
BACKBONE  (CSPDarknet with C2f blocks)  → multi-scale feature maps  P3 (80×80), P4 (40×40), P5 (20×20)
   │                                       SPPF block pools context at several scales
   ▼
NECK      (PAN-FPN)                      → fuses deep semantic features with shallow high-resolution features
   │                                       (top-down + bottom-up paths) so small AND large potholes are found
   ▼
HEAD      (decoupled, anchor-free)       → for every grid cell on each of the 3 scales:
                                              • box regression branch (distances to the 4 box sides, DFL)
                                              • classification branch (one score per class)
```

* **Anchor-free:** YOLOv8 predicts box edges directly relative to each grid cell. There are no hand-tuned anchor boxes.
* **Decoupled head:** classification and localisation use separate convolution branches, which improves accuracy.
* **Loss function:**
  * *Box loss* = CIoU loss (overlap, centre distance and aspect ratio) plus **Distribution Focal Loss (DFL)**, which models each box edge as a probability distribution.
  * *Classification loss* = binary cross-entropy.
  * The **Task-Aligned Assigner** decides which predictions count as positives for each ground-truth box during training.

These three losses appear in the dashboard as `box_loss`, `cls_loss` and `dfl_loss`.

---

## 4. Bounding boxes

A bounding box is the smallest axis-aligned rectangle that contains the object.

* **Pascal VOC format (RDD2022 XML):** `xmin, ymin, xmax, ymax` in pixels.
* **YOLO format (training labels):** `class x_center y_center width height`, all normalised to 0–1 by the image width and height.

Conversion used in `src/preprocessing.py`:

```
x_c = (xmin + xmax) / 2 / W        w = (xmax − xmin) / W
y_c = (ymin + ymax) / 2 / H        h = (ymax − ymin) / H
```

Normalised coordinates stay the same when an image is resized uniformly. That is why the pre-processing step can shrink images to 640 px without rewriting the labels.

---

## 5. Confidence scores

For every predicted box YOLOv8 outputs a class probability (sigmoid) for each class. The **confidence** of a detection is the highest of these probabilities. Read it as the model's estimated likelihood that the box contains that class.

* The **confidence threshold** (a slider in the app, default 0.25) throws away weak predictions.
  * Raise the threshold to get fewer false alarms (higher precision) but more missed potholes (lower recall).
  * Lower it to get the opposite.

---

## 6. Intersection over Union (IoU)

```
            area(A ∩ B)
IoU(A,B) = ─────────────
            area(A ∪ B)
```

IoU measures how much two boxes overlap. It runs from 0 (no overlap) to 1 (identical boxes). It is used in three places:

1. **Evaluation:** a prediction is a *true positive* only if its IoU with a ground-truth box is at least a threshold (0.50 for mAP@50).
2. **NMS:** IoU decides which duplicate boxes to remove.
3. **Training loss:** CIoU is a refinement of IoU.

---

## 7. Non-Maximum Suppression (NMS)

The network often predicts several overlapping boxes for the same pothole. NMS cleans them up:

1. Sort all boxes by confidence.
2. Keep the highest-scoring box.
3. Delete every remaining box whose IoU with the kept box is above the **NMS IoU threshold** (a slider in the app, default 0.45).
4. Repeat with the next remaining box.

A lower IoU threshold suppresses more aggressively. It can merge two potholes that sit next to each other. A higher threshold keeps more overlapping boxes.

---

## 8. Precision and recall

| Term | Meaning (pothole example) |
|---|---|
| True positive (TP) | a predicted pothole box that matches a real pothole (IoU ≥ threshold) |
| False positive (FP) | a predicted box where there is no pothole (a shadow, a patch, a manhole) |
| False negative (FN) | a real pothole the model missed |

```
Precision = TP / (TP + FP)     → "Of the potholes we reported, how many were real?"
Recall    = TP / (TP + FN)     → "Of the real potholes, how many did we find?"
```

For road maintenance, **recall** matters a lot because a missed pothole is a safety risk. Precision controls how many false alarms the maintenance crews have to deal with.

---

## 9. Mean Average Precision (mAP)

1. Sweep the confidence threshold from high to low. Each threshold gives one (precision, recall) point, and together they form the **precision–recall curve**.
2. **Average Precision (AP)** is the area under that curve, computed with 101-point interpolation in Ultralytics. AP is calculated for one class at one IoU threshold.
3. **mAP** is the mean of AP over all classes.

* **mAP@50:** IoU threshold 0.50. Asks whether the detection is roughly in the right place.
* **mAP@50-95:** AP averaged over IoU thresholds 0.50, 0.55, …, 0.95. This is the stricter COCO metric and rewards precise boxes.

Ultralytics picks the best checkpoint by **fitness = 0.1·mAP@50 + 0.9·mAP@50-95**.

---

## 10. Transfer learning

Training a detector from random weights needs hundreds of thousands of labelled images. **Transfer learning** reuses a network already trained on a large dataset:

1. Start from **`yolov8n.pt` pretrained on COCO** (118k images, 80 classes). Its backbone already extracts general features such as edges, textures and shapes.
2. **Replace the classification part of the head** with an output layer for our classes (1 class for pothole-only, 5 for multi-class).
3. **Fine-tune the whole network** on road images, starting with a learning-rate warm-up so the pretrained features are not destroyed early on.

Benefits: training converges in a few epochs, works with a few thousand images, and generalises better than training from scratch.

Note that the COCO-pretrained model on its own has **no pothole class**. The app labels it "pretrained · COCO" and does not produce a road severity with it.

---

## 11. Training details used in this project

| Setting | Value | Why |
|---|---|---|
| Base model | YOLOv8n | smallest YOLOv8, fast on CPU and consumer GPUs |
| Input size | 640 (GPU) / 512 (CPU) | standard YOLO resolution; smaller on CPU to save time |
| Epochs | 10 by default (configurable; the delivered model ran 9) | short schedule keeps CPU training quick |
| Batch | AutoBatch (GPU) / 16 (CPU) | fits the available memory |
| Optimiser | `auto` → AdamW for short schedules | Ultralytics heuristic |
| Augmentation | mosaic, HSV jitter, flips, scale/translate (Ultralytics defaults) | reduces overfitting |
| Early stopping | patience 5 | stops when validation fitness stops improving |
| Split | 80 / 10 / 10 | train / validation (model selection) / test (final unbiased score) |

### Overfitting control
* Separate validation split used for checkpoint selection and early stopping.
* A **held-out test split** that is never used during training gives the final reported score.
* Data augmentation.
* **Leakage prevention:** augmented "_aug" copies from RoadEye are excluded, exact duplicates are removed (MD5), and near-identical images (same perceptual hash) are always put in the same split.

---

## 12. Severity estimation (rule layer)

The neural network outputs boxes. `src/severity.py` turns them into an **indicative** road condition:

1. Ignore detections below `min_confidence`.
2. Count the potholes.
3. Compute the **damaged-area ratio** as the *union* of all damage boxes divided by the image area. The boxes are rasterised on a grid, so overlapping boxes are not counted twice.
4. Apply the rules (thresholds live in `config.yaml`):
   * **SEVERE:** at least 3 potholes, *or* damaged area ≥ 8 %, *or* one high-confidence pothole covering ≥ 5 % of the frame.
   * **MODERATE:** at least 1 pothole, *or* damaged area ≥ 1 %.
   * **GOOD:** otherwise.
5. A derived **damage index** (0–100) = (100/3)·Σ(pothole confidence) + 4·(damaged area %), capped at 100.

The result is an AI-generated indicator, **not** a certified engineering assessment. Box area in a perspective photo does not equal physical area, and image depth cannot be measured from a single image.
