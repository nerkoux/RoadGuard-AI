# RoadGuard AI: Viva Questions and Answers

Forty questions grouped by topic. Answers are kept short enough to say aloud in an exam, and they refer to this project wherever possible.

---

## A. Deep learning and neural networks

**1. What is deep learning?**
Deep learning is a branch of machine learning that uses neural networks with many layers. The network learns features directly from raw data such as image pixels, so no one has to design those features by hand. Each layer builds more abstract features from the layer before it.

**2. What is an artificial neural network?**
A network of connected neurons. Each neuron computes a weighted sum of its inputs plus a bias, then passes the result through a non-linear activation function. The network learns by adjusting its weights to reduce a loss function.

**3. What is back-propagation?**
The algorithm that computes the gradient of the loss with respect to every weight, using the chain rule from the output layer back to the input. An optimiser (SGD or AdamW) then updates each weight: `w ← w − lr · ∂L/∂w`.

**4. Why are activation functions needed?**
Without a non-linearity, any stack of layers collapses into a single linear transformation and could not model complex shapes. YOLOv8 uses **SiLU** (`x · sigmoid(x)`).

**5. What are epochs, batch size and learning rate?**
* *Epoch*: one full pass over the training set (10 by default here; the delivered model trained for 9).
* *Batch size*: the number of images processed before each weight update (AutoBatch on a GPU, 16 on a CPU).
* *Learning rate*: the size of each update step. Ultralytics uses a warm-up phase followed by a decaying schedule.

---

## B. CNNs

**6. What is a Convolutional Neural Network?**
A neural network that uses convolution layers. Small learnable filters slide over the image, so the same pattern detector is shared across all positions. This keeps the parameter count low and makes the network translation-equivariant, so a pothole is recognised wherever it appears.

**7. What do early and deep CNN layers learn?**
Early layers learn edges, colour changes and textures. Middle layers learn parts such as crack lines and pothole rims. Deep layers learn whole-object patterns such as a dark, irregular depression with a rim.

**8. What is pooling / stride and why is it used?**
Both reduce the spatial resolution of feature maps. This increases the receptive field, so each neuron sees more of the image, and it reduces computation. YOLOv8 down-samples with stride-2 convolutions and uses **SPPF** (spatial pyramid pooling, fast) to collect context at several scales.

**9. What is batch normalisation?**
It normalises the activations inside each mini-batch to zero mean and unit variance, then applies a learnable scale and shift. It stabilises and speeds up training.

---

## C. Object detection and YOLO

**10. How is object detection different from classification?**
Classification gives one label for the whole image. Detection finds every object and returns a class, a bounding box and a confidence for each one. We need detection because the project counts potholes and measures the area they cover.

**11. What does YOLO stand for and why is it fast?**
"You Only Look Once". It is a single-stage detector: one forward pass predicts every box and class at once, with no separate region-proposal stage as in Faster R-CNN. That makes it fast enough for real-time video.

**12. Explain the YOLOv8 architecture.**
* **Backbone (CSPDarknet with C2f blocks):** extracts features at three scales.
* **Neck (PAN-FPN):** fuses deep semantic features with shallow high-resolution features so it can find both small and large objects.
* **Head (decoupled, anchor-free):** has separate branches for box regression (using Distribution Focal Loss) and classification, evaluated at three output scales (80×80, 40×40 and 20×20 grids for a 640-pixel input).

**13. What does "anchor-free" mean?**
Older YOLO versions predicted offsets from predefined anchor boxes. YOLOv8 predicts the distance from each grid point to the four box sides directly. This removes anchor tuning and generalises better to unusual object shapes such as long cracks.

**14. Why YOLOv8n (nano)?**
It is the smallest variant: about 3.0 M parameters and 8.7 GFLOPs. It trains and runs in real time on a CPU or a modest GPU such as the GTX 1650 in our lab machine. The trade-off is slightly lower accuracy than the s, m and l variants.

**15. What loss functions does YOLOv8 use?**
* **Box loss:** CIoU, which accounts for overlap, centre distance and aspect ratio, plus Distribution Focal Loss (`box_loss`, `dfl_loss`).
* **Classification loss:** binary cross-entropy (`cls_loss`).
* The Task-Aligned Assigner decides which predictions are matched to which ground-truth boxes.

**16. What is a bounding box and what format do YOLO labels use?**
A bounding box is the smallest rectangle that encloses an object. A YOLO label line is `class x_center y_center width height`, with all values normalised to 0–1. The RDD2022 XML files use absolute `xmin, ymin, xmax, ymax`, and we convert them in `src/preprocessing.py`.

---

## D. Confidence, IoU, NMS

**17. What is a confidence score?**
The model's predicted probability that a box contains the given class. Detections below the **confidence threshold** (0.25 by default, adjustable with a slider in the app) are discarded.

**18. What happens if you raise or lower the confidence threshold?**
Raising it gives fewer false positives (higher precision) but more missed potholes (lower recall). Lowering it does the opposite. The app's slider shows this trade-off live.

**19. Define Intersection over Union.**
`IoU = area(A ∩ B) / area(A ∪ B)`. It runs from 0 to 1. It is used to decide whether a prediction matches the ground truth (true positive if IoU ≥ 0.5 for mAP@50), inside NMS, and in the CIoU loss.

**20. What is Non-Maximum Suppression?**
A post-processing step that removes duplicate boxes. It keeps the highest-confidence box, deletes any other box that overlaps it with IoU above the NMS threshold (0.45), and repeats with the remaining boxes. Without NMS a single pothole would be reported several times.

---

## E. Training, testing and dataset

**21. Which datasets did you use and how did you get them?**
1. **RDD2022** (CC BY 4.0, from India, Japan, Czech Republic and the USA), in Pascal VOC XML with classes D00, D10, D20, D40 and Repair. The official release is a single 13 GB zip. Our downloader uses HTTP range requests to read the zip directory and fetch only the selected images and annotations.
2. **RoadEye** (Hugging Face), YOLO format, pothole class only. Roboflow Universe projects are supported, but they need a free API key.

Every image is used only together with its annotation file.

**22. Why split the data into train, validation and test sets?**
* **Train** (80 %): the weights are learned on this set.
* **Validation** (10 %): monitors overfitting, picks the best checkpoint and triggers early stopping.
* **Test** (10 %): never touched during training. It gives an unbiased final score.

Reporting results on the training data would overestimate accuracy.

**23. What is data leakage and how did you prevent it?**
Leakage happens when the same, or a nearly identical, image appears in both training and test sets, which inflates the test score. We:
(a) excluded RoadEye's augmented `_aug` copies,
(b) removed exact duplicates using MD5 hashes,
(c) grouped visually identical images by perceptual hash (dHash) and kept each group within one split,
(d) automatically checked that no file name appears in two splits.

**24. What pre-processing did you perform?**
Converted VOC annotations to YOLO format, mapped every source's classes to one unified schema (pothole, longitudinal, transverse and alligator cracks, patch), clipped boxes and removed invalid or tiny ones, removed corrupt and duplicate images, resized images to a maximum side of 640 px, made an 80/10/10 split, capped background images at about 10 %, and generated `data.yaml`.

**25. What is overfitting and how do you detect and reduce it?**
Overfitting means the model memorises the training images and does poorly on new ones. You can spot it when training loss keeps falling but validation loss rises or validation mAP stops improving (see the curves on the Model Performance page). We reduce it with transfer learning, data augmentation (mosaic, flips, HSV and scale), early stopping (patience 5) and a reasonably sized dataset.

**26. What is data augmentation?**
Applying random transformations to training images, such as mosaic of four images, flips, colour (HSV) jitter, scaling and translation. This adds variety without new labels and makes the model robust to lighting and viewpoint changes.

**27. Why include background images (images without potholes)?**
They teach the model what is *not* a pothole, such as shadows, patches, manholes and cracks, which reduces false positives. We cap them at about 10 % so they do not dilute the positive examples.

---

## F. Metrics

**28. Define precision and recall.**
`Precision = TP / (TP + FP)`: of the potholes we reported, how many were real.
`Recall = TP / (TP + FN)`: of the real potholes, how many we found.

**29. What is mAP@50 and mAP@50-95?**
Average Precision (AP) is the area under the precision–recall curve for one class. mAP is the mean of AP over all classes.
* **mAP@50** uses an IoU threshold of 0.5.
* **mAP@50-95** averages AP over IoU thresholds from 0.50 to 0.95 in steps of 0.05, so it is stricter and rewards tighter boxes.

**30. What is a confusion matrix in object detection?**
A table of predicted classes against true classes, plus a "background" row and column. Background in the predicted column means a missed object (false negative). Background in the true row means a false alarm (false positive). We read it from the Ultralytics validation output on the test split.

**31. Why might mAP be modest for pothole detection?**
Potholes have fuzzy boundaries, so annotators disagree on box extent, which lowers IoU-based scores. They vary a lot in size and lighting, and they can be confused with patches, shadows and wet spots. On top of that, our training is short (9 epochs on a CPU with the nano model).

---

## G. Transfer learning, PyTorch, OpenCV

**32. What is transfer learning and why use it here?**
Reusing a model trained on a large dataset (COCO, 118k images) and fine-tuning it on our smaller pothole dataset. The pretrained backbone already extracts generic visual features, so the model converges in a few epochs with a few thousand images.

**33. Can the pretrained COCO YOLOv8n detect potholes?**
No. COCO has 80 classes such as car, person and truck, and none of them is a pothole. That is why the app labels the pretrained model "PRETRAINED · COCO" and reports road severity only with the fine-tuned weights. This shows the value of fine-tuning.

**34. What is PyTorch's role in this project?**
PyTorch is the deep-learning framework under Ultralytics YOLOv8. It provides tensors, GPU/CPU computation, automatic differentiation (autograd) for back-propagation, and the optimisers.

**35. What is OpenCV used for?**
Decoding and resizing images, drawing bounding boxes and labels, encoding the annotated output, reading webcam frames (`cv2.VideoCapture`) for live detection, and computing the perceptual hash used for de-duplication.

**36. What happens if no GPU is available?**
`src/detector.py` and `scripts/train.py` detect CUDA automatically. On a CPU, training uses a smaller input size (512) and a fixed batch of 16. Inference still runs at roughly 50–150 ms per image for YOLOv8n. If a GPU call fails, for example with out-of-memory, inference falls back to the CPU.

---

## H. Severity estimation and the application

**37. How is road severity calculated?**
A transparent rule layer in `src/severity.py` works on the real detections:
* Detections below the confidence floor are ignored.
* Potholes are counted.
* The damaged area is the union of the boxes divided by the image area.
* **SEVERE** if there are at least 3 potholes, *or* the area is at least 8 %, *or* one confident pothole covers at least 5 % of the image.
* **MODERATE** if there is at least 1 pothole *or* the area is at least 1 %.
* **GOOD** otherwise.

All thresholds are set in `config.yaml`.

**38. Why compute the union of boxes rather than the sum of box areas?**
Overlapping boxes would be counted twice and inflate the damaged-area percentage. We rasterise the boxes on a grid and measure the covered fraction.

**39. Is the severity output an engineering assessment?**
No. It is an AI-generated *indicative* classification from one 2-D image. Perspective distorts apparent area, depth cannot be measured, and the result depends on detection accuracy. It is useful for prioritising inspection, not for certification.

**40. What are the project's limitations and future scope?**
*Limitations:* CPU-only training with a short schedule, mixed annotation styles across sources, no depth or physical size estimation, and perspective effects on area.
*Future scope:* GPU training of larger models (YOLOv8s/m), segmentation masks for exact area, GPS tagging and a city-wide map, depth from stereo or LiDAR, video tracking to avoid double counting, and deployment on edge devices or dash-cams.
