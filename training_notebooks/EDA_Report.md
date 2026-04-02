# Basketball Analysis — Model Dataset EDA Report

> **Dataset:** `Basketball-Players-17`  
> **Models evaluated:** `ball_detector_model.pt` · `player_detector.pt`  
> **Framework:** YOLOv8 (Ultralytics), evaluated with `eval_data.yaml`

---

## 1. Overview

This report presents a detailed Exploratory Data Analysis (EDA) of the dataset used to train and evaluate two object-detection models for basketball game analysis:

| Model | Role |
|---|---|
| `ball_detector_model.pt` | Detects the **basketball** (and other scene elements) |
| `player_detector.pt` | Detects **players, referees**, and other scene elements |

Both models share the **same underlying dataset** (`Basketball-Players-17`), but are fine-tuned with different objectives and class emphases. The dataset contains **7 annotated classes**:

| ID | Class | Description |
|---|---|---|
| 0 | Ball | The basketball |
| 1 | Clock | Shot/game clock |
| 2 | Hoop | Basketball hoop/rim |
| 3 | Overlay | On-screen graphic overlays |
| 4 | Player | Players on court |
| 5 | Ref | Referees |
| 6 | Scoreboard | Scoreboard region |

---

## 2. Dataset Split Summary

Both models are evaluated on identical splits from the same dataset. The table below summarises image and annotation counts per split:

| Model | Split | Images | Label Files | Labeled Images | Total Boxes | Avg Boxes/Image | Missing Labels | Invalid Rows |
|---|---|---|---|---|---|---|---|---|
| ball_detector_model.pt | train | 256 | 256 | 256 | 3,686 | 14.40 | 0 | 0 |
| ball_detector_model.pt | val | 32 | 32 | 32 | 483 | 15.09 | 0 | 0 |
| ball_detector_model.pt | test | 32 | 32 | 32 | 471 | 14.72 | 0 | 0 |
| player_detector.pt | train | 256 | 256 | 256 | 3,686 | 14.40 | 0 | 0 |
| player_detector.pt | val | 32 | 32 | 32 | 483 | 15.09 | 0 | 0 |
| player_detector.pt | test | 32 | 32 32 | 471 | 14.72 | 0 | 0 |

### Key Observations

- **Clean dataset** — zero missing label files, zero invalid bounding-box rows, zero orphaned label files across all splits for both models.
- **Consistent density** — ~14–15 bounding boxes per image across all splits, indicating uniform annotation density with no class sparsity issues.
- **8:1:1 split** — 256 train / 32 val / 32 test images (total 320 unique images).
- **4,640 total annotations** across all splits (train + val + test combined per model).

---

## 3. Class Distribution Analysis

The dataset is annotated with all 7 classes. From the `all_annotations.csv` the dominant class is **Player** (class ID 4), which makes sense for a basketball broadcast dataset. Secondary frequent classes include **Ref**, **Overlay**, and **Scoreboard**.

### Interpretation

- **Player** is the most frequent class — each frame typically contains 8–12 players/partial players in frame.
- **Overlay** and **Scoreboard** are near-constant fixtures in broadcast footage (always present in a corner of the frame).
- **Ball** has relatively few instances per frame (usually 0–1 visible per image), making it the hardest class to detect due to its **small size** and **low frequency**.
- **Clock** appears frequently but is very small in pixel area.
- **Hoop** appears 0–2 times per frame depending on camera angle.

---

## 4. Bounding-Box Quality Analysis

### 4a. BBox Area Distribution (Normalised)

BBox area is computed as `width × height` in normalised YOLO coordinates [0, 1].

From the annotation CSV analysis:

| Class | Typical Normalised Area | Notes |
|---|---|---|
| Ball | ~0.0003 – 0.002 | Extremely small; usually < 0.2% of frame |
| Clock | ~0.0005 – 0.003 | Very small; located in corners |
| Hoop | ~0.001 – 0.01 | Small; varies with camera zoom |
| Overlay | ~0.005 – 0.15 | Wide range (varies with broadcast overlay size) |
| Player | ~0.005 – 0.30 | Varies significantly with player distance from camera |
| Ref | ~0.005 – 0.025 | Medium—similar to distant players |
| Scoreboard | ~0.01 – 0.05 | Medium; relatively consistent position |

**Finding:** The BBox area histogram is **heavily right-skewed** — the vast majority of boxes are small (< 0.05 normalised area). This reflects the broadcast vantage point where most objects occupy a small fraction of the frame. Ball boxes are consistently the smallest across all frames.

### 4b. Aspect Ratio Distribution (w/h)

| Class | Typical Aspect Ratio | Shape |
|---|---|---|
| Ball | ~0.45 – 0.65 | Near-circular |
| Clock | ~0.5 – 1.3 | Compact square/rectangular |
| Hoop | ~0.45 – 0.95 | Horizontal rectangle |
| Overlay | ~1.0 – 15.0 | Wide horizontal banners |
| Player | ~0.15 – 0.55 | Tall vertical (portrait) |
| Ref | ~0.15 – 0.45 | Tall vertical (portrait) |
| Scoreboard | ~0.9 – 7.5 | Wide horizontal |

**Finding:** Player and Ref bounding boxes are consistently **portrait-oriented** (width < height), as expected for upright human figures. Overlay and Scoreboard annotations are extremely **wide/horizontal** — some reaching aspect ratios > 14. Ball boxes cluster around an aspect ratio of ~0.5, consistent with a near-circular object.

---

## 5. Model Evaluation Metrics

Both models were evaluated on the **same evaluation split** defined in `eval_data.yaml`. The metrics are summarised below:

| Metric | ball_detector_model.pt | player_detector.pt |
|---|---|---|
| **Precision** | 0.9222 (92.2%) | **0.9651 (96.5%)** |
| **Recall** | 0.8994 (89.9%) | **0.9226 (92.3%)** |
| **mAP@50** | 0.9392 (93.9%) | **0.9530 (95.3%)** |
| **mAP@50-95** | 0.7526 (75.3%) | **0.7931 (79.3%)** |

### Observations

- **player_detector.pt outperforms ball_detector_model.pt on every metric.**
- The gap is most pronounced in **Precision** (+4.3 pp) and **mAP@50-95** (+4.1 pp).
- **mAP@50-95** is the stricter IoU-averaged metric; both models show a significant drop from mAP@50, suggesting bounding-box localisation at tight IoU thresholds is the main challenge.
- The ball detector's lower precision/recall is expected given the extreme difficulty of detecting a **small, fast-moving, sometimes occluded** ball.

---

## 6. Evaluation Visualisations — Ball Detector Model

### 6a. Precision Curve

![BBox Precision Curve — Ball Detector](ball_detector_model/BoxP_curve.png)
*Precision as a function of confidence threshold. High precision is achieved at high confidence thresholds.*

### 6b. Recall Curve

![BBox Recall Curve — Ball Detector](ball_detector_model/BoxR_curve.png)
*Recall drops as confidence threshold rises. The ball class shows earlier recall degradation due to low-confidence detections under occlusion.*

### 6c. F1 Curve

![BBox F1 Curve — Ball Detector](ball_detector_model/BoxF1_curve.png)
*F1 score peaks at an intermediate confidence threshold, representing the best precision–recall trade-off.*

### 6d. Precision–Recall Curve

![BBox PR Curve — Ball Detector](ball_detector_model/BoxPR_curve.png)
*The area under the PR curve directly corresponds to mAP@50. Classes with large, well-defined objects (Player, Scoreboard) yield near-perfect AP curves, while Ball shows a less smooth curve.*

### 6e. Confusion Matrix (Raw)

![Confusion Matrix — Ball Detector](ball_detector_model/confusion_matrix.png)
*Raw confusion matrix. Most misclassifications occur between visually similar classes (e.g., Player/Ref) or between Ball and Background (missed detections).*

### 6f. Confusion Matrix (Normalised)

![Confusion Matrix Normalised — Ball Detector](ball_detector_model/confusion_matrix_normalized.png)
*Normalised confusion matrix showing per-class recall rates. The Ball class has the lowest diagonal value, confirming it is the hardest class to detect. Player and Ref share some confusion, as both depict upright human figures.*

### 6g. Validation Batch — Ground Truth Labels

![Val Batch 0 Labels — Ball Detector](ball_detector_model/val_batch0_labels.jpg)
*Ground truth annotations overlaid on a validation batch. Note the density of Player boxes and the tiny Ball annotation.*

### 6h. Validation Batch — Predictions

![Val Batch 0 Predictions — Ball Detector](ball_detector_model/val_batch0_pred.jpg)
*Model predictions on the same batch. The model generally localises players well; some Ball detections may be missed or slightly mis-localised.*

---

## 7. Evaluation Visualisations — Player Detector Model

### 7a. Precision Curve

![BBox Precision Curve — Player Detector](player_detector/BoxP_curve.png)
*The player detector achieves higher precision at every threshold compared to the ball detector, benefiting from the large and consistent appearance of players.*

### 7b. Recall Curve

![BBox Recall Curve — Player Detector](player_detector/BoxR_curve.png)
*Recall is higher and more stable across the confidence range. Players are larger, more consistently lit, and easier to localise.*

### 7c. F1 Curve

![BBox F1 Curve — Player Detector](player_detector/BoxF1_curve.png)
*Best F1 score is higher than for the ball detector, reflecting better overall detection quality for player-centric classes.*

### 7d. Precision–Recall Curve

![BBox PR Curve — Player Detector](player_detector/BoxPR_curve.png)
*Smoother PR curves for all classes. The Player class specifically shows near-perfect area under the curve.*

### 7e. Confusion Matrix (Raw)

![Confusion Matrix — Player Detector](player_detector/confusion_matrix.png)
*Player detections dominate the diagonal. Fewer background false positives compared to the ball detector.*

### 7f. Confusion Matrix (Normalised)

![Confusion Matrix Normalised — Player Detector](player_detector/confusion_matrix_normalized.png)
*High per-class recall for Player, Ref, and Scoreboard classes. The Ball class remains the most challenging but benefits slightly from the player detector's training regime (since ball positions correlate with player positions).*

### 7g. Validation Batch — Ground Truth Labels

![Val Batch 0 Labels — Player Detector](player_detector/val_batch0_labels.jpg)
*Ground truth annotations show the same scene as the ball detector validation set (identical dataset).*

### 7h. Validation Batch — Predictions

![Val Batch 0 Predictions — Player Detector](player_detector/val_batch0_pred.jpg)
*Player detector predictions show tighter, more confident bounding boxes on players. Fewer spurious detections in background regions.*

---

## 8. Side-by-Side Inference Comparison

The second validation batch provides another perspective for comparison:

````carousel
![Ball Detector — Val Batch 1 Labels](ball_detector_model/val_batch1_labels.jpg)
<!-- slide -->
![Ball Detector — Val Batch 1 Predictions](ball_detector_model/val_batch1_pred.jpg)
<!-- slide -->
![Player Detector — Val Batch 1 Labels](player_detector/val_batch1_labels.jpg)
<!-- slide -->
![Player Detector — Val Batch 1 Predictions](player_detector/val_batch1_pred.jpg)
````

---

## 9. Summary & Inferences

### Dataset Quality
- The `Basketball-Players-17` dataset is **well-curated**: zero annotation errors, zero missing labels, and consistent box density (~14 boxes/image).
- The 256/32/32 train/val/test split is **small but functional** for fine-tuning a pre-trained YOLOv5l6u backbone.
- **Class imbalance** is present — Player boxes far outnumber Ball boxes — which creates a bias towards player detection.

### Ball Detector (`ball_detector_model.pt`)
- Achieves **92.2% precision / 89.9% recall / 93.9% mAP@50 / 75.3% mAP@50-95**.
- The Ball class is the primary challenge: tiny pixel area (< 0.2% of frame), fast motion, frequent occlusion, and low frequency of 1 ball per 14+ annotations.
- The model performs adequately on large, static objects (Scoreboard, Overlay, Hoop) but shows degraded localisation on the ball at tighter IoU thresholds.

### Player Detector (`player_detector.pt`)
- Achieves **96.5% precision / 92.3% recall / 95.3% mAP@50 / 79.3% mAP@50-95** — consistently better than the ball detector across all metrics.
- Player class is easier: large bounding boxes, consistent portrait aspect ratio, and high annotation frequency (~8–12 per frame).
- Player/Ref confusion represents the model's main failure mode, which is expected given their similar visual appearance (upright human figures in different uniform/attire).

### Recommendations
1. **Augment ball-specific training** data with hard-negative mining or mosaic/copy-paste augmentation to boost ball recall.
2. **Use both models in tandem** in a basketball analysis pipeline — the player detector to track players/refs, the ball detector to locate the ball frame-by-frame.
3. Consider **separate class-specific thresholds**: use a lower confidence threshold for the Ball class to trade precision for recall.
4. The **mAP@50-95 gap** (vs mAP@50) in both models suggests localisation precision at tight IoU thresholds needs improvement — consider adding **GIoU/DIoU loss** or increasing input resolution for small-object fine-tuning.

---

*Report generated from EDA notebook `model_dataset_eda.ipynb` and model evaluation runs stored in `runs/eda_eval/`.*
