# 4.5 Ablation Studies and Hyperparameter Tuning

This section presents a structured ablation and tuning strategy for the three project models:
- ball_detector_model.pt
- Player_detector.pt
- court_keypoint_detector.pt

The goal is to isolate which training decisions most improve detection quality, then converge to robust hyperparameter settings with repeatable gains.

## 4.5.1 Current Baseline Snapshot

From the current evaluation export:
- Source file: runs/eda_reports/model_eval_metrics.csv
- Ball detector baseline: Precision 0.922, Recall 0.899, mAP50 0.939, mAP50-95 0.753
- Player detector baseline: Precision 0.965, Recall 0.923, mAP50 0.953, mAP50-95 0.793
- Court keypoint detector baseline: not yet included in the current evaluation export

Inference:
- Player detection is currently stronger than ball detection on both mAP50 and mAP50-95.
- The larger gap at mAP50-95 indicates room for localization refinement, especially for the ball model where object scale is small and position sensitivity is high.
- A dedicated ablation sequence should prioritize localization improvements first, then confidence calibration.

## 4.5.2 Ablation Study Design

Use one-factor-at-a-time ablation with a fixed baseline recipe.
Each ablation changes only one variable while all others remain fixed.
Use 3 independent seeds per experiment and report mean plus standard deviation.

Recommended report metrics:
- Primary: mAP50-95
- Secondary: mAP50, Precision, Recall
- Stability: standard deviation across seeds
- Efficiency: training time per epoch and total wall-clock

### A. Ball Detector Ablations

Rationale:
Ball objects are small, fast, often motion-blurred, and partially occluded. Performance usually depends heavily on image scale and augmentation balance.

Ablation matrix:
1. Input size: 640, 768, 896
2. Mosaic intensity: 1.0, 0.5, 0.0 in final epochs
3. HSV augmentation strength: low, medium, high
4. Copy-paste or mixup: off versus on
5. Loss gain balance: box gain up by 10 to 20 percent
6. Confidence threshold for validation diagnostics: 0.001 to 0.25 sweep

Expected outcomes:
- Higher input size should improve recall for small ball instances.
- Too aggressive mosaic can hurt final localization precision.
- Moderate color augmentation should improve robustness under lighting variation.

### B. Player Detector Ablations

Rationale:
Players are larger objects with frequent crowding and partial overlap. NMS and augmentation for occlusion are critical.

Ablation matrix:
1. Input size: 640, 768
2. NMS IoU during validation sweep: 0.5, 0.6, 0.7
3. Mosaic schedule: always on versus disabled in last 10 to 20 epochs
4. Perspective and scale augmentation ranges: narrow versus wide
5. Label smoothing: 0.0 versus 0.05
6. Warmup epochs: 3 versus 5

Expected outcomes:
- Mildly higher IoU in NMS may reduce duplicate detections in player clusters.
- Late-stage reduction of heavy augmentation often improves final precision.

### C. Court Keypoint Detector Ablations

Rationale:
Court keypoints require geometric consistency and spatial precision under camera angle changes.

Ablation matrix:
1. Input size: 640, 896, 1024
2. Geometric augmentation: rotation and perspective low versus medium
3. Horizontal flip: off versus on only if keypoint semantics remain valid
4. Loss weight emphasis on localization components
5. Temporal frame sampling strategy in dataset creation: dense versus sparse
6. Background clutter suppression using targeted crops

Expected outcomes:
- Higher input size should improve keypoint localization if memory budget allows.
- Geometry-aware augmentation should improve viewpoint generalization.

## 4.5.3 Hyperparameter Tuning Strategy

Use staged tuning instead of full-grid search to reduce cost.

Stage 1: coarse search
- Learning rate: 0.001, 0.003, 0.01
- Weight decay: 0.0001, 0.0005, 0.001
- Batch size: 8, 16, 32 if memory permits
- Optimizer: SGD versus AdamW
- Epochs: fixed at moderate budget for comparison

Stage 2: local refinement around best Stage 1 configuration
- Learning rate neighborhood around best value
- Warmup epochs and momentum fine-tuning
- Augmentation intensity fine-tuning

Stage 3: robustness validation
- Re-run best settings with 3 to 5 seeds
- Report mean, standard deviation, and worst-case seed result

## 4.5.4 Model-Specific Tuning Priorities

Ball detector priority order:
1. Input size
2. Learning rate and warmup
3. Augmentation balance for small objects
4. Box loss gain

Player detector priority order:
1. Augmentation schedule and NMS behavior
2. Learning rate and optimizer
3. Batch size scaling
4. Label smoothing

Court keypoint detector priority order:
1. Input size and geometry augmentation
2. Localization-centric loss weighting
3. Dataset frame diversity and keypoint visibility filtering
4. Optimizer and learning rate schedule

## 4.5.5 Example Experiment Table Template

Use this template for each model.

| Experiment ID | Variable Changed | Value | mAP50 | mAP50-95 | Precision | Recall | Train Time | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| B0 | Baseline | Default | 0.939 | 0.753 | 0.922 | 0.899 | - | Current ball baseline |
| B1 | Input size | 768 |  |  |  |  |  |  |
| B2 | Input size | 896 |  |  |  |  |  |  |
| B3 | Mosaic final phase | Off in last epochs |  |  |  |  |  |  |

Create similar tables for player and court keypoint models.

## 4.5.6 Statistical and Practical Inference Rules

Use these rules before claiming improvement:
1. A change is practically useful only if mean mAP50-95 improves and standard deviation does not increase sharply.
2. Prefer configurations with balanced precision and recall over single-metric spikes.
3. If two settings are close in accuracy, choose the faster and more stable one.
4. Keep a held-out validation split unchanged across all experiments.

## 4.5.7 Recommended Final Narrative for Thesis or Report

Suggested wording:
- Ablation studies showed that input resolution and augmentation scheduling were the most influential factors for detection quality.
- For the ball detector, localization-sensitive settings produced the largest gain due to the small object scale.
- For the player detector, tuning crowd-robust settings improved precision without sacrificing recall.
- For the court keypoint detector, geometry-aware augmentation and high-resolution training were the most relevant factors for stable keypoint localization.
- Final hyperparameters were selected using multi-seed stability, not single-run best metrics, improving reproducibility.

## 4.5.8 Next Practical Steps

1. Add court keypoint model evaluation output into runs/eda_reports/model_eval_metrics.csv so all three models have baseline rows.
2. Run Stage 1 coarse tuning for each model with fixed seed set and identical split.
3. Fill the experiment tables above and finalize the best configuration per model.
4. Re-export final metrics and confusion matrices for the final report appendix.
