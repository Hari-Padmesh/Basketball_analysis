# 🏀 Basketball Video Analysis

An end-to-end computer vision pipeline that analyses basketball game footage and automatically extracts rich insights: player & ball tracking, team assignment, ball possession, passes, interceptions, a bird's-eye tactical view, and per-player speed / distance statistics.

---

## 📋 Table of Contents

1. [Features](#features)
2. [Process Flow](#process-flow)
3. [Technical Architecture](#technical-architecture)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Setup & Installation](#setup--installation)
7. [Usage](#usage)
8. [Configuration](#configuration)
9. [Models](#models)
10. [Output](#output)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Player Tracking** | Detects and tracks all players across every frame using YOLOv8 + ByteTrack |
| **Ball Tracking** | Detects the basketball and interpolates missing detections |
| **Team Assignment** | Classifies each player into a team using the Fashion-CLIP model |
| **Ball Possession** | Determines which player (and therefore which team) holds the ball at every frame |
| **Pass Detection** | Identifies successful intra-team ball transfers |
| **Interception Detection** | Identifies cross-team ball turnovers |
| **Tactical View** | Transforms the broadcast camera perspective into a top-down court view via homography |
| **Speed & Distance** | Calculates real-world player speed (km/h) and cumulative distance (m) |
| **Annotated Video** | Produces a fully annotated output video with all overlays |

---

## 🔄 Process Flow

```
Input Video
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Frame Extraction                             │
│                    (OpenCV VideoCapture)                            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│  Player Tracker  │   │   Ball Tracker   │   │ Court Keypoint Det. │
│  (YOLOv8 + ByteTrack)│ │ (YOLOv8)        │   │    (YOLOv8 Pose)    │
└────────┬─────────┘   └───────┬──────────┘   └──────────┬──────────┘
         │                     │                          │
         │             ┌───────▼──────────┐               │
         │             │ Remove Erroneous │               │
         │             │  Detections &    │               │
         │             │  Interpolate     │               │
         │             └───────┬──────────┘               │
         │                     │                          │
         ▼                     │               ┌──────────▼──────────┐
┌──────────────────┐           │               │  Keypoint Validation │
│  Team Assigner   │           │               │  (Proportion Check) │
│ (Fashion-CLIP)   │           │               └──────────┬──────────┘
└────────┬─────────┘           │                          │
         │                     │               ┌──────────▼──────────┐
         └──────────┬──────────┘               │ Tactical View Conv. │
                    │                          │   (Homography)      │
                    ▼                          └──────────┬──────────┘
         ┌──────────────────┐                             │
         │  Ball Acquisition│◄────────────────────────────┘
         │    Detector      │
         └────────┬─────────┘
                  │
        ┌─────────▼──────────┐
        │ Pass & Interception│
        │    Detector        │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │ Speed & Distance   │
        │    Calculator      │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  Drawing / Overlay │
        │  (All Drawers)     │
        └─────────┬──────────┘
                  │
                  ▼
           Output Video
```

### Step-by-Step Description

1. **Frame Extraction** – OpenCV reads the input video and returns a list of BGR frames.

2. **Player Tracking** – A fine-tuned YOLOv8 model detects players; SuperVision's *ByteTrack* assigns persistent IDs across frames. Detections are processed in batches of 20 frames.

3. **Ball Tracking** – A separate YOLOv8 model detects the basketball. The highest-confidence detection per frame is kept. Wrong detections (ball suddenly appearing far from its previous position) are removed and the remaining gaps are filled by linear interpolation using Pandas.

4. **Court Keypoint Detection** – A YOLOv8 Pose model detects 18 pre-defined court landmark points (corners, free-throw lines, mid-court, etc.) per frame, also in batches of 20.

5. **Keypoint Validation** – Proportional-distance checks filter out hallucinated keypoints (80% error margin).

6. **Team Assignment** – For each frame, every visible player's bounding box is cropped and passed to [Fashion-CLIP](https://huggingface.co/patrickjohncyh/fashion-clip) (`patrickjohncyh/fashion-clip`). The model scores the image against text prompts such as `"white shirt"` and `"dark red shirt"` to determine team membership. To avoid accumulating stale IDs over long sequences, the player-to-team mapping cache is reset every 50 frames while still classifying every player in every frame.

7. **Ball Possession / Acquisition Detection** – For each frame the system finds which player is closest to the ball (or whose bounding box contains the most of the ball). A player must hold possession for at least 10 consecutive frames before it is confirmed.

8. **Pass & Interception Detection** – Possession transfers between players of the *same* team are counted as **passes**; transfers between players of *different* teams are counted as **interceptions**.

9. **Tactical View Conversion** – The 18 detected court keypoints are matched against known real-world court coordinates. OpenCV's `findHomography` (RANSAC) computes a perspective-transform matrix. A consistency check between candidate and previous homography prevents sudden mirrored flips. Player foot positions are projected into the 300 × 161-pixel tactical canvas.

10. **Speed & Distance Calculation** – Tactical pixel positions are scaled to real-world metres (28 m × 15 m court). Per-frame distances are accumulated for cumulative distance; a rolling 5-frame window is used to estimate speed in km/h.

11. **Drawing & Annotation** – A set of dedicated `Drawer` classes renders:
    - Coloured ellipses and team colours under each player
    - A triangle indicator over the ball-holder
    - Court keypoints
    - Team ball-control percentage bar
    - Pass / interception event labels
    - Per-player speed and cumulative distance
    - Minimap (tactical view overlay) in the top-left corner
    - Frame counter

12. **Video Export** – Annotated frames are written to an AVI file using OpenCV's `VideoWriter` (XVID codec, 24 fps).

---

## 🏗️ Technical Architecture

### Models

| Model | Purpose | Format |
|---|---|---|
| `models/player_detector.pt` | YOLOv8 player detection | PyTorch |
| `models/ball_detector_model.pt` | YOLOv8 basketball detection | PyTorch |
| `models/court_keypoint_detector.pt` | YOLOv8-Pose court landmark detection | PyTorch |
| `patrickjohncyh/fashion-clip` | CLIP-based team jersey classification | HuggingFace Hub |

### Key Technologies

| Library | Version | Role |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | ≥ 8.0 | YOLOv8 inference & training |
| [Supervision](https://github.com/roboflow/supervision) | ≥ 0.18 | ByteTrack multi-object tracker, detection helpers |
| [OpenCV](https://opencv.org/) | ≥ 4.8 | Video I/O, homography, drawing |
| [HuggingFace Transformers](https://huggingface.co/docs/transformers) | ≥ 4.30 | Fashion-CLIP model & processor |
| [PyTorch](https://pytorch.org/) | ≥ 2.0 | Deep learning backend |
| [NumPy](https://numpy.org/) | ≥ 1.24 | Array operations, homography maths |
| [Pandas](https://pandas.pydata.org/) | ≥ 2.0 | Ball-position interpolation |
| [Pillow](https://python-pillow.org/) | ≥ 9.5 | Image conversion for CLIP |

### Stub / Cache System

Heavy inference passes (player tracking, ball tracking, keypoint detection, team assignment) can be **cached to disk** as pickle files (`stubs/` directory). On subsequent runs the system reads from the stub if the frame count matches, skipping the expensive forward pass entirely.

---

## 📁 Project Structure

```
Basketball_analysis/
├── main.py                          # Entry point – orchestrates the full pipeline
├── requirements.txt                 # Python dependencies
│
├── configs/
│   └── configs.py                   # Global paths & default constants
│
├── trackers/
│   ├── player_tracker.py            # YOLOv8 + ByteTrack player tracker
│   └── ball_tracker.py              # YOLOv8 ball tracker + interpolation
│
├── court_key_point_detector/
│   └── court_keypoint_detector.py   # YOLOv8-Pose court landmark detector
│
├── team_assigner/
│   └── team_assigner.py             # Fashion-CLIP team classification
│
├── ball_acquisition/
│   └── ball_acquisition_detector.py # Ball possession logic
│
├── pass_and_interception_detector/
│   └── pass_and_interception_detector.py  # Pass & interception detection
│
├── tactical_view_converter/
│   ├── tactical_view_converter.py   # Homography-based top-down view
│   └── homography.py                # Homography helper utilities
│
├── speed_and_distance_calculator/
│   └── speed_and_distance_calculator.py  # Real-world speed & distance
│
├── drawers/
│   ├── player_tracks_drawer.py      # Player bounding boxes & team colours
│   ├── ball_tracks_drawer.py        # Ball indicator
│   ├── court_key_points_drawer.py   # Court landmark overlay
│   ├── team_ball_control_drawer.py  # Team possession percentage
│   ├── pass_and_interceptions_drawer.py  # Pass / interception labels
│   ├── tactical_view_drawer.py      # Minimap overlay
│   ├── speed_and_distance_drawer.py # Speed & distance labels
│   ├── frame_number_drawer.py       # Frame counter
│   └── utils.py                     # Shared drawing helpers
│
├── utils/
│   ├── video_utils.py               # read_video / save_video
│   ├── bbox_utils.py                # Bounding box helpers
│   └── stub_utils.py                # Pickle stub read/write
│
├── stubs/                           # Cached inference results (auto-generated)
│   ├── player_track_stubs.pkl
│   ├── ball_track_stubs.pkl
│   ├── court_key_points_stub.pkl
│   └── player_assignment_stub.pkl
│
├── images/
│   └── basketball_court.png         # Top-down court image for tactical view
│
├── input_videos/                    # Place your input video(s) here
├── output_videos/                   # Annotated output videos (auto-generated)
│
├── models/                          # Pre-trained YOLO model weights (download separately)
│   ├── player_detector.pt
│   ├── ball_detector_model.pt
│   └── court_keypoint_detector.pt
│
├── training_notebooks/              # Jupyter notebooks for model training & EDA
│   ├── basketball_player_detection_training.ipynb
│   ├── basketball_ball_detection_training.ipynb
│   └── model_dataset_eda.ipynb
│
└── Documentation/                   # Research documentation & ablation studies
    └── Ablation_Studies_Hyperparameter_Tuning.md
```

---

## ✅ Prerequisites

- **Python 3.9 – 3.11** (recommended)
- **CUDA-capable GPU** (strongly recommended for real-time-ish processing; CPU-only is supported but slow)
- **FFmpeg** (optional, for playing back the output AVI)

---

## 🛠️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hari-Padmesh/Basketball_analysis.git
cd Basketball_analysis
```

### 2. Create and activate a virtual environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Make sure you install the CUDA-compatible version of PyTorch **before** running the command above, or replace the torch lines in `requirements.txt` with the wheel from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

### 4. Download model weights

Place the three YOLO model files in a `models/` directory at the project root:

```
models/
├── player_detector.pt
├── ball_detector_model.pt
└── court_keypoint_detector.pt
```

The Fashion-CLIP model (`patrickjohncyh/fashion-clip`) is downloaded automatically from the HuggingFace Hub on first run.

### 5. Add an input video

Copy your basketball game video into the `input_videos/` directory:

```
input_videos/
└── your_game.mp4
```

---

## 🚀 Usage

### Basic run

```bash
python main.py input_videos/your_game.mp4
```

### Custom output path

```bash
python main.py input_videos/your_game.mp4 --output_video output_videos/annotated_game.avi
```

### Custom stub directory

```bash
python main.py input_videos/your_game.mp4 --stub_path stubs/
```

### All options

```
usage: main.py [-h] [--output_video OUTPUT_VIDEO] [--stub_path STUB_PATH] input_video

Basketball Video Analysis

positional arguments:
  input_video           Path to input video file

optional arguments:
  -h, --help            show this help message and exit
  --output_video        Path to output video file (default: output_videos/output_video.avi)
  --stub_path           Path to stub directory (default: stubs)
```

### First run vs. subsequent runs

- **First run:** All detectors run inference and cache their results to `stubs/`. This may take several minutes depending on video length and hardware.
- **Subsequent runs:** The stubs are reused automatically, making reruns very fast.

To force re-inference (e.g., after changing a model), delete the relevant `.pkl` files in `stubs/`.

---

## ⚙️ Configuration

Edit `configs/configs.py` to change default paths:

```python
STUBS_DEFAULT_PATH = 'stubs'
PLAYER_DETECTOR_PATH = 'models/player_detector.pt'
BALL_DETECTOR_PATH = 'models/ball_detector_model.pt'
COURT_KEYPOINT_DETECTOR_PATH = 'models/court_keypoint_detector.pt'
OUTPUT_VIDEO_PATH = 'output_videos/output_video.avi'
```

To customise **team jersey descriptions** for a different game, pass arguments to `TeamAssigner`:

```python
team_assigner = TeamAssigner(
    team_1_class_name="white shirt",
    team_2_class_name="dark red shirt"
)
```

---

## 🎯 Models

### Training your own models

Training notebooks are provided in `training_notebooks/`:

| Notebook | Purpose |
|---|---|
| `basketball_player_detection_training.ipynb` | Fine-tune YOLOv8 for player detection |
| `basketball_ball_detection_training.ipynb` | Fine-tune YOLOv8 for ball detection |
| `model_dataset_eda.ipynb` | Exploratory data analysis of training datasets |

Refer to `Documentation/Ablation_Studies_Hyperparameter_Tuning.md` for hyperparameter tuning results and ablation study findings.

### Court Keypoints Layout

The system uses **18 court keypoints** mapped to real-world NBA/FIBA dimensions (28 m × 15 m):

```
Index  Location
─────  ────────────────────────────────────────────────────
 0     Left baseline – top corner
 1     Left baseline – 0.91 m from top
 2     Left baseline – top of left paint (5.18 m from top)
 3     Left baseline – bottom of left paint (10 m from top)
 4     Left baseline – 14.1 m from top
 5     Left baseline – bottom corner
 6     Mid-court line – bottom
 7     Mid-court line – top
 8     Left free-throw line – top (5.18 m from top)
 9     Left free-throw line – bottom (10 m from top)
10     Right baseline – bottom corner
11     Right baseline – 14.1 m from top
12     Right baseline – bottom of right paint (10 m from top)
13     Right baseline – top of right paint (5.18 m from top)
14     Right baseline – 0.91 m from top
15     Right baseline – top corner
16     Right free-throw line – top (5.18 m from top)
17     Right free-throw line – bottom (10 m from top)
```

---

## 📊 Output

The pipeline produces a single annotated video (`output_videos/output_video.avi`) containing:

- **Coloured ellipses** under each player (team colour coded)
- **Red triangle** above the player with ball possession
- **Team ball-control percentage** overlay (bottom-right)
- **Pass / interception** event labels
- **Speed (km/h) and distance (m)** displayed below each player
- **Tactical minimap** (top-left corner) showing a bird's-eye view of player positions on the court
- **Frame number** counter

---

## 📄 License

This project is provided for educational and research purposes.
