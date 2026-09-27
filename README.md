# Human Activity Recognition

Real-time webcam person detection, tracking, and activity classification
(Walking / Sitting / Standing) using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
and [ByteTrack](https://github.com/ifzhang/ByteTrack), built with OpenCV and PyTorch.

## Features

- Live webcam capture with configurable resolution / FPS
- YOLOv8-based person detection with persistent multi-object tracking (ByteTrack)
- Per-track activity classification (Walking, Sitting, Standing) using smoothed
  foot-point velocity and bounding-box aspect ratio, with a stability filter to
  avoid label flicker
- Automatic GPU acceleration (CUDA) when available, with FP16 + cuDNN/TF32 tuning
- On-screen FPS, inference time, people count, and active track count
- Adjustable confidence threshold and optional per-track movement trails
- Automatic cleanup of stale tracks

## Requirements

- Python 3.9+
- A webcam accessible to OpenCV
- (Optional but recommended) an NVIDIA GPU with CUDA for real-time performance

## Installation

```bash
git clone <this-repo-url>
cd <this-repo>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The first run will automatically download the `yolov8s.pt` weights via Ultralytics.

## Usage

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset tracking |
| `S` | Show / hide statistics overlay |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

## Configuration

All key parameters live at the top of `main.py`:

| Setting | Description | Default |
|---|---|---|
| `MODEL_PATH` | YOLO weights file | `yolov8s.pt` |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` / `CAMERA_FPS` | Requested camera capture settings | `1280x720 @ 60` |
| `PROCESSING_WIDTH` / `PROCESSING_HEIGHT` | Resolution the model runs inference on | `800x600` |
| `CONFIDENCE_THRESHOLD` | Minimum detection confidence | `0.35` |
| `IOU_THRESHOLD` | NMS IoU threshold | `0.45` |
| `MAX_DETECTIONS` | Max detections per frame | `15` |
| `WALKING_THRESHOLD` | Foot-point pixel velocity above which a track is considered walking | `2.5` |
| `ACTIVITY_STABILITY_FRAMES` | Frames a new activity must persist before it's confirmed | `3` |
| `TRACKER_CONFIG` | Ultralytics tracker config | `bytetrack.yaml` |
| `SHOW_STATS` / `SHOW_TRACK_LINES` | Display toggles | `True` / `False` |

## How activity classification works

1. Each tracked person's foot point (bottom-center of the bounding box) is
   recorded every frame in a short position history.
2. Frame-to-frame displacement is smoothed over a small velocity window.
3. If smoothed velocity exceeds `WALKING_THRESHOLD`, the raw activity is
   **Walking**; otherwise bounding-box aspect ratio is used to distinguish
   **Sitting** (wider/shorter box) from **Standing**.
3. A stability filter requires the new activity to be seen for
   `ACTIVITY_STABILITY_FRAMES` consecutive frames before it replaces the
   track's confirmed activity, reducing flicker between labels.

## Notes / Limitations

- Activity classification is a lightweight heuristic (velocity + aspect
  ratio), not a trained action-recognition model — it works best with a
  roughly front-on, stationary camera and may misclassify unusual poses or
  camera angles.
- Camera backend defaults to DirectShow on Windows and V4L2 on Linux.
- Only the `person` class (COCO class `0`) is detected/tracked.

## License

See [LICENSE](LICENSE).
