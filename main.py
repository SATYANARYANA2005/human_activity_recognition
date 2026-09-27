#pip install "ultralytics>=8.0.0" "opencv-python>=4.7.0" "numpy>=1.23.0" "torch>=2.0.0"
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolov8s.pt"

# Camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

# Processing resolution
PROCESSING_WIDTH = 800
PROCESSING_HEIGHT = 600

# Detection
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 15

# Activity detection
WALKING_THRESHOLD = 2.5
ACTIVITY_STABILITY_FRAMES = 3

# History
POSITION_HISTORY_LENGTH = 12
VELOCITY_HISTORY_LENGTH = 5

# Tracking
TRACKER_CONFIG = "bytetrack.yaml"

# Display
SHOW_STATS = True
SHOW_TRACK_LINES = False

# Classes
PERSON_CLASS = 0


# ============================================================
# GPU / DEVICE SETUP
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("PERSON ACTIVITY DETECTION")
print("=" * 60)

print(f"🚀 Device: {device.upper()}")

if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(
        f"   GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )

    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# LOAD MODEL
# ============================================================

print("\n🧠 Loading YOLO model...")

model = YOLO(MODEL_PATH)
model.to(device)

# Fuse layers if possible
try:
    model.fuse()
    print("✅ Model fused")
except Exception as e:
    print(f"⚠️ Model fusion skipped: {e}")


# ============================================================
# WARM-UP
# ============================================================

print("🔥 Warming up model...")

try:
    dummy = np.zeros(
        (PROCESSING_HEIGHT, PROCESSING_WIDTH, 3),
        dtype=np.uint8
    )

    with torch.inference_mode():
        for _ in range(3):
            model.predict(
                dummy,
                imgsz=640,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                device=device,
                half=(device == "cuda"),
                classes=[PERSON_CLASS],
                max_det=MAX_DETECTIONS
            )

    print("✅ Warm-up complete")

except Exception as e:
    print(f"⚠️ Warm-up failed: {e}")


# ============================================================
# ACTIVITY TRACK
# ============================================================

class ActivityTrack:
    """
    Stores movement and activity information for one person.
    """

    __slots__ = (
        "last_seen",
        "last_activity",
        "candidate_activity",
        "candidate_frames",
        "position_history",
        "velocity_history",
        "bbox_history",
        "last_bbox",
    )

    def __init__(self, bbox, center, frame_number):
        self.last_seen = frame_number

        self.last_activity = "Standing"

        self.candidate_activity = "Standing"
        self.candidate_frames = 0

        self.position_history = deque(
            maxlen=POSITION_HISTORY_LENGTH
        )

        self.velocity_history = deque(
            maxlen=VELOCITY_HISTORY_LENGTH
        )

        self.bbox_history = deque(maxlen=5)

        self.position_history.append(center)
        self.bbox_history.append(bbox)

        self.last_bbox = bbox


# Dictionary:
# YOLO tracking ID -> ActivityTrack
activity_tracks = {}


# ============================================================
# ACTIVITY COLORS
# ============================================================

ACTIVITY_COLORS = {
    "Walking": (255, 200, 0),
    "Sitting": (255, 165, 0),
    "Standing": (0, 255, 0),
}


def get_activity_color(activity):
    return ACTIVITY_COLORS.get(
        activity,
        (0, 255, 0)
    )


# ============================================================
# ACTIVITY CLASSIFICATION
# ============================================================

def classify_activity(
    track_id,
    x1,
    y1,
    x2,
    y2,
    frame_number
):
    """
    Classifies a person as:
        Walking
        Sitting
        Standing

    Uses smoothed movement + bounding box geometry.
    """

    # --------------------------------------------------------
    # Basic geometry
    # --------------------------------------------------------

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5

    # Bottom-center is useful for human movement tracking
    foot_x = center_x
    foot_y = y2

    center = (foot_x, foot_y)
    bbox = (x1, y1, x2, y2)

    aspect_ratio = width / height

    # --------------------------------------------------------
    # Create track
    # --------------------------------------------------------

    if track_id not in activity_tracks:
        activity_tracks[track_id] = ActivityTrack(
            bbox,
            center,
            frame_number
        )

    track = activity_tracks[track_id]

    # --------------------------------------------------------
    # Calculate movement
    # --------------------------------------------------------

    if len(track.position_history) > 0:

        previous_x, previous_y = track.position_history[-1]

        movement = np.sqrt(
            (foot_x - previous_x) ** 2 +
            (foot_y - previous_y) ** 2
        )

        track.velocity_history.append(movement)

    # Add current position
    track.position_history.append(center)

    # --------------------------------------------------------
    # Smoothed velocity
    # --------------------------------------------------------

    if len(track.velocity_history) > 0:

        average_velocity = float(
            np.mean(track.velocity_history)
        )

    else:
        average_velocity = 0.0

    # --------------------------------------------------------
    # Determine raw activity
    # --------------------------------------------------------

    raw_activity = "Standing"

    # Walking
    if (
        len(track.velocity_history) >= 3
        and average_velocity > WALKING_THRESHOLD
    ):
        raw_activity = "Walking"

    # Sitting
    elif (
        aspect_ratio > 1.20
        or (
            height < width * 1.15
            and aspect_ratio > 1.15
        )
    ):
        raw_activity = "Sitting"

    # Otherwise Standing
    else:
        raw_activity = "Standing"

    # --------------------------------------------------------
    # Activity stability filter
    # --------------------------------------------------------

    if raw_activity == track.last_activity:

        # Already stable
        track.candidate_activity = raw_activity
        track.candidate_frames = 0

    else:

        # New candidate activity
        if raw_activity == track.candidate_activity:
            track.candidate_frames += 1
        else:
            track.candidate_activity = raw_activity
            track.candidate_frames = 1

        # Confirm only after enough frames
        if track.candidate_frames >= ACTIVITY_STABILITY_FRAMES:

            track.last_activity = raw_activity
            track.candidate_frames = 0

    # --------------------------------------------------------
    # Update track
    # --------------------------------------------------------

    track.last_seen = frame_number
    track.last_bbox = bbox
    track.bbox_history.append(bbox)

    return (
        track.last_activity,
        average_velocity
    )


# ============================================================
# REMOVE OLD TRACKS
# ============================================================

def cleanup_old_tracks(
    current_frame,
    max_age=60
):
    """
    Removes activity tracks that haven't been detected
    for a while.
    """

    old_ids = []

    for track_id, track in activity_tracks.items():

        if current_frame - track.last_seen > max_age:
            old_ids.append(track_id)

    for track_id in old_ids:
        del activity_tracks[track_id]


# ============================================================
# CAMERA SETUP
# ============================================================

print("\n📷 Opening camera...")

if os.name == "nt":
    camera_backend = cv2.CAP_DSHOW
else:
    camera_backend = cv2.CAP_V4L2


cap = cv2.VideoCapture(
    0,
    camera_backend
)

if not cap.isOpened():

    print("❌ ERROR: Could not open webcam.")
    raise SystemExit


# ============================================================
# CAMERA SETTINGS
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

# MJPG can improve high-FPS USB webcam performance
try:
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
except Exception:
    pass


# ============================================================
# ACTUAL CAMERA PARAMETERS
# ============================================================

actual_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

actual_fps = cap.get(
    cv2.CAP_PROP_FPS
)

print(
    f"📷 Camera: "
    f"{actual_width}x{actual_height} "
    f"@ {actual_fps:.0f} FPS"
)

print(
    f"🎯 Processing: "
    f"{PROCESSING_WIDTH}x{PROCESSING_HEIGHT}"
)

print(
    f"🧠 Model: {MODEL_PATH}"
)

print(
    f"🎯 Confidence: "
    f"{CONFIDENCE_THRESHOLD:.2f}"
)

print(
    f"📊 IoU: "
    f"{IOU_THRESHOLD:.2f}"
)

print(
    f"🚶 Walking threshold: "
    f"{WALKING_THRESHOLD:.2f}"
)

print("\n" + "=" * 60)
print("CONTROLS")
print("=" * 60)
print("Q  = Quit")
print("R  = Reset tracking")
print("S  = Show/Hide statistics")
print("+  = Increase confidence")
print("-  = Decrease confidence")
print("=" * 60)
print()


# ============================================================
# SCALE FACTORS
# ============================================================

scale_x = actual_width / PROCESSING_WIDTH
scale_y = actual_height / PROCESSING_HEIGHT


# ============================================================
# PERFORMANCE VARIABLES
# ============================================================

frame_count = 0

fps_history = deque(maxlen=30)
inference_history = deque(maxlen=30)

show_stats = SHOW_STATS


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        loop_start = time.perf_counter()

        # ----------------------------------------------------
        # Read camera frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print("❌ Failed to capture frame.")
            break

        frame_count += 1

        # ----------------------------------------------------
        # Resize for processing
        # ----------------------------------------------------

        processing_frame = cv2.resize(
            frame,
            (
                PROCESSING_WIDTH,
                PROCESSING_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        inference_start = time.perf_counter()

        with torch.inference_mode():

            results = model.track(
                processing_frame,

                persist=True,

                tracker=TRACKER_CONFIG,

                conf=CONFIDENCE_THRESHOLD,

                iou=IOU_THRESHOLD,

                classes=[PERSON_CLASS],

                max_det=MAX_DETECTIONS,

                imgsz=640,

                verbose=False,

                device=device,

                half=(device == "cuda"),

                agnostic_nms=False,
            )

        inference_time = (
            time.perf_counter() -
            inference_start
        ) * 1000

        inference_history.append(
            inference_time
        )

        # ----------------------------------------------------
        # Prepare display
        # ----------------------------------------------------

        annotated = frame.copy()

        result = results[0]

        boxes = result.boxes

        detected_people = 0

        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        if boxes is not None and len(boxes) > 0:

            detected_people = len(boxes)

            # YOLO tracking IDs
            tracking_ids = boxes.id

            # Convert boxes to CPU
            xyxy = boxes.xyxy.cpu().numpy()

            confidences = (
                boxes.conf.cpu().numpy()
                if boxes.conf is not None
                else np.ones(len(xyxy))
            )

            # ------------------------------------------------
            # Process each person
            # ------------------------------------------------

            for index, box in enumerate(xyxy):

                # --------------------------------------------
                # Check ID
                # --------------------------------------------

                if tracking_ids is None:
                    continue

                track_id = int(
                    tracking_ids[index].item()
                )

                # --------------------------------------------
                # Coordinates in processing image
                # --------------------------------------------

                px1, py1, px2, py2 = box

                # --------------------------------------------
                # Convert to camera resolution
                # --------------------------------------------

                x1 = int(px1 * scale_x)
                y1 = int(py1 * scale_y)

                x2 = int(px2 * scale_x)
                y2 = int(py2 * scale_y)

                # Clamp coordinates
                x1 = max(
                    0,
                    min(x1, actual_width - 1)
                )

                y1 = max(
                    0,
                    min(y1, actual_height - 1)
                )

                x2 = max(
                    0,
                    min(x2, actual_width - 1)
                )

                y2 = max(
                    0,
                    min(y2, actual_height - 1)
                )

                # Invalid box
                if x2 <= x1 or y2 <= y1:
                    continue

                # --------------------------------------------
                # Activity
                # --------------------------------------------

                activity, velocity = classify_activity(
                    track_id,
                    x1,
                    y1,
                    x2,
                    y2,
                    frame_count
                )

                color = get_activity_color(
                    activity
                )

                # --------------------------------------------
                # Bounding box
                # --------------------------------------------

                thickness = 3

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness
                )

                # --------------------------------------------
                # Center / foot point
                # --------------------------------------------

                center_x = int(
                    (x1 + x2) * 0.5
                )

                foot_y = y2

                cv2.circle(
                    annotated,
                    (center_x, foot_y),
                    5,
                    color,
                    -1
                )

                # --------------------------------------------
                # Track line
                # --------------------------------------------

                if SHOW_TRACK_LINES:

                    track = activity_tracks.get(
                        track_id
                    )

                    if (
                        track is not None
                        and len(track.position_history) > 1
                    ):

                        points = list(
                            track.position_history
                        )

                        for i in range(
                            1,
                            len(points)
                        ):

                            pt1 = (
                                int(points[i - 1][0]),
                                int(points[i - 1][1])
                            )

                            pt2 = (
                                int(points[i][0]),
                                int(points[i][1])
                            )

                            cv2.line(
                                annotated,
                                pt1,
                                pt2,
                                color,
                                2
                            )

                # --------------------------------------------
                # Label
                # --------------------------------------------

                confidence = float(
                    confidences[index]
                )

                label = (
                    f"ID {track_id} | "
                    f"{activity} | "
                    f"{confidence:.2f}"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.60
                font_thickness = 2

                label_size = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    font_thickness
                )[0]

                label_width = (
                    label_size[0] + 10
                )

                label_height = (
                    label_size[1] + 12
                )

                # Put label above box if possible
                label_y1 = max(
                    0,
                    y1 - label_height
                )

                label_y2 = y1

                cv2.rectangle(
                    annotated,
                    (
                        x1,
                        label_y1
                    ),
                    (
                        x1 + label_width,
                        label_y2
                    ),
                    color,
                    -1
                )

                cv2.putText(
                    annotated,
                    label,
                    (
                        x1 + 5,
                        label_y2 - 6
                    ),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # Cleanup old activity tracks
        # ----------------------------------------------------

        cleanup_old_tracks(
            frame_count,
            max_age=60
        )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        loop_time = (
            time.perf_counter() -
            loop_start
        )

        current_fps = (
            1.0 / loop_time
            if loop_time > 0
            else 0
        )

        fps_history.append(
            current_fps
        )

        average_fps = float(
            np.mean(fps_history)
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if show_stats:

            average_inference = (
                float(np.mean(inference_history))
                if inference_history
                else 0
            )

            # Statistics background
            stats_x1 = 5
            stats_y1 = 5
            stats_x2 = 365
            stats_y2 = 145

            cv2.rectangle(
                annotated,
                (
                    stats_x1,
                    stats_y1
                ),
                (
                    stats_x2,
                    stats_y2
                ),
                (0, 0, 0),
                -1
            )

            cv2.rectangle(
                annotated,
                (
                    stats_x1,
                    stats_y1
                ),
                (
                    stats_x2,
                    stats_y2
                ),
                (0, 255, 0),
                2
            )

            # FPS
            cv2.putText(
                annotated,
                f"FPS: {average_fps:.1f}",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Inference
            cv2.putText(
                annotated,
                f"Inference: "
                f"{average_inference:.1f} ms",
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # People
            cv2.putText(
                annotated,
                f"People: {detected_people}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Tracking
            cv2.putText(
                annotated,
                f"Active Tracks: "
                f"{len(activity_tracks)}",
                (15, 117),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 0),
                2,
                cv2.LINE_AA
            )

            # Confidence
            cv2.putText(
                annotated,
                f"Confidence: "
                f"{CONFIDENCE_THRESHOLD:.2f}",
                (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 200, 255),
                1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "YOLO Person Activity Detection",
            annotated
        )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Q = quit
        if key == ord("q"):

            print("\n🛑 Exiting...")
            break

        # R = reset
        elif key == ord("r"):

            activity_tracks.clear()

            # Reinitialize tracker
            try:
                model.predictor = None
            except Exception:
                pass

            print("🔄 Activity tracking reset")

        # S = statistics
        elif key == ord("s"):

            show_stats = not show_stats

            print(
                f"📊 Statistics: "
                f"{'ON' if show_stats else 'OFF'}"
            )

        # + = increase confidence
        elif key in (
            ord("+"),
            ord("=")
        ):

            CONFIDENCE_THRESHOLD = min(
                0.90,
                CONFIDENCE_THRESHOLD + 0.05
            )

            print(
                f"📈 Confidence: "
                f"{CONFIDENCE_THRESHOLD:.2f}"
            )

        # - = decrease confidence
        elif key in (
            ord("-"),
            ord("_")
        ):

            CONFIDENCE_THRESHOLD = max(
                0.10,
                CONFIDENCE_THRESHOLD - 0.05
            )

            print(
                f"📉 Confidence: "
                f"{CONFIDENCE_THRESHOLD:.2f}"
            )


# ============================================================
# CLEANUP
# ============================================================

except KeyboardInterrupt:

    print("\n⚠️ Interrupted by user.")


finally:

    print("\n🧹 Cleaning up...")

    cap.release()

    cv2.destroyAllWindows()

    # Performance summary
    if fps_history:

        final_fps = float(
            np.mean(fps_history)
        )

        print(
            f"📊 Average FPS: "
            f"{final_fps:.1f}"
        )

    if inference_history:

        final_inference = float(
            np.mean(inference_history)
        )

        print(
            f"🧠 Average inference: "
            f"{final_inference:.1f} ms"
        )

    print(
        f"🎞️ Total frames: "
        f"{frame_count}"
    )

    print(
        "✅ Cleanup complete."
    )