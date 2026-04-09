"""
Data Collection Script for Gaze / Blink Classifier (Model 2).

This script auto-generates a labeled dataset from video files using MediaPipe.
For each frame it extracts 12 eye-landmark distances and labels them:
  - blink: 1 (eye closed) / 0 (eye open)
  - gaze:  1 (looking forward) / 0 (looking away)

Usage:
    python training/collect_gaze_dataset.py

Output:
    training/data/gaze_dataset.csv

This IS our own data — we are the ones collecting and labeling it!
"""
import cv2
import csv
import os
import mediapipe as mp
import numpy as np

# ── Load MediaPipe ───────────────────────────────────────────────────────────
import mediapipe.python.solutions.face_mesh as mp_face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Eye landmark indices from MediaPipe 468-landmark map
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133
RIGHT_EYE_TOP   = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT  = 362
RIGHT_EYE_RIGHT = 263

# Nose tip and cheekbones for gaze direction
NOSE_TIP     = 1
NOSE_BASE    = 4
FACE_LEFT    = 234
FACE_RIGHT   = 454


DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH  = os.path.join(DATA_DIR, "gaze_dataset.csv")
os.makedirs(DATA_DIR, exist_ok=True)

FEATURE_NAMES = [
    "left_ear",  "right_ear",          # Eye Aspect Ratio (blink indicator)
    "left_eye_x", "left_eye_y",        # Left eye center
    "right_eye_x", "right_eye_y",      # Right eye center
    "nose_x", "nose_y",                # Nose position
    "face_width",                      # Face width (for normalization)
    "gaze_horizontal",                 # How centered nose is horizontally
    "gaze_vertical",                   # Vertical gaze component
    "eye_symmetry"                     # Symmetry between left/right eye
]


def eye_aspect_ratio(landmarks, top_i, bottom_i, left_i, right_i):
    """Compute Eye Aspect Ratio (EAR). < 0.2 usually = blink."""
    vert  = abs(landmarks[top_i].y - landmarks[bottom_i].y)
    horiz = abs(landmarks[left_i].x - landmarks[right_i].x)
    return vert / (horiz + 1e-6)


def extract_features(landmarks):
    left_ear   = eye_aspect_ratio(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
    right_ear  = eye_aspect_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
    left_cx    = (landmarks[LEFT_EYE_LEFT].x + landmarks[LEFT_EYE_RIGHT].x) / 2
    left_cy    = (landmarks[LEFT_EYE_TOP].y  + landmarks[LEFT_EYE_BOTTOM].y) / 2
    right_cx   = (landmarks[RIGHT_EYE_LEFT].x + landmarks[RIGHT_EYE_RIGHT].x) / 2
    right_cy   = (landmarks[RIGHT_EYE_TOP].y  + landmarks[RIGHT_EYE_BOTTOM].y) / 2
    nose_x     = landmarks[NOSE_TIP].x
    nose_y     = landmarks[NOSE_TIP].y
    face_w     = abs(landmarks[FACE_LEFT].x - landmarks[FACE_RIGHT].x)
    gaze_h     = (landmarks[NOSE_TIP].x - landmarks[FACE_LEFT].x) / (face_w + 1e-6) - 0.5  # 0=center
    gaze_v     = landmarks[NOSE_TIP].y - landmarks[NOSE_BASE].y
    eye_sym    = abs(left_cy - right_cy)

    return [left_ear, right_ear, left_cx, left_cy, right_cx, right_cy,
            nose_x, nose_y, face_w, gaze_h, gaze_v, eye_sym]


def label_frame(features):
    """Auto-label a frame based on geometric thresholds."""
    left_ear, right_ear = features[0], features[1]
    gaze_h = features[9]

    # Blink: EAR < 0.15 for both eyes
    blink = 1 if (left_ear < 0.15 and right_ear < 0.15) else 0

    # Gaze: nose centered within ±0.15 of face midpoint means looking forward
    gaze_forward = 1 if abs(gaze_h) < 0.15 else 0

    return blink, gaze_forward


def process_video(video_path, writer, frame_skip=3):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    samples = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            features = extract_features(landmarks)
            blink, gaze = label_frame(features)
            writer.writerow(features + [blink, gaze])
            samples += 1
    cap.release()
    return samples


if __name__ == "__main__":
    # Collect all .mp4 / .mov from the project root
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    video_files = [
        os.path.join(PROJECT_ROOT, f)
        for f in os.listdir(PROJECT_ROOT)
        if f.lower().endswith((".mp4", ".mov", ".webm"))
    ]

    if not video_files:
        print("No video files found in project root. Add .mp4 files and rerun.")
        exit(1)

    print(f"Found {len(video_files)} video(s): {[os.path.basename(v) for v in video_files]}")

    total = 0
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["blink_label", "gaze_label"])
        for vp in video_files:
            print(f"  Processing {os.path.basename(vp)}...")
            n = process_video(vp, writer)
            total += n
            print(f"    → {n} labeled frames extracted")

    print(f"\n✓ Dataset saved to {CSV_PATH}")
    print(f"  Total samples: {total}")
    print("  Next: run  python training/train_gaze_mlp.py")
