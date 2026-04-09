"""
Model 2: Gaze / Blink Classifier MLP
Trains a Multi-Layer Perceptron on the auto-collected landmark dataset.

Usage:
    python training/collect_gaze_dataset.py   # First, collect data
    python training/train_gaze_mlp.py         # Then train

Output:
    trained_models/gaze_blink_model.pkl
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "gaze_dataset.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "gaze_blink_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "left_ear", "right_ear",
    "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
    "nose_x", "nose_y", "face_width", "gaze_horizontal",
    "gaze_vertical", "eye_symmetry"
]


def load_and_validate(path):
    if not os.path.exists(path):
        print(f"ERROR: Dataset not found at {path}")
        print("Run:  python training/collect_gaze_dataset.py  first!")
        exit(1)
    df = pd.read_csv(path).dropna()
    print(f"Loaded {len(df)} samples from dataset.")
    print("Class balance:")
    print(f"  Blink  — 1 (closed): {df['blink_label'].sum()}  0 (open): {(df['blink_label']==0).sum()}")
    print(f"  Gaze   — 1 (forward): {df['gaze_label'].sum()}  0 (away): {(df['gaze_label']==0).sum()}")
    return df


if __name__ == "__main__":
    df = load_and_validate(DATA_PATH)

    X = df[FEATURE_COLS].values
    y_blink = df["blink_label"].values
    y_gaze  = df["gaze_label"].values
    Y = np.column_stack([y_blink, y_gaze])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=y_gaze)

    # Build pipeline: StandardScaler → MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True
    )

    # Wrap in MultiOutput to predict both [blink, gaze] simultaneously
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MultiOutputClassifier(mlp))
    ])

    print("\nTraining Gaze/Blink MLP...")
    pipeline.fit(X_train, Y_train)

    Y_pred = pipeline.predict(X_test)

    print("\n── Blink Classification Report ──")
    print(classification_report(Y_test[:, 0], Y_pred[:, 0], target_names=["Open", "Blink"]))
    print("── Gaze Classification Report ──")
    print(classification_report(Y_test[:, 1], Y_pred[:, 1], target_names=["Looking Away", "Forward"]))

    blink_acc = accuracy_score(Y_test[:, 0], Y_pred[:, 0])
    gaze_acc  = accuracy_score(Y_test[:, 1], Y_pred[:, 1])
    print(f"  Blink Accuracy: {blink_acc*100:.2f}%")
    print(f"  Gaze Accuracy:  {gaze_acc*100:.2f}%")

    # Save pipeline (includes scaler + model)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\n✓ Model saved to {MODEL_PATH}")
