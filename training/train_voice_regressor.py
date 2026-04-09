"""
Model 3: Voice Confidence Regressor
Trains a Random Forest on RAVDESS audio features (MFCC, pitch, RMS, tempo)
to output a confidence/emotion-intensity score (0–10).

Usage:
    python training/download_datasets.py      # First download RAVDESS
    python training/train_voice_regressor.py  # Then train

Output:
    trained_models/voice_regressor.pkl
"""
import os
import glob
import pickle
import numpy as np
import librosa
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
RAVDESS_DIR = os.path.join(os.path.dirname(__file__), "data", "ravdess")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "trained_models")
MODEL_PATH  = os.path.join(MODEL_DIR, "voice_regressor.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# RAVDESS filename convention:
# ...-XX-XX-XX-XX-{intensity:01=normal,02=strong}-XX-XX.wav
# Emotion codes: 1=neutral,2=calm,3=happy,4=sad,5=angry,6=fear,7=disgust,8=surprised
# Confidence mapping: happy/calm/surprised = high; neutral/sad/fear = low
EMOTION_CONFIDENCE = {
    1: 5.0,   # neutral
    2: 6.5,   # calm
    3: 9.0,   # happy
    4: 2.5,   # sad
    5: 7.0,   # angry (high energy but negative)
    6: 3.0,   # fearful
    7: 2.0,   # disgust
    8: 8.0,   # surprised
}
INTENSITY_BONUS = {1: 0.0, 2: 1.0}  # strong intensity = +1 point


def extract_features(audio_path, sr=22050):
    """Extract 45-dimensional audio feature vector."""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=5.0)
    except Exception:
        return None

    # MFCC (40 coefficients — most important for speech)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)

    # RMS energy (volume)
    rms = np.mean(librosa.feature.rms(y=y))

    # Spectral centroid (brightness of voice)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Zero crossing rate (noisiness)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

    # Tempo / speech rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    # Pitch standard deviation (pitch variation — confident speakers vary pitch more)
    pitch = librosa.yin(y, fmin=50, fmax=400)
    pitch_std = np.std(pitch[pitch > 0]) if np.any(pitch > 0) else 0.0

    features = np.concatenate([mfcc_mean, [rms, centroid, zcr, tempo, pitch_std]])
    return features  # 45-dimensional vector


def parse_label(filename):
    """Parse confidence score from RAVDESS filename format."""
    parts = os.path.basename(filename).replace(".wav", "").split("-")
    if len(parts) < 7:
        return None
    emotion_id  = int(parts[2])
    intensity_id = int(parts[3])
    score = EMOTION_CONFIDENCE.get(emotion_id, 5.0) + INTENSITY_BONUS.get(intensity_id, 0.0)
    return min(score, 10.0)


def load_dataset():
    wav_files = glob.glob(os.path.join(RAVDESS_DIR, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"ERROR: No .wav files found in {RAVDESS_DIR}")
        print("Run:  python training/download_datasets.py")
        exit(1)

    print(f"Found {len(wav_files)} audio files. Extracting features...")
    X, y = [], []
    for i, path in enumerate(wav_files):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(wav_files)}...")
        label = parse_label(path)
        if label is None:
            continue
        features = extract_features(path)
        if features is None:
            continue
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_dataset()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Score range: {y.min():.1f} – {y.max():.1f} (mean: {y.mean():.2f})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\nTraining Random Forest Regressor...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\n── Results ──")
    print(f"  MAE (Mean Absolute Error): {mae:.3f} points")
    print(f"  R² Score:                  {r2:.3f}")

    # Cross-validation for robustness
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    print(f"  5-Fold CV R²:              {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importance
    rf = pipeline.named_steps["rf"]
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-5:][::-1]
    print(f"\nTop 5 most important features (indices): {top_idx.tolist()}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\n✓ Model saved to {MODEL_PATH}")
