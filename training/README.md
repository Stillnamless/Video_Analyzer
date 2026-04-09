# InterviewIQ — Custom Model Training Guide

## Models Overview

| Model | File | Dataset | Replaces |
|---|---|---|---|
| Emotion CNN | `train_emotion_cnn.py` | FER-2013 | DeepFace pre-trained model |
| Gaze/Blink MLP | `train_gaze_mlp.py` | **Self-collected** | MediaPipe hardcoded thresholds |
| Voice Regressor | `train_voice_regressor.py` | RAVDESS | Librosa heuristics |

---

## Step 1 — Setup Kaggle API (for FER-2013)

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → "Create New API Token"
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
3. `pip install kaggle`

---

## Step 2 — Download Datasets

```bash
python training/download_datasets.py
```

---

## Step 3 — Collect Gaze Dataset (our own data!)

```bash
# Ensure sample .mp4 videos are in the project root
python training/collect_gaze_dataset.py
# → Creates training/data/gaze_dataset.csv
```

---

## Step 4 — Train All Models

```bash
# Train Emotion CNN (~30 mins on GPU, ~2hrs on CPU)
python training/train_emotion_cnn.py

# Train Gaze + Blink Classifier (fast, ~2 mins)
python training/train_gaze_mlp.py

# Train Voice Confidence Regressor (~5 mins)
python training/train_voice_regressor.py
```

Trained models are saved to `trained_models/` automatically.

---

## Step 5 — Run the App

After training, the app automatically detects and uses the custom models:

```bash
streamlit run videoBot.py
```

> **Note:** If trained models are not found, the app falls back to the original pre-trained models. 
> This ensures the app always works, even before training.
