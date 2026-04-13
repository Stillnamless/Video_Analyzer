# Video Analyzer

An AI-powered interview analysis tool built with Streamlit. Upload interview recordings and receive automated scoring across facial expressions, eye contact, posture, voice confidence, speech fluency, and technical-answer quality.

---

## ML Models Used

| File | Model | Library | Task |
|------|-------|---------|------|
| `utils/expression_utils.py` | **DeepFace** (VGG-Face backend by default) | `deepface` | Classifies the dominant facial emotion per frame (happy, neutral, surprise, sad, angry, fear, disgust) |
| `utils/expression_utils.py` | **MediaPipe Face Mesh** | `mediapipe` | Detects 468 facial landmarks; used for gaze direction, blink-rate, and head-movement estimation |
| `utils/video_utils.py` | **MediaPipe Pose** (BlazePose) | `mediapipe` | Detects 33 body landmarks; used to assess shoulder alignment and upright posture |
| `utils/audio_utils.py` | **OpenAI Whisper "base"** | `openai-whisper` | Speech-to-text transcription of the audio track |
| `models/qa_model.py` | **SentenceTransformer "all-MiniLM-L6-v2"** | `sentence-transformers` | Encodes sentences into embeddings; cosine similarity scores technical interview answers |

### Model Details

#### 1. DeepFace — Emotion Detection (`utils/expression_utils.py`)
- **What it does**: Analyses each sampled video frame and returns a dominant emotion label.
- **Default backend**: VGG-Face (a 138 M-parameter CNN trained on the VGGFace dataset).
- **How it is used**: The emotion label is mapped to a score (e.g. `happy → 10`, `neutral → 8`) which contributes 35 % of the overall confidence score.

#### 2. MediaPipe Face Mesh — Facial Landmarks (`utils/expression_utils.py`)
- **What it does**: Fits a mesh of 468 3-D landmarks onto the detected face in real time.
- **How it is used**:
  - **Gaze estimation** — checks whether both eyes are visible and roughly symmetric (forward-facing proxy).
  - **Blink detection** — measures the vertical distance between upper and lower eyelid landmarks.
  - **Head movement** — tracks the nose-tip landmark across frames to compute lateral drift.

#### 3. MediaPipe Pose — Posture Detection (`utils/video_utils.py`)
- **What it does**: Detects 33 full-body pose landmarks (shoulders, hips, knees, etc.).
- **How it is used**: Checks that the candidate's shoulders are level and their head is above shoulder height as a proxy for an upright, confident posture.

#### 4. OpenAI Whisper "base" — Speech Transcription (`utils/audio_utils.py`)
- **What it does**: Converts the audio extracted from the video into a text transcript.
- **How it is used**: The transcript is analysed for filler-word frequency (fluency score) and passed to the QA scoring module.

#### 5. SentenceTransformer "all-MiniLM-L6-v2" — Technical Answer Scoring (`models/qa_model.py`)
- **What it does**: Encodes natural-language sentences into 384-dimensional embeddings.
- **How it is used**: Cosine similarity between the embedding of the candidate's best-matching sentence and the embedding of the expected answer gives a per-question score out of 10.

---

## Scoring Breakdown

| Component | Weight | Source |
|-----------|--------|--------|
| Facial expression | 35 % | DeepFace emotion label → fixed score map |
| Gaze / eye contact | 20 % | MediaPipe Face Mesh (forward-facing check) |
| Voice confidence | 20 % | Librosa pitch variance, volume, speech rate |
| Speech fluency | 15 % | Whisper transcript → filler-word ratio |
| Posture | 25 % | MediaPipe Pose (shoulder alignment) |
| Blink rate | 10 % | MediaPipe Face Mesh (eyelid landmarks) |
| Head movement | 10 % | MediaPipe Face Mesh (nose-tip tracking) |

> **Note**: weights currently sum to 135 % (35 + 20 + 20 + 15 + 25 + 10 + 10), so the raw `confidence_score` can exceed 10. Normalisation is left to the caller.

---

## Getting Started

```bash
pip install -r requirements.txt
streamlit run video.py
```

Upload one or more `.mp4` / `.webm` interview recordings and the app will display per-candidate scores and a ranked leaderboard.
