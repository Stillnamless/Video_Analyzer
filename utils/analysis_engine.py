from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import os
import pickle
import re
import subprocess
import tempfile
from typing import Any

import cv2
from deepface import DeepFace
import librosa
import mediapipe as mp
import numpy as np
from sentence_transformers import SentenceTransformer, util
import whisper


ROOT = Path(__file__).resolve().parent.parent
EMOTION_MODEL_PATH = ROOT / "trained_models" / "emotion_cnn.keras"
GAZE_MODEL_PATH = ROOT / "trained_models" / "gaze_blink_model.pkl"
VOICE_MODEL_PATH = ROOT / "trained_models" / "voice_regressor.pkl"

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

_L_TOP, _L_BOT, _L_LEFT, _L_RIGHT = 159, 145, 33, 133
_R_TOP, _R_BOT, _R_LEFT, _R_RIGHT = 386, 374, 362, 263
_NOSE, _NOSE_BASE, _FACE_L, _FACE_R = 1, 4, 234, 454


@dataclass(frozen=True)
class ModelStatus:
    key: str
    label: str
    active_backend: str
    detail: str
    ready: bool = True


@dataclass(frozen=True)
class CandidateAnalysis:
    dominant_emotion: str
    confidence_score: float
    technical_score: float
    overall_score: float
    grade: str
    grade_color: str
    transcript: str
    emotion_timeline: list[str]
    eye_contact_pct: float
    technical_results: list[dict[str, Any]]
    sampled_frames: int
    video_duration_s: float
    breakdown: dict[str, float]
    strengths: list[str]
    concerns: list[str]
    warnings: list[str]
    model_statuses: list[ModelStatus]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateAnalysis":
        model_statuses = [ModelStatus(**status) for status in payload["model_statuses"]]
        return cls(**{**payload, "model_statuses": model_statuses})


@lru_cache(maxsize=1)
def load_sentence_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def load_whisper_model():
    return whisper.load_model("base")


@lru_cache(maxsize=1)
def load_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    )


@lru_cache(maxsize=1)
def load_pose_detector():
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
    )


@lru_cache(maxsize=1)
def load_custom_emotion_model():
    if not EMOTION_MODEL_PATH.exists():
        return None
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(EMOTION_MODEL_PATH)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_custom_gaze_model():
    if not GAZE_MODEL_PATH.exists():
        return None
    try:
        with GAZE_MODEL_PATH.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_custom_voice_model():
    if not VOICE_MODEL_PATH.exists():
        return None
    try:
        with VOICE_MODEL_PATH.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def compute_grade(score: float) -> tuple[str, str]:
    if score >= 8.5:
        return "A+", "#34d399"
    if score >= 7.5:
        return "A", "#34d399"
    if score >= 6.5:
        return "B+", "#818cf8"
    if score >= 5.5:
        return "B", "#818cf8"
    if score >= 4.5:
        return "C", "#fbbf24"
    if score >= 3.5:
        return "D", "#fb923c"
    return "F", "#f87171"


def build_candidate_insights(
    confidence_score: float,
    technical_score: float,
    eye_contact_pct: float,
    fluency_score: float,
    voice_score: float,
    posture_score: float,
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    concerns: list[str] = []

    if confidence_score >= 7:
        strengths.append("Confidence signals stayed consistently strong across the sampled frames.")
    elif confidence_score <= 4.5:
        concerns.append("Overall confidence signals were weak and may need coaching before interviews.")

    if technical_score >= 7:
        strengths.append("Answer relevance tracked well against the expected interview prompts.")
    elif technical_score <= 4.5:
        concerns.append("Responses drifted from the target questions and need tighter structure.")

    if eye_contact_pct >= 65:
        strengths.append("Eye contact looked steady, which supports a more engaged on-camera presence.")
    elif eye_contact_pct <= 35:
        concerns.append("Eye contact was limited, which can make delivery feel less connected.")

    if fluency_score >= 7:
        strengths.append("Speech fluency stayed clean with relatively few filler-word interruptions.")
    elif fluency_score <= 4.5:
        concerns.append("Fluency dipped because filler words or pauses were frequent in the transcript.")

    if voice_score >= 7:
        strengths.append("Voice energy and pacing landed in a confident range.")
    elif voice_score <= 4.5:
        concerns.append("Voice delivery sounded tentative based on pacing, pitch variation, or volume.")

    if posture_score >= 0.75:
        strengths.append("Posture stayed upright for most of the sampled frames.")
    elif posture_score <= 0.45:
        concerns.append("Posture cues were inconsistent and may distract from the message.")

    if not strengths:
        strengths.append("The interview has a workable base and already shows a few stable signals to build on.")
    if not concerns:
        concerns.append("No major red flags were detected in the sampled interview segments.")

    return strengths[:3], concerns[:3]


def get_app_model_statuses() -> list[ModelStatus]:
    emotion_model = load_custom_emotion_model()
    gaze_model = load_custom_gaze_model()
    voice_model = load_custom_voice_model()
    return build_model_statuses(emotion_model, gaze_model, voice_model)


def build_model_statuses(emotion_model, gaze_model, voice_model) -> list[ModelStatus]:
    return [
        ModelStatus(
            key="emotion",
            label="Emotion Engine",
            active_backend="Custom Emotion CNN" if emotion_model is not None else "DeepFace Fallback",
            detail="7-class residual CNN trained on FER-2013." if emotion_model is not None else "Pretrained DeepFace emotion inference.",
        ),
        ModelStatus(
            key="gaze",
            label="Gaze + Blink",
            active_backend="Custom MLP Pipeline" if gaze_model is not None else "Geometry Fallback",
            detail="Multi-output MLP for blink and forward-gaze prediction." if gaze_model is not None else "MediaPipe landmark thresholds.",
        ),
        ModelStatus(
            key="voice",
            label="Voice Confidence",
            active_backend="Random Forest Regressor" if voice_model is not None else "Audio Heuristic",
            detail="Random forest scoring MFCC, tempo, pitch, and energy." if voice_model is not None else "Librosa-based heuristic scoring.",
        ),
        ModelStatus(
            key="speech",
            label="Speech-to-Text",
            active_backend="Whisper Base",
            detail="OpenAI Whisper base model for transcription.",
        ),
        ModelStatus(
            key="qa",
            label="Answer Matching",
            active_backend="MiniLM Embeddings",
            detail="SentenceTransformer all-MiniLM-L6-v2 semantic similarity.",
        ),
        ModelStatus(
            key="vision",
            label="Pose + Face Landmarks",
            active_backend="MediaPipe",
            detail="FaceMesh and Pose for nonverbal cues.",
        ),
    ]


def evaluate_technical_answers(transcript: str, qa_set: dict[str, str]) -> tuple[list[dict[str, Any]], float]:
    sentences = [s.strip() for s in re.split(r"[.!?,]", transcript.lower()) if len(s.strip()) > 8]
    if not sentences:
        sentences = [transcript.lower().strip() or "no spoken response detected"]

    questions = list(qa_set.keys())
    expected_answers = list(qa_set.values())

    model = load_sentence_model()
    sent_embeddings = model.encode(sentences, convert_to_tensor=True)
    q_embeddings = model.encode(questions, convert_to_tensor=True)
    exp_embeddings = model.encode(expected_answers, convert_to_tensor=True)

    sim_matrix = util.cos_sim(q_embeddings, sent_embeddings).cpu().numpy()

    assigned: dict[int, int] = {}
    used_indices: set[int] = set()
    order = sorted(range(len(questions)), key=lambda qi: sim_matrix[qi].max(), reverse=True)
    for qi in order:
        ranked = sim_matrix[qi].argsort()[::-1]
        for si in ranked:
            if si not in used_indices:
                assigned[qi] = int(si)
                used_indices.add(si)
                break
        else:
            assigned[qi] = int(sim_matrix[qi].argmax())

    results: list[dict[str, Any]] = []
    total_score = 0.0
    for qi, question in enumerate(questions):
        si = assigned[qi]
        best_sentence = sentences[si]
        best_emb = model.encode(best_sentence, convert_to_tensor=True)
        raw_sim = util.cos_sim(best_emb, exp_embeddings[qi]).item()
        q_relevance = float(sim_matrix[qi][si])
        combined = 0.6 * raw_sim + 0.4 * q_relevance
        score = round(max(0.0, combined) * 10, 2)
        total_score += score
        results.append(
            {
                "Question": question,
                "Best Match from Response": best_sentence,
                "Score": score,
                "Relevance": round(q_relevance * 10, 2),
            }
        )

    average_score = round(total_score / len(qa_set), 2) if qa_set else 0.0
    return results, average_score


def _predict_emotion_custom(frame_rgb: np.ndarray, emotion_model) -> str | None:
    try:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (48, 48))
        input_tensor = gray.astype("float32") / 255.0
        input_tensor = input_tensor.reshape(1, 48, 48, 1)
        probabilities = emotion_model.predict(input_tensor, verbose=0)[0]
        return EMOTION_LABELS[int(np.argmax(probabilities))]
    except Exception:
        return None


def _gaze_features(landmarks) -> list[float]:
    def ear(top_idx: int, bottom_idx: int, left_idx: int, right_idx: int) -> float:
        vertical = abs(landmarks[top_idx].y - landmarks[bottom_idx].y)
        horizontal = abs(landmarks[left_idx].x - landmarks[right_idx].x)
        return vertical / (horizontal + 1e-6)

    left_x = (landmarks[_L_LEFT].x + landmarks[_L_RIGHT].x) / 2
    left_y = (landmarks[_L_TOP].y + landmarks[_L_BOT].y) / 2
    right_x = (landmarks[_R_LEFT].x + landmarks[_R_RIGHT].x) / 2
    right_y = (landmarks[_R_TOP].y + landmarks[_R_BOT].y) / 2
    face_width = abs(landmarks[_FACE_L].x - landmarks[_FACE_R].x)
    gaze_horizontal = (landmarks[_NOSE].x - landmarks[_FACE_L].x) / (face_width + 1e-6) - 0.5
    gaze_vertical = landmarks[_NOSE].y - landmarks[_NOSE_BASE].y
    eye_symmetry = abs(left_y - right_y)
    return [
        ear(_L_TOP, _L_BOT, _L_LEFT, _L_RIGHT),
        ear(_R_TOP, _R_BOT, _R_LEFT, _R_RIGHT),
        left_x,
        left_y,
        right_x,
        right_y,
        landmarks[_NOSE].x,
        landmarks[_NOSE].y,
        face_width,
        gaze_horizontal,
        gaze_vertical,
        eye_symmetry,
    ]


def _predict_gaze(landmarks, gaze_model) -> tuple[bool, bool]:
    if gaze_model is not None:
        features = np.array([_gaze_features(landmarks)])
        prediction = gaze_model.predict(features)[0]
        return bool(prediction[0]), bool(prediction[1])

    blink = abs(landmarks[_L_TOP].y - landmarks[_L_BOT].y) < 0.015
    gaze_forward = abs(landmarks[33].x - landmarks[263].x) > 0.18
    return blink, gaze_forward


def _predict_voice_confidence(y_audio: np.ndarray, sample_rate: int, voice_model) -> float | None:
    try:
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        rms = np.mean(librosa.feature.rms(y=y_audio))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y_audio, sr=sample_rate))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_audio))
        onset_env = librosa.onset.onset_strength(y=y_audio, sr=sample_rate)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0]
        pitch = librosa.yin(y_audio, fmin=50, fmax=400)
        pitch_std = np.std(pitch[pitch > 0]) if np.any(pitch > 0) else 0.0
        features = np.concatenate([mfcc_mean, [rms, centroid, zcr, tempo, pitch_std]]).reshape(1, -1)
        score = float(voice_model.predict(features)[0])
        return max(0.0, min(10.0, score))
    except Exception:
        return None


def _voice_score_heuristic(y_audio: np.ndarray, sample_rate: int) -> float:
    pitch = librosa.yin(y_audio, fmin=50, fmax=300)
    volume = np.mean(np.abs(y_audio))
    onset_env = librosa.onset.onset_strength(y=y_audio, sr=sample_rate)
    speech_rate = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0]

    score = 0.0
    if 130 < speech_rate < 180:
        score += 3
    if volume > 0.02:
        score += 3
    if np.std(pitch) > 8:
        score += 4
    return round(score, 2)


def _transcribe_and_analyze_fluency(audio_path: str) -> tuple[str, float]:
    whisper_model = load_whisper_model()
    result = whisper_model.transcribe(audio_path)
    transcript = result.get("text", "").strip()

    filler_words = ["um", "uh", "like", "you know", "i mean", "so", "actually", "basically", "right", "okay"]
    transcript_lower = transcript.lower()
    total_words = len(transcript_lower.split())
    filler_count = sum(len(re.findall(rf"\b{re.escape(word)}\b", transcript_lower)) for word in filler_words)

    if total_words == 0:
        return transcript, 0.0

    filler_ratio = filler_count / total_words
    fluency_score = max(0.0, 10 - (filler_ratio * 50))
    return transcript, round(fluency_score, 2)


def _detect_head_movement(previous_positions: deque[float], current_nose: float) -> float:
    previous_positions.append(current_nose)
    if len(previous_positions) > 10:
        previous_positions.popleft()
    differences = [abs(previous_positions[i] - previous_positions[i - 1]) for i in range(1, len(previous_positions))]
    return float(np.mean(differences)) if differences else 0.0


def _detect_posture(frame: np.ndarray) -> float:
    pose_detector = load_pose_detector()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(rgb)
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        head_level = nose.y < left_shoulder.y
        return 1.0 if shoulder_diff < 0.07 and head_level else 0.0
    return 0.5


def _extract_audio_to_temp(video_path: str) -> str:
    if not shutil_which("ffmpeg"):
        raise FileNotFoundError("ffmpeg is not installed or not available on PATH.")

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    handle.close()
    output_path = handle.name

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "1",
        output_path,
    ]

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg failed to extract audio.")
    return output_path


def shutil_which(command: str) -> str | None:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / command
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def analyze_video(video_path: str, qa_set: dict[str, str], sample_frames: int = 30) -> CandidateAnalysis:
    warnings: list[str] = []
    emotion_model = load_custom_emotion_model()
    gaze_model = load_custom_gaze_model()
    voice_model = load_custom_voice_model()
    face_mesh = load_face_mesh()
    model_statuses = build_model_statuses(emotion_model, gaze_model, voice_model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the uploaded video for analysis.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    video_duration_s = round(frame_count / fps, 1) if fps > 0 and frame_count > 0 else 0.0
    interval = max(frame_count // sample_frames, 1) if frame_count else 1

    blink_count = 0
    total_blink_frames = 0
    sampled_frames_count = 0
    head_positions: deque[float] = deque()
    head_movement_values: list[float] = []
    expression_scores: list[float] = []
    gaze_scores: list[int] = []
    posture_scores: list[float] = []
    emotion_timeline: list[str] = []

    expression_score_map = {"happy": 10, "neutral": 7, "surprise": 5, "sad": 3, "angry": 2, "fear": 2, "disgust": 1}

    try:
        for frame_index in range(0, frame_count or sample_frames, interval):
            if frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            sampled_frames_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            emotion = _predict_emotion_custom(rgb, emotion_model) if emotion_model is not None else None
            if emotion is None:
                try:
                    analysis = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False, silent=True)
                    if isinstance(analysis, list):
                        emotion = analysis[0]["dominant_emotion"]
                    else:
                        emotion = analysis["dominant_emotion"]
                except Exception:
                    emotion = "neutral"

            expression_scores.append(expression_score_map.get(emotion, 5))
            emotion_timeline.append(emotion)

            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                is_blink, is_forward = _predict_gaze(face_landmarks, gaze_model)
                if is_blink:
                    blink_count += 1
                total_blink_frames += 1
                head_movement_values.append(_detect_head_movement(head_positions, face_landmarks[1].x))
                gaze_scores.append(1 if is_forward else 0)

            posture_scores.append(_detect_posture(frame))
    finally:
        cap.release()

    if not expression_scores:
        warnings.append("No readable frames were found in the uploaded video.")
        technical_results, technical_score = evaluate_technical_answers("", qa_set)
        grade, grade_color = compute_grade(0.0)
        strengths, concerns = build_candidate_insights(0.0, technical_score, 0.0, 0.0, 0.0, 0.0)
        return CandidateAnalysis(
            dominant_emotion="neutral",
            confidence_score=0.0,
            technical_score=technical_score,
            overall_score=round(technical_score / 2, 2),
            grade=grade,
            grade_color=grade_color,
            transcript="",
            emotion_timeline=[],
            eye_contact_pct=0.0,
            technical_results=technical_results,
            sampled_frames=0,
            video_duration_s=video_duration_s,
            breakdown={"expression": 0.0, "eye_contact": 0.0, "voice": 0.0, "fluency": 0.0, "posture": 0.0, "blink": 0.0, "head": 0.0},
            strengths=strengths,
            concerns=concerns,
            warnings=warnings,
            model_statuses=model_statuses,
        )

    avg_expression = float(np.mean(expression_scores))
    eye_contact_pct = round(float(np.mean(gaze_scores)) * 100, 1) if gaze_scores else 0.0
    avg_posture = float(np.mean(posture_scores)) if posture_scores else 0.0
    avg_blink_rate = blink_count / total_blink_frames if total_blink_frames > 0 else 0.0
    blink_score = max(0.0, min(10 - (avg_blink_rate * 100), 10))

    avg_head_movement = float(np.mean(head_movement_values)) if head_movement_values else 0.05
    head_score = max(0.0, min(10 - (avg_head_movement * 100), 10))

    audio_path = None
    transcript = ""
    fluency_score = 0.0
    voice_score = 0.0
    try:
        audio_path = _extract_audio_to_temp(video_path)
        transcript, fluency_score = _transcribe_and_analyze_fluency(audio_path)
        y_audio, sample_rate = librosa.load(audio_path)
        if voice_model is not None:
            voice_score = _predict_voice_confidence(y_audio, sample_rate, voice_model) or 0.0
        else:
            voice_score = _voice_score_heuristic(y_audio, sample_rate)
    except Exception as exc:
        warnings.append(f"Audio analysis fallback triggered: {exc}")
    finally:
        if audio_path and Path(audio_path).exists():
            Path(audio_path).unlink(missing_ok=True)

    technical_results, technical_score = evaluate_technical_answers(transcript, qa_set)
    confidence_score = round(
        0.25 * avg_expression
        + 0.20 * (eye_contact_pct / 10)
        + 0.20 * voice_score
        + 0.15 * fluency_score
        + 0.10 * (avg_posture * 10)
        + 0.05 * blink_score
        + 0.05 * head_score,
        2,
    )
    confidence_score = min(confidence_score, 10.0)
    overall_score = round((confidence_score + technical_score) / 2, 2)
    dominant_emotion = Counter(emotion_timeline).most_common(1)[0][0] if emotion_timeline else "neutral"
    grade, grade_color = compute_grade(overall_score)
    strengths, concerns = build_candidate_insights(
        confidence_score=confidence_score,
        technical_score=technical_score,
        eye_contact_pct=eye_contact_pct,
        fluency_score=fluency_score,
        voice_score=voice_score,
        posture_score=avg_posture,
    )

    breakdown = {
        "expression": round(avg_expression, 2),
        "eye_contact": round(eye_contact_pct / 10, 2),
        "voice": round(voice_score, 2),
        "fluency": round(fluency_score, 2),
        "posture": round(avg_posture * 10, 2),
        "blink": round(blink_score, 2),
        "head": round(head_score, 2),
    }

    return CandidateAnalysis(
        dominant_emotion=dominant_emotion,
        confidence_score=confidence_score,
        technical_score=technical_score,
        overall_score=overall_score,
        grade=grade,
        grade_color=grade_color,
        transcript=transcript,
        emotion_timeline=emotion_timeline,
        eye_contact_pct=eye_contact_pct,
        technical_results=technical_results,
        sampled_frames=sampled_frames_count,
        video_duration_s=video_duration_s,
        breakdown=breakdown,
        strengths=strengths,
        concerns=concerns,
        warnings=warnings,
        model_statuses=model_statuses,
    )
