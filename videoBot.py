import streamlit as st
import cv2
import subprocess
import librosa
import tempfile
import os
import whisper
from collections import deque
from deepface import DeepFace
import mediapipe as mp
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="InterviewIQ — AI Interview Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Premium CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ── */
*, html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.main { background: #0a0b14; }
.stApp { background: #0a0b14; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #0d0e1a; }
.block-container { padding-top: 1rem; max-width: 1200px; }
hr { border-color: rgba(99,102,241,0.15) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0b14; }
::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 4px; }

/* ── Animated Hero ── */
.hero-container {
    text-align: center;
    padding: 48px 20px 32px;
    position: relative;
}
.hero-badge {
    display: inline-block;
    padding: 6px 18px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 999px;
    color: #818cf8;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #c7d2fe 0%, #818cf8 30%, #6366f1 50%, #a78bfa 70%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s ease-in-out infinite alternate;
}
@keyframes shimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #6b7280;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Glass Card ── */
.glass-card {
    background: rgba(17,19,35,0.65);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
    border-color: rgba(99,102,241,0.3);
    box-shadow: 0 0 30px rgba(99,102,241,0.06);
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
}

/* ── Score Ring ── */
.score-ring-container { text-align: center; padding: 8px 0; }
.score-ring {
    position: relative;
    width: 110px;
    height: 110px;
    margin: 0 auto 8px;
}
.score-ring svg { transform: rotate(-90deg); }
.score-ring-label {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.6rem;
    font-weight: 800;
    color: #e2e8f0;
}
.score-ring-caption {
    font-size: 0.78rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* ── Emotion Chip ── */
.emotion-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 999px;
    margin-top: 8px;
}
.emotion-chip-emoji { font-size: 1.5rem; }
.emotion-chip-text {
    font-size: 0.88rem;
    font-weight: 600;
    color: #c7d2fe;
    text-transform: capitalize;
}

/* ── Q&A Cards ── */
.qa-card {
    background: rgba(17,19,35,0.5);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
    transition: all 0.3s;
    position: relative;
}
.qa-card:hover {
    border-color: rgba(99,102,241,0.25);
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}
.qa-question {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 10px;
    line-height: 1.5;
}
.qa-response-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    font-weight: 600;
    margin-bottom: 4px;
}
.qa-response {
    color: #a5b4fc;
    font-style: italic;
    font-size: 0.9rem;
    line-height: 1.5;
    padding: 10px 14px;
    background: rgba(99,102,241,0.06);
    border-radius: 8px;
    border-left: 3px solid #6366f1;
    margin-bottom: 12px;
}
.qa-score-row {
    display: flex;
    align-items: center;
    gap: 12px;
}
.score-badge {
    padding: 4px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
}
.badge-high  { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.badge-mid   { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
.badge-low   { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.2); }
.relevance-text {
    font-size: 0.78rem;
    color: #4b5563;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 32px 0 16px;
}
.section-header-icon {
    width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(99,102,241,0.1);
    border-radius: 10px;
    font-size: 1.1rem;
}
.section-header-text {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
}

/* ── Candidate Divider ── */
.candidate-label {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    background: linear-gradient(90deg, rgba(99,102,241,0.08), transparent);
    border-left: 3px solid #6366f1;
    border-radius: 0 12px 12px 0;
    margin: 24px 0 16px;
}
.candidate-label-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
}
.candidate-label-file {
    font-size: 0.82rem;
    color: #6b7280;
    font-weight: 400;
}

/* ── Leaderboard ── */
.lb-row {
    display: flex;
    align-items: center;
    padding: 16px 20px;
    background: rgba(17,19,35,0.5);
    border: 1px solid rgba(99,102,241,0.08);
    border-radius: 12px;
    margin: 8px 0;
    gap: 20px;
    transition: all 0.25s;
}
.lb-row:hover {
    border-color: rgba(99,102,241,0.2);
    background: rgba(17,19,35,0.7);
}
.lb-rank {
    font-size: 1.8rem;
    width: 48px;
    text-align: center;
    flex-shrink: 0;
}
.lb-name {
    flex: 1;
    font-weight: 600;
    color: #e2e8f0;
    font-size: 0.95rem;
}
.lb-scores {
    display: flex;
    gap: 24px;
}
.lb-score-item {
    text-align: center;
}
.lb-score-val {
    font-size: 1.2rem;
    font-weight: 800;
    color: #a5b4fc;
}
.lb-score-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4b5563;
    font-weight: 600;
}

/* ── Bar Chart ── */
.bar-track {
    width: 120px;
    height: 6px;
    background: rgba(99,102,241,0.1);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 4px;
}
.bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 80px 20px;
}
.empty-icon {
    font-size: 4rem;
    margin-bottom: 16px;
    opacity: 0.7;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}
.empty-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #4b5563;
    margin-bottom: 8px;
}
.empty-desc {
    font-size: 0.9rem;
    color: #374151;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: rgba(17,19,35,0.4);
    border: 2px dashed rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 16px;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.4);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(17,19,35,0.4);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 12px;
}
[data-testid="stExpander"] summary {
    color: #9ca3af;
    font-weight: 600;
}

/* ── Video ── */
[data-testid="stVideo"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.12);
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div { color: #818cf8 !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 40px 20px 24px;
    color: #374151;
    font-size: 0.78rem;
}
.footer a { color: #6366f1; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ─── Cached Model Loaders ────────────────────────────────────────────
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

sent_model = load_sentence_model()

# ─── Q&A Reference Set ───────────────────────────────────────────────
qa_set = {
    "Could you elaborate on your core competencies and key skills?": (
        "I have strong expertise in project management, stakeholder communication, "
        "technical problem solving, and team leadership. My core competencies include "
        "analytical thinking, attention to detail, and delivering results under pressure."
    ),
    "What activities or responsibilities do you find most fulfilling in your work?": (
        "I find the most fulfillment in mentoring team members, solving complex challenges, "
        "and delivering high-impact projects. Working collaboratively and seeing the results "
        "of our collective effort is very rewarding."
    ),
    "Could you provide an overview of your educational qualifications and professional experience?": (
        "I hold a degree in my field and have several years of professional experience "
        "working across various organizations. I have progressively taken on more responsibilities "
        "and developed expertise in my domain."
    ),
    "Can you share some of your significant achievements and how they were accomplished?": (
        "One of my key achievements was leading a project that significantly improved efficiency "
        "or delivered measurable results. I accomplished this through careful planning, "
        "cross-functional collaboration and clear communication."
    ),
    "What are your aspirations for the future, both professionally and personally?": (
        "Professionally, I aspire to grow into a leadership role where I can make a broader impact. "
        "Personally, I aim to continue learning, maintain a healthy work-life balance, "
        "and contribute meaningfully to my field."
    )
}

# ─── MediaPipe Setup ─────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
pose_detector = mp_pose.Pose(static_image_mode=True)


# ─── Analysis Functions ──────────────────────────────────────────────
def evaluate_technical_answers(transcript, qa_set):
    sentences = [s.strip() for s in re.split(r'[.!?,]', transcript.lower()) if len(s.strip()) > 8]
    if not sentences:
        sentences = [transcript.lower()]

    questions = list(qa_set.keys())
    expected_answers = list(qa_set.values())

    sent_embeddings = sent_model.encode(sentences, convert_to_tensor=True)
    q_embeddings = sent_model.encode(questions, convert_to_tensor=True)
    exp_embeddings = sent_model.encode(expected_answers, convert_to_tensor=True)

    sim_matrix = util.cos_sim(q_embeddings, sent_embeddings).cpu().numpy()

    assigned = {}
    used_indices = set()
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

    results = []
    total_score = 0

    for qi, question in enumerate(questions):
        si = assigned[qi]
        best_sentence = sentences[si]

        best_emb = sent_model.encode(best_sentence, convert_to_tensor=True)
        raw_sim = util.cos_sim(best_emb, exp_embeddings[qi]).item()
        q_relevance = float(sim_matrix[qi][si])

        combined = 0.6 * raw_sim + 0.4 * q_relevance
        score = round(combined * 10, 2)
        total_score += score

        results.append({
            "Question": question,
            "Best Match from Response": best_sentence,
            "Score": score,
            "Relevance": round(q_relevance * 10, 2),
        })

    average_score = round(total_score / len(qa_set), 2) if qa_set else 0.0
    return results, average_score


def extract_audio(video_path, audio_path="temp_audio.wav"):
    command = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 1 "{audio_path}"'
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def transcribe_and_analyze_fluency(audio_path):
    whisper_model = load_whisper_model()
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"].strip()

    filler_words = ["um", "uh", "like", "you know", "i mean", "so", "actually", "basically", "right", "okay"]
    transcript_lower = transcript.lower()
    total_words = len(transcript_lower.split())
    filler_count = sum(len(re.findall(rf"\b{re.escape(fw)}\b", transcript_lower)) for fw in filler_words)

    if total_words == 0:
        return transcript, 0.0

    filler_ratio = filler_count / total_words
    fluency_score = max(0, 10 - (filler_ratio * 50))
    return transcript, round(fluency_score, 2)


def detect_blink(landmarks):
    return abs(landmarks[159].y - landmarks[145].y) < 0.015


def detect_head_movement(prev_positions, current_nose):
    prev_positions.append(current_nose)
    if len(prev_positions) > 10:
        prev_positions.popleft()
    diffs = [abs(prev_positions[i] - prev_positions[i-1]) for i in range(1, len(prev_positions))]
    return np.mean(diffs) if diffs else 0.0


def analyze_voice_confidence(audio_path):
    y, sr = librosa.load(audio_path)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    volume = np.mean(np.abs(y))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    speech_rate = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    score = 0
    if 130 < speech_rate < 180:
        score += 3
    if volume > 0.02:
        score += 3
    if np.std(pitch) > 8:
        score += 4
    return round(score, 2)


def is_facing_forward(landmarks):
    return abs(landmarks[33].x - landmarks[263].x) > 0.18


def detect_posture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(rgb)
    if result.pose_landmarks:
        ls = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        shoulder_diff = abs(ls.y - rs.y)
        head_level = nose.y < ls.y
        return 1 if (shoulder_diff < 0.07 and head_level) else 0
    return 0.5


def analyze_confidence(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(frame_count // 30, 1)

    blink_count = 0
    total_blink_frames = 0
    head_positions = deque()
    head_movement_values = []
    expression_scores = []
    gaze_scores = []
    posture_scores = []

    expr_score_map = {"happy": 10, "neutral": 7, "surprise": 5, "sad": 3, "angry": 2, "fear": 2, "disgust": 1}

    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            analysis = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False, silent=True)
            emotion = analysis[0]["dominant_emotion"]
        except Exception:
            emotion = "neutral"

        expression_scores.append(expr_score_map.get(emotion, 5))

        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            if detect_blink(face.landmark):
                blink_count += 1
            total_blink_frames += 1
            nose_x = face.landmark[1].x
            head_movement_values.append(detect_head_movement(head_positions, nose_x))
            gaze_scores.append(1 if is_facing_forward(face.landmark) else 0)

        posture_scores.append(detect_posture(frame))

    cap.release()

    if not expression_scores:
        return "neutral", 0.0, ""

    avg_expression = np.mean(expression_scores)
    avg_gaze = np.mean(gaze_scores) if gaze_scores else 0.5
    avg_posture = np.mean(posture_scores)
    avg_blink_rate = blink_count / total_blink_frames if total_blink_frames > 0 else 0
    blink_score = max(0, min(10 - (avg_blink_rate * 100), 10))

    avg_head_movement = np.mean(head_movement_values) if head_movement_values else 0.05
    head_score = max(0, min(10 - (avg_head_movement * 100), 10))

    audio_path = extract_audio(video_path)
    transcript, fluency_score = transcribe_and_analyze_fluency(audio_path)
    voice_score = analyze_voice_confidence(audio_path)

    confidence_score = round(
        0.25 * avg_expression +
        0.20 * avg_gaze * 10 +
        0.20 * voice_score +
        0.15 * fluency_score +
        0.10 * avg_posture * 10 +
        0.05 * blink_score +
        0.05 * head_score, 2
    )
    confidence_score = min(confidence_score, 10.0)
    return emotion, confidence_score, transcript


# ─── Helper Functions ─────────────────────────────────────────────────
def score_badge_class(score):
    if score >= 7:
        return "badge-high"
    elif score >= 4:
        return "badge-mid"
    return "badge-low"


def score_color(score):
    if score >= 7:
        return "#34d399"
    elif score >= 4:
        return "#fbbf24"
    return "#f87171"


def emotion_emoji(emotion):
    return {"happy": "😄", "neutral": "😐", "sad": "😢", "angry": "😠",
            "fear": "😨", "surprise": "😲", "disgust": "🤢"}.get(emotion, "😐")


def render_score_ring(score, label, max_val=10):
    """SVG circular score gauge."""
    pct = min(score / max_val, 1.0)
    radius = 42
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - pct)
    color = score_color(score)

    return f"""
    <div class='score-ring-container'>
        <div class='score-ring'>
            <svg width='110' height='110' viewBox='0 0 110 110'>
                <circle cx='55' cy='55' r='{radius}' fill='none'
                    stroke='rgba(99,102,241,0.1)' stroke-width='8'/>
                <circle cx='55' cy='55' r='{radius}' fill='none'
                    stroke='{color}' stroke-width='8'
                    stroke-dasharray='{circumference}'
                    stroke-dashoffset='{offset}'
                    stroke-linecap='round'
                    style='transition: stroke-dashoffset 1s ease;'/>
            </svg>
            <div class='score-ring-label'>{score}</div>
        </div>
        <div class='score-ring-caption'>{label}</div>
    </div>
    """


def rank_medal(rank):
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


# ─── UI ───────────────────────────────────────────────────────────────

# Hero Section
st.markdown("""
<div class='hero-container'>
    <div class='hero-badge'>AI-Powered Analysis</div>
    <div class='hero-title'>InterviewIQ</div>
    <div class='hero-subtitle'>
        Upload interview recordings and get instant AI analysis of confidence,
        body language, voice quality, and answer relevance.
    </div>
</div>
""", unsafe_allow_html=True)

# Upload Area
uploaded_videos = st.file_uploader(
    "📁 Drop interview videos here",
    type=["mp4", "webm", "mov"],
    accept_multiple_files=True,
    help="Supports MP4, WebM, MOV · Multiple files for candidate comparison"
)

leaderboard = []

if uploaded_videos:
    for i, video in enumerate(uploaded_videos):
        # Candidate Header
        st.markdown(f"""
        <div class='candidate-label'>
            <span style='font-size:1.4rem'>🎬</span>
            <div>
                <div class='candidate-label-text'>Candidate {i+1}</div>
                <div class='candidate-label-file'>{video.name}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ext = os.path.splitext(video.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(video.read())
            video_path = tmp.name

        col_vid, col_space = st.columns([2, 1])
        with col_vid:
            st.video(video_path)

        with st.spinner(f"🔍 Analyzing Candidate {i+1} — this may take a minute..."):
            dominant_emotion, conf_score, transcript = analyze_confidence(video_path)
            qa_results, tech_score = evaluate_technical_answers(transcript, qa_set)
            total_score = round((conf_score + tech_score) / 2, 2)

        # ── Score Rings ──
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(render_score_ring(conf_score, "Confidence"), unsafe_allow_html=True)
        with c2:
            st.markdown(render_score_ring(tech_score, "Technical"), unsafe_allow_html=True)
        with c3:
            st.markdown(render_score_ring(total_score, "Overall"), unsafe_allow_html=True)
        with c4:
            emoji = emotion_emoji(dominant_emotion)
            st.markdown(f"""
            <div style='text-align:center; padding: 16px 0;'>
                <div class='emotion-chip'>
                    <span class='emotion-chip-emoji'>{emoji}</span>
                    <span class='emotion-chip-text'>{dominant_emotion}</span>
                </div>
                <div class='score-ring-caption' style='margin-top:12px'>Dominant Emotion</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Transcript ──
        with st.expander("📄 View Full Transcript"):
            st.write(transcript if transcript.strip() else "_No speech detected in video._")

        # ── Q&A Evaluation ──
        st.markdown("""
        <div class='section-header'>
            <div class='section-header-icon'>🧠</div>
            <div class='section-header-text'>Answer Evaluation</div>
        </div>
        """, unsafe_allow_html=True)

        for res in qa_results:
            sc = res["Score"]
            rel = res["Relevance"]
            badge = score_badge_class(sc)
            st.markdown(f"""
            <div class='qa-card'>
                <div class='qa-question'>❓ {res['Question']}</div>
                <div class='qa-response-label'>Candidate's Response</div>
                <div class='qa-response'>"{res['Best Match from Response']}"</div>
                <div class='qa-score-row'>
                    <span class='score-badge {badge}'>{sc}/10</span>
                    <span class='relevance-text'>Relevance: {rel}/10</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        leaderboard.append({
            "name": video.name,
            "emotion": dominant_emotion,
            "confidence": conf_score,
            "technical": tech_score,
            "total": total_score
        })

        st.divider()

    # ── Leaderboard ──
    if len(leaderboard) > 0:
        st.markdown("""
        <div class='section-header' style='margin-top:40px'>
            <div class='section-header-icon'>🏆</div>
            <div class='section-header-text'>Candidate Leaderboard</div>
        </div>
        """, unsafe_allow_html=True)

        sorted_lb = sorted(leaderboard, key=lambda x: x["total"], reverse=True)

        for rank, entry in enumerate(sorted_lb, 1):
            medal = rank_medal(rank)
            conf_pct = entry["confidence"] / 10 * 100
            tech_pct = entry["technical"] / 10 * 100
            tot_pct = entry["total"] / 10 * 100
            conf_clr = score_color(entry["confidence"])
            tech_clr = score_color(entry["technical"])
            tot_clr = score_color(entry["total"])
            emoji = emotion_emoji(entry["emotion"])

            st.markdown(f"""
            <div class='lb-row'>
                <div class='lb-rank'>{medal}</div>
                <div class='lb-name'>{emoji} {entry["name"]}</div>
                <div class='lb-scores'>
                    <div class='lb-score-item'>
                        <div class='lb-score-val' style='color:{conf_clr}'>{entry["confidence"]}</div>
                        <div class='lb-score-label'>Confidence</div>
                        <div class='bar-track'><div class='bar-fill' style='width:{conf_pct}%;background:{conf_clr}'></div></div>
                    </div>
                    <div class='lb-score-item'>
                        <div class='lb-score-val' style='color:{tech_clr}'>{entry["technical"]}</div>
                        <div class='lb-score-label'>Technical</div>
                        <div class='bar-track'><div class='bar-fill' style='width:{tech_pct}%;background:{tech_clr}'></div></div>
                    </div>
                    <div class='lb-score-item'>
                        <div class='lb-score-val' style='color:{tot_clr}'>{entry["total"]}</div>
                        <div class='lb-score-label'>Total</div>
                        <div class='bar-track'><div class='bar-fill' style='width:{tot_pct}%;background:{tot_clr}'></div></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Empty State
    st.markdown("""
    <div class='empty-state'>
        <div class='empty-icon'>🎬</div>
        <div class='empty-title'>No videos uploaded yet</div>
        <div class='empty-desc'>
            Upload interview recordings above to analyze confidence, expressions & answers
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    Built with ❤️ using Streamlit · Powered by Whisper, DeepFace, MediaPipe & Sentence-Transformers
</div>
""", unsafe_allow_html=True)
