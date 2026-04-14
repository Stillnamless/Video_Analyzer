import json
import os
import tempfile
from textwrap import dedent

import pandas as pd
import streamlit as st

from questions import qa_set
from utils.analysis_engine import CandidateAnalysis, analyze_video, get_app_model_statuses


st.set_page_config(
    page_title="InterviewIQ · AI Interview Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def html_block(markup: str) -> str:
    return dedent(markup).strip()


# ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown(
    html_block(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── TOKENS ─────────────────────────────────────────── */
:root {
    --bg:           #0e0d15;
    --bg-low:       #13131a;
    --surf:         #1b1b23;
    --surf-mid:     #1f1f27;
    --surf-high:    #2a2931;
    --surf-top:     #34343c;

    --primary:      #c0c1ff;
    --primary-dim:  #8083ff;
    --primary-raw:  #6366f1;
    --secondary:    #4cd7f6;
    --secondary-raw:#06b6d4;
    --tertiary:     #ffb783;
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;

    --on-bg:        #e4e1ec;
    --on-surf:      #e4e1ec;
    --on-muted:     #c7c4d7;
    --on-faint:     #908fa0;

    --outline:      rgba(192,193,255,0.10);
    --glow-ind:     rgba(99,102,241,0.18);
    --glow-cyan:    rgba(6,182,212,0.14);

    --radius-xl:    24px;
    --radius-lg:    18px;
    --radius-md:    12px;
    --radius-sm:    8px;
    --radius-pill:  999px;

    --blur:         blur(14px);
    --font:         'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── RESET & BASE ────────────────────────────────────── */
html, body, .stApp, button, input, textarea, select {
    font-family: var(--font) !important;
    color: var(--on-bg);
}

.stApp {
    background: var(--bg) !important;
}

/* ── ANIMATED STAR FIELD ─────────────────────────────── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 15% 22%, rgba(192,193,255,0.55) 0%, transparent 100%),
        radial-gradient(1px 1px at 72% 8%,  rgba(76,215,246,0.45) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 40% 55%, rgba(192,193,255,0.40) 0%, transparent 100%),
        radial-gradient(1px 1px at 88% 35%, rgba(255,183,131,0.30) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 78%, rgba(76,215,246,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 5%  90%, rgba(192,193,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 93% 70%, rgba(192,193,255,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at 28% 12%, rgba(76,215,246,0.30) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 42%, rgba(255,183,131,0.20) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 88%, rgba(192,193,255,0.25) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ── AMBIENT GLOW ORBS ───────────────────────────────── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 20% 10%,  rgba(99,102,241,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 35% at 85% 15%,  rgba(6,182,212,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 50% 90%,  rgba(99,102,241,0.08) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

header[data-testid="stHeader"] { background: transparent !important; }

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
    position: relative;
    z-index: 1;
}

hr { border-color: var(--outline) !important; }

/* ── SCROLLBAR ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-low); }
::-webkit-scrollbar-thumb {
    background: rgba(128,131,255,0.35);
    border-radius: 99px;
}

/* ════════════════════════════════════════════════════════
   HERO SECTION
   ════════════════════════════════════════════════════════ */
.hero-wrap {
    position: relative;
    padding: 48px 52px 44px;
    border-radius: var(--radius-xl);
    border: 1px solid var(--outline);
    background: linear-gradient(
        135deg,
        rgba(27,27,35,0.95) 0%,
        rgba(19,19,26,0.90) 100%
    );
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    overflow: hidden;
    margin-bottom: 32px;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at -10% -10%, rgba(99,102,241,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 110% 110%, rgba(6,182,212,0.12) 0%, transparent 50%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(192,193,255,0.5), rgba(76,215,246,0.5), transparent);
}

.hero-inner {
    display: grid;
    grid-template-columns: 1.55fr 1fr;
    gap: 40px;
    align-items: center;
    position: relative;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: var(--radius-pill);
    border: 1px solid rgba(192,193,255,0.22);
    background: rgba(128,131,255,0.10);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 20px;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--secondary);
    box-shadow: 0 0 6px 2px rgba(76,215,246,0.6);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

.hero-title {
    font-size: 4.2rem;
    font-weight: 900;
    letter-spacing: -0.05em;
    line-height: 0.94;
    margin-bottom: 18px;
    background: linear-gradient(135deg, #e1e0ff 10%, #c0c1ff 40%, #4cd7f6 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1.02rem;
    color: var(--on-muted);
    line-height: 1.75;
    max-width: 580px;
}

/* Hero metric panel */
.hero-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}
.hero-metric {
    padding: 18px 20px;
    border-radius: var(--radius-lg);
    border: 1px solid rgba(192,193,255,0.08);
    background: rgba(19,19,26,0.7);
    transition: border-color 0.25s, transform 0.25s;
}
.hero-metric:hover {
    border-color: rgba(192,193,255,0.22);
    transform: translateY(-2px);
}
.hero-metric-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--on-faint);
    margin-bottom: 8px;
}
.hero-metric-val {
    font-size: 1.06rem;
    font-weight: 800;
    color: var(--on-bg);
    margin-bottom: 6px;
}
.hero-metric-note {
    font-size: 0.78rem;
    color: var(--on-muted);
    line-height: 1.55;
}

/* ════════════════════════════════════════════════════════
   SECTION HEADERS
   ════════════════════════════════════════════════════════ */
.section-hd {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 36px 0 20px;
    position: relative;
}
.section-hd::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--outline), transparent);
    margin-left: 8px;
}
.section-hd-icon {
    width: 40px; height: 40px;
    display: flex; align-items: center; justify-content: center;
    border-radius: var(--radius-md);
    border: 1px solid var(--outline);
    background: rgba(27,27,35,0.9);
    font-size: 1.05rem;
    flex-shrink: 0;
    box-shadow: 0 0 18px rgba(99,102,241,0.15);
}
.section-hd-text {
    font-size: 1.18rem;
    font-weight: 800;
    color: var(--on-bg);
    letter-spacing: -0.02em;
}
.section-hd-sub {
    font-size: 0.82rem;
    color: var(--on-faint);
    margin-left: auto;
}

/* ════════════════════════════════════════════════════════
   MODEL STATUS CARDS
   ════════════════════════════════════════════════════════ */
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(172px, 1fr));
    gap: 14px;
    margin-bottom: 32px;
}
.model-card {
    padding: 18px 16px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--outline);
    background: var(--surf);
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
    position: relative;
    overflow: hidden;
}
.model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3), 0 0 20px var(--glow-ind);
}
.model-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    border-radius: 2px 2px 0 0;
}
.model-card.model-custom::before  { background: linear-gradient(90deg, var(--primary-raw), var(--primary)); }
.model-card.model-fallback::before { background: linear-gradient(90deg, var(--warning), #fbbf24); }
.model-card.model-core::before    { background: linear-gradient(90deg, var(--success), #34d399); }

.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: var(--radius-pill);
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.pill-custom   { background: rgba(99,102,241,0.12); color: var(--primary); border: 1px solid rgba(99,102,241,0.25); }
.pill-fallback { background: rgba(245,158,11,0.12); color: var(--warning); border: 1px solid rgba(245,158,11,0.25); }
.pill-core     { background: rgba(16,185,129,0.12); color: var(--success); border: 1px solid rgba(16,185,129,0.25); }

.model-label   { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase; color: var(--on-faint); }
.model-backend { font-size: 0.92rem; font-weight: 700; color: var(--on-bg); margin-bottom: 6px; line-height: 1.35; }
.model-detail  { font-size: 0.78rem; color: var(--on-muted); line-height: 1.55; }

/* ════════════════════════════════════════════════════════
   FILE UPLOADER
   ════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: rgba(27,27,35,0.8) !important;
    border: 2px dashed rgba(99,102,241,0.28) !important;
    border-radius: var(--radius-xl) !important;
    padding: 20px !important;
    box-shadow: 0 0 40px rgba(99,102,241,0.08) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.55) !important;
    box-shadow: 0 0 60px rgba(99,102,241,0.15) !important;
}
[data-testid="stFileUploader"] section { padding: 0 !important; }
[data-testid="stFileUploaderDropzone"] {
    background: rgba(14,13,21,0.6) !important;
    border: 1px solid rgba(192,193,255,0.08) !important;
    border-radius: var(--radius-lg) !important;
    min-height: 190px !important;
    padding: 30px 24px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p {
    margin: 0 !important;
    font-size: 1rem !important;
    color: var(--on-muted) !important;
    text-align: center !important;
    line-height: 1.6 !important;
}
[data-testid="stFileUploader"] button {
    min-width: 180px !important;
    min-height: 48px !important;
    border-radius: var(--radius-pill) !important;
    background: linear-gradient(135deg, #8083ff, #03b5d3) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    transition: opacity 0.2s, transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="stFileUploader"] button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.45) !important;
}
[data-testid="stFileUploader"] button p { margin: 0 !important; white-space: nowrap !important; }

/* ════════════════════════════════════════════════════════
   CANDIDATE HEADER
   ════════════════════════════════════════════════════════ */
.cand-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 24px;
    border-radius: var(--radius-lg);
    border: 1px solid rgba(128,131,255,0.22);
    border-left: 3px solid;
    border-image: linear-gradient(180deg, var(--primary-raw), var(--secondary-raw)) 1;
    background: rgba(27,27,35,0.92);
    margin: 36px 0 20px;
    backdrop-filter: var(--blur);
}
.cand-icon { font-size: 1.6rem; }
.cand-name  { font-size: 1.04rem; font-weight: 800; color: var(--on-bg); }
.cand-file  { font-size: 0.82rem; color: var(--on-muted); margin-top: 2px; }

/* ════════════════════════════════════════════════════════
   SCORE RING PANEL
   ════════════════════════════════════════════════════════ */
.scores-panel {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    padding: 28px 24px;
    border-radius: var(--radius-xl);
    border: 1px solid var(--outline);
    background: linear-gradient(180deg, var(--surf) 0%, var(--surf-mid) 100%);
    position: relative;
    overflow: hidden;
    margin-bottom: 28px;
}
.scores-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(192,193,255,0.35), rgba(76,215,246,0.35), transparent);
}
.score-col { text-align: center; }
.score-ring-wrap {
    position: relative;
    width: 112px; height: 112px;
    margin: 0 auto 10px;
}
.score-ring-wrap svg { transform: rotate(-90deg); }
.score-ring-val {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.7rem;
    font-weight: 900;
    color: var(--on-bg);
    letter-spacing: -0.04em;
}
.score-ring-cap {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--on-faint);
}

/* ════════════════════════════════════════════════════════
   EMOTION CHIP
   ════════════════════════════════════════════════════════ */
.emotion-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 18px;
    border-radius: var(--radius-pill);
    background: rgba(128,131,255,0.12);
    border: 1px solid rgba(128,131,255,0.22);
    margin-top: 6px;
}
.emotion-chip-text { font-size: 0.86rem; font-weight: 700; color: var(--primary); text-transform: capitalize; }
.grade-display {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    margin-top: 12px;
}
.grade-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--on-faint);
    margin-top: 2px;
}

/* ════════════════════════════════════════════════════════
   SUMMARY GRID CARDS
   ════════════════════════════════════════════════════════ */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}
.glass-card {
    padding: 22px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--outline);
    background: var(--surf);
    transition: border-color 0.25s, transform 0.25s;
}
.glass-card:hover {
    border-color: rgba(192,193,255,0.18);
    transform: translateY(-2px);
}
.card-title {
    font-size: 0.92rem;
    font-weight: 800;
    color: var(--on-bg);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-title-icon {
    width: 26px; height: 26px;
    border-radius: var(--radius-sm);
    background: rgba(128,131,255,0.15);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem;
}

.mini-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid rgba(192,193,255,0.05);
}
.mini-row:last-child { border-bottom: none; }
.mini-key { font-size: 0.76rem; color: var(--on-muted); font-weight: 500; }
.mini-val { font-size: 0.9rem; font-weight: 800; color: var(--on-bg); }

.insight-list {
    margin: 0; padding-left: 0;
    list-style: none;
}
.insight-list li {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    font-size: 0.83rem;
    color: var(--on-muted);
    line-height: 1.6;
    border-bottom: 1px solid rgba(192,193,255,0.04);
}
.insight-list li:last-child { border-bottom: none; }
.insight-list li::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 7px;
}
.insight-strength li::before { background: var(--success); box-shadow: 0 0 6px rgba(16,185,129,0.5); }
.insight-concern  li::before { background: var(--warning);  box-shadow: 0 0 6px rgba(245,158,11,0.5); }

/* ════════════════════════════════════════════════════════
   METRIC BREAKDOWN GRID
   ════════════════════════════════════════════════════════ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
    gap: 14px;
    margin-bottom: 28px;
}
.metric-card {
    padding: 20px 16px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--outline);
    background: var(--surf);
    text-align: center;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: rgba(192,193,255,0.22);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.metric-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(128,131,255,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.metric-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--on-faint);
    margin-bottom: 10px;
}
.metric-val {
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--primary-dim), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.04em;
    line-height: 1;
}

/* ════════════════════════════════════════════════════════
   EMOTION TIMELINE
   ════════════════════════════════════════════════════════ */
.timeline-wrap {
    padding: 24px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--outline);
    background: var(--surf);
    margin-bottom: 28px;
}
.timeline-bars {
    display: flex;
    gap: 12px;
    align-items: flex-end;
    min-height: 100px;
}
.timeline-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    flex: 1;
}
.timeline-bar-outer {
    width: 100%;
    max-width: 44px;
    border-radius: 6px 6px 2px 2px;
    display: flex;
    align-items: flex-end;
    overflow: hidden;
    transition: opacity 0.2s;
}
.timeline-bar-outer:hover { opacity: 0.8; }
.timeline-bar-inner {
    width: 100%;
    border-radius: 6px 6px 2px 2px;
    position: relative;
}
.timeline-bar-inner::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.15), transparent);
    border-radius: inherit;
}
.timeline-emo  { font-size: 0.68rem; color: var(--on-muted); font-weight: 600; letter-spacing: 0.04em; }
.timeline-cnt  { font-size: 0.76rem; font-weight: 800; }

/* ════════════════════════════════════════════════════════
   QA CARDS
   ════════════════════════════════════════════════════════ */
.qa-card {
    padding: 22px 26px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--outline);
    background: var(--surf);
    margin-bottom: 14px;
    transition: border-color 0.25s, transform 0.25s;
    position: relative;
    overflow: hidden;
}
.qa-card:hover {
    border-color: rgba(192,193,255,0.18);
    transform: translateY(-2px);
}
.qa-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--primary-raw), var(--secondary-raw));
    border-radius: 0 0 0 var(--radius-lg);
}
.qa-question {
    font-size: 0.94rem;
    font-weight: 700;
    color: var(--on-bg);
    margin-bottom: 10px;
    line-height: 1.5;
    padding-left: 14px;
}
.qa-response-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--on-faint);
    padding-left: 14px;
    margin-bottom: 8px;
}
.qa-response {
    font-style: italic;
    font-size: 0.88rem;
    color: var(--on-muted);
    line-height: 1.65;
    padding: 12px 16px;
    background: rgba(14,13,21,0.5);
    border-radius: var(--radius-md);
    border-left: 2px solid rgba(99,102,241,0.4);
    margin-left: 14px;
    margin-bottom: 14px;
}
.qa-score-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-left: 14px;
}
.score-pill {
    padding: 5px 14px;
    border-radius: var(--radius-pill);
    font-size: 0.76rem;
    font-weight: 800;
}
.pill-high { background: rgba(16,185,129,0.12);  color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.pill-mid  { background: rgba(245,158,11,0.12);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
.pill-low  { background: rgba(239,68,68,0.10);   color: #f87171; border: 1px solid rgba(239,68,68,0.20); }
.relevance-txt { font-size: 0.8rem; color: var(--on-faint); }

/* ════════════════════════════════════════════════════════
   LEADERBOARD
   ════════════════════════════════════════════════════════ */
.lb-card {
    padding: 18px 24px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--outline);
    background: var(--surf);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 20px;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}
.lb-card:hover {
    border-color: rgba(192,193,255,0.22);
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.lb-card.lb-first { border-color: rgba(128,131,255,0.35); background: rgba(128,131,255,0.07); }
.lb-rank  { font-size: 1.9rem; width: 48px; text-align: center; flex-shrink: 0; }
.lb-info  { flex: 1; min-width: 0; }
.lb-name  { font-size: 0.94rem; font-weight: 800; color: var(--on-bg); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lb-emo   { font-size: 0.78rem; color: var(--on-muted); margin-top: 3px; }
.lb-scores { display: flex; gap: 28px; flex-shrink: 0; }
.lb-score-col { text-align: right; min-width: 72px; }
.lb-score-val { font-size: 1.5rem; font-weight: 900; letter-spacing: -0.04em; }
.lb-score-lbl { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase; color: var(--on-faint); }
.bar-track {
    width: 100%;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
    margin-top: 6px;
}
.bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }

/* ════════════════════════════════════════════════════════
   EMPTY STATE
   ════════════════════════════════════════════════════════ */
.empty-state {
    text-align: center;
    padding: 100px 24px;
    border-radius: var(--radius-xl);
    border: 1px dashed rgba(99,102,241,0.22);
    background: rgba(27,27,35,0.6);
    position: relative;
    overflow: hidden;
}
.empty-state::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 50% at 50% 0%, rgba(99,102,241,0.10) 0%, transparent 65%);
}
.empty-icon { font-size: 4.5rem; line-height: 1; margin-bottom: 20px; opacity: 0.75; }
.empty-title {
    font-size: 1.55rem;
    font-weight: 900;
    color: var(--on-bg);
    margin-bottom: 12px;
    letter-spacing: -0.03em;
}
.empty-sub {
    font-size: 0.94rem;
    color: var(--on-muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ════════════════════════════════════════════════════════
   EXPANDER
   ════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: var(--surf) !important;
    border: 1px solid var(--outline) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stExpander"] summary {
    color: var(--on-bg) !important;
    font-weight: 700 !important;
}

/* ════════════════════════════════════════════════════════
   SPINNER & VIDEO
   ════════════════════════════════════════════════════════ */
[data-testid="stSpinner"] > div { color: var(--primary) !important; }
[data-testid="stVideo"] {
    border-radius: var(--radius-xl);
    overflow: hidden;
    border: 1px solid var(--outline);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}

/* ════════════════════════════════════════════════════════
   DOWNLOAD BUTTON
   ════════════════════════════════════════════════════════ */
[data-testid="stDownloadButton"] button {
    border-radius: var(--radius-pill) !important;
    background: linear-gradient(135deg, #8083ff, #03b5d3) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 14px 28px !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.35) !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
[data-testid="stDownloadButton"] button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
}

/* ════════════════════════════════════════════════════════
   FOOTER
   ════════════════════════════════════════════════════════ */
.app-footer {
    text-align: center;
    padding: 48px 24px 28px;
    font-size: 0.75rem;
    color: var(--on-faint);
    border-top: 1px solid var(--outline);
    margin-top: 40px;
}
.app-footer span { color: var(--primary-dim); }

/* ════════════════════════════════════════════════════════
   RESPONSIVE
   ════════════════════════════════════════════════════════ */
@media (max-width: 1024px) {
    .hero-inner { grid-template-columns: 1fr; }
    .scores-panel { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 768px) {
    .hero-title { font-size: 2.8rem; }
    .hero-wrap { padding: 32px 28px 28px; }
    .scores-panel { grid-template-columns: repeat(2, 1fr); }
    .lb-scores { display: none; }
    .hero-metrics { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 520px) {
    .hero-title { font-size: 2.2rem; }
    .hero-metrics, .metric-grid, .summary-grid { grid-template-columns: 1fr; }
    .model-grid { grid-template-columns: 1fr 1fr; }
    .scores-panel { grid-template-columns: 1fr 1fr; }
}
</style>
"""
    ),
    unsafe_allow_html=True,
)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def score_pill_class(score: float) -> str:
    if score >= 7:
        return "pill-high"
    if score >= 4:
        return "pill-mid"
    return "pill-low"


def score_color(score: float) -> str:
    if score >= 7:
        return "#34d399"
    if score >= 4:
        return "#fbbf24"
    return "#f87171"


def emotion_emoji(emotion: str) -> str:
    return {
        "happy":   "😄",
        "neutral": "😐",
        "sad":     "😢",
        "angry":   "😠",
        "fear":    "😨",
        "surprise":"😲",
        "disgust": "🤢",
    }.get(emotion, "😐")


def rank_medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


def render_score_ring(score: float, label: str, max_val: float = 10) -> str:
    pct = min(score / max_val, 1.0)
    radius = 44
    circ = 2 * 3.14159265 * radius
    offset = circ * (1 - pct)
    color = score_color(score)
    return html_block(
        f"""
        <div class='score-col'>
            <div class='score-ring-wrap'>
                <svg width='112' height='112' viewBox='0 0 112 112'>
                    <defs>
                        <filter id='glow-{int(score*100)}'>
                            <feGaussianBlur stdDeviation='3' result='coloredBlur'/>
                            <feMerge><feMergeNode in='coloredBlur'/><feMergeNode in='SourceGraphic'/></feMerge>
                        </filter>
                    </defs>
                    <circle cx='56' cy='56' r='{radius}' fill='none'
                        stroke='rgba(255,255,255,0.05)' stroke-width='8'/>
                    <circle cx='56' cy='56' r='{radius}' fill='none'
                        stroke='{color}' stroke-width='8'
                        stroke-dasharray='{circ}'
                        stroke-dashoffset='{offset}'
                        stroke-linecap='round'
                        filter='url(#glow-{int(score*100)})'
                        style='transition: stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1);'/>
                </svg>
                <div class='score-ring-val'>{score}</div>
            </div>
            <div class='score-ring-cap'>{label}</div>
        </div>
        """
    )


def render_model_cards() -> str:
    cards = []
    for st_obj in get_app_model_statuses():
        if "Fallback" in st_obj.active_backend or "Heuristic" in st_obj.active_backend:
            card_cls, pill_cls, pill_text = "model-fallback", "pill-fallback", "Fallback"
        elif "Custom" in st_obj.active_backend or "Random Forest" in st_obj.active_backend:
            card_cls, pill_cls, pill_text = "model-custom", "pill-custom", "Custom"
        else:
            card_cls, pill_cls, pill_text = "model-core", "pill-core", "Core"
        cards.append(
            html_block(
                f"""
                <div class='model-card {card_cls}'>
                    <div class='model-pill {pill_cls}'>{pill_text}</div>
                    <div class='model-label'>{st_obj.label}</div>
                    <div class='model-backend'>{st_obj.active_backend}</div>
                    <div class='model-detail'>{st_obj.detail}</div>
                </div>
                """
            )
        )
    return "<div class='model-grid'>" + "".join(cards) + "</div>"


def render_breakdown_grid(breakdown: dict) -> str:
    labels = {
        "expression":  "Expression",
        "eye_contact":  "Eye Contact",
        "voice":        "Voice",
        "fluency":      "Fluency",
        "posture":      "Posture",
        "blink":        "Blink",
        "head":         "Head",
    }
    cards = []
    for key, value in breakdown.items():
        cards.append(
            html_block(
                f"""
                <div class='metric-card'>
                    <div class='metric-lbl'>{labels.get(key, key.title())}</div>
                    <div class='metric-val'>{value}</div>
                </div>
                """
            )
        )
    return "<div class='metric-grid'>" + "".join(cards) + "</div>"


def render_insight_card(title: str, items: list, kind: str = "strength", icon: str = "✨") -> str:
    li_html = "".join(f"<li>{item}</li>" for item in items)
    return html_block(
        f"""
        <div class='glass-card'>
            <div class='card-title'>
                <div class='card-title-icon'>{icon}</div>
                {title}
            </div>
            <ul class='insight-list insight-{kind}'>{li_html}</ul>
        </div>
        """
    )


def render_snapshot_card(analysis: CandidateAnalysis) -> str:
    words = len(analysis.transcript.split())
    return html_block(
        f"""
        <div class='glass-card'>
            <div class='card-title'>
                <div class='card-title-icon'>📋</div>
                Review Snapshot
            </div>
            <div class='mini-row'>
                <span class='mini-key'>Overall Grade</span>
                <span class='mini-val' style='color:{analysis.grade_color}'>{analysis.grade}</span>
            </div>
            <div class='mini-row'>
                <span class='mini-key'>Duration</span>
                <span class='mini-val'>{analysis.video_duration_s:.1f}s</span>
            </div>
            <div class='mini-row'>
                <span class='mini-key'>Sampled Frames</span>
                <span class='mini-val'>{analysis.sampled_frames}</span>
            </div>
            <div class='mini-row'>
                <span class='mini-key'>Transcript Words</span>
                <span class='mini-val'>{words}</span>
            </div>
            <div class='mini-row'>
                <span class='mini-key'>Eye Contact</span>
                <span class='mini-val'>{analysis.eye_contact_pct:.0f}%</span>
            </div>
        </div>
        """
    )


def render_emotion_timeline(emotion_timeline: list) -> str:
    if not emotion_timeline:
        return "<div style='color:var(--on-muted);font-size:0.88rem;'>No emotion timeline data.</div>"

    order = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]
    color_map = {
        "happy":   "#34d399",
        "neutral": "#818cf8",
        "surprise":"#fbbf24",
        "sad":     "#60a5fa",
        "angry":   "#f87171",
        "fear":    "#c084fc",
        "disgust": "#a3e635",
    }
    counts = pd.Series(emotion_timeline).value_counts().reindex(order, fill_value=0)
    max_c = max(counts.values) + 1
    bars = []
    for emotion, count in counts.items():
        if count == 0:
            continue
        height = int((count / max_c) * 90)
        color = color_map.get(emotion, "#818cf8")
        bars.append(
            html_block(
                f"""
                <div class='timeline-col'>
                    <div class='timeline-bar-outer' style='height:{height}px;background:rgba(255,255,255,0.04);'>
                        <div class='timeline-bar-inner' style='height:{height}px;background:{color};'></div>
                    </div>
                    <div class='timeline-cnt' style='color:{color};'>{count}</div>
                    <div class='timeline-emo'>{emotion[:3]}</div>
                </div>
                """
            )
        )
    return "<div class='timeline-bars'>" + "".join(bars) + "</div>"


# ─── CACHED ANALYSIS ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def analyze_uploaded_video(video_bytes: bytes, filename: str) -> dict:
    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name
    try:
        return analyze_video(video_path, qa_set).to_dict()
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# ════════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(
    html_block(
        """
        <div class='hero-wrap'>
            <div class='hero-badge'>Interview Intelligence Platform</div>
            <div class='hero-inner'>
                <div class='hero-copy'>
                    <div class='hero-title'>InterviewIQ</div>
                    <div class='hero-sub'>
                        AI-powered interview analysis transforming candidate sessions into structured,
                        actionable intelligence — behavioral signals, vocal patterns, gaze tracking,
                        and semantic answer evaluation in one unified workspace.
                    </div>
                </div>
                <div class='hero-metrics'>
                    <div class='hero-metric'>
                        <div class='hero-metric-label'>Platform Coverage</div>
                        <div class='hero-metric-val'>Video · Audio · Answers</div>
                        <div class='hero-metric-note'>Unified pipeline for nonverbal, vocal, and semantic analysis.</div>
                    </div>
                    <div class='hero-metric'>
                        <div class='hero-metric-label'>Analysis Output</div>
                        <div class='hero-metric-val'>Structured Review</div>
                        <div class='hero-metric-note'>Scored summaries, signal breakdowns, and exportable reports.</div>
                    </div>
                    <div class='hero-metric'>
                        <div class='hero-metric-label'>Model Stack</div>
                        <div class='hero-metric-val'>Custom + Pretrained</div>
                        <div class='hero-metric-note'>Trained CNN, MLP, Random Forest alongside Whisper & MediaPipe.</div>
                    </div>
                    <div class='hero-metric'>
                        <div class='hero-metric-label'>Use Case</div>
                        <div class='hero-metric-val'>Recruiting Review</div>
                        <div class='hero-metric-note'>Faster screening, comparison, and structured candidate ranking.</div>
                    </div>
                </div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════════
# MODEL STACK
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(
    html_block(
        """
        <div class='section-hd'>
            <div class='section-hd-icon'>🧩</div>
            <div class='section-hd-text'>Model Stack</div>
            <div class='section-hd-sub'>Active inference backends</div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)
st.markdown(render_model_cards(), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(
    html_block(
        """
        <div class='section-hd'>
            <div class='section-hd-icon'>📤</div>
            <div class='section-hd-text'>Upload Interviews</div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

uploaded_videos = st.file_uploader(
    "Drop interview videos here",
    type=["mp4", "webm", "mov"],
    accept_multiple_files=True,
    help="Upload one or more interview clips. Cached analysis keeps reruns much faster.",
)

leaderboard: list[dict] = []
report_payload: list[dict] = []

# ════════════════════════════════════════════════════════════════════════════════
# CANDIDATE ANALYSIS LOOP
# ════════════════════════════════════════════════════════════════════════════════
if uploaded_videos:
    for index, video in enumerate(uploaded_videos, start=1):
        st.markdown(
            html_block(
                f"""
                <div class='cand-header'>
                    <span class='cand-icon'>🎬</span>
                    <div>
                        <div class='cand-name'>Candidate {index}</div>
                        <div class='cand-file'>{video.name}</div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        video_bytes = video.getvalue()
        st.video(video_bytes)

        try:
            with st.spinner(f"Analyzing Candidate {index} — building intelligence report…"):
                analysis = CandidateAnalysis.from_dict(
                    analyze_uploaded_video(video_bytes, video.name)
                )
        except Exception as exc:
            st.error(f"Analysis failed for **{video.name}**: {exc}")
            continue

        if analysis.warnings:
            for warning in analysis.warnings:
                st.warning(warning)

        # ── SCORE RINGS ──
        emoji = emotion_emoji(analysis.dominant_emotion)
        st.markdown(
            html_block(
                f"""
                <div class='scores-panel'>
                    {render_score_ring(analysis.confidence_score, "Confidence")}
                    {render_score_ring(analysis.technical_score, "Technical")}
                    {render_score_ring(analysis.overall_score, "Overall")}
                    {render_score_ring(round(analysis.eye_contact_pct / 10, 1), "Eye Contact")}
                    <div class='score-col'>
                        <div class='emotion-chip'><span>{emoji}</span><span class='emotion-chip-text'>{analysis.dominant_emotion}</span></div>
                        <div class='grade-display' style='color:{analysis.grade_color}'>{analysis.grade}</div>
                        <div class='grade-label'>Grade</div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        # ── SUMMARY CARDS ──
        st.markdown(
            "<div class='summary-grid'>"
            + render_snapshot_card(analysis)
            + render_insight_card("What Went Well", analysis.strengths, kind="strength", icon="✅")
            + render_insight_card("Coaching Focus", analysis.concerns, kind="concern", icon="🎯")
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── SIGNAL BREAKDOWN ──
        st.markdown(
            html_block(
                """
                <div class='section-hd'>
                    <div class='section-hd-icon'>📈</div>
                    <div class='section-hd-text'>Signal Breakdown</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        st.markdown(render_breakdown_grid(analysis.breakdown), unsafe_allow_html=True)

        # ── EMOTION TIMELINE ──
        st.markdown(
            html_block(
                """
                <div class='section-hd'>
                    <div class='section-hd-icon'>📊</div>
                    <div class='section-hd-text'>Emotion Timeline</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='timeline-wrap'>{render_emotion_timeline(analysis.emotion_timeline)}</div>",
            unsafe_allow_html=True,
        )

        # ── EXPANDERS ──
        with st.expander("🔬 Model Details For This Candidate"):
            for status in analysis.model_statuses:
                st.markdown(f"**{status.label}**: `{status.active_backend}`")
                st.caption(status.detail)

        with st.expander("📝 Transcript"):
            st.write(
                analysis.transcript if analysis.transcript.strip()
                else "_No speech detected in this video._"
            )

        # ── QA EVAL ──
        st.markdown(
            html_block(
                """
                <div class='section-hd'>
                    <div class='section-hd-icon'>🧠</div>
                    <div class='section-hd-text'>Answer Evaluation</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        for result in analysis.technical_results:
            badge = score_pill_class(result["Score"])
            st.markdown(
                html_block(
                    f"""
                    <div class='qa-card'>
                        <div class='qa-question'>❓ {result['Question']}</div>
                        <div class='qa-response-lbl'>Candidate Response Match</div>
                        <div class='qa-response'>"{result['Best Match from Response']}"</div>
                        <div class='qa-score-row'>
                            <span class='score-pill {badge}'>{result['Score']}/10</span>
                            <span class='relevance-txt'>Relevance: {result['Relevance']}/10</span>
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        leaderboard.append(
            {
                "name":       video.name,
                "emotion":    analysis.dominant_emotion,
                "confidence": analysis.confidence_score,
                "technical":  analysis.technical_score,
                "total":      analysis.overall_score,
            }
        )
        report_payload.append(
            {
                "candidate":        video.name,
                "emotion":          analysis.dominant_emotion,
                "grade":            analysis.grade,
                "confidence_score": analysis.confidence_score,
                "technical_score":  analysis.technical_score,
                "overall_score":    analysis.overall_score,
                "eye_contact_pct":  analysis.eye_contact_pct,
                "breakdown":        analysis.breakdown,
                "strengths":        analysis.strengths,
                "concerns":         analysis.concerns,
                "warnings":         analysis.warnings,
                "technical_results":analysis.technical_results,
            }
        )

        st.markdown("<br>", unsafe_allow_html=True)

    # ══ LEADERBOARD ══════════════════════════════════════════════════════════
    if leaderboard:
        st.markdown(
            html_block(
                """
                <div class='section-hd' style='margin-top:40px'>
                    <div class='section-hd-icon'>🏆</div>
                    <div class='section-hd-text'>Candidate Ranking</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        sorted_entries = sorted(leaderboard, key=lambda x: x["total"], reverse=True)
        for rank, entry in enumerate(sorted_entries, start=1):
            conf_col   = score_color(entry["confidence"])
            tech_col   = score_color(entry["technical"])
            total_col  = score_color(entry["total"])
            first_cls  = "lb-first" if rank == 1 else ""
            st.markdown(
                html_block(
                    f"""
                    <div class='lb-card {first_cls}'>
                        <div class='lb-rank'>{rank_medal(rank)}</div>
                        <div class='lb-info'>
                            <div class='lb-name'>{emotion_emoji(entry["emotion"])} {entry["name"]}</div>
                            <div class='lb-emo'>{entry["emotion"].title()} dominant</div>
                        </div>
                        <div class='lb-scores'>
                            <div class='lb-score-col'>
                                <div class='lb-score-val' style='color:{conf_col}'>{entry["confidence"]}</div>
                                <div class='lb-score-lbl'>Confidence</div>
                                <div class='bar-track'><div class='bar-fill' style='width:{entry["confidence"]*10}%;background:{conf_col}'></div></div>
                            </div>
                            <div class='lb-score-col'>
                                <div class='lb-score-val' style='color:{tech_col}'>{entry["technical"]}</div>
                                <div class='lb-score-lbl'>Technical</div>
                                <div class='bar-track'><div class='bar-fill' style='width:{entry["technical"]*10}%;background:{tech_col}'></div></div>
                            </div>
                            <div class='lb-score-col'>
                                <div class='lb-score-val' style='color:{total_col}'>{entry["total"]}</div>
                                <div class='lb-score-lbl'>Overall</div>
                                <div class='bar-track'><div class='bar-fill' style='width:{entry["total"]*10}%;background:{total_col}'></div></div>
                            </div>
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download Full Report (JSON)",
            data=json.dumps(report_payload, indent=2),
            file_name="interviewiq-report.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    # ── EMPTY STATE ───────────────────────────────────────────────────────────
    st.markdown(
        html_block(
            """
            <div class='empty-state'>
                <div class='empty-icon'>🎬</div>
                <div class='empty-title'>Upload interview recordings to begin.</div>
                <div class='empty-sub'>
                    InterviewIQ evaluates confidence, answer quality, voice delivery,
                    posture, gaze, and facial signals in one unified AI pass.
                    Supports MP4, MOV, and WEBM formats.
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(
    html_block(
        """
        <div class='app-footer'>
            <span>InterviewIQ</span> · Custom CNN · Custom MLP · Random Forest ·
            Whisper · MediaPipe · DeepFace · SentenceTransformers
        </div>
        """
    ),
    unsafe_allow_html=True,
)
