"""
QA / semantic-similarity model loader.

ML Model used in this module
-----------------------------
1. SentenceTransformer "all-MiniLM-L6-v2"
   - Library : sentence-transformers
   - Task    : Encodes sentences into dense vector embeddings; cosine
               similarity between embeddings is used to score how closely
               a candidate's spoken answer matches the expected answer for
               each interview question.
   - Model   : A 6-layer MiniLM Transformer fine-tuned for semantic
               textual similarity (~22 M parameters).
"""

from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_qa_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
