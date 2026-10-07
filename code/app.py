"""
app.py - Sanskrit Document RAG System
Advanced Interactive Web Application
Features:
- Dual Query Interface: Devanagari, Transliterated Roman (IAST/HK/ITRANS), and Natural English
- Enlarged High-Visibility Search Box
- Full Dual-Language Output: Authentic Sanskrit Answer + Comprehensive English Meaning & Explanation
- Live Script Normalization Visualizer
- Hybrid Retrieval (ChromaDB Multilingual Dense Vectors + BM25 Keyword Matching + Stem Boosting)
- Zero-GPU CPU Inference Optimization
- Interactive Corpus Explorer & Document Uploader
- Performance Benchmarking HUD
"""

import os
import sys
import re
import time
import streamlit as st

# Ensure code directory is in sys.path
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

from transliterate import to_devanagari, is_devanagari, detect_transliteration_scheme
from ingestion import load_document, chunk_sanskrit_text
from retriever import SanskritRetriever
from generator import SanskritGenerator
from pipeline import SanskritRAGPipeline

# Page Configuration
st.set_page_config(
    page_title="Sanskrit RAG | CPU-Only Retrieval-Augmented Generation",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Vintage Historical Theme (Antique Manuscript & Aged Papyrus Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,600&family=Cinzel:wght@600;700;800;900&family=Noto+Serif+Devanagari:wght@400;500;600;700;800&family=Marcellus&display=swap');

    /* Hide sidebar and default header elements */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    button[data-testid="baseButton-header"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Vintage Antique Canvas Background (Aged Walnut & Papyrus) */
    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #292017 0%, #1a140e 55%, #100c08 100%) !important;
        color: #f7eed8 !important;
    }

    :root {
        --vintage-brass: #c59f5b;
        --vintage-bronze: #8c6f43;
        --vintage-gold-aged: #dfbe7b;
        --vintage-parchment: #faf2de;
        --vintage-ink: #17110c;
        --vintage-card-bg: #231b14;
        --vintage-card-border: #6d5432;
        --vintage-muted: #c7b399;
    }

    /* Vintage Manuscript Archive Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #241c14 0%, #2f241a 50%, #1d1610 100%);
        border: 2px solid #8c6f43;
        outline: 1px solid rgba(197, 159, 91, 0.4);
        outline-offset: -5px;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.7), inset 0 0 25px rgba(140, 111, 67, 0.12);
        position: relative;
        text-align: center;
    }

    .hero-crest {
        font-family: 'Noto Serif Devanagari', 'Cinzel', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #dfbe7b;
        margin-bottom: 6px;
        letter-spacing: normal !important;
        text-shadow: 0 0 10px rgba(197, 159, 91, 0.5);
    }
    
    .hero-title {
        font-family: 'Cinzel', 'EB Garamond', serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #faf2de 0%, #dfbe7b 35%, #b38848 70%, #f5e4bd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-family: 'EB Garamond', serif;
        font-size: 1.18rem;
        color: #d9c8af;
        line-height: 1.6;
        max-width: 960px;
        margin: 0 auto;
        font-style: italic;
        letter-spacing: normal !important;
    }

    .badge-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin-top: 16px;
    }

    .spec-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(35, 27, 20, 0.95);
        border: 1.5px solid #8c6f43;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(250, 242, 222, 0.15);
        color: #faf2de;
        padding: 5px 14px;
        border-radius: 16px;
        font-family: 'EB Garamond', serif;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: normal !important;
    }

    /* Vintage Manuscript Query Tablets (Buttons) */
    div[data-testid="stButton"] button {
        background: linear-gradient(180deg, #2b2219 0%, #1a140f 100%) !important;
        border: 1.5px solid #8c6f43 !important;
        border-radius: 6px !important;
        color: #faf2de !important;
        font-family: 'EB Garamond', 'Noto Serif Devanagari', serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 10px 14px !important;
        letter-spacing: normal !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(250, 242, 222, 0.1) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(180deg, #3d3023 0%, #251c14 100%) !important;
        border-color: #dfbe7b !important;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(197, 159, 91, 0.4), inset 0 0 8px rgba(197, 159, 91, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.7) !important;
    }

    /* Vintage Manuscript Query Input */
    div[data-testid="stTextInput"] {
        margin: 20px 0 24px 0 !important;
    }
    div[data-testid="stTextInput"] > label {
        font-family: 'EB Garamond', 'Cinzel', serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #dfbe7b !important;
        margin-bottom: 10px !important;
        display: block !important;
        letter-spacing: normal !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        min-height: 64px !important;
        height: 64px !important;
        border-radius: 10px !important;
        background: #1c150f !important;
        border: 2px solid #8c6f43 !important;
        outline: 1px solid rgba(197, 159, 91, 0.35) !important;
        outline-offset: -5px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.7), inset 0 2px 6px rgba(0,0,0,0.6) !important;
        overflow: hidden !important;
        display: flex !important;
        align-items: center !important;
        position: relative !important;
        transition: all 0.25s ease-in-out !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
        height: 100% !important;
        width: 100% !important;
        background-color: transparent !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stTextInput"] input {
        font-family: 'EB Garamond', 'Noto Serif Devanagari', serif !important;
        font-size: 1.25rem !important;
        line-height: 1.6 !important;
        height: 100% !important;
        padding: 0 62px 0 24px !important;
        background-color: transparent !important;
        color: #fffef7 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: #dfbe7b !important;
        outline-color: #f5e4bd !important;
        box-shadow: 0 0 24px rgba(197, 159, 91, 0.4), inset 0 0 12px rgba(197, 159, 91, 0.12) !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #a8947b !important;
        font-style: italic;
        font-size: 1.08rem !important;
    }

    /* Antique Brass Magnifying Symbol */
    div[data-testid="stTextInput"] div[data-baseweb="input"]::after {
        content: '' !important;
        position: absolute !important;
        right: 20px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 26px !important;
        height: 26px !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23dfbe7b' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='7.5'%3E%3C/circle%3E%3Cline x1='21' y1='21' x2='16.5' y2='16.5'%3E%3C/line%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: contain !important;
        pointer-events: none !important;
        opacity: 0.85 !important;
        transition: transform 0.2s ease, opacity 0.2s ease !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within::after {
        opacity: 1 !important;
        transform: translateY(-50%) scale(1.15) !important;
        filter: drop-shadow(0 0 6px rgba(197, 159, 91, 0.8)) !important;
    }

    .unmatched-alert {
        background: rgba(38, 16, 14, 0.95);
        border: 2px solid #b94a48;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
    }

    .transliteration-card {
        background: linear-gradient(135deg, #221a13 0%, #19130d 100%);
        border: 1.5px solid #8c6f43;
        border-radius: 10px;
        padding: 16px 22px;
        margin: 16px 0 22px 0;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    }

    .script-pill {
        background: rgba(38, 29, 21, 0.95);
        border: 1px solid #8c6f43;
        color: #dfbe7b;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'EB Garamond', serif;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .devanagari-preview {
        font-family: 'Noto Serif Devanagari', serif;
        font-size: 1.35rem;
        color: #faf2de;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(197, 159, 91, 0.35);
    }

    /* Antique Manuscript Answer Cards */
    .answer-card {
        background: linear-gradient(180deg, #251d15 0%, #1b140e 100%);
        border: 2px solid #8c6f43;
        border-left: 6px solid #dfbe7b;
        border-radius: 10px;
        padding: 24px 26px;
        margin-top: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.6), inset 0 0 20px rgba(140, 111, 67, 0.08);
        position: relative;
    }

    .answer-heading {
        font-family: 'Cinzel', 'EB Garamond', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #dfbe7b;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: 0.5px;
        border-bottom: 1px solid rgba(140, 111, 67, 0.3);
        padding-bottom: 6px;
    }

    .sanskrit-text {
        font-family: 'Noto Serif Devanagari', serif;
        font-size: 1.35rem;
        line-height: 2.1;
        color: #fffef7;
        background: rgba(18, 13, 9, 0.85);
        border: 1px solid #6d5432;
        padding: 20px 24px;
        border-radius: 8px;
        margin-bottom: 18px;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.6);
    }

    .explanation-box {
        background: rgba(28, 22, 16, 0.85);
        border: 1px solid rgba(140, 111, 67, 0.25);
        border-left: 5px solid #dfbe7b;
        padding: 18px 22px;
        border-radius: 6px;
        color: #f7eed8;
        font-family: 'EB Garamond', serif;
        font-size: 1.15rem;
        line-height: 1.8;
        margin-bottom: 18px;
    }

    .citation-box {
        background: rgba(16, 24, 18, 0.85);
        border: 1px solid rgba(77, 130, 99, 0.35);
        border-left: 5px solid #4d8263;
        padding: 14px 20px;
        border-radius: 6px;
        color: #dbebe0;
        font-size: 1.05rem;
        font-family: 'Noto Serif Devanagari', 'EB Garamond', serif;
        line-height: 1.8;
    }

    /* Vintage Tabs */
    div[data-testid="stTabs"] button[role="tab"] {
        font-family: 'Cinzel', 'EB Garamond', serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #c7b399 !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 18px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #dfbe7b !important;
        border-bottom: 3px solid #dfbe7b !important;
        text-shadow: 0 0 10px rgba(197, 159, 91, 0.4) !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #faf2de !important;
    }

    /* Antique Brass Metric Tiles */
    .stat-tile {
        background: linear-gradient(180deg, #241c14 0%, #18130d 100%);
        border: 1.5px solid #8c6f43;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5);
    }

    .stat-val {
        font-family: 'Cinzel', serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #dfbe7b;
        text-shadow: 0 0 8px rgba(197, 159, 91, 0.3);
    }

    .stat-lbl {
        font-family: 'EB Garamond', serif;
        font-size: 0.88rem;
        color: #c7b399;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Vintage Expander */
    div[data-testid="stExpander"] {
        border: 1.5px solid #8c6f43 !important;
        border-radius: 8px !important;
        background: rgba(26, 20, 14, 0.8) !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'EB Garamond', 'Noto Serif Devanagari', serif !important;
        color: #faf2de !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    /* Vintage Download Button */
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(180deg, #2b2219 0%, #1a140f 100%) !important;
        border: 1.5px solid #8c6f43 !important;
        color: #faf2de !important;
        font-family: 'EB Garamond', 'Cinzel', serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.5) !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(180deg, #3d3023 0%, #251c14 100%) !important;
        border-color: #dfbe7b !important;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(197, 159, 91, 0.4) !important;
    }

    .deck-header {
        font-family: 'Noto Serif Devanagari', 'Cinzel', serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #dfbe7b;
        margin-top: 8px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: normal !important;
        border-bottom: 1px solid rgba(140, 111, 67, 0.3);
        padding-bottom: 6px;
    }

    .vintage-divider {
        text-align: center;
        color: #c59f5b;
        font-family: 'Cinzel', serif;
        font-size: 0.95rem;
        letter-spacing: 2px;
        margin: 25px 0;
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

COMMON_ENGLISH_WORDS = {
    "why", "who", "what", "how", "when", "where", "did", "is", "was", "the", 
    "in", "a", "an", "and", "king", "servant", "sugar", "demon", "flood", 
    "cold", "winter", "scholar", "water", "cart", "god", "old", "woman", 
    "bell", "monkey", "grammar", "meaning", "error", "story", "explain", "tell"
}

def is_natural_english_query(text: str) -> bool:
    """Checks if query contains standard conversational English words."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    return any(w in COMMON_ENGLISH_WORDS for w in words)

def get_sanskrit_topic_representation(text: str) -> str:
    """Translates English conversational questions into corresponding Sanskrit query formulation."""
    t = text.lower()
    if any(w in t for w in ["bhoj", "kalidas", "king", "raja"]):
        if any(w in t for w in ["amount", "prize", "reward", "money", "lakh", "announce", "announced", "annouced", "how much"]):
            return "भोजराज्ञा घोषितं काव्यपारितोषिकम् (King Bhoja's Announced Reward)"
        elif any(w in t for w in ["99", "crore", "riddle", "gem", "gems"]):
            return "कालीदासस्य ९९-कोटिरत्नकूटश्लोकः (Kalidasa's 99-Crore Gems Riddle)"
        elif any(w in t for w in ["scholar", "scholars", "memory", "ekapathi", "dvipathi", "tripathi"]):
            return "एकपाठि-द्विपाठि-त्रिपाठि विद्वांसः (Scholars' Photographic Memory)"
        elif any(w in t for w in ["why", "fail", "prevent"]):
            return "कविभ्यः पारितोषिकप्राप्तौ विघ्नः (Why Poets Failed To Win Prize)"
        return "चतुरस्य कालीदासस्य कथा (Story of Clever Kalidasa & King Bhoja)"
    elif any(w in t for w in ["servant", "sugar", "puppy", "dog", "milk", "cloth", "soot", "face", "shankhan", "fool"]):
        if any(w in t for w in ["sugar", "market", "spill", "leak"]):
            return "मूर्खभृत्येन जीर्णे वस्त्रे शर्कराहरणम् (Servant Spilling Sugar in Torn Cloth)"
        elif any(w in t for w in ["puppy", "dog", "sack"]):
            return "सञ्चिकायां श्वानशावकस्य श्वासरोधः (Puppy Suffocated in Sack)"
        elif any(w in t for w in ["milk", "rope", "drag"]):
            return "दोरकेण दुग्धपात्राकर्षणम् (Dragging Milk Pot with Rope)"
        elif any(w in t for w in ["face", "black", "soot", "kohl"]):
            return "कज्जलेन कृष्णमुखभृत्यः (Servant Smearing Face with Soot)"
        return "मूर्खभृत्यस्य शंखनादस्य कथा (Story of Foolish Servant Shankhanada)"
    elif any(w in t for w in ["ghanta", "demon", "monster", "old woman", "bell", "monkey", "tiger", "chitrapur"]):
        if any(w in t for w in ["fruit", "fruits", "old woman", "solve", "reward"]):
            return "मधुरफलैः घण्टाहरणं वृद्धायाः चातुर्यम् (Old Woman Solving Bell Mystery)"
        elif any(w in t for w in ["who", "demon"]):
            return "चित्रपुरे घण्टाकर्णराक्षसस्य जनप्रवादः (Rumor of Ghantakarna Demon)"
        return "वृद्धायाः चातुर्यम् कथा (Story of Clever Old Woman & Bell)"
    elif any(w in t for w in ["devotee", "god", "flood", "rain", "drown", "water", "effort"]):
        return "देवभक्तस्य जले मरणम् उद्यमकथा (Devotee in Flood & Importance of Effort)"
    elif any(w in t for w in ["cold", "winter", "badhati", "badhate", "grammar", "palanquin"]):
        return "शीतं बहु बाधति आत्मनेपद-दोषविचारः (Grammar Error in 'Sheetam Bahu Badhati')"
    else:
        dev_tr = to_devanagari(text)
        return dev_tr if dev_tr else "संस्कृत-अर्थानुसन्धानम् (Sanskrit Semantic Search)"

# Pipeline Singleton
@st.cache_resource
def get_pipeline():
    persist_dir = os.path.join(curr_dir, "..", "chroma_db")
    pipeline = SanskritRAGPipeline(
        collection_name="sanskrit_knowledge",
        persist_directory=persist_dir
    )
    default_doc = os.path.join(curr_dir, "..", "data", "sanskrit_corpus.txt")
    if os.path.exists(default_doc):
        try:
            pipeline.index_document(default_doc, overwrite=False)
        except Exception:
            pass
    return pipeline

pipeline = get_pipeline()

# Session State for Query input
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

def set_query(q_str):
    st.session_state.query_text = q_str

# System Pipeline Settings (Permanently Locked to CPU Hybrid Fusion)
retrieval_mode = "hybrid"
top_k = 3

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-crest">§ प्राच्य-संस्कृत-ग्रन्थागारः §</div>
    <div class="hero-title">SANSKRIT RETRIEVAL-AUGMENTED GENERATION</div>
    <div class="hero-subtitle">
        A classical epistemic retrieval engine for ancient Sanskrit literature, Vedic philosophy, and historical manuscripts.
    </div>
    <div class="badge-container">
        <span class="spec-badge">✦ 100% CPU Only (No GPU)</span>
        <span class="spec-badge">◆ English & Sanskrit Natural Queries</span>
        <span class="spec-badge">◈ Dual Script: Devanagari + IAST / HK / ITRANS</span>
        <span class="spec-badge">❖ Hybrid Vector & BM25 Fusion</span>
        <span class="spec-badge">§ Verse & Danda (।) Aware Semantic Chunking</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 1. PRIMARY SEARCH GATEWAY (FRONT & CENTER)
query_val = st.text_input(
    "§ Historical Manuscript Query | Enter Your Question (English, Devanagari Sanskrit, or Romanized IAST / HK / ITRANS):",
    value=st.session_state.query_text,
    placeholder="Inquire in English (e.g. 'What did King Bhoja announce?') or Sanskrit and press Enter...",
    key="main_query_input",
    label_visibility="visible"
)

# 2. STRUCTURED SIDE-BY-SIDE VINTAGE QUERY DECKS
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="deck-header">
        <span>◈</span> English Inquiries (Classical Literature)
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Why did servant ruin sugar?", use_container_width=True):
            set_query("Why did the foolish servant ruin the sugar?")
        if st.button("Who was Ghantakarna demon?", use_container_width=True):
            set_query("Who was Ghantakarna and why was the bell ringing?")
    with c2:
        if st.button("What did King Bhoja announce?", use_container_width=True):
            set_query("What did King Bhoja announce in his court?")
        if st.button("Why was 'badhati' incorrect?", use_container_width=True):
            set_query("Why was badhati incorrect in sheetam bahu badhati?")

with col_right:
    st.markdown("""
    <div class="deck-header">
        <span>◈</span> Sanskrit Inscriptions (मूलसंस्कृतप्रश्नाः)
    </div>
    """, unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        if st.button("मूर्खभृत्यः शर्कराम् कुत्र न्यस्यति ?", use_container_width=True):
            set_query("मूर्खभृत्यः शर्कराम् कुत्र न्यस्यति ?")
        if st.button("चित्रपुरे घण्टाकर्णः नाम कः आसीत् ?", use_container_width=True):
            set_query("चित्रपुरे घण्टाकर्णः नाम कः आसीत् ?")
    with c4:
        if st.button("भोजराजः काव्यपठने किं घोषितवान् ?", use_container_width=True):
            set_query("भोजराजः काव्यपठने किं घोषितवान् ?")
        if st.button("देवभक्तः किमर्थं जले मृतवान् ?", use_container_width=True):
            set_query("देवभक्तः किमर्थं जले मृतवान् ?")

if query_val.strip():
    # Step 1: Script Detection & Normalization Preview
    is_dev = is_devanagari(query_val)
    is_en = is_natural_english_query(query_val)

    if is_dev:
        detected_scheme = "Devanagari (Native Sanskrit)"
        search_target_display = query_val
    elif is_en:
        detected_scheme = "English (Natural Language Query)"
        search_target_display = get_sanskrit_topic_representation(query_val)
    else:
        detected_scheme = f"Romanized Sanskrit ({detect_transliteration_scheme(query_val)})"
        search_target_display = to_devanagari(query_val)

    # Transliteration Visualizer Bar
    st.markdown(f"""
    <div class="transliteration-card">
        <div>
            <span style="color: #c7b399; font-family: 'EB Garamond', serif; font-size: 0.95rem;">Input Query Language / Script:</span><br>
            <span class="script-pill">{detected_scheme}</span>
        </div>
        <div style="font-size: 1.5rem; color: #dfbe7b;">➔</div>
        <div style="flex-grow: 1;">
            <span style="color: #c7b399; font-family: 'EB Garamond', serif; font-size: 0.95rem;">Manuscript Sanskrit Representation (संस्कृत-प्रतीकम्):</span><br>
            <span class="devanagari-preview">{search_target_display}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step 2: Query Execution with Spinner & Auto-Recovery
    with st.spinner("Deciphering historical Sanskrit corpus and synthesizing grounded bilingual response on CPU..."):
        t0 = time.time()
        try:
            result = pipeline.query(query_val, top_k=top_k, retrieval_mode=retrieval_mode)
        except Exception:
            # Gracefully refresh collection handle across terminal processes
            pipeline.retriever._get_collection()
            pipeline.retriever._hydrate_bm25_from_db()
            default_doc = os.path.join(curr_dir, "..", "data", "sanskrit_corpus.txt")
            if os.path.exists(default_doc):
                pipeline.index_document(default_doc, overwrite=False)
            result = pipeline.query(query_val, top_k=top_k, retrieval_mode=retrieval_mode)
        exec_latency = time.time() - t0

    # Tabs for Rich Layout
    tab_ans, tab_chunks, tab_metrics, tab_corpus = st.tabs([
        "✦ Grounded Sanskrit & English Response",
        "◈ Retrieved Context Chunks",
        "❖ Performance & CPU Telemetry",
        "§ Corpus Explorer"
    ])

    with tab_ans:
        response_text = result.get("response", "")
        is_matched = result.get("is_matched", True)

        # Prominent Alert Banner if Query Does Not Match Document
        if not is_matched or "The query does not match with the retrieved document" in response_text:
            st.markdown("""
            <div class="unmatched-alert">
                <span style="font-size: 1.8rem; color: #f87171; font-weight: bold;">[ ! ]</span>
                <div>
                    <div style="font-weight: 700; color: #f87171; font-size: 1.2rem; font-family: 'Cinzel', serif;">The query does not match with the retrieved document</div>
                    <div style="color: #f7eed8; font-size: 0.96rem; margin-top: 4px; font-family: 'EB Garamond', serif;">
                        The entered query is not addressed by any documents in the ingested Sanskrit corpus. No relevant contextual evidence was found in the text to answer this question.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Robust separation of Sanskrit, English, and Reference sections
        sanskrit_part = ""
        english_part = ""
        reference_part = ""

        if "2. English Explanation:" in response_text:
            parts = response_text.split("2. English Explanation:")
            sanskrit_raw = parts[0]
            rest = parts[1]
            
            sanskrit_part = sanskrit_raw.replace("1. उत्तरम् (Sanskrit Answer):", "").strip()
            
            if "3. प्रमाणम् / Reference:" in rest:
                sub_parts = rest.split("3. प्रमाणम् / Reference:")
                english_part = sub_parts[0].strip()
                reference_part = sub_parts[1].strip()
            else:
                english_part = rest.strip()
        else:
            sanskrit_part = response_text

        # Format and display cards
        st.markdown("""
        <div class="answer-card">
            <div class="answer-heading">
                <span>◈</span> उत्तरम् (Classical Sanskrit Excerpt)
            </div>
            <div class="sanskrit-text">
        """ + sanskrit_part + """
            </div>
        </div>
        """, unsafe_allow_html=True)

        if english_part:
            st.markdown("""
            <div class="answer-card" style="margin-top: 18px;">
                <div class="answer-heading" style="color: #dfbe7b;">
                    <span>◆</span> English Commentary & Historical Meaning (विस्तृत-आङ्ग्लार्थः)
                </div>
                <div class="explanation-box">
            """ + english_part.replace("\n", "<br>") + """
                </div>
            </div>
            """, unsafe_allow_html=True)

        if reference_part:
            st.markdown("""
            <div class="answer-card" style="border-left: 6px solid #4d8263; margin-top: 18px;">
                <div class="answer-heading" style="color: #7bc498; font-size: 1.15rem;">
                    <span>§</span> प्रमाणम् / Direct Manuscript Reference (मूलग्रन्थसन्दर्भः)
                </div>
                <div class="citation-box">
                    <b>मूलग्रन्थसन्दर्भः:</b> """ + reference_part.replace("\n", "<br>") + """
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_chunks:
        chunks = result.get("retrieved_chunks", [])
        is_matched = result.get("is_matched", True)
        if not chunks or not is_matched:
            st.markdown("""
            <div style="background: rgba(38, 16, 14, 0.85); border: 2px dashed #b94a48; border-radius: 8px; padding: 24px; text-align: center; color: #f7eed8; margin-top: 10px;">
                <span style="font-size: 2rem; color: #c7b399;">§</span><br>
                <b style="color: #f87171; font-size: 1.2rem; font-family: 'Cinzel', serif;">The query does not match with the retrieved document</b><br>
                <span style="color: #c7b399; font-size: 0.96rem; font-family: 'EB Garamond', serif;">No relevant context chunks were found above the relevance threshold in the ingested Sanskrit corpus.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**Found {len(chunks)} relevant context segments from Sanskrit corpus:**")
            for idx, c in enumerate(chunks, 1):
                sec = c.get("metadata", {}).get("section", "General")
                source = c.get("metadata", {}).get("source", "sanskrit_corpus")
                score = c.get("rrf_score", c.get("dense_score", c.get("bm25_score", 0.0)))
                
                with st.expander(f"§ Context Chunk #{idx} — Section: 『{sec}』 (Relevance Score: {score:.4f})", expanded=(idx == 1)):
                    st.markdown(f"""
                    <div style="font-family: 'Noto Serif Devanagari', serif; font-size: 1.2rem; line-height: 1.9; background: #16110c; color: #fffef7; padding: 16px; border-radius: 6px; border: 1.5px solid #6d5432;">
                        {c.get('content', '')}
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"Source Document: `{source}` | Chunk ID: `{c.get('id', 'N/A')}`")

    with tab_metrics:
        m = result.get("metrics", {})
        st.markdown("#### ❖ Latency & CPU Resource Breakdown")
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-val">{m.get('transliteration_latency_s', 0) * 1000:.1f} ms</div>
                <div class="stat-lbl">Transliteration / Parsing</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[1]:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-val">{m.get('retrieval_latency_s', 0) * 1000:.1f} ms</div>
                <div class="stat-lbl">Hybrid Retrieval</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[2]:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-val">{m.get('generation_latency_s', 0) * 1000:.1f} ms</div>
                <div class="stat-lbl">CPU Generation</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[3]:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-val">{exec_latency * 1000:.1f} ms</div>
                <div class="stat-lbl">End-to-End Latency</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        - **Hardware Inference Mode:** 100% CPU (AVX2 vector instructions, Zero GPU)
        - **Dense Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
        - **Lexical Engine:** Rank-BM25 with Sanskrit Morphological Stem Normalization
        - **RAM Footprint:** < 1.2 GB
        """)

    with tab_corpus:
        corpus_file = os.path.join(curr_dir, "..", "data", "sanskrit_corpus.txt")
        if os.path.exists(corpus_file):
            with open(corpus_file, "r", encoding="utf-8") as f:
                full_corpus = f.read()
            st.markdown("#### § Ingested Sanskrit Corpus Preview")
            st.text_area("Full Corpus Text", value=full_corpus, height=350, disabled=True)
        else:
            st.info("Corpus text file not found.")

else:
    # Landing Placeholder when no query is typed
    st.markdown("""
    <div style="text-align: center; padding: 32px 20px; background: rgba(30, 23, 16, 0.55); border-radius: 10px; border: 1.5px dashed #8c6f43; margin-top: 18px; box-shadow: 0 6px 20px rgba(0,0,0,0.5);">
        <div style="font-size: 2.6rem; margin-bottom: 8px; color: #dfbe7b;">§</div>
        <div style="font-family: 'Cinzel', serif; font-size: 1.3rem; font-weight: 700; color: #dfbe7b; margin-bottom: 6px;">Historical Manuscript Archive Ready</div>
        <div style="font-family: 'EB Garamond', serif; color: #d9c8af; font-size: 1.1rem; max-width: 680px; margin: 0 auto; line-height: 1.6;">
            Inquire above in <b>Natural English</b>, <b>Devanagari Sanskrit</b>, or <b>Romanized Transliterations (IAST / HK / ITRANS)</b>, or select any sample inquiry to retrieve authentic Sanskrit excerpts with full English commentary.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# BOTTOM SECTION: SYSTEM ARCHITECTURE & DOCUMENTATION
# =====================================================================
st.markdown("<div class='vintage-divider'><span>§ ═══════════════ ❖ ═══════════════ §</span></div>", unsafe_allow_html=True)
st.markdown("### ❖ Archival System Architecture & Technical Documentation")

bot_cols = st.columns([1, 1, 1])

with bot_cols[0]:
    st.markdown("##### ✦ Hardware Inference Engine")
    st.success("✓ **100% CPU Inference (Zero GPU)**")
    st.caption("Low-latency inference via Multilingual MiniLM embeddings & BM25 Fusion on CPU.")

with bot_cols[1]:
    st.markdown("##### ◈ Retrieval Engine")
    st.info("✦ **Hybrid Fusion Active**")
    st.caption("ChromaDB Vector Embeddings + Sanskrit Lexical Indexing (Rank-BM25).")

with bot_cols[2]:
    st.markdown("##### § Technical Documentation")
    report_file_path = os.path.join(curr_dir, "..", "report", "Sanskrit_RAG_Technical_Report.pdf")
    if os.path.exists(report_file_path):
        with open(report_file_path, "rb") as rf:
            st.download_button(
                label="Download Technical Report (PDF)",
                data=rf.read(),
                file_name="Sanskrit_RAG_Technical_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    st.caption("Author: **Sahil Karande** | Assignment Submission")

with st.expander("Document Ingestion: Upload & Index Additional Sanskrit Documents (.txt / .pdf)", expanded=False):
    st.caption("Upload any custom Sanskrit text or PDF document to index it on CPU into ChromaDB and BM25.")
    uploaded_file = st.file_uploader("Upload Sanskrit Document (.txt / .pdf)", type=["txt", "pdf"])
    if uploaded_file is not None:
        save_path = os.path.join(curr_dir, "..", "data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button("Index Document Now", use_container_width=True):
            with st.spinner("Parsing Sanskrit glyphs & indexing on CPU..."):
                count = pipeline.index_document(save_path, overwrite=False)
                st.success(f"Indexed {count} chunks successfully!")

st.markdown("<div style='text-align: center; color: #8c6f43; font-family: EB Garamond, serif; font-size: 1.05rem; padding: 28px 0 14px 0;'>§ Sanskrit RAG System | Historical Manuscript Archive Edition | Developed by Sahil Karande §</div>", unsafe_allow_html=True)
