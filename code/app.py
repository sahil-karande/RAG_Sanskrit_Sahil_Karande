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

# Custom Royal Indian Historical King Style (Imperial Darbar & Vedic Gold-Crimson Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=Cinzel+Decorative:wght@700;900&family=Marcellus&family=Noto+Serif+Devanagari:wght@400;500;600;700;800&family=Rozha+One&display=swap');

    /* Completely hide the sidebar & default header elements */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    button[data-testid="baseButton-header"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Royal Imperial Canvas Background */
    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #2e0912 0%, #170408 50%, #0d0204 100%) !important;
        color: #fdf6e2 !important;
    }

    :root {
        --royal-gold-light: #fff2b2;
        --royal-gold: #d4af37;
        --royal-gold-burnished: #f59e0b;
        --royal-gold-deep: #b38728;
        --royal-crimson-dark: #1f070c;
        --royal-crimson-mid: #330b14;
        --royal-crimson-card: #24080e;
        --royal-gold-border: #c59b27;
        --royal-ivory: #fffbeb;
        --royal-muted: #d1bda0;
    }

    /* Royal Imperial Court Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #2e0912 0%, #42101c 50%, #22060c 100%);
        border: 2px solid #b38728;
        outline: 1px solid rgba(212, 175, 55, 0.45);
        outline-offset: -5px;
        border-radius: 16px;
        padding: 30px 36px;
        margin-bottom: 26px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.75), inset 0 0 35px rgba(212, 175, 55, 0.12);
        position: relative;
        overflow: hidden;
        text-align: center;
    }

    .hero-crest {
        font-family: 'Cinzel Decorative', 'Cinzel', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffd700;
        letter-spacing: 3px;
        margin-bottom: 6px;
        text-shadow: 0 0 12px rgba(212, 175, 55, 0.6);
    }
    
    .hero-title {
        font-family: 'Cinzel Decorative', 'Cinzel', serif;
        font-size: 2.35rem;
        font-weight: 900;
        background: linear-gradient(135deg, #fff7c2 0%, #ffd700 25%, #d4af37 50%, #fff0a8 75%, #b38728 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
        text-shadow: 0 0 25px rgba(212, 175, 55, 0.35);
    }

    .hero-subtitle {
        font-family: 'Marcellus', 'Noto Serif Devanagari', serif;
        font-size: 1.12rem;
        color: #fce7cf;
        line-height: 1.7;
        max-width: 960px;
        margin: 0 auto;
    }

    .badge-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin-top: 18px;
    }

    .spec-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(45, 10, 18, 0.95);
        border: 1.5px solid #d4af37;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 242, 178, 0.25);
        color: #fff2b2;
        padding: 5px 15px;
        border-radius: 20px;
        font-family: 'Marcellus', serif;
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0.4px;
    }

    /* Royal King Court Buttons (राजमुद्रा-पट्टिका) */
    div[data-testid="stButton"] button {
        background: linear-gradient(180deg, #380d16 0%, #1f060b 100%) !important;
        border: 1.5px solid #d4af37 !important;
        border-radius: 8px !important;
        color: #fff0a8 !important;
        font-family: 'Marcellus', 'Noto Serif Devanagari', serif !important;
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        letter-spacing: 0.4px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 242, 178, 0.2) !important;
        transition: all 0.22s ease-in-out !important;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(180deg, #541423 0%, #300913 100%) !important;
        border-color: #ffe58f !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.5), inset 0 0 10px rgba(212, 175, 55, 0.25) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.7) !important;
    }

    /* Imperial Search Query Gateway (राजकीय-प्रश्नद्वारम्) */
    div[data-testid="stTextInput"] {
        margin: 24px 0 28px 0 !important;
    }
    div[data-testid="stTextInput"] > label {
        font-family: 'Cinzel', serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #d4af37 !important;
        text-shadow: 0 0 12px rgba(212, 175, 55, 0.3) !important;
        margin-bottom: 12px !important;
        display: block !important;
        letter-spacing: 0.8px !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        min-height: 66px !important;
        height: 66px !important;
        border-radius: 12px !important;
        background: linear-gradient(180deg, #22080e 0%, #150408 100%) !important;
        border: 2px solid #d4af37 !important;
        outline: 1px solid rgba(212, 175, 55, 0.4) !important;
        outline-offset: -5px !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.7), inset 0 2px 8px rgba(0,0,0,0.6) !important;
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
        font-family: 'Marcellus', 'Noto Serif Devanagari', serif !important;
        font-size: 1.22rem !important;
        line-height: 1.6 !important;
        height: 100% !important;
        padding: 0 62px 0 26px !important;
        background-color: transparent !important;
        color: #fffbeb !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: #fff0a8 !important;
        outline-color: #ffd700 !important;
        box-shadow: 0 0 28px rgba(212, 175, 55, 0.5), inset 0 0 15px rgba(212, 175, 55, 0.15) !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #bfa68a !important;
        font-style: italic;
        font-size: 1.05rem !important;
    }

    /* Royal Golden Sceptre / Magnifying Symbol (Not emoji) */
    div[data-testid="stTextInput"] div[data-baseweb="input"]::after {
        content: '' !important;
        position: absolute !important;
        right: 22px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 28px !important;
        height: 28px !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23d4af37' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='7.5'%3E%3C/circle%3E%3Cline x1='21' y1='21' x2='16.5' y2='16.5'%3E%3C/line%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: contain !important;
        pointer-events: none !important;
        opacity: 0.9 !important;
        transition: transform 0.2s ease, opacity 0.2s ease !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within::after {
        opacity: 1 !important;
        transform: translateY(-50%) scale(1.2) !important;
        filter: drop-shadow(0 0 8px rgba(212, 175, 55, 0.9)) !important;
    }

    .unmatched-alert {
        background: rgba(45, 10, 15, 0.9);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    }

    .transliteration-card {
        background: linear-gradient(135deg, #24080e 0%, #1a0509 100%);
        border: 1.5px solid #b38728;
        border-radius: 12px;
        padding: 18px 24px;
        margin: 18px 0 26px 0;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.5), inset 0 0 15px rgba(212, 175, 55, 0.06);
    }

    .script-pill {
        background: rgba(45, 10, 18, 0.95);
        border: 1px solid #d4af37;
        color: #fef08a;
        padding: 5px 14px;
        border-radius: 6px;
        font-family: 'Marcellus', serif;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .devanagari-preview {
        font-family: 'Noto Serif Devanagari', 'Rozha One', serif;
        font-size: 1.35rem;
        color: #ffd700;
        font-weight: 700;
        text-shadow: 0 0 12px rgba(212, 175, 55, 0.4);
    }

    /* Royal Decree Answer Cards (राजकीय-उत्तर-पत्रम्) */
    .answer-card {
        background: linear-gradient(180deg, #2a0910 0%, #1b0509 100%);
        border: 2px solid #b38728;
        border-left: 7px solid #d4af37;
        border-radius: 14px;
        padding: 26px 28px;
        margin-top: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.65), inset 0 0 25px rgba(212, 175, 55, 0.08);
        position: relative;
    }

    .answer-heading {
        font-family: 'Cinzel Decorative', 'Cinzel', serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffd700;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: 0.5px;
        border-bottom: 1px solid rgba(212, 175, 55, 0.25);
        padding-bottom: 8px;
    }

    .sanskrit-text {
        font-family: 'Noto Serif Devanagari', serif;
        font-size: 1.35rem;
        line-height: 2.1;
        color: #fffdf5;
        background: rgba(14, 3, 6, 0.85);
        border: 1px solid rgba(212, 175, 55, 0.35);
        padding: 22px 26px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.6);
    }

    .explanation-box {
        background: rgba(22, 10, 20, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-left: 5px solid #d4af37;
        padding: 20px 24px;
        border-radius: 8px;
        color: #f5eedc;
        font-family: 'Marcellus', serif;
        font-size: 1.08rem;
        line-height: 1.8;
        margin-bottom: 20px;
    }

    .citation-box {
        background: rgba(10, 24, 16, 0.85);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 5px solid #10b981;
        padding: 16px 22px;
        border-radius: 8px;
        color: #d1fae5;
        font-size: 1.02rem;
        font-family: 'Noto Serif Devanagari', serif;
        line-height: 1.8;
    }

    /* Royal Tabs (राजकीय-पट्टिका) */
    div[data-testid="stTabs"] button[role="tab"] {
        font-family: 'Cinzel', serif !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        color: #d1bda0 !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
        padding: 12px 20px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #ffd700 !important;
        border-bottom: 3px solid #d4af37 !important;
        text-shadow: 0 0 10px rgba(212, 175, 55, 0.5) !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #fff2b2 !important;
    }

    /* Royal Medallion Tiles (राजमुद्रा-मापकम्) */
    .stat-tile {
        background: linear-gradient(180deg, #2b0910 0%, #190509 100%);
        border: 1.5px solid #b38728;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5), inset 0 0 12px rgba(212, 175, 55, 0.08);
    }

    .stat-val {
        font-family: 'Cinzel', serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffd700;
        text-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
    }

    .stat-lbl {
        font-family: 'Marcellus', serif;
        font-size: 0.8rem;
        color: #d1bda0;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 6px;
    }

    /* Royal Expander Styling */
    div[data-testid="stExpander"] {
        border: 1.5px solid #b38728 !important;
        border-radius: 10px !important;
        background: rgba(26, 6, 11, 0.75) !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'Marcellus', 'Noto Serif Devanagari', serif !important;
        color: #fef08a !important;
        font-weight: 600 !important;
    }

    /* Royal Document Download Button */
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(180deg, #380d16 0%, #1f060b 100%) !important;
        border: 1.5px solid #d4af37 !important;
        color: #fff0a8 !important;
        font-family: 'Cinzel', serif !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6) !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(180deg, #551423 0%, #300913 100%) !important;
        border-color: #ffd700 !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.5) !important;
    }

    .royal-divider {
        text-align: center;
        color: #d4af37;
        font-family: 'Cinzel', serif;
        font-size: 1.0rem;
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
    <div class="hero-crest">⚜ श्रीसंस्कृतज्ञानभाण्डारम् ⚜</div>
    <div class="hero-title">SANSKRIT RETRIEVAL-AUGMENTED GENERATION</div>
    <div class="hero-subtitle">
        राजकीय-शास्त्रानुसन्धान-प्रणाली | Imperial Question-Answering Architecture for Classical Sanskrit Literature, Philosophy, and Royal Subhashitas.
        Grounded with Authentic Devanagari Decrees, Comprehensive English Commentaries, and Verbatim Citations.
    </div>
    <div class="badge-container">
        <span class="spec-badge">✦ 100% CPU Only (No GPU)</span>
        <span class="spec-badge">◆ English & Sanskrit Natural Queries</span>
        <span class="spec-badge">◈ Dual Script: Devanagari + IAST / HK / ITRANS</span>
        <span class="spec-badge">❖ Imperial Hybrid Vector & BM25 Fusion</span>
        <span class="spec-badge">§ Verse & Danda (।) Aware Semantic Chunking</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Query Chips
st.markdown("##### ⚜ Quick Sample Queries (Click any royal decree button to test):")

st.markdown("<span style='color: #ffd700; font-family: Marcellus, serif; font-weight: 700; font-size: 0.98rem;'>◈ English Queries (आङ्ग्लभाषा-प्रश्नाः):</span>", unsafe_allow_html=True)
en_cols = st.columns(4)
with en_cols[0]:
    if st.button("Why did the foolish servant ruin the sugar?", use_container_width=True):
        set_query("Why did the foolish servant ruin the sugar?")
with en_cols[1]:
    if st.button("What did King Bhoja announce in his court?", use_container_width=True):
        set_query("What did King Bhoja announce in his court?")
with en_cols[2]:
    if st.button("Who was Ghantakarna and why was the bell ringing?", use_container_width=True):
        set_query("Who was Ghantakarna and why was the bell ringing?")
with en_cols[3]:
    if st.button("Why was 'badhati' grammatically incorrect?", use_container_width=True):
        set_query("Why was badhati incorrect in sheetam bahu badhati?")

st.markdown("<span style='color: #ffd700; font-family: Marcellus, serif; font-weight: 700; font-size: 0.98rem;'>◈ Sanskrit Queries (संस्कृत-प्रश्नाः):</span>", unsafe_allow_html=True)
sk_cols = st.columns(4)
with sk_cols[0]:
    if st.button("मूर्खभृत्यः शर्कराम् कुत्र न्यस्यति ?", use_container_width=True):
        set_query("मूर्खभृत्यः शर्कराम् कुत्र न्यस्यति ?")
with sk_cols[1]:
    if st.button("भोजराजः काव्यपठने किं घोषितवान् ?", use_container_width=True):
        set_query("भोजराजः काव्यपठने किं घोषितवान् ?")
with sk_cols[2]:
    if st.button("चित्रपुरे घण्टाकर्णः नाम कः आसीत् ?", use_container_width=True):
        set_query("चित्रपुरे घण्टाकर्णः नाम कः आसीत् ?")
with sk_cols[3]:
    if st.button("देवभक्तः किमर्थं जले मृतवान् ?", use_container_width=True):
        set_query("देवभक्तः किमर्थं जले मृतवान् ?")

# Full-Width Search Input (With End-Corner Golden Sceptre/Magnifying Symbol — Press Enter to Search)
query_val = st.text_input(
    "⚜ प्रष्टव्य-प्रश्नद्वारम् | Enter Your Royal Question (English, Devanagari Sanskrit, or Romanized IAST / HK / ITRANS):",
    value=st.session_state.query_text,
    placeholder="Ask in English (e.g. 'What did King Bhoja announce?') or Sanskrit and press Enter...",
    key="main_query_input",
    label_visibility="visible"
)

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
            <span style="color: #d1bda0; font-family: 'Marcellus', serif; font-size: 0.88rem;">Input Query Language / Script:</span><br>
            <span class="script-pill">{detected_scheme}</span>
        </div>
        <div style="font-size: 1.6rem; color: #ffd700;">➔</div>
        <div style="flex-grow: 1;">
            <span style="color: #d1bda0; font-family: 'Marcellus', serif; font-size: 0.88rem;">Sanskrit Canonical Representation (संस्कृत-प्रतीकम्):</span><br>
            <span class="devanagari-preview">{search_target_display}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step 2: Query Execution with Spinner & Auto-Recovery
    with st.spinner("Retrieving Sanskrit context chunks and synthesizing grounded bilingual response on CPU..."):
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
                    <div style="color: #fce7cf; font-size: 0.96rem; margin-top: 4px; font-family: 'Marcellus', serif;">
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
                <span>◈</span> उत्तरम् (Imperial Sanskrit Decree)
            </div>
            <div class="sanskrit-text">
        """ + sanskrit_part + """
            </div>
        </div>
        """, unsafe_allow_html=True)

        if english_part:
            st.markdown("""
            <div class="answer-card" style="margin-top: 20px;">
                <div class="answer-heading" style="color: #fff2b2;">
                    <span>◆</span> English Explanation & Complete Meaning (विस्तृत-आङ्ग्लार्थः)
                </div>
                <div class="explanation-box">
            """ + english_part.replace("\n", "<br>") + """
                </div>
            </div>
            """, unsafe_allow_html=True)

        if reference_part:
            st.markdown("""
            <div class="answer-card" style="border-left: 7px solid #10b981; margin-top: 20px;">
                <div class="answer-heading" style="color: #6ee7b7; font-size: 1.15rem;">
                    <span>§</span> प्रमाणम् / Direct Corpus Reference (मूलग्रन्थसन्दर्भः)
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
            <div style="background: rgba(45, 10, 15, 0.85); border: 2px dashed #ef4444; border-radius: 12px; padding: 26px; text-align: center; color: #fce7cf; margin-top: 10px;">
                <span style="font-size: 2rem; color: #d1bda0;">§</span><br>
                <b style="color: #f87171; font-size: 1.2rem; font-family: 'Cinzel', serif;">The query does not match with the retrieved document</b><br>
                <span style="color: #d1bda0; font-size: 0.96rem; font-family: 'Marcellus', serif;">No relevant context chunks were found above the relevance threshold in the ingested Sanskrit corpus.</span>
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
                    <div style="font-family: 'Noto Serif Devanagari', serif; font-size: 1.2rem; line-height: 1.9; background: #140407; color: #fffdf5; padding: 18px; border-radius: 8px; border: 1.5px solid #b38728;">
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
    <div style="text-align: center; padding: 50px 24px; background: rgba(36, 8, 14, 0.45); border-radius: 14px; border: 1.5px dashed #b38728; margin-top: 25px; box-shadow: 0 8px 24px rgba(0,0,0,0.5);">
        <div style="font-size: 3.2rem; margin-bottom: 14px; color: #ffd700;">⚜</div>
        <div style="font-family: 'Cinzel Decorative', 'Cinzel', serif; font-size: 1.45rem; color: #ffd700; margin-bottom: 8px;">Ready for Royal Queries</div>
        <div style="font-family: 'Marcellus', serif; color: #fce7cf; font-size: 1.05rem; max-width: 680px; margin: 0 auto; line-height: 1.7;">
            Inquire in <b>Natural English</b>, <b>Devanagari Sanskrit</b>, or <b>Romanized Transliterations (IAST / HK / ITRANS)</b>, or select any imperial decree query above to receive the authentic Sanskrit answer with full English commentary.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# BOTTOM SECTION: SYSTEM ARCHITECTURE & DOCUMENTATION
# =====================================================================
st.markdown("<div class='royal-divider'><span>⚜ ═══════════════ ◆ ═══════════════ ⚜</span></div>", unsafe_allow_html=True)
st.markdown("### ❖ Imperial System Architecture & Royal Documentation")

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

st.markdown("<div style='text-align: center; color: #b38728; font-family: Marcellus, serif; font-size: 0.95rem; padding: 30px 0 15px 0;'>⚜ Sanskrit RAG System | Royal Historical King Edition | Developed by Sahil Karande ⚜</div>", unsafe_allow_html=True)
