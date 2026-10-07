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
import base64
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

    /* Vintage Antique Parchment Canvas with Soft Gentle Amber Fades */
    .stApp {
        background: radial-gradient(ellipse at 50% 35%, #fdf8ee 0%, #f7ecd7 35%, #ebd8b8 70%, #dfc79f 90%, #d5b88c 100%) fixed !important;
        background-color: #f7ecd7 !important;
        color: #2b180d !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        color: #2b180d !important;
        font-family: 'Cinzel', serif !important;
        letter-spacing: 0.5px !important;
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
        background: linear-gradient(180deg, #2a1f15 0%, #17110c 100%) !important;
        border: 1.5px solid #8c6f43 !important;
        border-left: 5px solid #dfbe7b !important;
        border-radius: 6px !important;
        color: #fffdf5 !important;
        font-family: 'EB Garamond', 'Noto Serif Devanagari', serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 8px 12px !important;
        min-height: 52px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        letter-spacing: normal !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(250, 242, 222, 0.1) !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
    }
    div[data-testid="stButton"] button *,
    div[data-testid="stButton"] button p,
    div[data-testid="stButton"] button span,
    div[data-testid="stButton"] button div,
    div[data-testid="stButton"] button [data-testid="stMarkdownContainer"],
    div[data-testid="stButton"] button [data-testid="stMarkdownContainer"] p {
        color: #fffdf5 !important;
        font-family: 'EB Garamond', 'Noto Serif Devanagari', serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8) !important;
        letter-spacing: normal !important;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(180deg, #443222 0%, #291d13 100%) !important;
        border-color: #dfbe7b !important;
        border-left-color: #ffd782 !important;
        box-shadow: 0 6px 18px rgba(197, 159, 91, 0.45) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stButton"] button:hover *,
    div[data-testid="stButton"] button:hover p,
    div[data-testid="stButton"] button:hover span {
        color: #ffffff !important;
        text-shadow: 0 0 8px rgba(255, 215, 130, 0.6) !important;
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.7) !important;
    }

    /* Vintage Manuscript Query Input */
    div[data-testid="stTextInput"] {
        margin: 0 0 20px 0 !important;
    }
    div[data-testid="stTextInput"] > label {
        display: none !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        min-height: 64px !important;
        height: 64px !important;
        border-radius: 0 0 10px 10px !important;
        background: #1c150f !important;
        border: 2px solid #8c6f43 !important;
        border-top: none !important;
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

    /* 1. Luminous Aged Parchment Manuscript Answer Card (English Commentary) */
    .parchment-english-card {
        background: linear-gradient(180deg, #fdf8eb 0%, #f4e7cd 100%) !important;
        border: 2px solid #a3814e !important;
        border-left: 8px solid #c59f5b !important;
        border-radius: 10px !important;
        padding: 24px 28px !important;
        margin-top: 18px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.7), inset 0 0 40px rgba(184, 148, 87, 0.15) !important;
        position: relative !important;
        display: block !important;
    }
    .parchment-english-card .card-badge {
        display: inline-block !important;
        background: #362415 !important;
        color: #dfbe7b !important;
        font-family: 'Cinzel', serif !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        padding: 4px 14px !important;
        border-radius: 4px !important;
        letter-spacing: 1px !important;
        margin-bottom: 12px !important;
    }
    .parchment-english-card .answer-heading {
        font-family: 'Cinzel', 'EB Garamond', serif !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        color: #2b1c0e !important;
        margin-bottom: 14px !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        border-bottom: 2px solid rgba(140, 111, 67, 0.4) !important;
        padding-bottom: 8px !important;
    }
    .parchment-english-card .explanation-box {
        background: rgba(255, 255, 255, 0.6) !important;
        border: 1px solid #d4c2a1 !important;
        border-left: 5px solid #8c6f43 !important;
        padding: 20px 24px !important;
        border-radius: 6px !important;
        color: #1a1106 !important;
        font-family: 'EB Garamond', Georgia, serif !important;
        font-size: 1.25rem !important;
        line-height: 1.85 !important;
        box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.05) !important;
    }

    /* 2. Deep Sanctum Vedic Manuscript Card (Sanskrit Excerpt) */
    .vedic-sanskrit-card {
        background: linear-gradient(180deg, #221810 0%, #150f09 100%) !important;
        border: 2px solid #8c6f43 !important;
        border-left: 8px solid #dfbe7b !important;
        border-radius: 10px !important;
        padding: 24px 28px !important;
        margin-top: 18px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.7), inset 0 0 25px rgba(223, 190, 123, 0.08) !important;
        position: relative !important;
        display: block !important;
    }
    .vedic-sanskrit-card .card-badge {
        display: inline-block !important;
        background: #3d2417 !important;
        color: #fca86c !important;
        font-family: 'Cinzel', serif !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        padding: 4px 14px !important;
        border-radius: 4px !important;
        letter-spacing: 1px !important;
        margin-bottom: 12px !important;
    }
    .vedic-sanskrit-card .answer-heading {
        font-family: 'Noto Serif Devanagari', 'Cinzel', serif !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #dfbe7b !important;
        margin-bottom: 14px !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        border-bottom: 1.5px solid rgba(223, 190, 123, 0.3) !important;
        padding-bottom: 8px !important;
        letter-spacing: normal !important;
    }
    .vedic-sanskrit-card .sanskrit-text {
        font-family: 'Noto Serif Devanagari', serif !important;
        font-size: 1.4rem !important;
        line-height: 2.2 !important;
        color: #fffef7 !important;
        background: rgba(12, 8, 5, 0.95) !important;
        border: 1.5px solid #6d5432 !important;
        padding: 22px 26px !important;
        border-radius: 8px !important;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.7) !important;
        letter-spacing: normal !important;
    }

    /* 3. Archival Seal Reference Card */
    .reference-citation-card {
        background: linear-gradient(180deg, #151e17 0%, #0d150e 100%) !important;
        border: 2px solid #3d694f !important;
        border-left: 8px solid #4d8263 !important;
        border-radius: 10px !important;
        padding: 22px 26px !important;
        margin-top: 18px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6) !important;
        display: block !important;
    }
    .reference-citation-card .card-badge {
        display: inline-block !important;
        background: #1c3123 !important;
        color: #8ed6aa !important;
        font-family: 'Cinzel', serif !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        padding: 3px 12px !important;
        border-radius: 4px !important;
        letter-spacing: 1px !important;
        margin-bottom: 10px !important;
    }
    .reference-citation-card .answer-heading {
        font-family: 'Cinzel', 'Noto Serif Devanagari', serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #7bc498 !important;
        margin-bottom: 12px !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        border-bottom: 1px solid rgba(77, 130, 99, 0.4) !important;
        padding-bottom: 6px !important;
    }
    .reference-citation-card .citation-box {
        background: rgba(7, 13, 8, 0.85) !important;
        border: 1px solid rgba(77, 130, 99, 0.4) !important;
        padding: 16px 22px !important;
        border-radius: 6px !important;
        color: #e2f5e8 !important;
        font-size: 1.12rem !important;
        font-family: 'Noto Serif Devanagari', 'EB Garamond', serif !important;
        line-height: 1.85 !important;
    }

    /* Vintage Tabs resting on Parchment */
    div[data-testid="stTabs"] div[role="tablist"] {
        border-bottom: 2px solid #8c6f43 !important;
        gap: 6px !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        font-family: 'Cinzel', 'EB Garamond', serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #4a331e !important;
        background: rgba(45, 32, 20, 0.12) !important;
        border: 1.5px solid rgba(140, 111, 67, 0.45) !important;
        border-bottom: none !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #ffd782 !important;
        background: #2b1d12 !important;
        border: 1.5px solid #8c6f43 !important;
        border-bottom: 3px solid #dfbe7b !important;
        box-shadow: 0 -4px 12px rgba(45, 30, 15, 0.3) !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #1a0f06 !important;
        background: rgba(45, 32, 20, 0.22) !important;
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
        background: linear-gradient(180deg, #281d13 0%, #1a120b 100%) !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 14px rgba(45, 30, 15, 0.3) !important;
    }
    div[data-testid="stExpander"] summary {
        background: transparent !important;
    }
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span {
        font-family: 'EB Garamond', 'Noto Serif Devanagari', serif !important;
        color: #ffd782 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stExpander"] summary svg {
        fill: #ffd782 !important;
        width: 18px !important;
        height: 18px !important;
        margin-right: 8px !important;
        flex-shrink: 0 !important;
    }
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background: #1b130b !important;
        border-top: 1px solid #5a4225 !important;
        padding: 20px 24px !important;
    }
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] p,
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] span,
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] label {
        color: #f7eed8 !important;
    }
    div[data-testid="stExpander"] div[data-testid="stCaptionContainer"] p,
    div[data-testid="stExpander"] .stCaption {
        color: #dfbe7b !important;
        font-family: 'EB Garamond', serif !important;
        font-size: 1.02rem !important;
        font-weight: 500 !important;
    }

    /* Vintage Document Ingestion & File Uploader */
    div[data-testid="stFileUploader"] {
        background: #16100a !important;
        border: 1.5px solid #6d5432 !important;
        border-radius: 8px !important;
        padding: 18px 20px !important;
        margin: 14px 0 18px 0 !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5) !important;
    }
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] label p {
        color: #ffd782 !important;
        font-family: 'Cinzel', serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin-bottom: 10px !important;
    }
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] {
        background: #231911 !important;
        border: 1.5px dashed #8c6f43 !important;
        border-radius: 6px !important;
        padding: 24px 20px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stFileUploaderDropzone"]:hover,
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #dfbe7b !important;
        background: #2c2016 !important;
    }
    section[data-testid="stFileUploaderDropzone"] div,
    section[data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stFileUploaderDropzone"] p {
        color: #faf2de !important;
        font-family: 'EB Garamond', serif !important;
        font-size: 1.08rem !important;
    }
    section[data-testid="stFileUploaderDropzone"] small {
        color: #dfbe7b !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        display: block !important;
        margin-top: 4px !important;
    }
    section[data-testid="stFileUploaderDropzone"] svg {
        fill: #dfbe7b !important;
        color: #dfbe7b !important;
        width: 34px !important;
        height: 34px !important;
        margin-bottom: 4px !important;
    }
    section[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(180deg, #3d2c1c 0%, #221810 100%) !important;
        border: 1.5px solid #8c6f43 !important;
        border-radius: 6px !important;
        color: #fffdf5 !important;
        font-family: 'Cinzel', serif !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        padding: 8px 20px !important;
        min-height: auto !important;
        height: auto !important;
        cursor: pointer !important;
        margin-top: 8px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
    }
    section[data-testid="stFileUploaderDropzone"] button:hover {
        border-color: #dfbe7b !important;
        color: #ffffff !important;
        background: linear-gradient(180deg, #523b26 0%, #322216 100%) !important;
    }

    /* Uploaded File Data Row */
    div[data-testid="stFileUploaderFileData"],
    div[data-testid="stFileUploaderFileData"] > div {
        background: #1f160e !important;
        border: 1px solid #8c6f43 !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
        margin-top: 12px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 14px !important;
    }
    div[data-testid="stFileUploaderFileData"] span,
    div[data-testid="stFileUploaderFileData"] p {
        color: #fffdf5 !important;
        font-family: 'EB Garamond', serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploaderFileData"] small {
        color: #dfbe7b !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stFileUploaderFileData"] button {
        background: transparent !important;
        border: none !important;
        color: #e07a38 !important;
        cursor: pointer !important;
    }
    div[data-testid="stFileUploaderFileData"] button:hover {
        color: #f87171 !important;
    }

    /* Vintage Download Button */
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(180deg, #2a1f15 0%, #17110c 100%) !important;
        border: 1.5px solid #8c6f43 !important;
        border-left: 5px solid #dfbe7b !important;
        border-radius: 6px !important;
        color: #fffdf5 !important;
        font-family: 'EB Garamond', 'Cinzel', serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] button *,
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stDownloadButton"] button span,
    div[data-testid="stDownloadButton"] button div,
    div[data-testid="stDownloadButton"] button [data-testid="stMarkdownContainer"],
    div[data-testid="stDownloadButton"] button [data-testid="stMarkdownContainer"] p {
        color: #fffdf5 !important;
        font-family: 'EB Garamond', 'Cinzel', serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8) !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(180deg, #423120 0%, #2a1c11 100%) !important;
        border-color: #dfbe7b !important;
        box-shadow: 0 0 16px rgba(197, 159, 91, 0.4) !important;
    }
    div[data-testid="stDownloadButton"] button:hover *,
    div[data-testid="stDownloadButton"] button:hover p,
    div[data-testid="stDownloadButton"] button:hover span {
        color: #ffffff !important;
    }

    /* Antique Alert Cards (Success & Info Boxes) */
    div[data-testid="stAlert"],
    div.stAlert {
        background: linear-gradient(135deg, #271c13 0%, #1a120b 100%) !important;
        border: 1.5px solid #8c6f43 !important;
        border-left: 6px solid #dfbe7b !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(45, 30, 15, 0.25) !important;
        color: #fffdf5 !important;
    }
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] span,
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] strong,
    div.stAlert [data-testid="stMarkdownContainer"] p,
    div.stAlert [data-testid="stMarkdownContainer"] span,
    div.stAlert [data-testid="stMarkdownContainer"] strong {
        color: #fffdf5 !important;
        font-size: 1rem !important;
        font-family: 'EB Garamond', serif !important;
    }
    div[data-testid="stAlert"] svg,
    div.stAlert svg {
        fill: #dfbe7b !important;
        color: #dfbe7b !important;
        flex-shrink: 0 !important;
    }

    /* Caption Styling for Archival Clarity on Parchment */
    div[data-testid="stCaptionContainer"] *,
    div[data-testid="stCaptionContainer"] p,
    .stCaption,
    small {
        color: #3b2311 !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
    }

    /* Primary Search Console Gateway Banner */
    .search-gateway-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, rgba(62, 45, 29, 0.95) 0%, rgba(32, 23, 15, 0.95) 100%);
        border: 2px solid #8c6f43;
        border-left: 8px solid #dfbe7b;
        border-radius: 10px 10px 0 0;
        padding: 12px 20px;
        margin-top: 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5);
    }
    .gateway-title {
        font-family: 'Cinzel', serif;
        font-size: 1.15rem;
        font-weight: 800;
        color: #dfbe7b;
        letter-spacing: 1px;
    }
    .gateway-hint {
        font-family: 'EB Garamond', serif;
        font-size: 0.98rem;
        color: #d9c8af;
        font-style: italic;
    }

    div[data-testid="stButton"] {
        margin-bottom: 10px !important;
    }

    .directory-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 24px;
        margin-bottom: 14px;
        border-bottom: 1.5px solid rgba(140, 111, 67, 0.35);
        padding-bottom: 8px;
    }
    .directory-title {
        font-family: 'Cinzel', serif;
        font-size: 1.18rem;
        font-weight: 800;
        color: #dfbe7b;
        letter-spacing: 1px;
    }
    .directory-badge {
        font-family: 'EB Garamond', serif;
        font-size: 0.95rem;
        color: #dfbe7b;
        background: rgba(54, 40, 27, 0.9);
        border: 1px solid #8c6f43;
        padding: 3px 12px;
        border-radius: 4px;
        font-style: italic;
    }
    .shelf-label-en {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Cinzel', serif;
        font-size: 0.96rem;
        font-weight: 700;
        color: #ffd782;
        background: rgba(54, 40, 27, 0.95);
        border: 1px solid #8c6f43;
        border-left: 5px solid #dfbe7b;
        padding: 5px 14px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 14px;
    }
    .shelf-label-sa {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Noto Serif Devanagari', serif;
        font-size: 1.02rem;
        font-weight: 700;
        color: #ffb17a;
        background: rgba(50, 26, 18, 0.95);
        border: 1px solid #b56839;
        border-left: 5px solid #e07a38;
        padding: 5px 14px;
        border-radius: 4px;
        margin-top: 24px;
        margin-bottom: 14px;
        letter-spacing: normal !important;
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

# Sacred Indian Kolam / Yantra Background Watermark (Low-intensity brown motif like Bharatiya GPT)
kolam_file = os.path.join(curr_dir, "..", "assets", "sacred_kolam_watermark.svg")
kolam_b64 = ""
if os.path.exists(kolam_file):
    with open(kolam_file, "rb") as kf:
        kolam_b64 = base64.b64encode(kf.read()).decode("utf-8")

if kolam_b64:
    st.markdown(f"""
    <style>
        .stApp::before {{
            content: "" !important;
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -46%) !important;
            width: 520px !important;
            height: 520px !important;
            max-width: 85vw !important;
            max-height: 85vh !important;
            background-image: url("data:image/svg+xml;base64,{kolam_b64}") !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            opacity: 0.05 !important; /* Soft, subtle low-intensity watermark */
            pointer-events: none !important;
            z-index: 0 !important;
        }}
        .main, .block-container, div[data-testid="stVerticalBlock"] {{
            position: relative !important;
            z-index: 1 !important;
        }}
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
    if any(w in t for w in ["bhoj", "kalidas", "king", "raja", "poet", "reward", "prize", "court", "lakh", "rupee", "poem", "verse"]):
        if any(w in t for w in ["why", "fail", "prevent", "unable", "not able", "originally able", "win", "who"]):
            return "कविभ्यः पारितोषिकप्राप्तौ विघ्नः (Why Poets Failed To Win Prize)"
        elif any(w in t for w in ["amount", "prize", "reward", "money", "lakh", "announce", "announced", "annouced", "how much"]):
            return "भोजराज्ञा घोषितं काव्यपारितोषिकम् (King Bhoja's Announced Reward)"
        elif any(w in t for w in ["99", "crore", "riddle", "gem", "gems"]):
            return "कालीदासस्य ९९-कोटिरत्नकूटश्लोकः (Kalidasa's 99-Crore Gems Riddle)"
        elif any(w in t for w in ["scholar", "scholars", "memory", "ekapathi", "dvipathi", "tripathi"]):
            return "एकपाठि-द्विपाठि-त्रिपाठि विद्वांसः (Scholars' Photographic Memory)"
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
        return "संस्कृत-अर्थानुसन्धानम् (Cross-Lingual Semantic Concept Retrieval)"

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
    <div class="hero-crest">◈ प्राच्य-संस्कृत-ग्रन्थागारः ◈</div>
    <div class="hero-title">SANSKRIT RETRIEVAL-AUGMENTED GENERATION</div>
    <div class="hero-subtitle">
        A classical epistemic retrieval engine for ancient Sanskrit literature, Vedic philosophy, and historical manuscripts.
    </div>
    <div class="badge-container">
        <span class="spec-badge">✦ 100% CPU Only (No GPU)</span>
        <span class="spec-badge">◆ English & Sanskrit Natural Queries</span>
        <span class="spec-badge">◈ Dual Script: Devanagari + IAST / HK / ITRANS</span>
        <span class="spec-badge">❖ Hybrid Vector & BM25 Fusion</span>
        <span class="spec-badge">◈ Verse & Danda (।) Aware Semantic Chunking</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 1. PRIMARY SEARCH GATEWAY (FRONT & CENTER)
st.markdown("""
<div class="search-gateway-banner">
    <span class="gateway-title">✦ PRIMARY INQUIRY CONSOLE | ग्रन्थ-सन्धानम्</span>
    <span class="gateway-hint">Type below in English, Devanagari Sanskrit, or Romanized IAST / HK / ITRANS</span>
</div>
""", unsafe_allow_html=True)

query_val = st.text_input(
    "Historical Manuscript Query | Enter Your Question:",
    value=st.session_state.query_text,
    placeholder="Inquire in English (e.g. 'Why was no poet able to win the reward?') or Sanskrit and press Enter...",
    key="main_query_input",
    label_visibility="collapsed"
)

# 2. STRUCTURED VINTAGE QUERY DIRECTORY
st.markdown("""
<div class="directory-banner">
    <span class="directory-title">◈ HISTORICAL QUERY DIRECTORY | शीघ्र-ग्रन्थ-प्रश्नावली</span>
    <span class="directory-badge">Click any tablet below to load authentic query</span>
</div>
<div class="shelf-label-en">◆ Classical English Inquiries (आङ्ग्लप्रश्नाः)</div>
""", unsafe_allow_html=True)

ec1, ec2, ec3, ec4 = st.columns(4)
with ec1:
    if st.button("◆ Why did servant ruin sugar?", use_container_width=True):
        set_query("Why did the foolish servant ruin the sugar?")
with ec2:
    if st.button("◆ What did King Bhoja announce?", use_container_width=True):
        set_query("What did King Bhoja announce in his court?")
with ec3:
    if st.button("◆ Who was Ghantakarna demon?", use_container_width=True):
        set_query("Who was Ghantakarna and why was the bell ringing?")
with ec4:
    if st.button("◆ Why was 'badhati' incorrect?", use_container_width=True):
        set_query("Why was badhati incorrect in sheetam bahu badhati?")

st.markdown("""
<div class="shelf-label-sa">◈ Native Sanskrit Inscriptions (मूलसंस्कृतप्रश्नाः)</div>
""", unsafe_allow_html=True)

sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    if st.button("◈ मूर्खभृत्यः शर्कराम् कुत्र न्यस्यति ?", use_container_width=True):
        set_query("मूर्खभृत्यः शर्कराम् कुत्र न्यस्यति ?")
with sc2:
    if st.button("◈ भोजराजः काव्यपठने किं घोषितवान् ?", use_container_width=True):
        set_query("भोजराजः काव्यपठने किं घोषितवान् ?")
with sc3:
    if st.button("◈ चित्रपुरे घण्टाकर्णः नाम कः आसीत् ?", use_container_width=True):
        set_query("चित्रपुरे घण्टाकर्णः नाम कः आसीत् ?")
with sc4:
    if st.button("◈ देवभक्तः किमर्थं जले मृतवान् ?", use_container_width=True):
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
        "◈ Corpus Explorer"
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

        # Format and display cards:
        formatted_english = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', english_part.strip()).replace("\n", "<br>")
        formatted_sanskrit = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', sanskrit_part.strip()).replace("\n", "<br>")
        formatted_ref = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', reference_part.strip()).replace("\n", "<br>")

        # 1. English Commentary & Historical Meaning (UP SIDE - Radiant Aged Parchment)
        if formatted_english:
            st.markdown(f"""
            <div class="parchment-english-card">
                <div class="card-badge">MANUSCRIPT EXEGESIS & HISTORICAL COMMENTARY</div>
                <div class="answer-heading">
                    <span>◆</span> English Commentary & Historical Meaning (विस्तृत-आङ्ग्लार्थः)
                </div>
                <div class="explanation-box">
                    {formatted_english}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Classical Sanskrit Excerpt (DOWN OF THE ENGLISH - Sacred Vedic Tablet)
        if formatted_sanskrit:
            st.markdown(f"""
            <div class="vedic-sanskrit-card">
                <div class="card-badge">AUTHENTIC SANSKRIT CORPUS PASSAGE</div>
                <div class="answer-heading">
                    <span>◈</span> उत्तरम् (Classical Sanskrit Excerpt)
                </div>
                <div class="sanskrit-text">
                    {formatted_sanskrit}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Direct Manuscript Reference (DOWN OF SANSKRIT - Archival Seal)
        if formatted_ref:
            st.markdown(f"""
            <div class="reference-citation-card">
                <div class="card-badge">VERBATIM MANUSCRIPT CITATION</div>
                <div class="answer-heading">
                    <span>◈</span> प्रमाणम् / Direct Manuscript Reference (मूलग्रन्थसन्दर्भः)
                </div>
                <div class="citation-box">
                    <b>मूलग्रन्थसन्दर्भः:</b> {formatted_ref}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_chunks:
        chunks = result.get("retrieved_chunks", [])
        is_matched = result.get("is_matched", True)
        if not chunks or not is_matched:
            st.markdown("""
            <div style="background: rgba(38, 16, 14, 0.85); border: 2px dashed #b94a48; border-radius: 8px; padding: 24px; text-align: center; color: #f7eed8; margin-top: 10px;">
                <span style="font-size: 2rem; color: #c7b399;">◈</span><br>
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
                
                with st.expander(f"◈ Context Chunk #{idx} — Section: 『{sec}』 (Relevance Score: {score:.4f})", expanded=(idx == 1)):
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
            st.markdown("#### ◈ Ingested Sanskrit Corpus Preview")
            st.text_area("Full Corpus Text", value=full_corpus, height=350, disabled=True)
        else:
            st.info("Corpus text file not found.")

else:
    # Landing Placeholder when no query is typed
    st.markdown("""
    <div style="text-align: center; padding: 34px 24px; background: linear-gradient(135deg, #261b12 0%, #1a120c 100%); border-radius: 10px; border: 2px solid #8c6f43; outline: 1px solid rgba(223, 190, 123, 0.35); outline-offset: -5px; margin-top: 18px; box-shadow: 0 10px 28px rgba(45, 28, 14, 0.4);">
        <div style="font-size: 2.4rem; margin-bottom: 8px; color: #ffd782; text-shadow: 0 0 12px rgba(223, 190, 123, 0.5);">❖</div>
        <div style="font-family: 'Cinzel', serif; font-size: 1.35rem; font-weight: 800; color: #ffd782; letter-spacing: 1px; margin-bottom: 8px;">HISTORICAL MANUSCRIPT ARCHIVE READY</div>
        <div style="font-family: 'EB Garamond', serif; color: #fbf4e6; font-size: 1.15rem; max-width: 720px; margin: 0 auto; line-height: 1.7;">
            Inquire above in <b style="color: #ffd782;">Natural English</b>, <b style="color: #ffd782;">Devanagari Sanskrit</b>, or <b style="color: #ffd782;">Romanized Transliterations (IAST / HK / ITRANS)</b>, or select any sample inquiry above to retrieve authentic Sanskrit excerpts with full English commentary.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# BOTTOM SECTION: SYSTEM ARCHITECTURE & DOCUMENTATION
# =====================================================================
st.markdown("<div class='vintage-divider'><span>❖ ═════════════════════════════ ❖</span></div>", unsafe_allow_html=True)
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
    st.markdown("##### ◈ Technical Documentation")
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

st.markdown("<div style='text-align: center; color: #5a381b; font-family: EB Garamond, serif; font-size: 1.1rem; font-weight: 700; padding: 28px 0 14px 0;'>◈ Sanskrit RAG System | Historical Manuscript Archive Edition | Developed by Sahil Karande ◈</div>", unsafe_allow_html=True)
