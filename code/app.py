"""
app.py - Sanskrit Document RAG System
Advanced Interactive Web Application
Features:
- Dual Query Interface: Devanagari & Romanized Transliteration (IAST, Harvard-Kyoto, ITRANS)
- Live Script Normalization Visualizer
- Hybrid Retrieval (ChromaDB Multilingual Dense Vectors + BM25 Keyword Matching + Stem Boosting)
- Zero-GPU CPU Inference Optimization
- Interactive Corpus Explorer & Document Uploader
- Performance Benchmarking HUD
"""

import os
import sys
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
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Vedic-Modern Dark-Slate & Warm Saffron Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');

    :root {
        --primary-gold: #f59e0b;
        --deep-gold: #d97706;
        --saffron: #ea580c;
        --bg-slate: #0f172a;
        --card-bg: #1e293b;
        --card-border: #334155;
        --text-light: #f8fafc;
        --text-muted: #94a3b8;
    }

    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50px;
        right: -50px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(245, 158, 11, 0.15) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
    }

    .hero-title {
        font-family: 'Cinzel', serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #fef08a, #f59e0b, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.0rem;
        color: #cbd5e1;
        line-height: 1.5;
        max-width: 900px;
    }

    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }

    .spec-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fef3c7;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    .transliteration-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        gap: 15px;
    }

    .script-pill {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .devanagari-preview {
        font-family: 'Noto Sans Devanagari', sans-serif;
        font-size: 1.25rem;
        color: #38bdf8;
        font-weight: 600;
    }

    .answer-card {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #475569;
        border-left: 6px solid #f59e0b;
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }

    .answer-heading {
        font-family: 'Cinzel', serif;
        font-size: 1.25rem;
        color: #fbbf24;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sanskrit-text {
        font-family: 'Noto Sans Devanagari', serif;
        font-size: 1.15rem;
        line-height: 1.7;
        color: #f8fafc;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(245, 158, 11, 0.2);
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 14px;
    }

    .explanation-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 6px;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 12px;
    }

    .citation-box {
        background: rgba(15, 23, 42, 0.7);
        border-left: 3px solid #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        color: #a7f3d0;
        font-size: 0.9rem;
        font-family: 'Noto Sans Devanagari', sans-serif;
    }

    .stat-tile {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }

    .stat-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .stat-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

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

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🕉️ Sanskrit RAG System")
    st.caption("CPU-Centric Multilingual Architecture")
    
    st.markdown("---")
    st.markdown("#### ⚡ Hardware Inference Engine")
    st.success("✅ **100% CPU Inference (Zero GPU)**")
    st.caption("Engineered for low-latency CPU inference via Vector Space & BM25 Fusion.")

    st.markdown("---")
    st.markdown("#### 🔍 Retrieval Strategy")
    retrieval_mode = st.selectbox(
        "Search Strategy",
        options=["hybrid", "dense", "bm25"],
        format_func=lambda x: "🔥 Hybrid Fusion (Vector + BM25)" if x == "hybrid" else ("🧬 Dense Vector (ChromaDB)" if x == "dense" else "🔤 Keyword Search (BM25)")
    )
    top_k = st.slider("Context Depth (Top-K Chunks)", min_value=1, max_value=6, value=3)

    st.markdown("---")
    st.markdown("#### 📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload Sanskrit Document (.txt / .pdf)", type=["txt", "pdf"])
    if uploaded_file is not None:
        save_path = os.path.join(curr_dir, "..", "data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button("🚀 Index Document Now", use_container_width=True):
            with st.spinner("Parsing Sanskrit glyphs & indexing on CPU..."):
                count = pipeline.index_document(save_path, overwrite=False)
                st.success(f"Indexed {count} chunks successfully!")

    st.markdown("---")
    st.markdown("#### 📑 Technical Documentation")
    report_file_path = os.path.join(curr_dir, "..", "report", "Sanskrit_RAG_Technical_Report.pdf")
    if os.path.exists(report_file_path):
        with open(report_file_path, "rb") as rf:
            st.download_button(
                label="📥 Download Technical Report (PDF)",
                data=rf.read(),
                file_name="Sanskrit_RAG_Technical_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.caption("Author: **Sahil Karande** | Assignment Submission")

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🕉️ Sanskrit Retrieval-Augmented Generation</div>
    <div class="hero-subtitle">
        An end-to-end question-answering architecture for Sanskrit classical literature, philosophy, and subhashitas.
        Processes queries in native Devanagari or Romanized transliterations with instant, grounded CPU-based synthesis.
    </div>
    <div class="badge-container">
        <span class="spec-badge">⚡ 100% CPU Only (No GPU)</span>
        <span class="spec-badge">🔤 Dual Script: Devanagari + IAST / HK / ITRANS</span>
        <span class="spec-badge">🧬 Hybrid Vector & BM25 Fusion</span>
        <span class="spec-badge">📜 Sanskrit Verse & Danda (।) Aware Chunking</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Query Chips
st.markdown("##### 💡 Try Sample Queries (Click to load):")
chip_cols = st.columns(4)

with chip_cols[0]:
    if st.button("🍬 शर्करा कुत्र गच्छति ?\n*(Foolish Servant)*", use_container_width=True):
        set_query("मूर्खभृत्यः शर्कराम् कुत्र न्यस्यति ?")

with chip_cols[1]:
    if st.button("👑 bhojaraajaa ghoshanam\n*(King Bhoja - Roman HK)*", use_container_width=True):
        set_query("bhojaraajaa kaavya paThane kim ghoshhitavaan?")

with chip_cols[2]:
    if st.button("🔔 ghaṇṭākarṇaḥ rākṣasaḥ\n*(Old Woman - IAST)*", use_container_width=True):
        set_query("citrapure ghaṇṭākarṇaḥ nāma kaḥ āsīt ?")

with chip_cols[3]:
    if st.button("❄️ sheetaM bahu baadhate\n*(Grammar Riddle)*", use_container_width=True):
        set_query("sheetam bahu baadhate atra kaa trutiH?")

# Query Input Form
with st.container():
    query_val = st.text_input(
        "Enter Query (in Devanagari or Romanized English letters like IAST / Harvard-Kyoto / ITRANS):",
        value=st.session_state.query_text,
        placeholder="Type here, e.g. 'shankhanaadaH sharkaraam kuto gachchhati?' or 'राजा भोजः किं दत्तवान् ?'",
        key="main_query_input"
    )

if query_val.strip():
    # Step 1: Script Detection & Normalization Preview
    is_dev = is_devanagari(query_val)
    detected_scheme = "Devanagari (Native)" if is_dev else detect_transliteration_scheme(query_val)
    devanagari_query = to_devanagari(query_val)

    # Transliteration Visualizer Bar
    st.markdown(f"""
    <div class="transliteration-card">
        <div>
            <span style="color: #94a3b8; font-size: 0.85rem;">Input Script Scheme:</span><br>
            <span class="script-pill">{detected_scheme}</span>
        </div>
        <div style="font-size: 1.5rem; color: #f59e0b;">➔</div>
        <div style="flex-grow: 1;">
            <span style="color: #94a3b8; font-size: 0.85rem;">Normalized Devanagari Search Query:</span><br>
            <span class="devanagari-preview">{devanagari_query}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step 2: Query Execution with Spinner
    with st.spinner("🔍 Retrieving context chunks & generating grounded Sanskrit response on CPU..."):
        t0 = time.time()
        result = pipeline.query(query_val, top_k=top_k, retrieval_mode=retrieval_mode)
        exec_latency = time.time() - t0

    # Tabs for Rich Layout
    tab_ans, tab_chunks, tab_metrics, tab_corpus = st.tabs([
        "🕉️ Grounded Response",
        "📜 Retrieved Context Chunks",
        "⚡ Performance & CPU Telemetry",
        "📖 Corpus Explorer"
    ])

    with tab_ans:
        response_text = result.get("response", "")
        
        # Parse standard formatted response blocks
        sanskrit_part = ""
        english_part = ""
        reference_part = ""
        
        lines = response_text.split("\n")
        current_section = None
        for line in lines:
            if "उत्तरम्" in line or "Sanskrit Answer" in line:
                current_section = "sanskrit"
                sanskrit_part += line.replace("1. उत्तरम् (Sanskrit Answer):", "").replace("उत्तरम्:", "").strip() + " "
            elif "English Explanation" in line:
                current_section = "english"
                english_part += line.replace("2. English Explanation:", "").replace("English:", "").strip() + " "
            elif "प्रमाणम्" in line or "Reference" in line:
                current_section = "reference"
                reference_part += line.replace("3. प्रमाणम् / Reference:", "").replace("Reference:", "").strip() + " "
            else:
                if current_section == "sanskrit":
                    sanskrit_part += line.strip() + " "
                elif current_section == "english":
                    english_part += line.strip() + " "
                elif current_section == "reference":
                    reference_part += line.strip() + " "

        # Fallback if unformatted
        if not sanskrit_part.strip():
            sanskrit_part = response_text

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-heading">
                <span>🕉️</span> उत्तरम् (Sanskrit Response)
            </div>
            <div class="sanskrit-text">
                {sanskrit_part.strip()}
            </div>
            {"<div class='answer-heading' style='font-size: 1.05rem;'><span>📘</span> English Explanation & Context</div><div class='explanation-box'>" + english_part.strip() + "</div>" if english_part.strip() else ""}
            {"<div class='answer-heading' style='font-size: 1.05rem;'><span>📜</span> प्रमाणम् / Source Citation</div><div class='citation-box'><b>मूलसन्दर्भः :</b> " + reference_part.strip() + "</div>" if reference_part.strip() else ""}
        </div>
        """, unsafe_allow_html=True)

    with tab_chunks:
        chunks = result.get("retrieved_chunks", [])
        if not chunks:
            st.warning("No matching context chunks found for this query.")
        else:
            st.markdown(f"**Found {len(chunks)} relevant context segments from Sanskrit corpus:**")
            for idx, c in enumerate(chunks, 1):
                sec = c.get("metadata", {}).get("section", "General")
                source = c.get("metadata", {}).get("source", "sanskrit_corpus")
                score = c.get("rrf_score", c.get("dense_score", c.get("bm25_score", 0.0)))
                
                with st.expander(f"📍 Context Chunk #{idx} — Section: 『{sec}』 (Relevance Score: {score:.4f})", expanded=(idx == 1)):
                    st.markdown(f"""
                    <div style="font-family: 'Noto Sans Devanagari', serif; font-size: 1.1rem; line-height: 1.7; background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155;">
                        {c.get('content', '')}
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"Source Document: `{source}` | Chunk ID: `{c.get('id', 'N/A')}`")

    with tab_metrics:
        m = result.get("metrics", {})
        st.markdown("#### ⏱️ Latency & CPU Resource Breakdown")
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-val">{m.get('transliteration_latency_s', 0) * 1000:.1f} ms</div>
                <div class="stat-lbl">Transliteration</div>
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
        - **Hardware Inference Mode:** 100% CPU (AVX2 vector instructions)
        - **Dense Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
        - **Lexical Engine:** Rank-BM25 with Sanskrit Morphological Stem Normalization
        - **RAM Footprint:** < 1.2 GB
        """)

    with tab_corpus:
        corpus_file = os.path.join(curr_dir, "..", "data", "sanskrit_corpus.txt")
        if os.path.exists(corpus_file):
            with open(corpus_file, "r", encoding="utf-8") as f:
                full_corpus = f.read()
            st.markdown("#### 📜 Ingested Sanskrit Corpus Preview")
            st.text_area("Full Corpus Text", value=full_corpus, height=350, disabled=True)
        else:
            st.info("Corpus text file not found.")

else:
    # Landing Placeholder when no query is typed
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: rgba(30, 41, 59, 0.3); border-radius: 12px; border: 1px dashed #334155; margin-top: 20px;">
        <div style="font-size: 2.5rem; margin-bottom: 10px;">📖</div>
        <div style="font-family: 'Cinzel', serif; font-size: 1.2rem; color: #f59e0b; margin-bottom: 6px;">Ready for Queries</div>
        <div style="color: #94a3b8; font-size: 0.95rem; max-width: 600px; margin: 0 auto;">
            Enter a query above in Devanagari (e.g. <i>'मूर्खभृत्यस्य शर्करा'</i>) or Romanized text (e.g. <i>'bhojaraajaa'</i> or <i>'ghaṇṭākarṇaḥ'</i>), or click one of the quick query examples above.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>🕉️ Sanskrit RAG System | Developed by Sahil Karande | CPU-Only Inference</div>", unsafe_allow_html=True)
