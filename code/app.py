"""
app.py - Streamlit Interactive Web Application for Sanskrit RAG
Features:
- Devanagari & Transliterated Roman Query Support (IAST, ITRANS, HK)
- Real-time Transliteration Preview
- Hybrid Search (ChromaDB Vector + BM25 Keyword Search)
- CPU-only local inference
- Context chunk inspector with similarity metrics
"""

import os
import sys
import time
import streamlit as st

# Add code directory to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

from transliterate import to_devanagari, is_devanagari, detect_transliteration_scheme
from ingestion import load_document, chunk_sanskrit_text
from retriever import SanskritRetriever
from generator import SanskritGenerator
from pipeline import SanskritRAGPipeline

st.set_page_config(
    page_title="Sanskrit RAG System | CPU Inference",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #d97706;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    """Initializes and caches the Sanskrit RAG Pipeline."""
    persist_dir = os.path.join(curr_dir, "..", "chroma_db")
    pipeline = SanskritRAGPipeline(
        collection_name="sanskrit_knowledge",
        persist_directory=persist_dir
    )
    
    # Auto-index default document if present
    default_doc = os.path.join(curr_dir, "..", "data", "sanskrit_corpus.txt")
    if os.path.exists(default_doc):
        try:
            pipeline.index_document(default_doc, overwrite=False)
        except Exception as e:
            print(f"Index notice: {e}")
            
    return pipeline

pipeline = get_pipeline()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Om_symbol.svg/200px-Om_symbol.svg.png", width=70)
    st.title("⚙️ RAG Configuration")
    
    st.markdown("### 🖥️ Hardware Inference")
    st.success("✅ **CPU-Only Mode** (Zero GPU)")
    st.caption("Powered by `sentence-transformers` & `llama-cpp` / CPU Synthesizer")

    st.markdown("---")
    st.markdown("### 🔍 Retrieval Settings")
    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        options=["hybrid", "dense", "bm25"],
        format_func=lambda x: "Hybrid (Vector + BM25 RRF)" if x == "hybrid" else ("Dense Vector (ChromaDB)" if x == "dense" else "Keyword Search (BM25)")
    )
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=6, value=3)

    st.markdown("---")
    st.markdown("### 📂 Ingest New Document")
    uploaded_file = st.file_uploader("Upload Sanskrit Document", type=["txt", "pdf"])
    if uploaded_file is not None:
        save_path = os.path.join(curr_dir, "..", "data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button("Index Uploaded Document"):
            with st.spinner("Ingesting & Indexing on CPU..."):
                count = pipeline.index_document(save_path, overwrite=False)
                st.success(f"Indexed {count} chunks successfully!")

# Main Header
st.markdown('<div class="main-title">🕉️ Sanskrit Document RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Retrieval-Augmented Generation with Transliteration Support & 100% CPU Inference</div>', unsafe_allow_html=True)

# Example Queries
st.markdown("**💡 Quick Query Examples (Click to try):**")
cols = st.columns(4)
example_q = ""
if cols[0].button(" मूर्खभृत्यस्य शर्करा"):
    example_q = "मूर्खभृत्यः शर्कराम् कुत्र न्यस्यति ?"
if cols[1].button("👑 भोजराजस्य सभा (HK)"):
    example_q = "bhojaraajaa kavibhyaH kiM daatuM ghoshhitavaan?"
if cols[2].button("🔔 घण्टाकर्णः राक्षसः (IAST)"):
    example_q = "ghaṇṭākarṇaḥ kuto vināśitaḥ ?"
if cols[3].button("📖 शीतं बहु बाधति (Grammar)"):
    example_q = "sheetaM bahu baadhate - atra kaa trutiH?"

# Query Input
user_query = st.text_input(
    "Enter Sanskrit Query (Devanagari or Romanized text like IAST / Harvard-Kyoto / ITRANS):",
    value=example_q if example_q else "",
    placeholder="e.g. 'shankhanaadah kim aakarot ?' or 'राजा भोजः किं सुवर्णं दत्तवान् ?'"
)

if user_query:
    # Transliteration Preview
    is_dev = is_devanagari(user_query)
    detected_scheme = "Devanagari" if is_dev else detect_transliteration_scheme(user_query)
    devanagari_query = to_devanagari(user_query)

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.info(f"**Detected Script / Scheme:** `{detected_scheme}`")
    with col_t2:
        st.success(f"**Devanagari Processed Query:** {devanagari_query}")

    # Run Query
    with st.spinner("Searching Sanskrit knowledge base on CPU..."):
        result = pipeline.query(user_query, top_k=top_k, retrieval_mode=retrieval_mode)

    # Response Display
    st.markdown("### 📝 Generated Response")
    st.markdown(f"""
    <div style="background-color: #fdf6e2; border-left: 5px solid #d97706; padding: 15px; border-radius: 5px; font-size: 1.05rem; line-height: 1.6;">
        {result['response'].replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    # Performance Metrics
    st.markdown("### ⚡ Performance & CPU Benchmarks")
    m = result["metrics"]
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Transliteration Time", f"{m['transliteration_latency_s'] * 1000:.1f} ms")
    m_col2.metric("Retrieval Latency", f"{m['retrieval_latency_s'] * 1000:.1f} ms")
    m_col3.metric("Generation Latency", f"{m['generation_latency_s'] * 1000:.1f} ms")
    m_col4.metric("Total Latency", f"{m['total_latency_s']:.2f} s")

    # Context Inspector
    st.markdown("### 📚 Retrieved Context Chunks")
    chunks = result["retrieved_chunks"]
    if not chunks:
        st.warning("No context chunks retrieved.")
    else:
        for idx, chunk in enumerate(chunks, 1):
            sec = chunk.get("metadata", {}).get("section", "General")
            score = chunk.get("rrf_score", chunk.get("dense_score", chunk.get("bm25_score", 0.0)))
            with st.expander(f"Chunk #{idx} | Section: {sec} | Score: {score:.4f}", expanded=(idx == 1)):
                st.write(chunk.get("content", ""))
                st.caption(f"Source: `{chunk.get('metadata', {}).get('source', 'corpus')}` | ID: `{chunk.get('id', '')}`")

st.markdown("---")
st.caption("Sanskrit RAG System | Developed by Sahil Karande | Evaluation Deliverables")
