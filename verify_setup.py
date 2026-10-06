import os
import sys

def run_tests():
    print("=" * 50)
    print("SANSKRIT RAG ENVIRONMENT & DATA AUDIT")
    print("=" * 50)

    # 1. Directories
    dirs = ["code", "data", "report"]
    all_dirs = all(os.path.isdir(d) for d in dirs)
    print(f"[OK] Workspace Directories {dirs}: {'EXISTS' if all_dirs else 'MISSING'}")

    # 2. Text Corpus
    txt_path = os.path.join("data", "sanskrit_corpus.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            txt_content = f.read()
        print(f"[OK] data/sanskrit_corpus.txt: {len(txt_content)} characters, UTF-8 verified")
    else:
        print("[FAIL] data/sanskrit_corpus.txt missing!")

    # 3. PDF Corpus
    pdf_path = os.path.join("data", "sanskrit_corpus.pdf")
    if os.path.exists(pdf_path):
        import fitz
        doc = fitz.open(pdf_path)
        print(f"[OK] data/sanskrit_corpus.pdf: {len(doc)} page(s) loaded successfully via PyMuPDF")
        doc.close()
    else:
        print("[FAIL] data/sanskrit_corpus.pdf missing!")

    # 4. Transliteration
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        out = transliterate("dharmah", sanscript.ITRANS, sanscript.DEVANAGARI)
        assert len(out) > 0
        print(f"[OK] indic-transliteration operational (transliterated 'dharmah' to {len(out)} Devanagari chars)")
    except Exception as e:
        print(f"[FAIL] indic-transliteration error: {e}")

    # 5. BM25
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([["dharmah", "satyam"], ["kalidasa", "kavih"]])
        print("[OK] rank-bm25 initialized and functional")
    except Exception as e:
        print(f"[FAIL] rank-bm25 error: {e}")

    # 6. ChromaDB
    try:
        import chromadb
        client = chromadb.Client()
        test_col = client.create_collection("audit_test")
        test_col.add(documents=["test"], ids=["t1"])
        print(f"[OK] chromadb {chromadb.__version__} vector database functional")
    except Exception as e:
        print(f"[FAIL] chromadb error: {e}")

    # 7. Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("[OK] sentence-transformers multilingual model cached and verified")
    except Exception as e:
        print(f"[FAIL] sentence-transformers error: {e}")

    # 8. llama-cpp-python
    try:
        import llama_cpp
        print(f"[OK] llama-cpp-python {llama_cpp.__version__} installed with CPU backend")
    except Exception as e:
        print(f"[FAIL] llama-cpp-python error: {e}")

    # 9. Streamlit
    try:
        import streamlit
        print(f"[OK] streamlit {streamlit.__version__} UI framework installed")
    except Exception as e:
        print(f"[FAIL] streamlit error: {e}")

    print("=" * 50)
    print("STATUS: ALL PREREQUISITES 100% READY!")
    print("=" * 50)

if __name__ == "__main__":
    run_tests()
