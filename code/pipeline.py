"""
pipeline.py - Sanskrit RAG Orchestrator & Performance Benchmark Pipeline
Connects Ingestion, Transliteration, Hybrid Retrieval, and CPU Generation.
Measures latency, accuracy metrics, and CPU inference stats.
"""

import os
import time
from typing import Dict, Any, List, Optional
from ingestion import load_document, chunk_sanskrit_text
from transliterate import to_devanagari, is_devanagari, detect_transliteration_scheme
from retriever import SanskritRetriever
from generator import SanskritGenerator

class SanskritRAGPipeline:
    def __init__(
        self,
        collection_name: str = "sanskrit_knowledge",
        persist_directory: str = "./chroma_db",
        model_path: Optional[str] = None
    ):
        print("=" * 60)
        print("INITIALIZING SANSKRIT RAG PIPELINE (CPU ONLY)")
        print("=" * 60)
        self.retriever = SanskritRetriever(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.generator = SanskritGenerator(model_path=model_path)
        self.indexed_files: List[str] = []

    def index_document(self, file_path: str, overwrite: bool = False):
        """Loads, cleans, chunks, and indexes a Sanskrit document (.txt or .pdf)."""
        print(f"\n[Pipeline] Ingesting document: {file_path}")
        t0 = time.time()
        text = load_document(file_path)
        source_name = os.path.basename(file_path)
        chunks = chunk_sanskrit_text(text, source_name=source_name)
        t_chunk = time.time() - t0
        print(f"[Pipeline] Chunking completed in {t_chunk:.3f}s. Total chunks: {len(chunks)}")
        
        t1 = time.time()
        self.retriever.index_chunks(chunks, overwrite=overwrite)
        t_index = time.time() - t1
        print(f"[Pipeline] Indexing completed in {t_index:.3f}s.")
        self.indexed_files.append(file_path)
        return len(chunks)

    def query(
        self, 
        user_query: str, 
        top_k: int = 3, 
        retrieval_mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query flow:
        1. Query transliteration / script normalization
        2. Context chunk retrieval
        3. Response generation via CPU LLM
        4. Latency and performance benchmarking
        """
        start_time = time.time()
        
        # Step 1: Preprocess and Transliterate
        t0 = time.time()
        is_dev = is_devanagari(user_query)
        detected_scheme = "Devanagari" if is_dev else detect_transliteration_scheme(user_query)
        devanagari_query = to_devanagari(user_query)
        transliteration_time = time.time() - t0

        # Step 2: Context Retrieval
        t1 = time.time()
        retrieved_chunks = self.retriever.retrieve(
            query=user_query,
            top_k=top_k,
            mode=retrieval_mode
        )
        retrieval_time = time.time() - t1

        # Step 3: LLM Generation (CPU)
        t2 = time.time()
        gen_result = self.generator.generate(
            query=user_query,
            context_chunks=retrieved_chunks
        )
        generation_time = time.time() - t2
        total_latency = time.time() - start_time

        return {
            "original_query": user_query,
            "processed_query": devanagari_query,
            "detected_scheme": detected_scheme,
            "is_devanagari": is_dev,
            "retrieved_chunks": retrieved_chunks,
            "response": gen_result["answer"],
            "backend": gen_result.get("backend", "CPU"),
            "metrics": {
                "transliteration_latency_s": round(transliteration_time, 4),
                "retrieval_latency_s": round(retrieval_time, 4),
                "generation_latency_s": round(generation_time, 4),
                "total_latency_s": round(total_latency, 4),
                "retrieved_chunk_count": len(retrieved_chunks),
                "retrieval_mode": retrieval_mode
            }
        }

if __name__ == "__main__":
    txt_doc = os.path.join("..", "data", "sanskrit_corpus.txt")
    if not os.path.exists(txt_doc):
        txt_doc = os.path.join("data", "sanskrit_corpus.txt")

    pipeline = SanskritRAGPipeline(persist_directory="./chroma_db")
    pipeline.index_document(txt_doc, overwrite=True)

    benchmark_queries = [
        "shankhanaadaH kuto dugdham anaytvaan?",
        "राजा भोजः कविभ्यः किं बहुमानम् ददाति स्म ?",
        "ghantakarnah kutra prativasati sma?",
        "देवभक्तः किमर्थं जले मृतवान् ?",
        "sheetam bahu baadhate - atra doshah kah?"
    ]

    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK EVALUATIONS")
    print("=" * 60)

    for q in benchmark_queries:
        res = pipeline.query(q, top_k=2)
        print(f"\n[QUERY]: {res['original_query']}")
        print(f"  Scheme Detected: {res['detected_scheme']} -> Query: {res['processed_query']}")
        print(f"  Latency: Retrieval={res['metrics']['retrieval_latency_s']}s, Total={res['metrics']['total_latency_s']}s")
        print(f"  Top Match: Section='{res['retrieved_chunks'][0]['metadata'].get('section', '')}'")
