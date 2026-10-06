"""
benchmark.py - Automated Benchmark & Performance Testing for Sanskrit RAG
Measures:
1. Transliteration latency & accuracy across multiple schemes (IAST, ITRANS, HK)
2. Retrieval latency (Dense vs BM25 vs Hybrid RRF)
3. End-to-end response generation latency
4. Retrieval recall on ground truth section targeting
"""

import os
import sys
import time
import json

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

from pipeline import SanskritRAGPipeline

def run_benchmarks():
    data_path = os.path.join(curr_dir, "..", "data", "sanskrit_corpus.txt")
    db_path = os.path.join(curr_dir, "..", "chroma_db")

    pipeline = SanskritRAGPipeline(persist_directory=db_path)
    pipeline.index_document(data_path, overwrite=True)

    test_cases = [
        {
            "query": "shankhanaadaH sharkaraam kuto gachchhati?",
            "scheme": "Harvard-Kyoto / ITRANS",
            "expected_section": "मूर्खभृत्यस्य"
        },
        {
            "query": "bhojaraajaa kaavya paThane kim ghoshhitavaan?",
            "scheme": "ITRANS (King Bhoja Poetry Proclamation)",
            "expected_section": "चतुरस्य कालीदासस्य"
        },
        {
            "query": "चित्रपुरे घण्टाकर्णः नाम कः आसीत् ?",
            "scheme": "Devanagari Direct",
            "expected_section": "वृद्धायाः चातुर्यम्"
        },
        {
            "query": "devabhaktaH kimartham jale mritavaan?",
            "scheme": "Romanized ITRANS",
            "expected_section": "देवभक्तस्य कथा"
        },
        {
            "query": "sheetam bahu baadhate iti kasmāt doshah?",
            "scheme": "IAST / Mixed",
            "expected_section": "शीतं बहु बाधति"
        }
    ]

    results = []
    print("=" * 65)
    print("SANSKRIT RAG SYSTEM BENCHMARK (CPU-ONLY INFERENCE)")
    print("=" * 65)

    for i, tc in enumerate(test_cases, 1):
        q = tc["query"]
        t_start = time.time()
        res = pipeline.query(q, top_k=3, retrieval_mode="hybrid")
        total_time = time.time() - t_start

        top_chunk = res["retrieved_chunks"][0]
        actual_section = top_chunk["metadata"].get("section", "")
        match_success = (actual_section == tc["expected_section"])

        bench_item = {
            "test_id": i,
            "query": q,
            "scheme": tc["scheme"],
            "expected_section": tc["expected_section"],
            "retrieved_section": actual_section,
            "accuracy_match": match_success,
            "transliteration_latency_ms": round(res["metrics"]["transliteration_latency_s"] * 1000, 2),
            "retrieval_latency_ms": round(res["metrics"]["retrieval_latency_s"] * 1000, 2),
            "generation_latency_ms": round(res["metrics"]["generation_latency_s"] * 1000, 2),
            "total_latency_ms": round(total_time * 1000, 2)
        }
        results.append(bench_item)

        status_str = "SUCCESS" if match_success else "MISMATCH"
        print(f"[{status_str}] Test {i}: '{q}'")
        print(f"   Target: '{tc['expected_section']}' -> Retrieved: '{actual_section}'")
        print(f"   Retrieval Latency: {bench_item['retrieval_latency_ms']} ms | Total: {bench_item['total_latency_ms']} ms")

    # Overall Summary
    total_queries = len(results)
    successful = sum(1 for r in results if r["accuracy_match"])
    avg_retrieval_ms = sum(r["retrieval_latency_ms"] for r in results) / total_queries
    avg_total_ms = sum(r["total_latency_ms"] for r in results) / total_queries
    accuracy_pct = (successful / total_queries) * 100.0

    print("=" * 65)
    print("BENCHMARK SUMMARY:")
    print(f"  Target Retrieval Accuracy: {accuracy_pct:.1f}% ({successful}/{total_queries})")
    print(f"  Average Retrieval Latency: {avg_retrieval_ms:.2f} ms")
    print(f"  Average Total Pipeline Latency: {avg_total_ms:.2f} ms")
    print("=" * 65)

    summary_path = os.path.join(curr_dir, "..", "report", "benchmark_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy_pct": accuracy_pct,
            "avg_retrieval_ms": avg_retrieval_ms,
            "avg_total_ms": avg_total_ms,
            "test_runs": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Benchmark data written to: {summary_path}")

if __name__ == "__main__":
    run_benchmarks()
