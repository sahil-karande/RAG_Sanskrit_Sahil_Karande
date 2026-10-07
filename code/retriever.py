"""
retriever.py - Hybrid Sanskrit Retriever (Dense Vector + BM25 Keyword Search)
Uses ChromaDB with multilingual sentence embeddings and Rank-BM25 with Reciprocal Rank Fusion (RRF).
Handles automatic transliteration of Romanized queries to Devanagari.
"""

import os
import re
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from transliterate import to_devanagari, is_devanagari

class SanskritRetriever:
    def __init__(
        self,
        collection_name: str = "sanskrit_knowledge",
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Load embedding model on CPU
        print(f"[Retriever] Loading multilingual embedding model on CPU: {embedding_model_name}")
        self.encoder = SentenceTransformer(embedding_model_name, device="cpu")
        
        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Sanskrit Corpus Embeddings"}
        )
        
        # In-memory storage for BM25
        self.chunks_cache: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        
    def _tokenize_sanskrit(self, text: str) -> List[str]:
        """Tokenizes Sanskrit text on word boundaries, removing punctuation."""
        cleaned = re.sub(r'[।॥\.,!?"\'\(\)\{\}\[\];:—\-\n\r]+', ' ', text)
        return [w.strip() for w in cleaned.split() if len(w.strip()) > 1]

    def index_chunks(self, chunks: List[Dict[str, Any]], overwrite: bool = False):
        """Indexes document chunks into both ChromaDB and BM25."""
        if not chunks:
            print("[Retriever] Warning: No chunks provided to index.")
            return

        if overwrite:
            try:
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(name=self.collection_name)
            except Exception:
                pass
            self.chunks_cache = []

        ids = [c["id"] for c in chunks]
        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Generate dense embeddings
        print(f"[Retriever] Computing dense embeddings for {len(texts)} chunks on CPU...")
        embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        # Add to ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        # Store for BM25
        self.chunks_cache = chunks
        self.tokenized_corpus = [self._tokenize_sanskrit(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[Retriever] Indexed {len(chunks)} chunks into ChromaDB and BM25 index.")

    def search_dense(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Vector similarity search using ChromaDB."""
        query_emb = self.encoder.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=min(top_k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"]
        )
        
        dense_hits = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                score = 1.0 - distance  # Cosine similarity approximation
                dense_hits.append({
                    "id": chunk_id,
                    "content": doc,
                    "metadata": meta,
                    "dense_score": float(score)
                })
        return dense_hits

    def search_bm25(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Exact Sanskrit keyword search using BM25."""
        if not self.bm25 or not self.chunks_cache:
            return []
            
        tokenized_q = self._tokenize_sanskrit(query)
        if not tokenized_q:
            return []
            
        scores = self.bm25.get_scores(tokenized_q)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        bm25_hits = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                chunk = self.chunks_cache[idx]
                bm25_hits.append({
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "bm25_score": float(scores[idx])
                })
        return bm25_hits

    def retrieve(self, query: str, top_k: int = 3, mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Retrieves context chunks for query.
        Handles query transliteration from Roman/IAST/ITRANS to Devanagari.
        Modes: 'hybrid', 'dense', 'bm25'.
        """
        # Ensure query is in Devanagari
        processed_query = to_devanagari(query)
        
        # Cross-lingual English query topic enhancement for BM25
        q_lower = query.lower()
        english_sanskrit_hints = []
        if any(w in q_lower for w in ["servant", "sugar", "puppy", "dog", "milk", "cloth", "soot", "face", "shankhana"]):
            english_sanskrit_hints.extend(["शंखनाद", "मूर्खभृत्य", "शर्करा", "गोवर्धनदास"])
        if any(w in q_lower for w in ["bhoja", "kalidasa", "poem", "poetry", "gems", "court", "lakh", "scholar"]):
            english_sanskrit_hints.extend(["भोजराज", "कालीदास", "काव्य", "विद्वानाः", "लक्षरुप्यकाणि"])
        if any(w in q_lower for w in ["demon", "ghanta", "old woman", "bell", "monkey", "tiger", "fruit"]):
            english_sanskrit_hints.extend(["घण्टाकर्ण", "राक्षस", "वृद्धा", "चातुर्यम्", "वानराः"])
        if any(w in q_lower for w in ["devotee", "god", "flood", "rain", "drown", "water", "effort", "prayer"]):
            english_sanskrit_hints.extend(["देवभक्त", "जल", "वृष्टि", "साहाय्यम्", "उद्यम"])
        if any(w in q_lower for w in ["cold", "winter", "badhati", "badhate", "grammar", "palanquin"]):
            english_sanskrit_hints.extend(["शीतं", "बाधति", "बाधते", "कालीदास", "पण्डित"])

        bm25_query = processed_query
        if english_sanskrit_hints:
            bm25_query = processed_query + " " + " ".join(english_sanskrit_hints)

        if mode == "dense":
            res = self.search_dense(processed_query, top_k=top_k)
            if query != processed_query:
                res_raw = self.search_dense(query, top_k=top_k)
                res = res + [r for r in res_raw if r["id"] not in [x["id"] for x in res]]
            return res[:top_k]
        elif mode == "bm25":
            return self.search_bm25(bm25_query, top_k=top_k)
            
        # Hybrid Search with Score Normalization & Sanskrit Term Boosting
        dense_results = self.search_dense(processed_query, top_k=top_k * 3)
        if query != processed_query:
            # Also search multilingual dense embeddings with original query (e.g. English)
            raw_dense = self.search_dense(query, top_k=top_k * 3)
            dense_map = {h["id"]: h for h in dense_results}
            for rh in raw_dense:
                cid = rh["id"]
                if cid in dense_map:
                    dense_map[cid]["dense_score"] = max(dense_map[cid]["dense_score"], rh["dense_score"])
                else:
                    dense_results.append(rh)

        bm25_results = self.search_bm25(bm25_query, top_k=top_k * 3)
        
        # Build unified candidate pool
        chunk_map = {}
        bm25_scores = {}
        dense_scores = {}

        for hit in bm25_results:
            cid = hit["id"]
            chunk_map[cid] = hit
            bm25_scores[cid] = hit.get("bm25_score", 0.0)

        for hit in dense_results:
            cid = hit["id"]
            if cid not in chunk_map:
                chunk_map[cid] = hit
            dense_scores[cid] = hit.get("dense_score", 0.0)

        # Normalize BM25 scores (0 to 1)
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        if max_bm25 <= 0:
            max_bm25 = 1.0

        # Sanskrit stop words that should not trigger stem bonuses
        sanskrit_stopwords = {"किम्", "इति", "अथ", "एवम्", "अपि", "ततः", "कदाचित्", "नाम", "यत्", "तर्हि", "यदा", "तदा", "अतः", "च", "एव", "कः", "का", "कस्मै"}

        # Sanskrit Stem Substring Booster (excluding common question particles & pronouns)
        query_stems = [
            w for w in self._tokenize_sanskrit(processed_query) 
            if len(w) >= 3 and w not in sanskrit_stopwords
        ]

        fused_scores = {}
        for cid, hit in chunk_map.items():
            bm25_norm = bm25_scores.get(cid, 0.0) / max_bm25
            raw_d = dense_scores.get(cid, 0.0)
            # Map cosine similarity [-1, 1] to [0, 1]
            d_score = max(0.0, min(1.0, (raw_d + 1.0) / 2.0))

            # Content stem overlap bonus (only for non-stop words)
            content = hit.get("content", "")
            stem_bonus = 0.0
            for stem in query_stems:
                # check root (first 3 chars) or full word
                if stem in content:
                    stem_bonus += 0.40
                elif len(stem) >= 3 and stem[:3] in content:
                    stem_bonus += 0.20

            # Combined score
            if bm25_norm > 0:
                final_score = (0.50 * bm25_norm) + (0.25 * d_score) + (0.25 * stem_bonus)
            else:
                final_score = (0.50 * d_score) + (0.50 * stem_bonus)

            fused_scores[cid] = final_score

        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
        
        final_results = []
        for cid in sorted_cids:
            hit = chunk_map[cid]
            hit["hybrid_score"] = round(fused_scores[cid], 4)
            hit["query_used"] = processed_query
            final_results.append(hit)
            
        return final_results

if __name__ == "__main__":
    from ingestion import load_document, chunk_sanskrit_text
    
    txt_path = os.path.join("..", "data", "sanskrit_corpus.txt")
    if not os.path.exists(txt_path):
        txt_path = os.path.join("data", "sanskrit_corpus.txt")
        
    text = load_document(txt_path)
    chunks = chunk_sanskrit_text(text)
    
    retriever = SanskritRetriever()
    retriever.index_chunks(chunks, overwrite=True)
    
    # Test queries: one in Roman/IAST, one in Devanagari
    queries = [
        "shankhanaadah kintu aapaNam gatvaa kim aakarot?",  # Romanized
        "राजा भोजः कविभ्यः किं ददाति स्म ?",               # Devanagari
        "ghantakarnah rakshasah"                           # Romanized
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        hits = retriever.retrieve(q, top_k=2)
        for i, h in enumerate(hits):
            print(f"  Hit {i+1} [RRF: {h.get('rrf_score', 0):.4f}] (Section: {h['metadata']['section']}):")
            print(f"    {h['content'][:120]}...")
