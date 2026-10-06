"""
ingestion.py - Document Ingestion and Sanskrit-Aware Chunking Pipeline
Supports .txt and .pdf Sanskrit documents.
Implements domain-tailored chunking respecting shloka markers (।, ॥) and story headings.
"""

import os
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF
from transliterate import normalize_sanskrit

def load_txt(file_path: str) -> str:
    """Loads a UTF-8 text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def load_pdf(file_path: str) -> str:
    """Extracts text from PDF preserving Devanagari Unicode glyphs."""
    doc = fitz.open(file_path)
    text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(text_parts)

def load_document(file_path: str) -> str:
    """Loads document based on file extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        raw = load_txt(file_path)
    elif ext == ".pdf":
        raw = load_pdf(file_path)
    elif ext == ".docx":
        import docx
        doc = docx.Document(file_path)
        raw = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .txt, .pdf, .docx")
    
    return normalize_sanskrit(raw)

def chunk_sanskrit_text(
    text: str, 
    source_name: str = "sanskrit_doc",
    max_chunk_chars: int = 500, 
    overlap_chars: int = 100
) -> List[Dict[str, Any]]:
    """
    Chunks Sanskrit text intelligently by:
    1. Splitting on major sections / story headings / double line breaks.
    2. Splitting within large paragraphs along shloka/danda boundaries (॥, ।).
    3. Retaining overlap for retrieval continuity.
    """
    # Identify distinct sections/stories
    sections = re.split(r'\n{2,}', text)
    chunks = []
    chunk_index = 0

    current_story = "General"

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # Check if line looks like a story title (short line without dandas)
        lines = [line.strip() for line in sec.split("\n") if line.strip()]
        if len(lines) == 1 and len(lines[0]) < 60 and ("।" not in lines[0] and "॥" not in lines[0]):
            current_story = lines[0]
            continue

        # If section is small enough, keep as single chunk
        if len(sec) <= max_chunk_chars:
            chunks.append({
                "id": f"{source_name}_chunk_{chunk_index}",
                "content": sec,
                "metadata": {
                    "source": source_name,
                    "chunk_id": chunk_index,
                    "section": current_story,
                    "char_count": len(sec)
                }
            })
            chunk_index += 1
        else:
            # Split by shlokas (॥) or sentence dandas (।)
            sentences = re.split(r'(?<=[॥।])\s+', sec)
            curr_chunk = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(curr_chunk) + len(sent) + 1 <= max_chunk_chars:
                    curr_chunk = f"{curr_chunk} {sent}".strip()
                else:
                    if curr_chunk:
                        chunks.append({
                            "id": f"{source_name}_chunk_{chunk_index}",
                            "content": curr_chunk,
                            "metadata": {
                                "source": source_name,
                                "chunk_id": chunk_index,
                                "section": current_story,
                                "char_count": len(curr_chunk)
                            }
                        })
                        chunk_index += 1
                        # Retain overlap from end of curr_chunk
                        curr_chunk = curr_chunk[-overlap_chars:] + " " + sent
                    else:
                        # Single sentence longer than max_chunk_chars
                        chunks.append({
                            "id": f"{source_name}_chunk_{chunk_index}",
                            "content": sent,
                            "metadata": {
                                "source": source_name,
                                "chunk_id": chunk_index,
                                "section": current_story,
                                "char_count": len(sent)
                            }
                        })
                        chunk_index += 1
                        curr_chunk = ""

            if curr_chunk.strip():
                chunks.append({
                    "id": f"{source_name}_chunk_{chunk_index}",
                    "content": curr_chunk.strip(),
                    "metadata": {
                        "source": source_name,
                        "chunk_id": chunk_index,
                        "section": current_story,
                        "char_count": len(curr_chunk.strip())
                    }
                })
                chunk_index += 1

    return chunks

if __name__ == "__main__":
    txt_path = os.path.join("..", "data", "sanskrit_corpus.txt")
    if not os.path.exists(txt_path):
        txt_path = os.path.join("data", "sanskrit_corpus.txt")
    
    text = load_document(txt_path)
    chunks = chunk_sanskrit_text(text, source_name="sanskrit_corpus")
    print(f"Loaded document from {txt_path}: {len(text)} characters.")
    print(f"Generated {len(chunks)} Sanskrit chunks.")
    print(f"Sample Chunk 0 (Section: {chunks[0]['metadata']['section']}):")
    print(f"'{chunks[0]['content'][:150]}...'")
