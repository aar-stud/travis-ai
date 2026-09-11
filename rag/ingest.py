"""
ingest.py — Run this ONCE (or on any knowledge base update).

Reads markdown/text files in knowledge_base, chunks them using a recursive
text splitter (to handle generic policies instead of just Q&A),
embeds, and stores in ChromaDB.
"""

import os
import re

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
KB_DIR     = os.path.abspath(os.path.join(BASE_DIR, "..", "knowledge_base"))
CHROMA_DIR = os.path.join(KB_DIR, "chroma_store")
COLLECTION = "travis_bank_policy"
MODEL_NAME = "all-MiniLM-L6-v2"

def chunk_text(text: str, max_size: int = 600, overlap: int = 100) -> list[str]:
    """
    Context-aware Markdown chunking.
    Tracks headers (##, ###) and injects them into the chunks so the embedding
    model has full semantic context for smaller, more granular text blocks.
    This saves LLM tokens and improves embedding match accuracy.
    """
    lines = text.split('\n')
    chunks = []
    
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    
    current_text = ""
    
    def finalize_chunk(text_to_add):
        ctx = []
        if current_h1: ctx.append(current_h1)
        if current_h2: ctx.append(current_h2)
        if current_h3: ctx.append(current_h3)
        context_str = "Topic: " + " > ".join(ctx) + "\n" if ctx else ""
        return context_str + text_to_add.strip()

    for line in lines:
        if line.startswith("# "):
            current_h1 = line[2:].strip()
            current_h2 = ""
            current_h3 = ""
            continue
        elif line.startswith("## "):
            current_h2 = line[3:].strip()
            current_h3 = ""
            continue
        elif line.startswith("### "):
            current_h3 = line[4:].strip()
            continue
            
        if not line.strip():
            continue
            
        if len(current_text) + len(line) > max_size and current_text.strip():
            chunks.append(finalize_chunk(current_text))
            # Keep overlap by grabbing the last few words
            words = current_text.split()
            current_text = " ".join(words[-20:]) + " " + line + "\n"
        else:
            current_text += line + "\n"
            
    if current_text.strip():
        chunks.append(finalize_chunk(current_text))
        
    return chunks

def load_and_chunk(kb_dir: str) -> list:
    """
    Load .md files in kb_dir (ignoring old faqs.txt).
    Split them into semantic overlapping chunks.
    """
    all_chunks = []

    for fname in sorted(os.listdir(kb_dir)):
        # Only process markdown files, ignore txt files like faqs.txt
        if not fname.endswith(".md"):
            continue
            
        fpath = os.path.join(kb_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()

        # Generate chunks with smaller max_size to save LLM tokens
        blocks = chunk_text(text, max_size=600, overlap=100)

        for i, block in enumerate(blocks):
            # Clean up the ID to be alphanumeric
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '', fname.replace('.md', ''))
            all_chunks.append({
                "id":     f"{clean_name}_chunk{i:04d}",
                "source": fname,
                "text":   block,
            })

        print(f"[ingest] {fname}: {len(blocks)} chunks created")

    return all_chunks


def main():
    # Lazy imports — only loaded when ingest actually runs
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("\n=== TRAVIS RAG Ingest ===\n")

    chunks = load_and_chunk(KB_DIR)
    if not chunks:
        print("[ingest] ERROR: no .md blocks found. Check knowledge_base/*.md")
        return

    print(f"\n[ingest] Total chunks to index: {len(chunks)}")

    print("\n[ingest] Sample chunks:")
    for c in chunks[:3]:
        preview = c["text"][:100].replace("\n", " ")
        print(f"  [{c['id']}] {preview}...")

    # Embed
    print(f"\n[ingest] Loading model '{MODEL_NAME}' ...")
    model      = SentenceTransformer(MODEL_NAME)
    texts      = [c["text"] for c in chunks]
    print("[ingest] Embedding ...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    # Write to ChromaDB (always start fresh)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    from chromadb.config import Settings
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=settings)

    try:
        client.delete_collection(COLLECTION)
        print(f"[ingest] Deleted old collection '{COLLECTION}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids        = [c["id"]     for c in chunks],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [{"source": c["source"]} for c in chunks],
    )

    print(f"\n[ingest] Done — {len(chunks)} chunks indexed into '{COLLECTION}'.")
    print(f"[ingest] ChromaDB path: {CHROMA_DIR}\n")


if __name__ == "__main__":
    main()