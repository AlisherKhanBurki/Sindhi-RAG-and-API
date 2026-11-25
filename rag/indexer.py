import os
import json
import time
import urllib.request
import numpy as np
import faiss
from config import (
    OLLAMA_EMBED_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    SHAH_JO_RISALO_JSONL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)

def load_shah_jo_risalo_documents(limit=500):
    """
    Load structured verses and explanations from Shah Jo Risalo.
    """
    documents = []
    if not os.path.exists(SHAH_JO_RISALO_JSONL):
        raise FileNotFoundError(f"Dataset not found at {SHAH_JO_RISALO_JSONL}")

    print(f"Reading up to {limit} verses from Shah Jo Risalo...")
    with open(SHAH_JO_RISALO_JSONL, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            try:
                row = json.loads(line.strip())
                text_content = (
                    f"سر: {row.get('melody', '')}\n"
                    f"شاعريءَ جو قسم: {row.get('peotry_type', '')}\n"
                    f"شعر: {row.get('poetry_text', '')}\n"
                    f"تشريح ۽ معني: {row.get('explanation', '')}"
                )
                documents.append({
                    "id": f"sjr_{idx}",
                    "source": "شاهه جو رسالو",
                    "melody": row.get("melody", ""),
                    "poetry_type": row.get("peotry_type", ""),
                    "compiler": row.get("compiler", ""),
                    "content": text_content,
                    "explanation": row.get("explanation", ""),
                    "verse": row.get("poetry_text", "")
                })
            except Exception:
                continue

    print(f"Loaded {len(documents)} document chunks.")
    return documents

def embed_documents_in_batches(texts, batch_size=25):
    """
    Embed texts using local BGE-M3 on NVIDIA RTX A4000 GPU via Ollama.
    """
    all_vectors = []
    total = len(texts)
    print(f"Generating GPU embeddings for {total} documents using {EMBEDDING_MODEL} (NVIDIA RTX A4000)...")

    for i in range(0, total, batch_size):
        batch = texts[i:min(i + batch_size, total)]
        payload = json.dumps({
            "model": EMBEDDING_MODEL,
            "input": batch
        }).encode("utf-8")

        req = urllib.request.Request(
            OLLAMA_EMBED_URL,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for emb in data["embeddings"]:
                all_vectors.append(emb)

        print(f"  GPU Embedded {min(i + batch_size, total)}/{total} chunks...", flush=True)

    vectors_np = np.array(all_vectors, dtype=np.float32)
    faiss.normalize_L2(vectors_np)
    return vectors_np

def build_index(doc_limit=500):
    """
    Build and persist the local FAISS index + metadata on GPU.
    """
    start_time = time.time()
    docs = load_shah_jo_risalo_documents(limit=doc_limit)
    texts = [d["content"] for d in docs]

    vectors = embed_documents_in_batches(texts, batch_size=25)

    print("Building FAISS Index (IndexFlatIP for cosine similarity on 1024 dims)...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    print(f"FAISS Index built with {index.ntotal} local vectors in {time.time()-start_time:.1f}s.")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")

    # Save metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")

if __name__ == "__main__":
    build_index(doc_limit=500)
