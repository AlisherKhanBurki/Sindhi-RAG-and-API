import os
import sys
import json
import time
import urllib.request
import numpy as np
import faiss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from config import (
    OLLAMA_EMBED_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    SHAH_JO_RISALO_JSONL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)

CHECKPOINT_INTERVAL = 500
BATCH_SIZE = 30

def load_all_verses():
    print(f"Reading all verses from {SHAH_JO_RISALO_JSONL}...")
    all_docs = []
    with open(SHAH_JO_RISALO_JSONL, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                text_content = (
                    f"سر: {row.get('melody', '')}\n"
                    f"شاعريءَ جو قسم: {row.get('peotry_type', '')}\n"
                    f"شعر: {row.get('poetry_text', '')}\n"
                    f"تشريح ۽ معني: {row.get('explanation', '')}"
                )
                all_docs.append({
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
    print(f"Total verses loaded from file: {len(all_docs):,}")
    return all_docs

def embed_batch_on_gpu(batch_texts):
    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": batch_texts
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["embeddings"]

def run_full_ingestion():
    print("=" * 65)
    print("  FULL SHAH JO RISALO INGESTION ON NVIDIA RTX A4000 GPU")
    print("  Model: BGE-M3 (1024 dims) | Target: 43,000+ Verses")
    print("=" * 65, flush=True)

    all_docs = load_all_verses()
    total_target = len(all_docs)

    # Check for existing checkpoint
    existing_count = 0
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            existing_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
            if existing_index.ntotal == len(existing_meta) and existing_index.d == EMBEDDING_DIM:
                existing_count = existing_index.ntotal
                index = existing_index
                metadata = existing_meta
                print(f"Found existing checkpoint: {existing_count:,} verses already embedded! Resuming...", flush=True)
            else:
                index = faiss.IndexFlatIP(EMBEDDING_DIM)
                metadata = []
        except Exception:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            metadata = []
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = []

    if existing_count >= total_target:
        print(f"All {total_target:,} verses are already indexed! Nothing left to ingest.")
        return

    remaining_docs = all_docs[existing_count:]
    start_time = time.time()
    processed_in_run = 0

    print(f"Starting GPU embedding for remaining {len(remaining_docs):,} verses...", flush=True)

    new_vectors = []
    new_metadata = []

    for i in range(0, len(remaining_docs), BATCH_SIZE):
        batch = remaining_docs[i:min(i + BATCH_SIZE, len(remaining_docs))]
        batch_texts = [d["content"] for d in batch]

        try:
            embeddings = embed_batch_on_gpu(batch_texts)
            new_vectors.extend(embeddings)
            new_metadata.extend(batch)
            processed_in_run += len(batch)

            current_total = existing_count + processed_in_run
            elapsed = time.time() - start_time
            speed = processed_in_run / elapsed if elapsed > 0 else 0
            eta_seconds = (len(remaining_docs) - processed_in_run) / speed if speed > 0 else 0
            eta_mins = eta_seconds / 60

            # Progress log
            if processed_in_run % 60 == 0 or processed_in_run >= len(remaining_docs):
                pct = (current_total / total_target) * 100
                print(
                    f"[{current_total:,}/{total_target:,} | {pct:.1f}%] "
                    f"Speed: {speed:.1f} verses/sec | ETA: {eta_mins:.1f} mins | "
                    f"Sur: {batch[-1].get('melody', '')}",
                    flush=True
                )

            # Checkpoint save
            if len(new_vectors) >= CHECKPOINT_INTERVAL:
                vecs_np = np.array(new_vectors, dtype=np.float32)
                faiss.normalize_L2(vecs_np)
                index.add(vecs_np)
                metadata.extend(new_metadata)

                faiss.write_index(index, FAISS_INDEX_PATH)
                with open(METADATA_PATH, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False)

                print(f"  💾 Checkpoint saved: {index.ntotal:,} vectors written to disk.", flush=True)
                new_vectors = []
                new_metadata = []

        except Exception as e:
            print(f"  Warning on batch {i}: {e}. Retrying after 2s...", flush=True)
            time.sleep(2)

    # Final flush
    if new_vectors:
        vecs_np = np.array(new_vectors, dtype=np.float32)
        faiss.normalize_L2(vecs_np)
        index.add(vecs_np)
        metadata.extend(new_metadata)

        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"✓ INGESTION COMPLETE! Indexed {index.ntotal:,} verses into FAISS.")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Index path: {FAISS_INDEX_PATH}")
    print("=" * 65, flush=True)

if __name__ == "__main__":
    run_full_ingestion()
