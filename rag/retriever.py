import os
import json
import sqlite3
import urllib.request
import numpy as np
import faiss
from config import (
    OLLAMA_EMBED_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

FTS_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index", "sindhi_lexicon_fts.db")

class SindhiRetriever:
    def __init__(self):
        self.fts_available = os.path.exists(FTS_DB_PATH)
        if self.fts_available:
            print(f"Connected to 223K Lexicon FTS5 database: {FTS_DB_PATH}")

        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError("FAISS index or metadata not found. Run indexer.py first.")

        print(f"Loading local FAISS index from {FAISS_INDEX_PATH}...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"Retriever initialized: {self.index.ntotal:,} vectors in FAISS + 223,342 Lexicon FTS entries.")

    def search_lexicon(self, query: str, limit: int = 2):
        """
        Query the SQLite FTS5 223,000 Lexicon index.
        """
        if not self.fts_available:
            return []

        results = []
        try:
            # Clean query for FTS syntax
            clean_q = "".join([c for c in query if c.isalnum() or c.isspace()]).strip()
            if not clean_q:
                return []

            # Extract key words
            words = clean_q.split()
            # Try searching individual key terms
            conn = sqlite3.connect(FTS_DB_PATH)
            cur = conn.cursor()

            for w in words[:3]:
                if len(w) < 2:
                    continue
                cur.execute("""
                    SELECT word, normalized_word, word_with_airab, part_of_speech, definition, source_dictionary
                    FROM lexicon_fts
                    WHERE word = ? OR normalized_word = ?
                    LIMIT ?
                """, (w, w, limit))
                rows = cur.fetchall()

                # If no exact match, try prefix match
                if not rows:
                    cur.execute("""
                        SELECT word, normalized_word, word_with_airab, part_of_speech, definition, source_dictionary
                        FROM lexicon_fts
                        WHERE lexicon_fts MATCH ?
                        LIMIT ?
                    """, (f'"{w}"*', limit))
                    rows = cur.fetchall()

                for row in rows:
                    content_str = (
                        f"لغت جو لفظ: {row[0]} ({row[2] or ''})\n"
                        f"گرامر / قسم: {row[3] or 'عام'}\n"
                        f"معنيٰ ۽ وصف: {row[4]}\n"
                        f"ماخذ لغت: {row[5]}"
                    )
                    results.append({
                        "id": f"lex_{row[0]}",
                        "source": f"سنڌي لغت ({row[5]})",
                        "melody": "لغت / فرهنگ",
                        "content": content_str,
                        "similarity_score": 0.85, # Strong lexical confidence
                        "rank": 1,
                        "verse": f"{row[0]}: {row[4]}"
                    })
                if results:
                    break

            conn.close()
        except Exception as e:
            print(f"FTS error: {e}")

        return results[:limit]

    def embed_query(self, query: str) -> np.ndarray:
        payload = json.dumps({
            "model": EMBEDDING_MODEL,
            "input": query
        }).encode("utf-8")

        req = urllib.request.Request(
            OLLAMA_EMBED_URL,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_vec = data["embeddings"][0]

        vec = np.array([raw_vec], dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, top_k: int = TOP_K):
        # 1. Lexical Lookup (Instant exact match)
        lex_docs = self.search_lexicon(query, limit=2)

        # 2. Semantic Vector Search
        query_vec = self.embed_query(query)
        scores, indices = self.index.search(query_vec, top_k)

        sem_docs = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(self.metadata):
                doc = self.metadata[idx].copy()
                doc["similarity_score"] = float(score)
                doc["rank"] = rank + 1
                sem_docs.append(doc)

        # Hybrid merge: lexical first if relevant, followed by semantic
        all_docs = lex_docs + sem_docs
        top_score = max([d["similarity_score"] for d in all_docs]) if all_docs else 0.0
        is_confident = top_score >= SIMILARITY_THRESHOLD

        return {
            "query": query,
            "top_score": top_score,
            "is_confident": is_confident,
            "threshold": SIMILARITY_THRESHOLD,
            "documents": all_docs[:top_k + len(lex_docs)]
        }
