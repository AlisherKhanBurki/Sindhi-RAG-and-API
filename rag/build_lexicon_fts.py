import os
import sys
import json
import sqlite3
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets")
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

LEXICON_JSONL = os.path.join(DATASETS_DIR, "sindhi_open_lexicon", "data", "sindhi_open_lexicon_master_223342.jsonl")
FTS_DB_PATH = os.path.join(INDEX_DIR, "sindhi_lexicon_fts.db")

def build_lexicon_fts():
    print(f"Opening Lexicon file: {LEXICON_JSONL}")
    if not os.path.exists(LEXICON_JSONL):
        raise FileNotFoundError(f"Lexicon file not found: {LEXICON_JSONL}")

    # Remove old DB if rebuilding
    if os.path.exists(FTS_DB_PATH):
        try:
            os.remove(FTS_DB_PATH)
        except Exception:
            pass

    conn = sqlite3.connect(FTS_DB_PATH)
    cur = conn.cursor()

    # Create FTS5 virtual table
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS lexicon_fts USING fts5(
            word,
            normalized_word,
            word_with_airab,
            part_of_speech,
            definition,
            source_dictionary,
            tokenize = 'unicode61'
        )
    """)

    print("Ingesting 223,342 lexicon entries into SQLite FTS5...")
    start_time = time.time()
    batch = []
    total_count = 0

    with open(LEXICON_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                word = row.get("word", "")
                norm_word = row.get("normalized_word", "")
                airab = row.get("word_with_airab_or_variant", "")
                pos = row.get("part_of_speech", "")
                defn = row.get("definition", "")
                source = row.get("source_dictionary", "")

                batch.append((word, norm_word, airab, pos, defn, source))
                total_count += 1

                if len(batch) >= 10000:
                    cur.executemany("""
                        INSERT INTO lexicon_fts(word, normalized_word, word_with_airab, part_of_speech, definition, source_dictionary)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, batch)
                    conn.commit()
                    batch = []
                    print(f"  Ingested {total_count:,} entries...", flush=True)

            except Exception as e:
                continue

    if batch:
        cur.executemany("""
            INSERT INTO lexicon_fts(word, normalized_word, word_with_airab, part_of_speech, definition, source_dictionary)
            VALUES (?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()

    conn.close()
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(FTS_DB_PATH) / (1024 * 1024)
    print(f"\n✓ Done! Successfully indexed {total_count:,} lexicon entries into FTS5.")
    print(f"  Database Path: {FTS_DB_PATH}")
    print(f"  Database Size: {file_size_mb:.2f} MB")
    print(f"  Elapsed Time: {elapsed:.1f}s ({round(total_count/elapsed, 1)} entries/sec)")

if __name__ == "__main__":
    build_lexicon_fts()
