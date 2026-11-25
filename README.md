# Sindhi Literature Hybrid RAG System

A hybrid Retrieval-Augmented Generation (RAG) system for the Sindhi language (`سنڌي`), combining **local GPU vector search** (FAISS + Ollama `bge-m3`), **SQLite FTS5 lexical dictionary search** (223K+ lexicon entries), and an **automatic Gemini API fallback** when local context has low relevance.

## Features

- **Hybrid Retrieval:**
  - **Semantic Vector Search:** Local `FAISS` (`IndexFlatIP` cosine similarity) powered by `bge-m3` embeddings running on local GPU via Ollama (43,828 verses and commentaries from *Shah Jo Risalo*).
  - **Lexical Full-Text Search:** `SQLite FTS5` index over 223,342 entries from the *Sindhi Open Lexicon* for instant exact/prefix dictionary lookups.
- **Smart Gemini Fallback:** Evaluates retrieval confidence (`SIMILARITY_THRESHOLD = 0.50`) and context sufficiency; automatically falls back to Gemini (`gemini-3.5-flash` / `gemini-3.5-flash-lite`) when local knowledge does not cover the query.
- **RTL Sindhi Web UI & CLI:** Clean RTL web interface (`rag/web/index.html`) served via Python's `ThreadingHTTPServer`, plus an interactive terminal CLI (`rag/cli_rag.py`).

## Project Structure

```text
├── download_fast.py                # Download Sindhi corpora from Hugging Face
├── download_datasets.py            # Alternative datasets-library downloader
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
└── rag/
    ├── config.py                   # Configuration & paths (loads .env)
    ├── build_lexicon_fts.py        # Builds SQLite FTS5 index (223K lexicon entries)
    ├── indexer.py                  # Quick sample FAISS indexer
    ├── ingest_full_shah_jo_risalo.py # Resumable full GPU indexer (43K+ verses)
    ├── retriever.py                # Hybrid FTS5 + FAISS retriever
    ├── rag_engine.py               # Grounded RAG + Gemini fallback orchestrator
    ├── server.py                   # Web UI & REST API server (port 7860)
    ├── cli_rag.py                  # Interactive CLI chatbot
    └── web/
        └── index.html              # RTL Sindhi web chat frontend
```

## Setup

1. **Install dependencies & pull embedding model:**
   ```bash
   pip install -r requirements.txt
   ollama pull bge-m3
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY in .env
   ```

3. **Download Sindhi datasets:**
   ```bash
   python download_fast.py
   ```
   *(Note: If you also want the Sindhi Mega Corpus 118M zip from Google Drive, ensure `gdown` is installed or download it per the Hugging Face repo README.)*

4. **Build local indexes:**
   ```bash
   python rag/build_lexicon_fts.py
   python rag/ingest_full_shah_jo_risalo.py
   ```

5. **Run the Web UI or CLI:**
   ```bash
   python rag/server.py
   # Open http://localhost:7860
   ```
   Or via terminal:
   ```bash
   python rag/cli_rag.py
   ```
