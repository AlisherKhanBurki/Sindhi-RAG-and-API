import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")
os.makedirs(INDEX_DIR, exist_ok=True)

# Gemini API Key loaded from environment / .env (used for grounded generation & smart fallback)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Local GPU Embeddings via Ollama (e.g., NVIDIA RTX A4000)
EMBEDDING_PROVIDER = "ollama"
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = 1024  # BGE-M3 native dimension

# Candidate generation models for RAG grounding / Fallback
GENERATION_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

# Paths
SHAH_JO_RISALO_JSONL = os.path.join(DATASETS_DIR, "shah_jo_risalo", "train.jsonl")
CORPUS_505M_DIR = os.path.join(DATASETS_DIR, "sindhi_corpus_505m", "data")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "sindhi_faiss.bin")
METADATA_PATH = os.path.join(INDEX_DIR, "sindhi_metadata.json")

# RAG Quality Thresholds for BGE-M3 unit-normalized vectors
# For BGE-M3, cosine similarity >= 0.50 represents strong semantic relevance.
SIMILARITY_THRESHOLD = 0.50
TOP_K = 3
