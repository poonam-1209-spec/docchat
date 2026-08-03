# config.py — central place for all tunable parameters

import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", 3))       # final chunks after reranking
RERANK_FETCH_K = int(os.environ.get("RERANK_FETCH_K", 10))  # candidates fetched before reranking

# Chunking
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150))

# Retrieval
DEFAULT_K = int(os.environ.get("DEFAULT_K", 4))
MIN_K = 2
MAX_K = 10

# Similarity score threshold (FAISS L2 distance — lower = more similar)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 2.0))

# Multi-query: number of alternative question variants to generate
MULTI_QUERY_COUNT = int(os.environ.get("MULTI_QUERY_COUNT", 3))

# Logging
LOG_FILE = os.environ.get("LOG_FILE", "docchat.log")
CACHE_DIR = os.environ.get("CACHE_DIR", "faiss_cache")

# OCR (for scanned/image-based PDFs)
TESSERACT_PATH = os.environ.get("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH = os.environ.get("POPPLER_PATH", r"C:\poppler\Library\bin")
