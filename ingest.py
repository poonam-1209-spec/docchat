"""
ingest.py — Step 1 & 2 of the DocChat pipeline.

Loads a PDF, splits it into overlapping chunks, embeds those chunks with a
free local sentence-transformers model, and saves a FAISS index to disk so
app.py (or any other script) can load it later without re-embedding.

Usage:
    python ingest.py path/to/document.pdf
    python ingest.py path/to/document.pdf --index-dir my_index
"""

import argparse
import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_index(pdf_path: str, index_dir: str = "faiss_index") -> FAISS:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No such file: {pdf_path}")

    print(f"[1/4] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # one Document per page, with page metadata
    print(f"      Loaded {len(documents)} pages")

    print("[2/4] Splitting into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # characters per chunk
        chunk_overlap=150,    # overlap so context isn't lost at chunk boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"      Created {len(chunks)} chunks")

    print("[3/4] Embedding chunks with sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("[4/4] Building FAISS index")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    print(f"      Saved index to ./{index_dir}/")

    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into a FAISS index")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--index-dir", default="faiss_index", help="Where to save the FAISS index"
    )
    args = parser.parse_args()

    try:
        build_index(args.pdf_path, args.index_dir)
        print("\nDone. Load it later with:")
        print(
            "  FAISS.load_local("
            f"'{args.index_dir}', embeddings, allow_dangerous_deserialization=True)"
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
