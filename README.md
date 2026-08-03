# DocChat — Chat with your PDF using RAG

A small retrieval-augmented generation (RAG) pipeline that lets you ask
questions about any PDF and get answers grounded in that document, instead
of an LLM guessing from general training data.

## Problem

LLMs hallucinate when asked about content they weren't trained on — a
specific syllabus, a contract, a research paper. RAG fixes this by
retrieving the most relevant chunks of the actual document and feeding
them to the LLM as context before it answers.

## Approach

```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter (chunks)
    → HuggingFace all-MiniLM-L6-v2 embeddings → FAISS vector store
    → retriever (top-k similarity search) → LLM (Groq / Llama 3.1)
    → grounded answer + source pages
```

1. **Load**: `PyPDFLoader` reads the PDF page by page.
2. **Chunk**: `RecursiveCharacterTextSplitter` splits pages into ~1000-char
   chunks with 150-char overlap, so an answer that spans a chunk boundary
   doesn't lose context.
3. **Embed**: each chunk is turned into a vector with a free, local
   `sentence-transformers` model (`all-MiniLM-L6-v2`) — no API key, runs on
   CPU.
4. **Store & retrieve**: vectors go into a local FAISS index. At query
   time, the question is embedded and FAISS returns the top-4 most similar
   chunks.
5. **Generate**: those chunks are stuffed into a prompt template along with
   the question, sent to an LLM (Groq's free-tier Llama 3.1 by default),
   and the model is instructed to answer only from that context.

## Setup

```bash
git clone <your-repo-url>
cd docchat
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # then paste in a free key from https://console.groq.com/keys
streamlit run app.py
```

Open the local URL Streamlit prints, upload a PDF in the sidebar, click
**Process document**, then ask questions in the chat box.

### Using the ingestion script standalone

```bash
python ingest.py path/to/document.pdf --index-dir my_index
```

This builds and saves a FAISS index to disk without launching the UI —
useful if you want to pre-build indexes for several documents.

## Example Q&A

*(from testing against a university course syllabus)*

| Question | Answer |
|---|---|
| "What's the late submission policy?" | Grounded answer quoting the penalty-per-day rule, with the source page cited |
| "When is the midterm?" | Correct date pulled directly from the schedule table |
| "What's the professor's opinion on AI tools?" | Model correctly said this wasn't covered in the document rather than guessing |

## What I'd improve next

- **Chunking**: fixed-size chunking can split tables and multi-column PDFs
  awkwardly — a layout-aware splitter (e.g. `unstructured`) would help.
- **Retrieval quality**: add a re-ranker (e.g. cross-encoder) on top of the
  initial FAISS top-k to improve precision on longer documents.
- **Evaluation**: right now quality is eyeballed against 5-10 manual
  questions; a small eval set with expected answers (RAGAS or similar)
  would make this measurable.
- **Multi-document support**: currently one PDF per session; could extend
  to a persistent index across multiple documents with per-document
  filtering.
- **Streaming responses**: swap `.invoke()` for `.stream()` so answers
  appear token-by-token instead of all at once.

## Stack

Python · LangChain · FAISS · Hugging Face `sentence-transformers` ·
Groq (Llama 3.1) · Streamlit
