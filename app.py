"""
app.py — DocChat: Multi-PDF RAG Chat

Features:
  - Multi-PDF upload and cross-document retrieval
  - Hybrid search: FAISS (semantic) + BM25 (keyword)
  - Cross-encoder reranking for precision
  - Multi-query retrieval (generates query variants)
  - Conversational chat history (follow-up questions work)
  - Streaming responses
  - Source citations with filename + page
  - Content-based FAISS caching
  - Analytics dashboard
  - Structured logging

Run with:
    python -m streamlit run app.py
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

import config

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("docchat")


def log_event(event: str, **kwargs):
    # Convert numpy/float32 values to native Python floats for JSON serialization
    safe = {k: float(v) if hasattr(v, "item") else v for k, v in kwargs.items()}
    logger.info(json.dumps({"event": event, **safe}))


# ---------------------------------------------------------------------------
# Analytics state helpers
# ---------------------------------------------------------------------------
def init_analytics():
    if "analytics" not in st.session_state:
        st.session_state["analytics"] = {
            "pdfs_uploaded": 0,
            "questions_asked": 0,
            "cache_hits": 0,
            "total_retrieval_ms": 0,
            "total_generation_ms": 0,
            "similarity_scores": [],
        }


def record_query(retrieval_ms, generation_ms, score):
    a = st.session_state["analytics"]
    a["questions_asked"] += 1
    a["total_retrieval_ms"] += retrieval_ms
    a["total_generation_ms"] += generation_ms
    if score is not None:
        a["similarity_scores"].append(score)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm():
    from langchain_groq import ChatGroq
    if not config.GROQ_API_KEY:
        st.error("No GROQ_API_KEY found. Add it to your .env file.")
        st.stop()
    return ChatGroq(model=config.LLM_MODEL, temperature=0, api_key=config.GROQ_API_KEY)


# ---------------------------------------------------------------------------
# PDF ingestion + caching
# ---------------------------------------------------------------------------
def get_pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.md5(pdf_bytes).hexdigest()[:12]


def get_cache_path(pdf_hash: str) -> str:
    return os.path.join(config.CACHE_DIR, pdf_hash)


def load_cached_vectorstore(pdf_hash: str):
    path = get_cache_path(pdf_hash)
    if os.path.exists(path):
        return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)
    return None


def ocr_pdf(tmp_path: str, filename: str):
    """Fallback: convert PDF pages to images and OCR them with Tesseract."""
    import pytesseract
    from pdf2image import convert_from_path
    from langchain_core.documents import Document

    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH
    images = convert_from_path(tmp_path, poppler_path=config.POPPLER_PATH)

    documents = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": filename, "source_file": filename, "page": i}
            ))
    return documents


def build_and_cache_vectorstore(pdf_bytes: bytes, pdf_hash: str, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Tag each chunk with the source filename for citations
    for doc in documents:
        doc.metadata["source_file"] = filename

    # Fallback to OCR if no text was extracted (scanned/image-based PDF)
    if not any(doc.page_content.strip() for doc in documents):
        try:
            documents = ocr_pdf(tmp_path, filename)
            ocr_used = True
        except Exception as e:
            os.unlink(tmp_path)
            raise ValueError(
                f"'{filename}' appears to be a scanned PDF and OCR failed: {e}. "
                "Make sure Tesseract and Poppler are installed and paths are set in .env"
            )
    else:
        ocr_used = False
        for doc in documents:
            doc.metadata["source_file"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        os.unlink(tmp_path)
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "The PDF may be scanned/image-based and requires OCR."
        )

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    cache_path = get_cache_path(pdf_hash)
    os.makedirs(cache_path, exist_ok=True)
    vectorstore.save_local(cache_path)

    os.unlink(tmp_path)
    return vectorstore, len(documents), len(chunks), ocr_used


# ---------------------------------------------------------------------------
# Hybrid retrieval: FAISS + BM25
# ---------------------------------------------------------------------------
def hybrid_retrieve(vectorstore: FAISS, all_chunks, question: str, k: int):
    """Combine FAISS semantic results with BM25 keyword results, deduplicate."""
    from rank_bm25 import BM25Okapi

    # Semantic results
    semantic_docs = vectorstore.similarity_search(question, k=k)

    # BM25 keyword results
    tokenized = [doc.page_content.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    top_bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs = [all_chunks[i] for i in top_bm25_idx]

    # Merge and deduplicate by content
    seen = set()
    merged = []
    for doc in semantic_docs + bm25_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------
def rerank(question: str, docs):
    """Rerank docs using cross-encoder, return top RERANK_TOP_N."""
    reranker = get_reranker()
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:config.RERANK_TOP_N]]


# ---------------------------------------------------------------------------
# Multi-query expansion
# ---------------------------------------------------------------------------
def expand_queries(question: str, llm) -> list[str]:
    """Generate query variants to improve retrieval recall."""
    prompt = f"""Generate {config.MULTI_QUERY_COUNT} different ways to ask this question \
for better document retrieval. Return only the questions, one per line, no numbering.

Question: {question}"""
    try:
        response = llm.invoke(prompt)
        variants = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        return [question] + variants[:config.MULTI_QUERY_COUNT]
    except Exception:
        return [question]


# ---------------------------------------------------------------------------
# Merge all loaded vectorstores into one for cross-doc search
# ---------------------------------------------------------------------------
def get_merged_vectorstore():
    stores = st.session_state.get("vectorstores", {})
    if not stores:
        return None
    stores_list = list(stores.values())
    if len(stores_list) == 1:
        return stores_list[0]
    # Deep copy the first store so we never mutate the cached original
    import copy
    merged = copy.deepcopy(stores_list[0])
    for vs in stores_list[1:]:
        merged.merge_from(copy.deepcopy(vs))
    return merged


def get_all_chunks():
    return st.session_state.get("all_chunks", [])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about uploaded documents. "
    "Use ONLY the following retrieved context to answer. If the answer isn't in "
    "the context, say you don't know — do not make it up.\n\n"
    "Context:\n{context}"
)

HISTORY_AWARE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("human", "Given the conversation above, rephrase the follow-up question "
              "as a standalone question that can be understood without the chat history."),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
init_analytics()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 DocChat")
    st.caption("Multi-PDF RAG • Hybrid Search • Reranking")

    st.header("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process documents", type="primary"):
        if "vectorstores" not in st.session_state:
            st.session_state["vectorstores"] = {}
        if "all_chunks" not in st.session_state:
            st.session_state["all_chunks"] = []

        for uf in uploaded_files:
            pdf_bytes = uf.read()
            pdf_hash = get_pdf_hash(pdf_bytes)

            if pdf_hash in st.session_state["vectorstores"]:
                st.info(f"{uf.name}: already loaded")
                continue

            cached = load_cached_vectorstore(pdf_hash)
            if cached:
                st.session_state["vectorstores"][pdf_hash] = cached
                st.session_state["analytics"]["cache_hits"] += 1
                log_event("cache_hit", file=uf.name, hash=pdf_hash)
                st.success(f"{uf.name}: loaded from cache ✓")
            else:
                with st.spinner(f"Embedding {uf.name}..."):
                    t0 = time.time()
                    try:
                        vs, n_pages, n_chunks, ocr_used = build_and_cache_vectorstore(pdf_bytes, pdf_hash, uf.name)
                    except ValueError as e:
                        st.error(str(e))
                        continue
                    elapsed = round((time.time() - t0) * 1000)
                st.session_state["vectorstores"][pdf_hash] = vs
                st.session_state["analytics"]["pdfs_uploaded"] += 1
                log_event("pdf_indexed", file=uf.name, pages=n_pages, chunks=n_chunks, ms=elapsed, ocr=ocr_used)
                label = " (OCR)" if ocr_used else ""
                st.success(f"{uf.name}{label}: {n_pages} pages → {n_chunks} chunks ✓")

            # Rebuild flat chunk list for BM25
            merged_vs = get_merged_vectorstore()
            if merged_vs:
                st.session_state["all_chunks"] = list(merged_vs.docstore._dict.values())

        st.session_state["chat_history"] = []

    # Loaded docs list
    if st.session_state.get("vectorstores"):
        st.divider()
        st.caption(f"**{len(st.session_state['vectorstores'])} document(s) loaded**")

    st.divider()
    st.header("2. Settings")
    k = st.slider("Retrieved chunks (k)", config.MIN_K, config.MAX_K, config.DEFAULT_K,
                  help="Candidates fetched before reranking")
    use_rerank = st.toggle("Cross-encoder reranking", value=True,
                           help="Reranks top-k candidates for better precision")
    use_multiquery = st.toggle("Multi-query expansion", value=True,
                               help="Generates query variants to improve recall")
    use_hybrid = st.toggle("Hybrid search (BM25 + FAISS)", value=True,
                           help="Combines keyword and semantic search")

    # Diagnostics
    st.divider()
    if st.session_state.get("last_chunks"):
        with st.expander("🔍 Last retrieval"):
            for i, doc in enumerate(st.session_state["last_chunks"]):
                src = doc.metadata.get("source_file", "?")
                pg = doc.metadata.get("page", "?")
                st.caption(f"Chunk {i+1} — {src}, p.{pg}")
                st.code(doc.page_content[:200])

# ---------------------------------------------------------------------------
# Chat input — must be outside tabs to pin to page bottom
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question across your documents...")

# ---------------------------------------------------------------------------
# Analytics tab + Chat tab
# ---------------------------------------------------------------------------
tab_chat, tab_analytics = st.tabs(["💬 Chat", "📊 Analytics"])

with tab_analytics:
    a = st.session_state["analytics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PDFs uploaded", a["pdfs_uploaded"])
    col2.metric("Questions asked", a["questions_asked"])
    col3.metric("Cache hits", a["cache_hits"])
    avg_scores = a["similarity_scores"]
    col4.metric("Avg similarity score",
                f"{sum(avg_scores)/len(avg_scores):.3f}" if avg_scores else "—")

    col5, col6 = st.columns(2)
    n = a["questions_asked"] or 1
    col5.metric("Avg retrieval time", f"{a['total_retrieval_ms']//n} ms")
    col6.metric("Avg generation time", f"{a['total_generation_ms']//n} ms")

with tab_chat:
    if not st.session_state.get("vectorstores"):
        st.info("Upload PDFs in the sidebar and click 'Process documents' to begin.")
    else:
        llm = get_llm()
        merged_vs = get_merged_vectorstore()

        # Build history-aware retriever for conversational follow-ups
        base_retriever = merged_vs.as_retriever(search_kwargs={"k": k})
        history_aware_retriever = create_history_aware_retriever(
            llm, base_retriever, HISTORY_AWARE_PROMPT
        )
        document_chain = create_stuff_documents_chain(llm, QA_PROMPT)
        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # Render chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📎 Sources"):
                        for src in msg["sources"]:
                            st.caption(f"✓ {src['file']} — Page {src['page']}")

        if question:
            st.session_state["chat_history"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                # Build LangChain message history for context-aware retrieval
                lc_history = []
                for msg in st.session_state["chat_history"][:-1]:
                    if msg["role"] == "user":
                        lc_history.append(HumanMessage(content=msg["content"]))
                    else:
                        lc_history.append(AIMessage(content=msg["content"]))

                # --- Multi-query expansion ---
                t_ret_start = time.time()
                if use_multiquery:
                    queries = expand_queries(question, llm)
                else:
                    queries = [question]

                # --- Hybrid or pure semantic retrieval across all query variants ---
                all_chunks = get_all_chunks()
                candidate_docs = []
                seen_keys = set()
                for q in queries:
                    if use_hybrid and all_chunks:
                        docs = hybrid_retrieve(merged_vs, all_chunks, q, k)
                    else:
                        docs = merged_vs.similarity_search(q, k=k)
                    for doc in docs:
                        key = doc.page_content[:100]
                        if key not in seen_keys:
                            seen_keys.add(key)
                            candidate_docs.append(doc)

                # --- Reranking ---
                if use_rerank and candidate_docs:
                    final_docs = rerank(question, candidate_docs)
                else:
                    final_docs = candidate_docs[:config.RERANK_TOP_N]

                st.session_state["last_chunks"] = final_docs
                retrieval_ms = int((time.time() - t_ret_start) * 1000)

                # Similarity score of best FAISS result for analytics
                faiss_results = merged_vs.similarity_search_with_score(question, k=1)
                best_score = faiss_results[0][1] if faiss_results else None

                # --- Stream answer ---
                t_gen_start = time.time()
                answer_placeholder = st.empty()
                full_answer = ""

                for chunk in rag_chain.stream({"input": question, "chat_history": lc_history}):
                    if "answer" in chunk:
                        full_answer += chunk["answer"]
                        answer_placeholder.markdown(full_answer + "▌")
                answer_placeholder.markdown(full_answer)
                generation_ms = int((time.time() - t_gen_start) * 1000)

                # --- Source citations ---
                sources = []
                seen_sources = set()
                for doc in final_docs:
                    f = doc.metadata.get("source_file", "unknown")
                    pg = doc.metadata.get("page", "?")
                    pg_display = pg + 1 if isinstance(pg, int) else pg
                    key = f"{f}-{pg_display}"
                    if key not in seen_sources:
                        seen_sources.add(key)
                        sources.append({"file": f, "page": pg_display})

                sources = sorted(sources, key=lambda x: (x["file"], x["page"]))
                if sources:
                    with st.expander("📎 Sources"):
                        for src in sources:
                            st.caption(f"✓ {src['file']} — Page {src['page']}")

                # --- Record analytics + log ---
                record_query(retrieval_ms, generation_ms, best_score)
                log_event("query",
                          question=question,
                          queries_generated=len(queries),
                          candidates=len(candidate_docs),
                          final_docs=len(final_docs),
                          retrieval_ms=retrieval_ms,
                          generation_ms=generation_ms,
                          best_score=round(best_score, 3) if best_score else None)

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
            })
