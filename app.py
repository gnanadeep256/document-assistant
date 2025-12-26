import shutil
from pathlib import Path
import streamlit as st
import json

from ingest.pdf_loader import extract_pdf_pages
from preprocessing.chunker import chunk_pages
from indexing.index_chunks import main as index_chunks_main

from retrieval.retriever import Retriever
from retrieval.query_intent import detect_intent, QueryIntent
from retrieval.aggregation import aggregate_section, aggregate_global

from llm.offline_ollama import OfflineLLM
from llm.online_gemini import OnlineGeminiLLM

# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VECTOR_DIR = Path("data/vector_db")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PAGES_FILE = PROCESSED_DIR / "pages.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

# Streamlit config
st.set_page_config(
    page_title="Document Intelligence Copilot",
    layout="wide",
)

st.title("Document Intelligence Copilot")
st.caption("Chat with your documents — Non-LLM | Offline LLaMA | Online Gemini")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Inference mode",
        [
            "Retrieval only (Non-LLM)",
            "Offline LLM (LLaMA-3 via Ollama)",
            "Online LLM (Gemini)",
        ],
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if st.button("Reset chat"):
        st.session_state.chat = []
        st.session_state.indexed = False
        st.rerun()

# Session state
if "chat" not in st.session_state:
    st.session_state.chat = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# PDF ingestion & indexing 

if uploaded_file and not st.session_state.indexed:
    with st.spinner("Indexing document… This may take a minute."):

        # Clear previous data
        shutil.rmtree(RAW_DIR, ignore_errors=True)
        shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
        shutil.rmtree(VECTOR_DIR, ignore_errors=True)

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Save uploaded PDF
        pdf_path = RAW_DIR / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # ---- PDF → pages.json ----
        pages = extract_pdf_pages(pdf_path)
        with open(PAGES_FILE, "w", encoding="utf-8") as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)

        # ---- pages → chunks.json ----
        chunks = chunk_pages(pages)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        # ---- chunks → vector DB ----
        index_chunks_main()

        st.session_state.indexed = True

    st.success("Document indexed successfully. Ask away!")

# Load retriever & LLM
retriever = Retriever()
llm = None

if mode == "Offline LLM (LLaMA-3 via Ollama)":
    llm = OfflineLLM()
elif mode == "Online LLM (Gemini)":
    llm = OnlineGeminiLLM()

# Render chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask a question about the document…")

if query:
    if not st.session_state.indexed:
        st.warning("Please upload and index a PDF first.")
    else:
        st.session_state.chat.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        intent = detect_intent(query)
        results = retriever.search(query, k=25)

        if not results:
            answer = "No relevant content found in the document."
        else:
            if intent == QueryIntent.SECTION:
                section_id = "".join(c for c in query if c.isdigit() or c == ".").strip(".")
                agg = aggregate_section(results, section_id)
                context = agg["text"]

                if not context.strip():
                    answer = "No content found for that section."
                elif llm:
                    with st.spinner("Thinking…"):
                        answer = llm.answer(query, context, intent.name)
                else:
                    answer = context
            else:
                agg = aggregate_global(results)
                context = agg["text"]

                if llm:
                    with st.spinner("Thinking…"):
                        answer = llm.answer(query, context, intent.name)
                else:
                    answer = context

        st.session_state.chat.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
