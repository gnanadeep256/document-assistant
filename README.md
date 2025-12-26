# Document Assistant

Document Assistant is a local-first PDF question-answering system that lets you upload documents and chat with them using intelligent retrieval and optional LLM reasoning.

It supports **three modes**:

1. Retrieval-only (no LLM)
2. Offline LLMs (LLaMA via Ollama)
3. Online LLMs (Google Gemini)

The system is designed to work **fully offline** by default, with online models used only when explicitly selected.

---

## Key Features

- Upload any PDF and ask natural language questions
- Section-aware document chunking
- Semantic search using vector embeddings
- Chat-style interface (Streamlit)
- Intent detection for:
  - Definitions
  - Explanations
  - Section-specific queries
  - Comparisons
  - “Why” questions
  - Paper summary / contribution
- Multiple inference backends:
  - Non-LLM retrieval
  - Offline LLaMA (via Ollama)
  - Online Gemini

---

## How It Works

Conceptual pipeline:

1. PDF upload  
2. Text extraction (PyMuPDF)  
3. Section-aware chunking  
4. Embedding generation (Sentence Transformers)  
5. Vector storage (Chroma)  
6. Retrieval  
7. Optional LLM reasoning  
8. Answer generation

---

## Setup & Usage

For installation, configuration, and how to run the app, see:

- [`SETUP.pdf`]

---

## Privacy

- Documents are processed locally
- No data is uploaded in offline mode
- In online mode, only the selected context is sent to Gemini
