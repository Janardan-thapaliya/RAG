# RAG Pipelines

Hands-on notebooks for building Retrieval-Augmented Generation systems with LangChain and LangGraph — from raw document ingestion to full self-correcting pipelines.

---

## Repository Structure

```
RAG/
├── Document Loader/        # Loading documents from PDFs, CSVs, JSON, web, etc.
├── Text Splitters/         # Chunking strategies for splitting documents
├── Embeddings/             # Embedding models (OpenAI, Ollama)
└── Advanced RAG/           # Self-RAG and Corrective RAG with LangGraph
```

---

## Modules

### [`Document Loader/`](./Document%20Loader)

Explores 7 loaders: `TextLoader`, `PyPDFLoader` (plain + OCR via RapidOCR/Tesseract), `PDFMinerLoader`, `PDFPlumberLoader`, `CSVLoader`, `JSONLoader`, and `WebBaseLoader`.

→ [See README](./Document%20Loader/readme.md)

---

### [`Text Splitters/`](./Text%20Splitters)

Covers 8 splitting strategies: character-based, token-based, recursive (plain + language-aware for Python/Markdown/JSON), semantic (embedding-based), and LLM-based chunking.

→ [See README](./Text%20Splitters/readme.md)

---

### [`Embeddings/`](./Embeddings)

OpenAI embeddings (`text-embedding-3-large/small` with matryoshka truncation) and Ollama local embeddings, covering `embed_query` vs `embed_documents`.

→ [See README](./Embeddings/readme.md)

---

### [`Advanced RAG/`](./Advanced%20RAG)

Two production-grade pipelines built with LangGraph:

#### Self-RAG
Hallucination-resistant chatbot with smart retrieval routing, relevance grading, IsSUP grounding verification, answer revision loop, and IsUSE usefulness check.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janardan-thapaliya/RAG/blob/main/Advanced%20RAG/Self_RAG.ipynb)

#### Corrective RAG
Pipeline that scores retrieved docs and falls back to Tavily web search when local documents are insufficient.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janardan-thapaliya/RAG/blob/main/Advanced%20RAG/Corrective_RAG.ipynb)

→ [See README](./Advanced%20RAG/readme.md)

---

## Tech Stack

LangChain · LangGraph · OpenAI (`gpt-4o-mini`, `text-embedding-3-large`) · Ollama · FAISS · PyPDF · Pydantic · Tavily · Python

---

## Setup

```bash
pip install langchain_community langchain_openai langchain_text_splitters \
            langchain_core langgraph pypdf faiss-cpu python-dotenv
```

Create a `.env` file and add your API keys:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key   # required for Corrective RAG only
```
