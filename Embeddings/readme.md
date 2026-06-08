# Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janardan-thapaliya/RAG/blob/main/Embeddings/Embeddings.ipynb)

Embeddings are the **third step in a RAG pipeline** — they convert text chunks (produced by the Text Splitter) into dense numerical vectors that capture semantic meaning, enabling similarity-based retrieval from a vector store.

## What's Covered

### OpenAI Embeddings

Uses `OpenAIEmbeddings` from `langchain-openai` with two model variants:

| Model | Dimensions (default) | Notes |
|---|---|---|
| `text-embedding-3-large` | 3072 | Higher accuracy, larger vectors |
| `text-embedding-3-small` | 1536 | Faster and cheaper |

Both models support **custom dimensions** via the `dimensions` parameter (e.g. `dimensions=256`), which truncates the output vector using matryoshka representation learning — useful for reducing storage and compute costs without fully retraining.

### Ollama (Local) Embeddings

Uses `embeddinggemma` pulled locally via Ollama — runs fully **on-device** with no API key required. Two usage patterns are shown:

- **Native Ollama SDK** (`ollama.embed`) — returns raw embedding vectors directly; the `dimensions` parameter is also supported for truncation.
- **LangChain integration** (`OllamaEmbeddings`) — wraps the Ollama model in a LangChain-compatible interface, required when building LangChain or LangGraph pipelines.

## Core Methods

| Method | Input | Output |
|---|---|---|
| `embed_query(text)` | Single string | One vector (list of floats) |
| `embed_documents(texts)` | List of strings | List of vectors |

## End-to-End Workflow Demonstrated

1. Load a PDF with `PyPDFLoader`
2. Split into chunks with `RecursiveCharacterTextSplitter` (`chunk_size=300`, `chunk_overlap=50`)
3. Extract `page_content` strings from each chunk
4. Embed all chunks with `embed_documents()`
5. Inspect embedding dimensions and vector values

## Dependencies

```bash
pip install langchain langchain-core langchain-openai langchain-text-splitters \
            langchain-community langchain-ollama python-dotenv pypdf ollama
```

> **Note:** Running Ollama embeddings in Google Colab requires installing the Ollama daemon and pulling the model at runtime (see notebook setup cells).

## Position in RAG Pipeline

```
Chunks  →  [Embedding Model]  →  Vectors  →  Vector Store  →  Retrieval
```
