# Advanced RAG Pipelines

Production-style Retrieval-Augmented Generation systems built with LangChain 
and LangGraph — from basic retrieval to full self-correcting pipelines.

## What's Inside

### Self-RAG
A hallucination-resistant RAG chatbot over internal company PDFs with:
- **Smart retrieval routing** — decides if retrieval is even needed
- **Document relevance grading** — filters irrelevant chunks per query
- **IsSUP verification loop** — checks if answer is grounded in context
- **Answer revision loop** — rewrites using only direct quotes if unsupported
- **IsUSE check** — validates the answer actually addresses the user's question
- **Query rewrite loop** — reformulates query and retries if answer is not useful
- 
  ## Run in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dXLO7ikZdNL-Bfa3rN5MDCQmvcoK7EJ2)

<img width="792" height="926" alt="image" src="https://github.com/user-attachments/assets/cfd3c547-e338-41e3-a918-7ef859f1de6a" />

### Corrective RAG
Advanced retrieval correction with web search fallback on retrieval failure.

## Run in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z93GT8q6oI7iUw2Z0_yI1nk7Ncy-2aWD)

<img width="218" height="801" alt="image" src="https://github.com/user-attachments/assets/bd2878ec-c663-4896-833b-63f60e41d0a6" />

### Supporting Modules
- `Embeddings/` — embedding model experiments
- `Document Loader/` — multi-source document ingestion
- `Text Splitters/` — chunking strategy comparisons

## Tech Stack
LangChain · LangGraph · OpenAI (gpt-4o-mini, text-embedding-3-large) · 
FAISS · PyPDF · Pydantic · Python

## Key Concepts
- Structured LLM output with Pydantic schemas
- LangGraph StateGraph with conditional routing
- Multi-loop self-correction (IsSUP + IsUSE + query rewrite)
- Hallucination detection and grounding enforcement

## Setup
```bash
pip install langchain_community langchain_openai langchain_text_splitters \
            langchain_core langgraph pypdf faiss-cpu python-dotenv
```
Add your `OPENAI_API_KEY` to a `.env` file and provide PDF documents.
