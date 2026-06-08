# Advanced RAG

> **Pipeline position:** Documents → Chunks → Embeddings → Vector Store → **[Advanced RAG Pipelines]** → Answer

This folder contains production-grade RAG pipelines built with LangChain and LangGraph. Both implement multi-step self-correction loops that go well beyond basic retrieve-and-generate.

---

## Notebooks

### `Self_RAG.ipynb`

A hallucination-resistant RAG chatbot over internal company PDFs.

**Pipeline stages:**

| Stage | What it does |
|---|---|
| `decide_retrieval` | Decides whether retrieval is even needed for the question |
| `retrieve` | Fetches top-k chunks from FAISS using the (possibly rewritten) query |
| `is_relevant` | Per-document relevance filter — drops chunks unrelated to the topic |
| `generate_from_context` | Generates an answer grounded in the filtered context |
| `is_sup` (IsSUP) | Verifies every claim in the answer is explicitly supported by context |
| `revise_answer` | Rewrites the answer using only direct quotes if partially/unsupported |
| `is_use` (IsUSE) | Checks whether the answer actually resolves the user's question |
| `rewrite_question` | Reformulates the retrieval query and retries if the answer is not useful |

**Key parameters:**
- `chunk_size=600`, `chunk_overlap=150`
- `k=4` retrieved docs
- `MAX_RETRIES = 10` (IsSUP revision loop)
- `MAX_REWRITE_TRIES = 3` (IsUSE rewrite loop)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janardan-thapaliya/RAG/blob/main/Advanced%20RAG/Self_RAG.ipynb)

<img width="742" height="857" alt="image" src="https://github.com/user-attachments/assets/5adda9de-9a01-4133-9011-d704c93798cc" />

---

### `Corrective_RAG.ipynb`

A RAG pipeline that self-evaluates retrieval quality and falls back to web search (Tavily) when local documents fall short.

**Pipeline stages:**

| Stage | What it does |
|---|---|
| `retrieve` | Fetches top-k chunks from FAISS |
| `eval_each_doc` | Scores each doc `[0.0–1.0]`; classifies overall as `CORRECT`, `AMBIGUOUS`, or `INCORRECT` |
| `rewrite_query` | Rewrites question into a keyword-optimised web search query |
| `web_search` | Runs a Tavily search and returns results as `Document` objects |
| `refine` | Decomposes context into sentences and filters to only relevant ones |
| `generate` | Generates final answer from the refined context |

**Routing logic:**
- `CORRECT` → skip web search, go straight to `refine`
- `INCORRECT` / `AMBIGUOUS` → `rewrite_query` → `web_search` → `refine`

**Key thresholds:**
- `UPPER_TH = 0.7` — doc scores above this → CORRECT
- `LOWER_TH = 0.3` — all docs below this → INCORRECT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janardan-thapaliya/RAG/blob/main/Advanced%20RAG/Corrective_RAG.ipynb)

<img width="210" height="736" alt="image" src="https://github.com/user-attachments/assets/90dbfae7-7098-4a81-8d2c-3a9401861c11" />

---

## Setup

```bash
pip install langchain_community langchain_openai langchain_text_splitters \
            langchain_core langgraph pypdf faiss-cpu python-dotenv
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key   # required for Corrective RAG only
```

Upload your PDF files to `/content/` (Colab) or adjust the paths in the notebook before running.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain_community` | `PyPDFLoader`, `FAISS`, `TavilySearchResults` |
| `langchain_openai` | `ChatOpenAI`, `OpenAIEmbeddings` |
| `langchain_core` | `Document`, `ChatPromptTemplate` |
| `langgraph` | `StateGraph`, conditional routing |
| `pydantic` | Structured LLM output schemas |
| `faiss-cpu` | Vector similarity search |
| `python-dotenv` | API key management |
