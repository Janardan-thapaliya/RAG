# Text Splitters

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janardan-thapaliya/RAG/blob/main/Text%20Splitters/Text_Splitters.ipynb)

Text Splitters are the **second step in a RAG pipeline** — they break long documents (loaded in the previous step) into smaller, overlapping chunks that fit within an embedding model's context window while preserving enough context for meaningful retrieval.

## Splitters Covered

| Splitter | Class | Best For |
|---|---|---|
| **Character** | `CharacterTextSplitter` | Simple, fast splitting by a chosen separator (e.g. space, newline) |
| **Token-based** | `CharacterTextSplitter.from_tiktoken_encoder` | Splitting by token count (e.g. `cl100k_base` for GPT-4 models) rather than characters |
| **Recursive Character** | `RecursiveCharacterTextSplitter` | General-purpose text — tries `\n\n` → `\n` → ` ` → `""` progressively to keep paragraphs/sentences intact |
| **Language-aware** | `RecursiveCharacterTextSplitter.from_language` | Source code — uses language-specific separators (Python class/def boundaries, Markdown headers, etc.) |
| **JSON** | `RecursiveJsonSplitter` | Nested JSON objects — splits along JSON hierarchy to stay under `max_chunk_size` |
| **Markdown Header** | `MarkdownHeaderTextSplitter` | Markdown documents — splits at heading boundaries and propagates header hierarchy into metadata |
| **Semantic** | `SemanticChunker` (langchain-experimental) | Groups sentences by embedding similarity; boundaries are placed where meaning shifts |
| **LLM-based** | Custom chain with `ChatOpenAI` + structured output | Uses an LLM to identify natural topic boundaries and simultaneously generate chunk summaries |

## Key Parameters

- **`chunk_size`** — maximum characters (or tokens) per chunk.
- **`chunk_overlap`** — number of characters/tokens shared between consecutive chunks to prevent context loss at boundaries.
- **`separator`** — the string used to split text in `CharacterTextSplitter`.
- **`breakpoint_threshold_type`** — controls how aggressively `SemanticChunker` splits (`percentile`, `standard_deviation`, `interquartile`, `gradient`).

## Methods

Both `split_text(text: str)` and `split_documents(docs: list[Document])` are supported. The latter preserves the original document's metadata in every chunk.

## Dependencies

```bash
pip install langchain langchain-core langchain-text-splitters \
            langchain-experimental langchain-openai tiktoken python-dotenv
```

## Position in RAG Pipeline

```
Documents  →  [Text Splitter]  →  Chunks  →  Embeddings  →  Vector Store
```
