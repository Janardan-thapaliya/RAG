# Document Loader

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Janardan-thapaliya/RAG/blob/main/Document%20Loader/Document_Loader.ipynb)

Document Loaders are the **first step in a RAG pipeline** — they read raw source data from various formats and convert it into LangChain `Document` objects, each containing `page_content` and `metadata`.

## What's Covered

| Loader | Class | Key Behaviour |
|---|---|---|
| **Text** | `TextLoader` | Loads entire `.txt` file as a single `Document` |
| **PDF (PyPDF)** | `PyPDFLoader` | Loads PDFs page-by-page; can extract images via OCR (`RapidOCRBlobParser` / `TesseractBlobParser`) |
| **PDF (Miner)** | `PDFMinerLoader` | Better layout preservation and image/table extraction than PyPDF |
| **PDF (Plumber)** | `PDFPlumberLoader` | Produces rich metadata (bounding boxes, fonts, etc.) per page |
| **CSV** | `CSVLoader` | Each row becomes one `Document`; supports `source_column`, `metadata_columns`, and `content_columns` |
| **JSON** | `JSONLoader` | Uses `jq_schema` to target nested fields; `metadata_func` lets you enrich metadata from record values |
| **Web** | `WebBaseLoader` | Fetches and parses one or more URLs using BeautifulSoup; each URL becomes one `Document` |

## Key Concepts

- **`page_content`** — the extracted text that gets embedded and retrieved.
- **`metadata`** — structured information attached to each document (source path, page number, author, etc.) used for filtering during retrieval.
- **Loading mode** — PDF loaders support `mode='page'` to split by page, giving you fine-grained control over chunk granularity.
- **Image extraction** — `PyPDFLoader` and `PDFMinerLoader` support `extract_images=True` combined with an OCR parser to pull text from embedded figures and diagrams.

## Dependencies

```bash
pip install langchain langchain_community pypdf rapidocr-onnxruntime \
            pytesseract pdfplumber pdfminer.six jq beautifulsoup4 lxml
```

## Position in RAG Pipeline

```
Raw Files / URLs  →  [Document Loader]  →  Documents  →  Text Splitter  →  Embeddings  →  Vector Store
```
