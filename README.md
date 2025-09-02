# 🔐🚀 ConfidRAG — Private, Local RAG Demo

A minimal Retrieval-Augmented Generation (RAG) **demo UI** built with **Streamlit**, using:
- 🧩 **LangChain** loaders/splitters
- 🧠 **Sentence-Transformers** (`all-MiniLM-L6-v2`) for embeddings (384-dim, cosine)
- 📚 **FAISS** for a **local, on-disk** vector index (per-model folder)
- 💬 Simple chat UI + previews

> **Why?** Ingest `.txt` files/folders/URLs → **chunk** → **embed** → **index** → **search** locally.  
> No external services needed after the first model download.

---

## ✨ Features

- **Ingestion** 📥
  - Single **.txt file** by path
  - **Folder** of `.txt` (recursive via `**/*.txt`)
  - Single **URL** (`UnstructuredURLLoader`)
- **Chunking** ✂️
  - `RecursiveCharacterTextSplitter` with **1000 chars** size and **200 overlap**
  - Provenance in metadata: `source_tag`, `chunk_id`, `char_len`
- **Embeddings** 🧠
  - `sentence-transformers/all-MiniLM-L6-v2` (fast on CPU)
  - Unit-normalized (cosine-ready)
- **Vector DB** 🗂️
  - **FAISS** (inner product = cosine with normalized vectors)
  - Per-model index folder: `confidrag_index/<model_id_sanitized>/`
  - Dedup by `chunk_id`
  - Persisted to disk, auto-loaded on app start
- **Search** 🔎
  - Query embedded with the same model
  - **Top-K** similarity with scores
  - Shows snippet, `source_tag`, `chunk_id`
  - **Previews auto-hide** when you click **Send**
- **Reset** ♻️
  - **Clear All** wipes in-memory state **and** deletes the FAISS index on disk

---

## ⚡ Quickstart

**Requirements:** Python **3.10+** recommended (Windows/macOS/Linux).

```bash
# 1) Create & activate a venv
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
# python -m venv .venv
# source .venv/bin/activate

# 2) Install dependencies
pip install -U streamlit langchain langchain-community langchain-text-splitters
pip install -U sentence-transformers torch faiss-cpu
pip install -U "unstructured[all-docs]" lxml html5lib beautifulsoup4

# (If PyTorch wheels fail, CPU-only wheels:)
# pip install --index-url https://download.pytorch.org/whl/cpu torch

# (If faiss-cpu fails on Windows with pip, try conda:)
# conda install -c pytorch faiss-cpu

# 3) Run the app
streamlit run UIWindow.py


## 🧭 How to Use (UI Flow)

- **Ingest 📥**
- Enter a path or URL in the input box. Click “.txt file”, “.txt folder”, or “url upload”.
- You’ll see document and chunk previews.
- **Embed + Index 🧠➡️🗂️**
- Click “Embed Chunks” (embeds new chunks for diagnostics) and indexes chunks into FAISS
- (deduped by chunk_id).
- **Ask ❓**
- Type a question and click Send. The app embeds your question, retrieves Top-K (default 5),
- and shows scored hits. Previews are hidden after Send for a cleaner chat area.
- **Reset ♻️**
- Click “Clear All” to wipe in-memory state and delete the FAISS index on disk.

