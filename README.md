# 🔐🚀 ConfidRAG — Private, Local RAG Demo

A minimal Retrieval-Augmented Generation (RAG) **demo UI** built with **Streamlit**, using:

- 🧩 **LangChain** loaders/splitters
- 🧠 **Sentence-Transformers** (`all-MiniLM-L6-v2`) for embeddings (384‑dim, cosine)
- 📚 **FAISS** for a **local, on-disk** vector index (per-model folder)
- 🤖 **Ollama** for a fully local LLM (Llama/Qwen/Phi/Gemma/etc.)
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
  - `RecursiveCharacterTextSplitter` with **1000** size / **200** overlap
  - Provenance in metadata: `source_tag`, `chunk_id`, `char_len`
- **Embeddings** 🧠
  - `sentence-transformers/all-MiniLM-L6-v2` (fast on CPU)
  - Unit-normalized (cosine-ready)
- **Vector DB** 🗂️
  - **FAISS** (inner product = cosine with normalized vectors)
  - Per-model index folder: `confidrag_index/<model_id_sanitized>/`
  - Dedup by `chunk_id`, persisted to disk, auto-loaded
- **Search** 🔎
  - Query embedded with the same model
  - **Top-K** similarity with scores
  - Shows `source_tag`, `chunk_id`, and scores in UI
- **Reset** ♻️
  - **Clear All** wipes in-memory state **and** deletes the FAISS index on disk

---

## ⚡ Quickstart

**Requirements:** Python **3.10+** (Windows/macOS/Linux)

```bash
# 1) Create & activate a venv
# Windows
python -m venv .venv
.\.venv\Scriptsctivate

# macOS/Linux
# python -m venv .venv
# source .venv/bin/activate

# 2) Install Python deps
pip install -U streamlit langchain langchain-community langchain-text-splitters
pip install -U sentence-transformers torch faiss-cpu
pip install -U "unstructured[all-docs]" lxml html5lib beautifulsoup4
pip install -U ollama

# (If PyTorch wheels fail, CPU-only wheels:)
# pip install --index-url https://download.pytorch.org/whl/cpu torch

# (If faiss-cpu fails on Windows with pip, try conda:)
# conda install -c pytorch faiss-cpu
```

---

## 🤖 Set up the local LLM with **Ollama** (must be running in the background)

1) **Start the Ollama server** (keep it running):
```bash
ollama serve
```
Sanity check in a browser: open `http://127.0.0.1:11434` → should return `{"status":"OK"}`.

2) **Pull a small, CPU-friendly model** (recommended defaults):
```bash
# default used in the code
ollama pull llama3.2:1b

# good alternatives:
# ollama pull phi3:mini
# ollama pull qwen2.5:1.5b
# ollama pull tinyllama
```

3) **(Optional) Test the model**
```bash
ollama run llama3.2:1b "Say 'ready'."
```

> The app connects to `http://127.0.0.1:11434` and uses `llama3.2:1b` by default.  
> If you change the model, update `model=` in `ollama_local.py` (e.g., `"phi3:mini"`).

---

## ▶️ Run the app

```bash
streamlit run UIWindow.py
```

---

## 🧭 How to Use

1. **Load data** in the left column:
   - **“.txt file”** (path to a single file)
   - **“.txt folder”** (recursive)
   - **“url upload”** (fetch from a URL)
2. Click **Embed Chunks** to compute embeddings and index in FAISS.
3. Ask a question in **ASK** → **Send**.
4. The chat area shows **User/Bot** messages plus **Sources**, **Chunk IDs**, and **Scores**.

**Notes**
- Previews of the last document/chunk/embedding show when **Preview** is on; they auto-hide after **Send**.
- **Clear All** resets memory and deletes the on-disk FAISS index.

---

## ⚙️ Configuration & Defaults

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly, 384-dim)
- **Vector index dir**: `./confidrag_index/<model_id_sanitized>/`
- **Chunking**: size `1000`, overlap `200`
- **Search K** (default): `3` (tunable via `st.session_state.search_k`)
- **Ollama host**: `http://127.0.0.1:11434` (change with `OLLAMA_HOST` or in code)
- **LLM options** (in `ollama_local.py`):
```python
options={"num_ctx": 2048, "temperature": 0.2}
# You can also cap output tokens:
# options={"num_ctx": 2048, "temperature": 0.2, "num_predict": 256}
```

---

## 🛠️ Troubleshooting

- **`ConnectionError: Failed to connect to Ollama`**  
  Start the server: `ollama serve` (keep it running).  
  Check `http://127.0.0.1:11434` returns JSON OK.

- **`model "llama3.2:1b" not found, try pulling it first (404)`**  
  Run `ollama pull llama3.2:1b` (or whichever model you set).

- **Very slow answers on CPU**  
  Use **smaller models** (`llama3.2:1b`, `phi3:mini`, `qwen2.5:1.5b`, `tinyllama`),  
  reduce **Top‑K** (2–3 chunks), shorten sent text (keep only relevant paragraphs),  
  lower `num_ctx` (e.g., 1024–2048).

- **Windows + older AMD GPUs (e.g., RX 560X)**  
  Expect **CPU mode**. The app works well with small models; GPU acceleration isn’t required.

---

## 🧩 Project Layout (key files)

```
ConfidRAG/
├─ UIWindow.py           # Streamlit UI (ingest, chunk, embed, index, search, chat)
├─ ollama_local.py       # Ollama client; builds prompt from top-K chunks and asks the model
├─ confidrag_index/      # FAISS indices (auto-created, per embedding model)
└─ DataSet/              # (optional) example .txt files
```

---

## 🔒 Privacy

- All retrieval, embedding, vector search, and generation are local.
- No document content leaves your machine (aside from first-time model downloads).
