# pip install fastapi uvicorn langchain langchain-community langchain-text-splitters chromadb
# Choose an embedding model you like:
# pip install sentence-transformers   # for HF example below

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import asyncio
import os

from langchain_community.document_loaders import TextLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

PERSIST_DIR = "./vectordb"  # Chroma persists here

# ---------- App setup ----------
app = FastAPI()
app.state.lock = asyncio.Lock()
app.state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
app.state.embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
app.state.vs = None  # filled on startup


@app.on_event("startup")
async def startup():
    # Load existing persistent DB or create new
    if os.path.isdir(PERSIST_DIR):
        app.state.vs = Chroma(
            embedding_function=app.state.embed,
            persist_directory=PERSIST_DIR
        )
    else:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        app.state.vs = Chroma(
            embedding_function=app.state.embed,
            persist_directory=PERSIST_DIR
        )


# ---------- Models ----------
class UrlsIn(BaseModel):
    urls: List[str]


class QueryIn(BaseModel):
    question: str
    k: int = 4


# ---------- Helpers ----------
def _chunk(docs: List[Document], splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    return splitter.split_documents(docs)


async def _add_documents(docs: List[Document]):
    # Add with lock to serialize writes
    async with app.state.lock:
        app.state.vs.add_documents(docs)
        app.state.vs.persist()  # flush to disk


# ---------- Endpoints ----------
@app.post("/ingest/textfile")
async def ingest_text_file(file: UploadFile = File(...)):
    # Save to a temp path (you can also stream-read then delete)
    tmp_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # Load -> chunk -> add
    docs = TextLoader(tmp_path, encoding="utf-8").load()
    chunks = _chunk(docs, app.state.splitter)
    await _add_documents(chunks)

    return {"status": "ok", "file": file.filename, "chunks_added": len(chunks)}


@app.post("/ingest/urls")
async def ingest_urls(payload: UrlsIn):
    url_docs = UnstructuredURLLoader(urls=payload.urls).load()
    chunks = _chunk(url_docs, app.state.splitter)

    # Tag source to allow future deletions by filter
    for d in chunks:
        d.metadata["ingest_source"] = "url"
        d.metadata.setdefault("source", d.metadata.get("source", ""))
    await _add_documents(chunks)

    return {"status": "ok", "urls": payload.urls, "chunks_added": len(chunks)}


@app.post("/ask")
async def ask(payload: QueryIn):
    retriever = app.state.vs.as_retriever(search_kwargs={"k": payload.k})
    docs = retriever.get_relevant_documents(payload.question)
    # Here you would pass docs into your LLM; for demo we just return sources/snippets
    return {
        "matches": [
            {
                "source": d.metadata.get("source", ""),
                "start_index": d.metadata.get("start_index"),
                "snippet": d.page_content[:300]
            }
            for d in docs
        ]
    }
