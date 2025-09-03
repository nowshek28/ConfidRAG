import streamlit as st
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os, json
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import shutil
import ollama_local 


st.set_page_config(page_title="Nested Layout", layout="wide")

# ---------- Session State ----------
for k, v in {
    "list_items": [],
    "chat_item": [],
    "Input_value": "",
    "ask_value": "",
    "docs": [],
    "split_docs": [],
    "chunk_auto_id": 0,
    "emb_store": {},
    "emb_dim":0,
    "Preview": True,
    "search_k": 5,
    "search_results": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Correct arg names (lowercase)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # easy to swap later
INDEX_ROOT = "./confidrag_index"

# ---------- Functions ----------
def _model_dir(model_id: str) -> str:
    safe = model_id.replace("/", "__")
    return os.path.join(INDEX_ROOT, safe)

def add_item(prefix, value):
    value = (value or "").strip()
    if value:
        st.session_state.list_items.append(f"{prefix}: {value}")
    return value

def add_chat(prefix, value):
    if value:
        st.session_state.chat_item.append(f"{prefix}: {value}")

def ingest_and_chunk(new_docs, source_tag=None):
    if not new_docs:
        return 0

    # Optional provenance on raw docs (inherited by chunks)
    if source_tag:
        for d in new_docs:
            d.metadata = {**d.metadata, "source_tag": source_tag}

    # Chunk
    chunks = text_splitter.split_documents(new_docs)

    # Assign stable metadata
    base = st.session_state.chunk_auto_id
    for i, c in enumerate(chunks):
        c.metadata = {
            **c.metadata,
            "chunk_id": base + i,
            "char_len": len(c.page_content),
        }

    # Persist
    st.session_state.chunk_auto_id += len(chunks)
    st.session_state.split_docs.extend(chunks)
    return len(chunks)

def get_embedder(model_id: str = MODEL_ID) -> HuggingFaceEmbeddings:
    # normalize_embeddings=True → cosine-ready (inner product on unit vectors)
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": "cpu"},               # use "cuda" if you have it
        encode_kwargs={"normalize_embeddings": True},
    )

def embed_new_chunks_from_state():
    """
    Embed only chunks in st.session_state.split_docs that are not yet embedded.
    Stores embeddings in st.session_state.emb_store keyed by chunk_id (str).
    Returns (added_count, total_embedded).
    """
    chunks = st.session_state.split_docs
    if not chunks:
        return 0, len(st.session_state.emb_store)

    # collect texts for any chunk_id not already embedded
    to_embed_texts, to_embed_ids = [], []
    for d in chunks:
        cid = str(d.metadata.get("chunk_id"))
        if cid not in st.session_state.emb_store:
            to_embed_texts.append(d.page_content)
            to_embed_ids.append(cid)

    if not to_embed_texts:
        return 0, len(st.session_state.emb_store)

    embedder = get_embedder()
    vectors = embedder.embed_documents(to_embed_texts)  
    if vectors:
        st.session_state.emb_dim = len(vectors[0])

    for cid, vec in zip(to_embed_ids, vectors):
        st.session_state.emb_store[cid] = vec

    return len(to_embed_ids), len(st.session_state.emb_store)

@st.cache_resource(show_spinner=False)
def get_vectordb(model_id: str = MODEL_ID) -> FAISS:
    """
    Load existing FAISS index for this model, or create a truly empty one.
    Cached so it survives Streamlit reruns.
    """
    embedder = get_embedder(model_id)
    index_dir = _model_dir(model_id)  
    os.makedirs(index_dir, exist_ok=True)

    # Try to load a previously saved index
    try:
        return FAISS.load_local(
            index_dir,
            embedder,
            allow_dangerous_deserialization=True
        )
    except Exception:
        # Bootstrap an empty FAISS index with correct dim
        # Derive the dimension by embedding a tiny dummy string
        dim = len(embedder.embed_query("bootstrap"))
        index = faiss.IndexFlatIP(dim)  # cosine via inner product on unit-norm vectors
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}

        return FAISS(
            embedding_function=embedder,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
    
def persist_vectordb(vs: FAISS, model_id: str = MODEL_ID):
    vs.save_local(_model_dir(model_id))

def upsert_to_vectordb(chunks, model_id: str = MODEL_ID) -> int:
    """
    Add chunk Documents to FAISS (dedup by chunk_id). Returns count added.
    """
    if not chunks:
        return 0
    vs = get_vectordb(model_id)

    # Dedup: avoid re-adding the same chunk IDs
    try:
        existing_ids = set(vs.docstore._dict.keys())  # internal but fine for POC
    except Exception:
        existing_ids = set()

    docs_to_add, ids_to_add = [], []
    for d in chunks:
        cid = str(d.metadata.get("chunk_id"))
        if cid and cid not in existing_ids:
            docs_to_add.append(d)
            ids_to_add.append(cid)

    if not docs_to_add:
        return 0

    vs.add_documents(docs_to_add, ids=ids_to_add)  # FAISS embeds internally with your HF embedder
    persist_vectordb(vs, model_id)
    return len(docs_to_add)

def search_vectordb(query: str, k: int = 5):
    """Return [(Document, score), ...] or [] if index empty/failed."""
    q = (query or "").strip()
    if not q:
        st.warning("Type a question before searching.")
        return []

    try:
        vs = get_vectordb(MODEL_ID)  # uses the same embedder internally
    except Exception as e:
        st.error(f"Vector DB not ready: {e}")
        return []

    # If index is empty, bail early
    try:
        if not getattr(vs, "docstore", None) or not getattr(vs.docstore, "_dict", {}):
            st.info("Index is empty. Ingest & index chunks first.")
            return []
    except Exception:
        pass

    try:
        return vs.similarity_search_with_score(q, k=k)
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def clear_vectordb(model_id: str = MODEL_ID):
    # 1) drop the cached FAISS object
    try:
        get_vectordb.clear()
    except Exception:
        pass
    # 2) delete the saved index folder
    dirpath = _model_dir(model_id)
    try:
        shutil.rmtree(dirpath)
    except FileNotFoundError:
        pass
    # 3) (optional) re-bootstrap an empty index so future calls work immediately
    _ = get_vectordb(model_id)

# ---------- LAYOUT ----------
st.title("ConfidRAG")
st.divider()
col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.session_state.Input_value = st.text_input("Enter Here:", value=st.session_state.Input_value)

    # .txt file
    if st.button(".txt file"):
        path = add_item("txt_file", st.session_state.Input_value)
        if not path:
            st.warning("Please enter a file path.")
        else:
            try:
                with st.spinner("Loading file..."):
                    new_docs = TextLoader(path, encoding="utf-8").load()
                st.session_state.docs.extend(new_docs)
                st.success(f"Loaded {len(new_docs)} document(s). Total: {len(st.session_state.docs)}")
                n = ingest_and_chunk(new_docs, source_tag=path)
                st.success(f"Added {n} chunk(s). Total: {len(st.session_state.split_docs)}")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
        st.session_state.Input_value = ""

    # .txt folder
    if st.button(".txt folder"):
        folder = add_item("txt_folder", st.session_state.Input_value)
        if not folder:
            st.warning("Please enter a folder path.")
        else:
            try:
                with st.spinner("Loading folder..."):
                    loader = DirectoryLoader(
                        folder,
                        glob="**/*.txt",
                        loader_cls=TextLoader,
                        loader_kwargs={"encoding": "utf-8"},
                        show_progress=True,
                        use_multithreading=True,
                        recursive=True,
                    )
                new_docs = loader.load()
                st.session_state.docs.extend(new_docs)
                st.success(f"Loaded {len(new_docs)} document(s). Total: {len(st.session_state.docs)}")
                n = ingest_and_chunk(new_docs, source_tag=folder)
                st.success(f"Added {n} chunk(s). Total: {len(st.session_state.split_docs)}")
            except Exception as e:
                st.error(f"Failed to load folder: {e}")
        st.session_state.Input_value = ""

    # URL upload (single URL)
    if st.button("url upload"):
        url = add_item("url_upload", st.session_state.Input_value)
        if not url:
            st.warning("Please enter a URL.")
        else:
            try:
                with st.spinner("Loading Url..."):
                    loader = UnstructuredURLLoader(urls=[url])
                new_docs = loader.load()
                st.session_state.docs.extend(new_docs)
                st.success(f"Loaded {len(new_docs)} document(s). Total: {len(st.session_state.docs)}")
                n = ingest_and_chunk(new_docs, source_tag=url)
                st.success(f"Added {n} chunk(s). Total: {len(st.session_state.split_docs)}")
            except Exception as e:
                st.error(f"Failed to load URL: {e}")
        st.session_state.Input_value = ""

    if st.button("Clear All"):
        st.session_state.docs.clear()
        st.session_state.split_docs.clear()
        st.session_state.chunk_auto_id = 0
        st.session_state.list_items.clear()
        st.session_state.chat_item.clear()
        st.session_state.emb_store.clear()      
        st.session_state.emb_dim = 0 
        st.session_state.Preview = True
        st.session_state.search_results = []
        clear_vectordb(MODEL_ID)  
        st.success("Cleared all data.")

    
    if st.button("Embed Chunks"):
        if not st.session_state.split_docs:
            st.warning("No chunks to embed. Ingest a file/folder/URL first.")
        else:
            with st.spinner("Embedding chunks..."):
                added, total = embed_new_chunks_from_state()
            if added == 0:
                st.info(f"No new chunks to embed. Already have {total} embedded "
                        f"(dim={st.session_state.emb_dim}).")
            else:
                st.success(f"Embedded {added} new chunk(s). Total embedded: {total} "
                        f"(dim={st.session_state.emb_dim}).")
                
            if not st.session_state.split_docs:
                st.warning("No chunks to index. Ingest and chunk first.")
            else:
                with st.spinner("Indexing chunks into FAISS..."):
                    added = upsert_to_vectordb(st.session_state.split_docs, model_id=MODEL_ID)
                if added == 0:
                    st.info("No new chunks to index (deduped by chunk_id).")
                else:
                    st.success(f"Indexed {added} new chunk(s) to FAISS.")

    st.subheader("Uploads")
    if st.session_state.list_items:
        st.write(st.session_state.list_items)
    else:
        st.caption("No data uploaded yet...")

with col2:
    top_area = st.container()     # removed invalid args
    bottom_area = st.container()

    with top_area:
        st.subheader("Chat Area")
        chat_window = st.container()
        with chat_window:
            if st.session_state.chat_item:
                st.write(st.session_state.chat_item[-1::-2])  # reverse order
            else:
                st.caption("No messages yet. Type below to start chatting.")

           
            if st.session_state.docs and st.session_state.Preview:
                st.markdown("**Preview of last updated document**")
                doc0 = st.session_state.docs[-1]
                st.write(doc0.metadata)
                st.code(doc0.page_content[:100])
                
            if st.session_state.split_docs and st.session_state.Preview:
                st.markdown("**Preview of last chunk**")
                c = st.session_state.split_docs[-1]
                st.write({"chunk_id": c.metadata.get("chunk_id"),
                          "source_tag": c.metadata.get("source_tag"),
                          "char_len": c.metadata.get("char_len")})
                st.code(c.page_content[:100])

            if st.session_state.emb_store and st.session_state.Preview:
                st.markdown("**Preview of last embedding**")
                last_id = list(st.session_state.emb_store.keys())[-1]
                vec = st.session_state.emb_store[last_id]
                st.write({"chunk_id": last_id, "dim": st.session_state.emb_dim})
                st.code(str(vec[:8]) + (" ..."))


    with bottom_area:
        leftcol, rightcol = st.columns([6, 1], vertical_alignment="bottom")
        with leftcol:
            st.subheader("ASK")
            st.session_state.ask_value = st.text_input(
                "Enter question here",
                value=st.session_state.ask_value,
                key="ask"
            )
        with rightcol:
            if st.button("Send"):
                st.session_state.Preview = False
                question = (st.session_state.ask_value or "").strip()
                if question:
                    add_chat("User", question)
                    with st.spinner("Searching knowledge base..."):
                        st.session_state.search_results = search_vectordb(question, k=st.session_state.search_k)
                        Bot_answer = ollama_local.list_to_string_with_ollama(st.session_state.search_results, question)
                        add_chat("Bot", st.session_state.search_results)
                else:
                    st.warning("Please enter a question.")
                st.session_state.ask_value = ""
