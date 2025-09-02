import streamlit as st
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Correct arg names (lowercase)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# ---------- Functions ----------
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

# ---------- LAYOUT ----------
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
                for msg in st.session_state.chat_item:
                    st.write(msg)
            else:
                st.caption("No messages yet. Type below to start chatting.")

           
            if st.session_state.docs:
                st.markdown("**Preview of last updated document**")
                doc0 = st.session_state.docs[-1]
                st.write(doc0.metadata)
                st.code(doc0.page_content[:400])
                
            if st.session_state.split_docs:
                st.markdown("**Preview of last chunk**")
                c = st.session_state.split_docs[-1]
                st.write({"chunk_id": c.metadata.get("chunk_id"),
                          "source_tag": c.metadata.get("source_tag"),
                          "char_len": c.metadata.get("char_len")})
                st.code(c.page_content[:300])

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
                add_chat("User", st.session_state.ask_value)
                st.session_state.ask_value = ""
