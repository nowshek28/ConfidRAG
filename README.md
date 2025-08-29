# ConfidRAG
ConfidRAG is a privacy-focused Retrieval-Augmented Generation (RAG) system that lets you search and query your local text files and URLs without sending data to external services.

📂 ConfidRAG (WIP)

ConfidRAG is a work-in-progress project aimed at building a local, confidential Retrieval-Augmented Generation (RAG) system.
The main idea is to allow users to upload and manage their local text files, split them into smaller chunks, and later perform search and retrieval on these chunks through a simple FastAPI backend.

🔍 Current Scope

FastAPI backend setup.

Document ingestion using LangChain (TextLoader, DirectoryLoader).

Basic document management (add, remove files).

Document chunking with RecursiveCharacterTextSplitter.

🛣️ Planned Features

Embedding generation (HuggingFace / OpenAI / Gemini).

Persistent local vector database (Chroma / FAISS).

Query endpoint to search documents (top-K nearest chunks).

Support for additional sources (URLs, PDFs, etc.).

Optional LLM integration for full RAG answers.


