"""
RAG Pipeline — Retrieval-Augmented Generation
Uses OpenAI embeddings + FAISS vector store + LangChain
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()


# ── Config ──────────────────────────────────────────────────────────────────

DOCS_DIR = "docs"          # Drop your .txt files here
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ── Document Loading ─────────────────────────────────────────────────────────

def load_documents(docs_dir: str) -> list:
    """Load all .txt files from the docs directory."""
    Path(docs_dir).mkdir(exist_ok=True)

    # Create a sample doc if directory is empty
    sample = Path(docs_dir) / "sample.txt"
    if not any(Path(docs_dir).iterdir()):
        sample.write_text(
            "LangChain is a framework for building LLM-powered applications.\n"
            "RAG stands for Retrieval-Augmented Generation. It grounds LLM responses "
            "in external documents, reducing hallucinations.\n"
            "FAISS is a library for efficient similarity search over dense vectors."
        )
        print(f"[info] Created sample doc at {sample}")

    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"[info] Loaded {len(docs)} document(s)")
    return docs


# ── Chunking & Embedding ─────────────────────────────────────────────────────

def build_vector_store(docs: list, store_path: str) -> FAISS:
    """Chunk documents and embed them into a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[info] Split into {len(chunks)} chunk(s)")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(store_path)
    print(f"[info] Vector store saved to '{store_path}/'")
    return store


def load_vector_store(store_path: str) -> FAISS:
    """Load an existing FAISS vector store from disk."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    print(f"[info] Loaded vector store from '{store_path}/'")
    return store


# ── RAG Chain ────────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY the
context provided below. If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:""",
)


def build_rag_chain(store: FAISS) -> RetrievalQA:
    """Wire up the retriever and LLM into a QA chain."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    retriever = store.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )
    return chain


# ── CLI Loop ─────────────────────────────────────────────────────────────────

def run():
    store_path = Path(VECTOR_STORE_PATH)

    if store_path.exists():
        store = load_vector_store(VECTOR_STORE_PATH)
    else:
        docs = load_documents(DOCS_DIR)
        store = build_vector_store(docs, VECTOR_STORE_PATH)

    chain = build_rag_chain(store)
    print("\n📚 RAG Pipeline ready. Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = chain.invoke({"query": query})
        print(f"\nAssistant: {result['result']}")

        sources = {doc.metadata.get("source", "unknown") for doc in result["source_documents"]}
        print(f"[sources] {', '.join(sources)}\n")


if __name__ == "__main__":
    run()
