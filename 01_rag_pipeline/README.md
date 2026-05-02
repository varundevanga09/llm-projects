# 01 · RAG Pipeline

A minimal Retrieval-Augmented Generation pipeline using **LangChain**, **FAISS**, and **OpenAI**.

## How it works

```
Your .txt files → chunked → embedded → FAISS index
                                             ↓
              User query → similarity search → top-k chunks → GPT → Answer
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
python rag.py
```

Drop any `.txt` files into the `docs/` folder before running to use your own knowledge base.
A sample document is auto-created on first run if the folder is empty.

## Key concepts

| Concept | What it does |
|---|---|
| `RecursiveCharacterTextSplitter` | Splits docs into overlapping chunks |
| `OpenAIEmbeddings` | Converts chunks to dense vectors |
| `FAISS` | Nearest-neighbour search over vectors |
| `RetrievalQA` | Fetches top-k chunks and passes to LLM |
