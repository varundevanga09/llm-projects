# LLM Projects

Three hands-on Python projects for working with Large Language Models.

## Projects

| # | Project | Key tech |
|---|---|---|
| 01 | [RAG Pipeline](./01_rag_pipeline/) | LangChain · FAISS · OpenAI Embeddings |
| 02 | [Chatbot with Memory](./02_chatbot_with_memory/) | LangChain · Buffer & Summary Memory |
| 03 | [LLM API Wrapper & Prompt Chaining](./03_llm_api_wrapper/) | OpenAI SDK · Sequential / Parallel / Conditional chains |

## Quick start

Each project is self-contained. Navigate into any folder and:

```bash
pip install -r requirements.txt
# Add your key to a .env file: OPENAI_API_KEY=sk-...
python <main_script>.py
```

## .env setup (shared across projects)

```
OPENAI_API_KEY=sk-...
```

You can place a single `.env` at the repo root and each project will pick it up.
