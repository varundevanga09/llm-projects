<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=LLM%20Projects&fontSize=70&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Hands-on%20Python%20AI%20%7C%20RAG%20%C2%B7%20Memory%20%C2%B7%20Prompt%20Chains&descAlignY=55&descSize=18" />

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=6AD5F7&center=true&vCenter=true&width=600&lines=Building+with+Large+Language+Models+%F0%9F%A4%96;RAG+%7C+Memory+%7C+Prompt+Chaining;Clean+%2C+Readable+%2C+Production-style+Python)](https://github.com/varundevanga09/llm-projects)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0082C9?style=for-the-badge)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 🗂️ Projects at a Glance

<div align="center">

| 🔢 | 🚀 Project | 📝 Description | ⚙️ Stack |
|:---:|:---|:---|:---|
| `01` | [**RAG Pipeline**](#-01--rag-pipeline) | Ground LLM answers in your own docs | LangChain · FAISS · Embeddings |
| `02` | [**Chatbot with Memory**](#-02--chatbot-with-memory) | Context-aware chat across turns | LangChain · Buffer & Summary Memory |
| `03` | [**LLM API Wrapper**](#-03--llm-api-wrapper--prompt-chaining) | Reusable wrapper + 3 chain patterns | OpenAI SDK · Dataclasses |

</div>

---

## ⚙️ Tech Stack

<div align="center">

![Python](https://skillicons.dev/icons?i=python,github,vscode,git&theme=dark)

</div>

<div align="center">

| Layer | Tools |
|:---:|:---|
| 🧠 **LLM** | `gpt-4o-mini` · `text-embedding-3-small` |
| 🔗 **Framework** | `LangChain 0.3` · `LangChain-Community` |
| 🗄️ **Vector Store** | `FAISS` (local, no server needed) |
| 🔐 **Config** | `python-dotenv` |
| 📦 **Packaging** | `pip` + `requirements.txt` per project |

</div>

---

## 🔧 Setup

```bash
# Clone the repo
git clone https://github.com/varundevanga09/llm-projects.git
cd llm-projects

# Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env
```

> ⚠️ `.env` is already in `.gitignore` — your key stays safe.

---

## 📦 01 · RAG Pipeline

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=11&height=3&section=header" width="100%"/>

> **Retrieval-Augmented Generation** — answers questions using *your* documents, not just the model's training data.

### 🔄 Architecture

```
 ┌─────────────┐    chunk     ┌──────────────┐   embed   ┌──────────────┐
 │  .txt files │ ──────────► │ Text Chunks  │ ────────► │ FAISS Index  │
 └─────────────┘             └──────────────┘           └──────┬───────┘
                                                                │  similarity
                                                                ▼  search
 ┌─────────────┐   answer    ┌──────────────┐  top-k   ┌──────────────┐
 │    User     │ ◄────────── │  GPT-4o-mini │ ◄─────── │  Retrieved   │
 │   Query     │             │  + Sources   │          │   Chunks     │
 └─────────────┘             └──────────────┘          └──────────────┘
```

### ▶️ Run

```bash
cd 01_rag_pipeline
pip install -r requirements.txt
python rag.py
```

Drop any `.txt` files into `docs/` before running to use your own knowledge base.
A sample file is auto-created if the folder is empty.

---

## 💬 02 · Chatbot with Memory

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=20&height=3&section=header" width="100%"/>

> A conversational chatbot that **remembers what you said** — with two pluggable memory backends.

### 🧠 Memory Strategies

```
Buffer Memory                         Summary Memory
─────────────────────────────────     ──────────────────────────────────────
Turn 1: "Hi, I'm Varun"               [Summary]: User is Varun, a software
Turn 2: "I work in AI"                engineer working in AI who asked
Turn 3: "What do I do?"               about RAG systems.
         ↓                                      ↓
  Full history kept verbatim            Old turns compressed by LLM
  ✅ Perfect recall                     ✅ Handles very long sessions
  ⚠️  Context grows unbounded           ✅ Stays within token limits
```

### ▶️ Run

```bash
cd 02_chatbot_with_memory
pip install -r requirements.txt

python chatbot.py                  # Buffer (default)
python chatbot.py --memory summary # Summary mode
```

**In-session commands:** type `memory` to inspect state · `quit` to exit

---

## 🔗 03 · LLM API Wrapper & Prompt Chaining

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6&height=3&section=header" width="100%"/>

> A **reusable OpenAI wrapper** with auto-retry on rate limits, token tracking, and three real-world chain patterns.

### 📁 Files

| File | What it does |
|------|-------------|
| `llm_wrapper.py` | `LLMWrapper` class · `LLMResponse` dataclass · `UsageTracker` |
| `prompt_chains.py` | Sequential · Parallel · Conditional chain demos |

### ▶️ Run

```bash
cd 03_llm_api_wrapper
pip install -r requirements.txt
python prompt_chains.py
```

### ⛓️ Chain Patterns Visualised

```
SEQUENTIAL                     PARALLEL                    CONDITIONAL
──────────────────────         ─────────────────────────   ──────────────────────
topic                          product                     user input
  │                              ├──► pros                   │
  ▼                              ├──► cons      ─► synth      ▼  classify
outline                          └──► audience              TECHNICAL?──► engineer prompt
  │                                                         CREATIVE? ──► writer prompt
  ▼                                                         GENERAL?  ──► default prompt
draft
  │
  ▼
polished post
```

### 💡 Quick usage

```python
from llm_wrapper import LLMWrapper

llm = LLMWrapper(model="gpt-4o-mini", temperature=0.5)

response = llm.complete("Explain RAG in one paragraph.")
print(response)       # content + tokens + latency
print(llm.usage())    # session-level stats
```

---

## 🗂️ Repo Structure

```
llm-projects/
├── 📁 01_rag_pipeline/
│   ├── rag.py
│   ├── requirements.txt
│   └── README.md
├── 📁 02_chatbot_with_memory/
│   ├── chatbot.py
│   ├── requirements.txt
│   └── README.md
├── 📁 03_llm_api_wrapper/
│   ├── llm_wrapper.py
│   ├── prompt_chains.py
│   ├── requirements.txt
│   └── README.md
├── .gitignore
└── README.md  ← you are here
```

---

## 🗺️ Roadmap

- [x] RAG pipeline with FAISS
- [x] Chatbot with buffer & summary memory
- [x] LLM wrapper with retry + token tracking
- [x] Sequential, parallel, conditional chains
- [ ] Async support in LLM wrapper
- [ ] Swap FAISS for ChromaDB (persistent)
- [ ] FastAPI layer over RAG pipeline
- [ ] Streaming responses in chatbot
- [ ] Tool / function calling chains

---

## 👤 Author

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=2" width="60%"/>

**Varun Devanga Ampabathini**
*Software Engineer · M.S. Computer Engineering · CSUCI '25*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-varundevanga09-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/varundevanga09)
[![GitHub](https://img.shields.io/badge/GitHub-varundevanga09-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/varundevanga09)
[![Email](https://img.shields.io/badge/Email-varundevanga09@icloud.com-D14836?style=for-the-badge&logo=apple&logoColor=white)](mailto:varundevanga09@icloud.com)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling"/>

*⭐ Star this repo if you found it useful!*

</div>
