# 02 · Chatbot with Memory

A conversational chatbot using **LangChain** with two pluggable memory strategies.

## Memory Strategies

| Strategy | How it works | Best for |
|---|---|---|
| `buffer` | Stores full conversation verbatim | Short sessions |
| `summary` | Compresses history into a rolling summary | Long sessions |

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
# Default (buffer memory)
python chatbot.py

# With summary memory
python chatbot.py --memory summary
```

## Special commands

| Command | Action |
|---|---|
| `memory` | Print current memory state |
| `quit` / `exit` | End the session |
