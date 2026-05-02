# 03 · LLM API Wrapper & Prompt Chaining

A clean, reusable OpenAI wrapper with token tracking, retry logic,
and three prompt chaining patterns.

## Files

| File | Purpose |
|---|---|
| `llm_wrapper.py` | Reusable `LLMWrapper` class + `LLMResponse` dataclass |
| `prompt_chains.py` | Sequential, parallel, and conditional chains |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## Run the chain demo

```bash
python prompt_chains.py
```

## Use the wrapper directly

```python
from llm_wrapper import LLMWrapper

llm = LLMWrapper(model="gpt-4o-mini", temperature=0.5)
response = llm.complete("Explain transformers in one paragraph.")
print(response)          # includes token counts + latency
print(llm.usage())       # session-level stats
```

## Chain Patterns

### Sequential
Each step's output feeds the next:
```
topic → outline → draft → polished post
```

### Parallel
Independent calls synthesised at the end:
```
product → [pros | cons | audience] → positioning statement
```

### Conditional
Classify intent first, then route to a specialist prompt:
```
user input → TECHNICAL | CREATIVE | GENERAL → tailored response
```
