"""
Chatbot with Memory — LangChain
Supports two memory strategies: buffer (full history) and summary (compressed).
"""

import os
import argparse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

CHAT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7

# ── Prompt ───────────────────────────────────────────────────────────────────

CHAT_PROMPT_TEMPLATE = (
    "You are a sharp, concise AI assistant with access to the conversation "
    "history below. Use it to give contextually aware responses.\n\n"
    "History:\n{history}\n\nHuman: {input}\nAI:"
)

CHAT_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=CHAT_PROMPT_TEMPLATE,
)

# ── Memory Factory ────────────────────────────────────────────────────────────

def build_memory(strategy: str, llm: ChatOpenAI):
    """Return a memory object based on the chosen strategy."""
    if strategy == "buffer":
        # Stores the full conversation verbatim — good for short sessions
        return ConversationBufferMemory(
            memory_key="history",
            return_messages=False,
        )
    elif strategy == "summary":
        # Compresses old turns into a running summary — good for long sessions
        return ConversationSummaryMemory(
            llm=llm,
            memory_key="history",
            return_messages=False,
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: buffer | summary")


# ── Chain Builder ─────────────────────────────────────────────────────────────

def build_chain(strategy: str) -> ConversationChain:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE)
    memory = build_memory(strategy, llm)
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=CHAT_PROMPT,
        verbose=False,
    )
    return chain


# ── CLI ───────────────────────────────────────────────────────────────────────

def run(strategy: str):
    chain = build_chain(strategy)
    strategy_label = "Buffer Memory" if strategy == "buffer" else "Summary Memory"
    print(f"\n💬 Chatbot ready [{strategy_label}]. Type 'quit' to exit. Type 'memory' to inspect state.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "memory":
            print(f"\n[memory state]\n{chain.memory.load_memory_variables({})}\n")
            continue

        if not user_input:
            continue

        response = chain.invoke({"input": user_input})
        print(f"\nAssistant: {response['response']}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain Chatbot with Memory")
    parser.add_argument(
        "--memory",
        choices=["buffer", "summary"],
        default="buffer",
        help="Memory strategy (default: buffer)",
    )
    args = parser.parse_args()
    run(strategy=args.memory)
