"""
Prompt Chaining
Demonstrates sequential, parallel, and conditional LLM chains
using the LLMWrapper from llm_wrapper.py.
"""

from llm_wrapper import LLMWrapper


# ── Chain Primitives ──────────────────────────────────────────────────────────

def sequential_chain(llm: LLMWrapper, topic: str) -> dict:
    """
    Sequential chain: each step's output feeds the next.
    Step 1 → outline  →  Step 2 → draft  →  Step 3 → polish
    """
    print(f"\n[chain:sequential] Topic: '{topic}'")

    # Step 1: Generate an outline
    outline = llm.complete(
        f"Create a 3-point outline for a short blog post about: {topic}. "
        "Return just the outline, no extra text."
    )
    print(f"  Step 1 (outline): {outline.content[:80]}...")

    # Step 2: Expand the outline into a draft
    draft = llm.complete(
        f"Expand this outline into a short blog post (150 words):\n{outline.content}"
    )
    print(f"  Step 2 (draft):   {draft.content[:80]}...")

    # Step 3: Polish the draft
    final = llm.complete(
        f"Polish and improve this blog post for clarity and tone:\n{draft.content}"
    )
    print(f"  Step 3 (final):   {final.content[:80]}...")

    return {"outline": outline.content, "draft": draft.content, "final": final.content}


def parallel_chain(llm: LLMWrapper, product: str) -> dict:
    """
    Parallel chain: run independent prompts, then synthesise.
    (Simulated sequential here — use asyncio / threading for true parallelism.)
    """
    print(f"\n[chain:parallel] Product: '{product}'")

    pros = llm.complete(f"List 3 pros of: {product}. Be concise.")
    cons = llm.complete(f"List 3 cons of: {product}. Be concise.")
    audience = llm.complete(f"Who is the ideal customer for: {product}? One sentence.")

    synthesis = llm.complete(
        f"Given these notes about '{product}':\n"
        f"Pros: {pros.content}\n"
        f"Cons: {cons.content}\n"
        f"Audience: {audience.content}\n\n"
        "Write a 2-sentence product positioning statement."
    )

    print(f"  Pros:        {pros.content[:60]}...")
    print(f"  Cons:        {cons.content[:60]}...")
    print(f"  Audience:    {audience.content[:60]}")
    print(f"  Synthesis:   {synthesis.content}")

    return {
        "pros": pros.content,
        "cons": cons.content,
        "audience": audience.content,
        "positioning": synthesis.content,
    }


def conditional_chain(llm: LLMWrapper, user_input: str) -> str:
    """
    Conditional chain: classify intent first, then route to specialist prompt.
    """
    print(f"\n[chain:conditional] Input: '{user_input}'")

    # Step 1: Classify intent
    intent = llm.complete(
        f"Classify this user message into exactly one category: "
        f"TECHNICAL | CREATIVE | GENERAL\n\nMessage: {user_input}\n\nCategory:",
        system_prompt="Reply with only the category label. Nothing else.",
    ).content.strip().upper()

    print(f"  Detected intent: {intent}")

    # Step 2: Route to a tailored system prompt
    routing = {
        "TECHNICAL": "You are a senior software engineer. Give precise, code-aware answers.",
        "CREATIVE": "You are a creative writing coach. Respond with imagination and flair.",
        "GENERAL":  "You are a friendly, knowledgeable assistant. Be clear and helpful.",
    }
    system = routing.get(intent, routing["GENERAL"])

    response = llm.complete(user_input, system_prompt=system)
    print(f"  Response: {response.content[:100]}...")
    return response.content


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    llm = LLMWrapper(model="gpt-4o-mini", temperature=0.7)

    # 1. Sequential chain
    seq_result = sequential_chain(llm, topic="the future of renewable energy")
    print("\n── Final blog post ──")
    print(seq_result["final"])

    # 2. Parallel chain
    par_result = parallel_chain(llm, product="a smart water bottle")
    print("\n── Product positioning ──")
    print(par_result["positioning"])

    # 3. Conditional chain
    test_inputs = [
        "How do I reverse a linked list in Python?",
        "Write me a haiku about autumn.",
        "What's the capital of Portugal?",
    ]
    print("\n── Conditional routing ──")
    for msg in test_inputs:
        conditional_chain(llm, msg)

    # Session summary
    print(f"\n── {llm.usage()} ──")


if __name__ == "__main__":
    main()
