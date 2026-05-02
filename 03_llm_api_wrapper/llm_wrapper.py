"""
LLM API Wrapper
A clean, reusable wrapper around the OpenAI API with retry logic,
token tracking, and structured response handling.
"""

import os
import time
import functools
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

load_dotenv()


# ── Response Model ────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float

    def __str__(self):
        return (
            f"{self.content}\n"
            f"[model={self.model} | tokens={self.total_tokens} "
            f"(p:{self.prompt_tokens} + c:{self.completion_tokens}) | "
            f"latency={self.latency_ms:.0f}ms]"
        )


# ── Session Tracker ───────────────────────────────────────────────────────────

@dataclass
class UsageTracker:
    total_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def update(self, response: LLMResponse):
        self.total_calls += 1
        self.total_tokens += response.total_tokens
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens

    def report(self) -> str:
        return (
            f"Session: {self.total_calls} call(s) | "
            f"{self.total_tokens} total tokens "
            f"({self.total_prompt_tokens} prompt + {self.total_completion_tokens} completion)"
        )


# ── LLM Wrapper ───────────────────────────────────────────────────────────────

class LLMWrapper:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        system_prompt: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.tracker = UsageTracker()

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Send a single prompt and return a structured LLMResponse."""
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._call_with_retry(messages)

    def chat(self, messages: list[dict]) -> LLMResponse:
        """Send a full messages array (for multi-turn conversations)."""
        return self._call_with_retry(messages)

    def _call_with_retry(self, messages: list[dict]) -> LLMResponse:
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency = (time.monotonic() - t0) * 1000

                result = LLMResponse(
                    content=resp.choices[0].message.content.strip(),
                    model=resp.model,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    total_tokens=resp.usage.total_tokens,
                    latency_ms=latency,
                )
                self.tracker.update(result)
                return result

            except RateLimitError:
                if attempt == self.max_retries:
                    raise
                wait = self.retry_delay * attempt
                print(f"[warn] Rate limited. Retrying in {wait}s... (attempt {attempt}/{self.max_retries})")
                time.sleep(wait)

            except APIError as e:
                print(f"[error] OpenAI API error: {e}")
                raise

    def usage(self) -> str:
        return self.tracker.report()
