from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cache = Cache(str(_CACHE_DIR))


def _make_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", judge_model: str = "gpt-4o"):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        self.model = model
        self.judge_model = judge_model

    def _retry_call(self, fn, *, max_retries: int = 4, backoff: float = 1.5):
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                sleep_s = backoff ** attempt
                print(f"[llm_client] attempt {attempt + 1} failed ({e}); sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[dict] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = None,
        use_cache: bool = True,
    ) -> dict:
        model = model or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        key = _make_key(payload)
        if use_cache and key in _cache:
            return _cache[key]

        def _call():
            resp = self.client.chat.completions.create(**payload)
            return resp.model_dump()

        result = self._retry_call(_call)
        if use_cache:
            _cache[key] = result
        return result

    def chat_text(self, messages: list[dict[str, str]], **kwargs) -> str:
        resp = self.chat(messages, **kwargs)
        return resp["choices"][0]["message"]["content"] or ""

    def chat_json(self, messages: list[dict[str, str]], **kwargs) -> dict:
        kwargs.setdefault("response_format", {"type": "json_object"})
        text = self.chat_text(messages, **kwargs)
        return json.loads(text)

    def chat_tool(
        self,
        messages: list[dict[str, str]],
        tool: dict,
        **kwargs,
    ) -> dict:
        resp = self.chat(
            messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool["function"]["name"]}},
            **kwargs,
        )
        calls = resp["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            raise RuntimeError("LLM returned no tool call.")
        return json.loads(calls[0]["function"]["arguments"])

    def embed(self, texts: list[str], *, model: str = "text-embedding-3-small") -> list[list[float]]:
        key = _make_key({"model": model, "texts": texts})
        if key in _cache:
            return _cache[key]

        def _call():
            resp = self.client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]

        result = self._retry_call(_call)
        _cache[key] = result
        return result
