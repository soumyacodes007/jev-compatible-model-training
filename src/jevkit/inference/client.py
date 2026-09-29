"""Client for a deployed decision model behind an OpenAI-compatible API (vLLM)."""

from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI, OpenAI

from jevkit.prompts import DEFAULT_FORMAT, build_messages, tag_options


def request_kwargs(model: str, row: dict[str, Any], fmt: str = DEFAULT_FORMAT) -> dict[str, Any]:
    """Chat-completion arguments for one record, constrained to its option letters."""

    labels = [option["label"] for option in row["options"]]
    return {
        "model": model,
        "messages": build_messages(row["state"], row["question"], row["options"], fmt=fmt),
        "temperature": 0,
        "max_tokens": 8,
        "extra_body": {
            "structured_outputs": {"choice": labels},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def as_options(options: list) -> list[dict[str, str]]:
    """Accept plain option strings or already-lettered option dicts."""

    if options and isinstance(options[0], dict):
        return options
    return tag_options([(str(text), str(text)) for text in options])


class DecisionClient:
    """Use the same ``prompt_format`` the model was trained with (see the build manifest)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 20 * 60,
        prompt_format: str = DEFAULT_FORMAT,
    ):
        self.model = model
        self.prompt_format = prompt_format
        self.sync = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.async_ = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0
        )

    def decide(self, state: Any, question: str, options: list) -> dict[str, Any]:
        row = {"state": state, "question": question, "options": as_options(options)}
        started = time.perf_counter()
        response = self.sync.chat.completions.create(
            **request_kwargs(self.model, row, self.prompt_format)
        )
        elapsed = time.perf_counter() - started
        answer = (response.choices[0].message.content or "").strip()
        selected = next((o for o in row["options"] if o["label"] == answer), None)
        return {"answer": answer, "option": selected, "elapsed_seconds": elapsed}

    async def adecide_record(self, row: dict[str, Any]) -> str:
        response = await self.async_.chat.completions.create(
            **request_kwargs(self.model, row, self.prompt_format)
        )
        return (response.choices[0].message.content or "").strip()
