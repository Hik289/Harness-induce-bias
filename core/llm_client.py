from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4-mini"


@dataclass
class CallStats:

    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    raw_response: Optional[str] = None
    error: Optional[str] = None


class LLMClient:

    def __init__(
        self,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        min_interval_s: float = 0.3,
        max_retries: int = 4,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY 未设置; 请通过环境变量注入 (不要硬编码到 agent 配置)"
            )
        base_url = (
            endpoint
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self._client = OpenAI(base_url=base_url.rstrip("/"), api_key=key)
        self._deployment = (
            deployment
            or os.environ.get("OPENAI_MODEL")
            or DEFAULT_MODEL
        )
        self._min_interval = min_interval_s
        self._max_retries = max_retries
        self._last_call_ts: float = 0.0
        self.total_calls = 0
        self.total_failures = 0

    def _respect_rate_limit(self) -> None:
        delta = time.time() - self._last_call_ts
        if delta < self._min_interval:
            time.sleep(self._min_interval - delta)
        self._last_call_ts = time.time()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format_json: bool = False,
        seed: Optional[int] = None,
    ) -> tuple[str, CallStats]:
        stats = CallStats()
        last_err: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            self._respect_rate_limit()
            self.total_calls += 1
            t0 = time.time()
            try:
                kwargs: dict[str, Any] = {
                    "model": self._deployment,
                    "messages": messages,
                }
                if max_tokens is not None:
                    kwargs["max_completion_tokens"] = max_tokens
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if response_format_json:
                    kwargs["response_format"] = {"type": "json_object"}
                if seed is not None:
                    kwargs["seed"] = seed

                resp = self._client.chat.completions.create(**kwargs)
                stats.latency_s = time.time() - t0
                stats.retries = attempt
                if resp.usage is not None:
                    stats.prompt_tokens = resp.usage.prompt_tokens
                    stats.completion_tokens = resp.usage.completion_tokens
                    stats.total_tokens = resp.usage.total_tokens
                content = resp.choices[0].message.content or ""
                stats.raw_response = content
                return content, stats

            except (RateLimitError, APIConnectionError, APIError) as e:
                last_err = e
                stats.error = f"{type(e).__name__}: {e}"
                if attempt >= self._max_retries:
                    break
                backoff = (2**attempt) + random.uniform(0, 0.5)
                time.sleep(backoff)
            except Exception as e:
                last_err = e
                stats.error = f"{type(e).__name__}: {e}"
                break

        self.total_failures += 1
        stats.latency_s = time.time() - t0
        raise RuntimeError(f"LLM 调用失败 ({self._max_retries+1} 次): {last_err}")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> tuple[dict, CallStats]:
        content, stats = self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format_json=True,
            seed=seed,
        )
        try:
            return json.loads(content), stats
        except json.JSONDecodeError as e:
            retry_msgs = messages + [
                {"role": "user", "content": (
                    "前一次回复无法解析为 JSON。请只输出一个合法的 JSON 对象, "
                    "不要任何 markdown / 解释。错误: " + str(e)
                )}
            ]
            content2, stats2 = self.chat(
                retry_msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format_json=True,
                seed=seed,
            )
            stats2.retries += stats.retries + 1
            try:
                return json.loads(content2), stats2
            except json.JSONDecodeError as e2:
                stats2.error = f"JSONDecodeError: {e2}"
                raise RuntimeError(f"两次都无法解析 JSON: {e2}\nraw={content2[:500]}")
