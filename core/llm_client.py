"""Azure GPT-5.4-mini API wrapper.

设计原则:
- API key 通过环境变量 AZURE_OPENAI_API_KEY 注入 (绝不写到 agent 配置或日志)
- 提供 retry + exponential backoff (per readme §Risks 段)
- 提供 strict-JSON 输出模式: 调用方传 schema, client 负责重试 + 校验
- 计费/调用元数据 (latency, tokens, cost) 由 client 写回, 上游统一记录
- rate limit: 默认 sleep 0.3s between calls (低于 spec 的 1 req/s 上限)

readme.md §0 给的 DefaultAzureCredential 路径要求 hpc 上有 Azure CLI / managed
identity, 实际环境无此条件; 经实测, 把 idea.md 中给出的 API key 作为 OpenAI
client 的 api_key (走 Azure 部署的 OpenAI v1 endpoint) 可直接通; 本项目固定走
此路径。
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError


# Endpoint 是 deployment 全局共享的常量, 没有敏感性, 可写代码里
AZURE_ENDPOINT = "https://YOUR_AZURE_ENDPOINT/openai/v1"
DEFAULT_DEPLOYMENT = "gpt-5.4-mini"


@dataclass
class CallStats:
    """单次 LLM 调用的元数据 (供 logger 写到 JSONL)."""

    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    raw_response: Optional[str] = None
    error: Optional[str] = None


class LLMClient:
    """瘦封装。所有 belief rollout / harness 调用都过这一层。"""

    def __init__(
        self,
        deployment: str = DEFAULT_DEPLOYMENT,
        api_key: Optional[str] = None,
        endpoint: str = AZURE_ENDPOINT,
        min_interval_s: float = 0.3,
        max_retries: int = 4,
    ) -> None:
        key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY 未设置; 请通过环境变量注入 (不要硬编码到 agent 配置)"
            )
        self._client = OpenAI(base_url=endpoint, api_key=key)
        self._deployment = deployment
        self._min_interval = min_interval_s
        self._max_retries = max_retries
        self._last_call_ts: float = 0.0
        # 简单计数, 供 anchor_1 50-call 测试用
        self.total_calls = 0
        self.total_failures = 0

    # --------------------------------------------------------- private
    def _respect_rate_limit(self) -> None:
        delta = time.time() - self._last_call_ts
        if delta < self._min_interval:
            time.sleep(self._min_interval - delta)
        self._last_call_ts = time.time()

    # ---------------------------------------------------------- public
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format_json: bool = False,
        seed: Optional[int] = None,
    ) -> tuple[str, CallStats]:
        """返回 (content, stats). 自动 retry + backoff.

        `seed`: 通过 chat.completions `seed` 参数传给 API. 实测 (DAY3 Step A 启动前
        sanity) Azure gpt-5.4-mini 接受 seed 参数但 system_fingerprint=None 且实际
        不强制 determinism. 仍然传, 因为: (a) Azure 后续可能启用 (b) Phase 1 主表
        需要 3 个 seed 估方差, seed 的不同值至少给了 prompt-cache miss/hit 的随机
        分支变化.
        """
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
            except Exception as e:  # noqa: BLE001
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
        """要求模型返回严格 JSON. 解析失败会自动 retry 1 次."""
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
