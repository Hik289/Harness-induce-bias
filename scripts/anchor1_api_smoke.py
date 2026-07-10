"""anchor_1: OpenAI-compatible API 50 次调用成功率 >= 95%.

每次调用要求模型返回一个 {n: int, parity: "odd"|"even"} 的合法 JSON, 用于同
时验证 (a) 链路可用 (b) JSON mode 稳定 (c) rate limit 不被触发。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 让脚本能 import sibling 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skeleton.core.llm_client import LLMClient  # noqa: E402


JST = timezone(timedelta(hours=9))


def main(n_calls: int, out_path: str) -> int:
    llm = LLMClient(min_interval_s=0.4, max_retries=3)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    successes = 0
    failures = 0
    t_start = time.time()
    for i in range(n_calls):
        n = i + 1
        prompt = [
            {"role": "system", "content": "你只输出一个合法 JSON 对象, 无 markdown."},
            {
                "role": "user",
                "content": (
                    f"请输出 JSON: {{\"n\": {n}, \"parity\": <\"odd\" 或 \"even\">}} "
                    f"对应数字 {n} 的奇偶性。"
                ),
            },
        ]
        rec: dict = {"i": i, "n": n, "ts": datetime.now(JST).isoformat(timespec="seconds")}
        try:
            obj, stats = llm.chat_json(prompt, max_tokens=64)
            ok = (
                isinstance(obj, dict)
                and obj.get("n") == n
                and obj.get("parity") in ("odd", "even")
                and (obj["parity"] == "even") == (n % 2 == 0)
            )
            rec.update(
                ok=ok,
                latency_s=round(stats.latency_s, 3),
                prompt_tokens=stats.prompt_tokens,
                completion_tokens=stats.completion_tokens,
                retries=stats.retries,
                response=obj,
            )
            if ok:
                successes += 1
            else:
                failures += 1
                rec["error"] = "answer mismatch"
        except Exception as e:  # noqa: BLE001
            failures += 1
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        results.append(rec)
        if (i + 1) % 10 == 0:
            print(
                f"  [{i+1}/{n_calls}] ok={successes} fail={failures} "
                f"elapsed={time.time()-t_start:.1f}s",
                flush=True,
            )

    rate = successes / n_calls
    summary = {
        "n_calls": n_calls,
        "successes": successes,
        "failures": failures,
        "success_rate": rate,
        "anchor_passed": rate >= 0.95,
        "elapsed_s": round(time.time() - t_start, 2),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "results": results,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    return 0 if summary["anchor_passed"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", default="logs/anchor1_api_smoke.json")
    a = p.parse_args()
    sys.exit(main(a.n, a.out))
