"""Thread-naive JSONL logger; 一行一 step. readme §15 格式."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))


def now_jst_iso() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


class JSONLLogger:
    """每个 run 一个 logger; 写到 logs/<run_id>.jsonl. 多 thread 安全."""

    def __init__(self, log_path: str | os.PathLike[str]) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # touch
        self._path.touch(exist_ok=True)
        self._lock = threading.Lock()
        self.line_count = 0

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self.line_count += 1

    def write_many(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
                    self.line_count += 1
