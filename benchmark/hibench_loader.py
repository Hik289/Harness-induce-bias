"""HIBench-Code v0 toy task loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DEFAULT_TASKS_PATH = Path(
    "./data/hibench_code/v0_toy/tasks.json"
)
# 在 hpc 上的镜像位置
HPC_TASKS_PATH = Path(
    "./data/hibench_code/v0_toy/tasks.json"
)


def load_tasks(path: Optional[str | Path] = None) -> list[dict]:
    """加载 v0_toy 任务列表; 优先用显式 path, 其次 GCP, 再 hpc 镜像."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend([DEFAULT_TASKS_PATH, HPC_TASKS_PATH])
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data["tasks"]
    raise FileNotFoundError(f"找不到 v0_toy tasks.json; 试过: {[str(c) for c in candidates]}")


def get_task(task_id: str, path: Optional[str | Path] = None) -> dict:
    for t in load_tasks(path):
        if t["task_id"] == task_id:
            return t
    raise KeyError(task_id)
