"""实时对局观察页的数据读取工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stats import DEFAULT_ACTION_LOG_PATH


def load_dashboard_events(log_path: str | Path = DEFAULT_ACTION_LOG_PATH, limit: int = 160) -> list[dict[str, Any]]:
    """读取最近动作；损坏或旧格式日志会被忽略，保证观察页始终可用。"""
    target = Path(log_path)
    if not target.exists():
        return []
    events: list[dict[str, Any]] = []
    for index, line in enumerate(target.read_text(encoding="utf-8").splitlines()[-limit:]):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event["event_id"] = index
        events.append(event)
    return events


def dashboard_payload(log_path: str | Path = DEFAULT_ACTION_LOG_PATH) -> dict[str, Any]:
    events = load_dashboard_events(log_path)
    latest = events[-1] if events else None
    return {
        "events": events,
        "latest": latest,
        "event_count": len(events),
    }
