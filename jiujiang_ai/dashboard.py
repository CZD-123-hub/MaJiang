"""实时对局观察页的数据读取工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stats import DEFAULT_ACTION_LOG_PATH, DEFAULT_ROUND_LOG_PATH


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
        "recent_rounds": load_recent_rounds(),
    }


def load_recent_rounds(log_path: str | Path = DEFAULT_ROUND_LOG_PATH, limit: int = 16) -> list[dict[str, Any]]:
    """读取最近已结算对局；测试服重复回调按 room_id 只展示一次。"""
    target = Path(log_path)
    if not target.exists():
        return []
    results: list[dict[str, Any]] = []
    seen_rooms: set[str] = set()
    for line in reversed(target.read_text(encoding="utf-8").splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            continue
        room_id = data.get("room_id")
        if room_id is None or str(room_id) in seen_rooms:
            continue
        seen_rooms.add(str(room_id))
        results.append(_round_result(payload, data))
        if len(results) >= limit:
            break
    return results


def _round_result(payload: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    winner = _valid_position(data.get("win_player_position", data.get("winner")))
    is_draw = data.get("end_type") == 2 or winner is None
    return {
        "room_id": data.get("room_id"),
        "timestamp": payload.get("timestamp"),
        "winner_position": None if is_draw else winner,
        "win_type": "流局" if is_draw else _win_type_label(data.get("win_type")),
    }


def _valid_position(value: object) -> int | None:
    try:
        position = int(value)
    except (TypeError, ValueError):
        return None
    return position if 0 <= position <= 3 else None


def _win_type_label(value: object) -> str:
    labels = {"zimo": "自摸", "dianpao": "点炮", "gangkai": "杠开", "qianggang": "抢杠"}
    return labels.get(str(value or "").strip().lower(), "胡牌")
