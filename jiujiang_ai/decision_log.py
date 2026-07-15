"""弃牌决策日志：为离线回放和策略 A/B 对比保留可解释摘要。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .rules import ACTION_DISCARD


DEFAULT_DECISION_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "jiujiang_decisions.jsonl"


def append_decision_log(
    data: dict,
    *,
    action_type: int,
    action_card: list[int],
    strategy: str,
    decision: object | None,
    log_path: str | Path | None = None,
) -> Path:
    """追加一条决策摘要；调用方决定是否启用，函数本身不修改游戏状态。"""
    target = Path(log_path) if log_path is not None else DEFAULT_DECISION_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "action_type": action_type,
        "action_card": list(action_card),
        "context": _context_summary(data),
        "decision": _decision_summary(decision),
    }
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=str))
        file.write("\n")
    return target


def load_decision_logs(log_path: str | Path | None = None) -> list[dict[str, Any]]:
    target = Path(log_path) if log_path is not None else DEFAULT_DECISION_LOG_PATH
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("decision log line must decode to a JSON object")
        records.append(record)
    return records


def _context_summary(data: dict) -> dict[str, Any]:
    position = int(data.get("acting_do_player_position", 0))
    hands = data.get("player_hand_cards") or []
    hand = list(hands[position]) if 0 <= position < len(hands) else []
    action_cards = data.get("action_cards") or {}
    discard_candidates = action_cards.get(ACTION_DISCARD, action_cards.get(str(ACTION_DISCARD), []))
    wall = data.get("remain_card_stack")
    return {
        "room_id": data.get("room_id"),
        "acting_position": position,
        "hand": hand,
        "candidates": [cards[0] for cards in discard_candidates if cards],
        "turn": sum(1 for action in data.get("action_seq") or [] if _is_discard(action)),
        "remaining_wall_count": len(wall) if isinstance(wall, list) else None,
    }


def _decision_summary(decision: object | None) -> dict[str, Any]:
    if decision is None:
        return {}
    fields = (
        "discard",
        "score",
        "shanten_after_discard",
        "effective_count",
        "effective_tiles",
        "winning_tiles",
        "safety_score",
        "progress_probability",
        "expected_win_value",
        "risk_score",
        "flexibility",
        "route_count",
        "retained_route_count",
        "expected_path_value",
        "expected_hu_value",
        "expected_ting_value",
        "expected_improvement_value",
    )
    result: dict[str, Any] = {}
    for field in fields:
        if hasattr(decision, field):
            value = getattr(decision, field)
            result[field] = _json_key_safe(value)
    if hasattr(decision, "route_summary"):
        route_summary = getattr(decision, "route_summary")
        result["route_summary"] = {
            field: _json_key_safe(getattr(route_summary, field))
            for field in (
                "route_count",
                "retained_route_count",
                "effective_count",
                "flexibility",
                "progress_probability",
                "expected_win_value",
                "route_value",
            )
            if hasattr(route_summary, field)
        }
    return result


def _json_key_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_key_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_key_safe(item) for item in value]
    return value


def _is_discard(action: object) -> bool:
    return isinstance(action, (list, tuple)) and len(action) >= 2 and action[1] == ACTION_DISCARD
