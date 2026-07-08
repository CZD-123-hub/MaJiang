from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WinContext:
    # 轻量版只识别常见胡牌方式，后续结算模块再继续扩展。
    win_type: str
    winners: list[int | str]
    dianpao_player: int | str | None
    is_multi_win: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_win_context(data: dict[str, Any]) -> WinContext:
    """从 round_end 数据中提取轻量版胡牌方式上下文。"""
    winners = _extract_winners(data)
    dianpao_player = _extract_dianpao_player(data)

    explicit_win_type = _normalize_win_type(data.get("win_type"))
    if explicit_win_type != "unknown":
        win_type = explicit_win_type
    elif bool(data.get("zimo")) or bool(data.get("self_draw")):
        win_type = "zimo"
    elif winners and dianpao_player is not None:
        win_type = "dianpao"
    else:
        win_type = "unknown"

    return WinContext(
        win_type=win_type,
        winners=winners,
        dianpao_player=dianpao_player,
        is_multi_win=len(winners) > 1,
    )


def _normalize_win_type(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"zimo", "self_draw", "自摸"}:
        return "zimo"
    if text in {"dianpao", "点炮", "点胡"}:
        return "dianpao"
    if text in {"qianggang", "qiang_gang", "rob_gang", "抢杠", "抢杠胡"}:
        return "qianggang"
    if text in {"gangkai", "gang_kai", "杠开", "杠上开花"}:
        return "gangkai"
    return "unknown"


def _extract_winners(data: dict[str, Any]) -> list[int | str]:
    if "winners" in data and isinstance(data["winners"], list):
        return data["winners"]
    if "hu_players" in data and isinstance(data["hu_players"], list):
        return data["hu_players"]
    for key in ("winner", "hu_player", "win_player"):
        if key in data and data[key] is not None:
            return [data[key]]
    return []


def _extract_dianpao_player(data: dict[str, Any]) -> int | str | None:
    for key in ("dianpao_player", "pao_player", "discard_player", "loser"):
        if key in data and data[key] is not None:
            return data[key]
    return None
