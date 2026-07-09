from __future__ import annotations

from typing import Any

from .win_context import detect_win_context


def detect_round_flow(data: dict[str, Any]) -> dict[str, Any]:
    """提取当前对局的庄家、胡牌与黄庄状态。"""
    context = detect_win_context(data)
    dealer = _dealer_player(data)
    player_count = _player_count(data)
    is_huangzhuang = _is_huangzhuang(data, context.winners)

    return {
        "dealer": dealer,
        "player_count": player_count,
        "has_winner": bool(context.winners),
        "winners": context.winners,
        "dianpao_player": context.dianpao_player,
        "is_multi_win": context.is_multi_win,
        "is_huangzhuang": is_huangzhuang,
        "is_draw_round": is_huangzhuang,
    }


def resolve_next_dealer(data: dict[str, Any]) -> dict[str, Any]:
    """根据九江规则计算下一局庄家。"""
    flow = detect_round_flow(data)
    dealer = flow["dealer"]
    player_count = flow["player_count"]
    winners = flow["winners"]
    dianpao_player = flow["dianpao_player"]

    if flow["is_huangzhuang"]:
        next_dealer = ((dealer or 0) + 1) % max(player_count, 1)
        return {
            **flow,
            "next_dealer": next_dealer,
            "reason": "draw_next_player",
        }

    if flow["is_multi_win"] and dianpao_player is not None:
        return {
            **flow,
            "next_dealer": int(dianpao_player),
            "reason": "dianpao_player_is_dealer",
        }

    if winners:
        return {
            **flow,
            "next_dealer": int(winners[0]),
            "reason": "winner_is_dealer",
        }

    return {
        **flow,
        "next_dealer": dealer if dealer is not None else 0,
        "reason": "keep_current_dealer",
    }


def _is_huangzhuang(data: dict[str, Any], winners: list[int | str]) -> bool:
    # 黄庄口径：无人胡牌且牌墙摸空；同时兼容显式的流局字段。
    if winners:
        return False
    if any(_truthy(data.get(key)) for key in ("huangzhuang", "is_huangzhuang", "draw_round", "liuju")):
        return True
    remain_card_stack = data.get("remain_card_stack")
    if isinstance(remain_card_stack, list):
        return len(remain_card_stack) == 0
    return False


def _dealer_player(data: dict[str, Any]) -> int | None:
    for key in ("dealer", "dealer_player", "banker", "zhuang", "zhuangjia"):
        value = _as_int_or_none(data.get(key))
        if value is not None:
            return value
    return 0


def _player_count(data: dict[str, Any]) -> int:
    for field in (
        "player_hand_cards",
        "played_cards",
        "player_chi_cards",
        "player_peng_cards",
        "player_gang_cards",
        "player_bugang_cards",
        "player_angang_cards",
    ):
        groups = data.get(field)
        if isinstance(groups, list) and groups:
            return len(groups)
    return 4


def _as_int_or_none(value: int | str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y", "shi", "kaiqi"}
    return bool(value)
