from __future__ import annotations

from typing import Any

from .tiles import HONGZHONG, JIUJIANG_TILE_SET, tile_rank


def calculate_zama_score(
    data: dict[str, Any],
    winner: int | str | None = None,
) -> dict[str, Any]:
    """计算扎码分。"""
    resolved_winner = _as_int_or_none(winner if winner is not None else _default_winner(data))
    zama_count = _zama_count(data)
    zama_cards = _zama_cards(data)

    if resolved_winner is None or zama_count <= 0:
        return {
            "winner": resolved_winner,
            "requested_count": zama_count,
            "extra_for_no_hongzhong": False,
            "extra_for_four_hongzhong": 0,
            "awarded_cards": [],
            "raw_zama_score": 0,
            "base_score_multiplier": 1,
            "zama_score": 0,
        }

    winner_hand = _winner_hand(data, resolved_winner)
    extra_for_no_hongzhong = HONGZHONG not in winner_hand
    awarded_count = zama_count + (1 if extra_for_no_hongzhong else 0)
    awarded_cards = zama_cards[:awarded_count]
    raw_zama_score = sum(_zama_tile_score(tile) for tile in awarded_cards)

    # 四红中直接胡牌时，按任务书要求额外增加 4 分码分。
    extra_for_four_hongzhong = 4 if winner_hand.count(HONGZHONG) >= 4 else 0
    raw_zama_score += extra_for_four_hongzhong

    # 开启“码跟底分”时，再按底分做一次倍率放大。
    base_score_multiplier = _base_score(data) if _zama_follow_base_score(data) else 1
    zama_score = raw_zama_score * base_score_multiplier

    return {
        "winner": resolved_winner,
        "requested_count": zama_count,
        "extra_for_no_hongzhong": extra_for_no_hongzhong,
        "extra_for_four_hongzhong": extra_for_four_hongzhong,
        "awarded_cards": awarded_cards,
        "raw_zama_score": raw_zama_score,
        "base_score_multiplier": base_score_multiplier,
        "zama_score": zama_score,
    }


def _zama_count(data: dict[str, Any]) -> int:
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    for source in option_sources:
        value = _as_int_or_none(source.get("zama_count"))
        if value is not None:
            return max(value, 0)
    return 0


def _zama_cards(data: dict[str, Any]) -> list[int]:
    # 优先兼容几种常见字段名，只保留九江麻将合法牌。
    for key in ("zama_cards", "zama_draw_cards", "ma_cards", "zha_ma_cards"):
        values = data.get(key)
        if isinstance(values, list):
            return [tile for tile in values if tile in JIUJIANG_TILE_SET]
    return []


def _winner_hand(data: dict[str, Any], winner: int) -> list[int]:
    hands = data.get("player_hand_cards") or []
    if 0 <= winner < len(hands):
        return list(hands[winner] or [])
    explicit_hand = data.get("winning_hand_cards")
    if isinstance(explicit_hand, list):
        return list(explicit_hand)
    return []


def _zama_follow_base_score(data: dict[str, Any]) -> bool:
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = ("zama_follow_base_score", "ma_follow_base_score", "码跟底分")
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _base_score(data: dict[str, Any]) -> int:
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    for source in option_sources:
        value = _as_int_or_none(source.get("base_score"))
        if value is not None and value > 0:
            return value
    return 1


def _zama_tile_score(tile: int) -> int:
    if tile == HONGZHONG:
        return 10
    return tile_rank(tile)


def _default_winner(data: dict[str, Any]) -> int | str | None:
    if "winner" in data and data["winner"] is not None:
        return data["winner"]
    winners = data.get("winners")
    if isinstance(winners, list) and winners:
        return winners[0]
    hu_players = data.get("hu_players")
    if isinstance(hu_players, list) and hu_players:
        return hu_players[0]
    return None


def _as_int_or_none(value: int | str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y", "shi", "kaiqi"}
    return bool(value)
