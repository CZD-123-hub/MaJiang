"""基于可见局面的首版弃牌风险评估。

该模块只使用公开弃牌、副露、听牌动作和牌墙长度；它不假装知道对手暗手牌。
因此结果是可解释的启发式风险，而不是精确放铳概率。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .rules import ACTION_DISCARD, ACTION_TING
from .tiles import JIUJIANG_TILE_SET


@dataclass(frozen=True)
class DiscardRisk:
    tile: int
    score: float
    dangerous_opponents: tuple[int, ...]
    reasons: tuple[str, ...]


def evaluate_discard_risks(
    data: dict,
    *,
    acting_position: int,
    candidates: Iterable[int],
    opponent_positions: Iterable[int] | None = None,
) -> dict[int, DiscardRisk]:
    """评估候选牌的相对风险，返回范围为 0 到 1 的保守分数。"""
    visible_counts, opponent_discards = _visible_discards(data)
    opponent_threats = _opponent_threats(data, acting_position, opponent_positions=opponent_positions)
    risks: dict[int, DiscardRisk] = {}
    for tile in set(candidates):
        if tile not in JIUJIANG_TILE_SET:
            continue
        risks[tile] = _tile_risk(tile, visible_counts, opponent_discards, opponent_threats)
    return risks


def _tile_risk(
    tile: int,
    visible_counts: Counter[int],
    opponent_discards: dict[int, set[int]],
    opponent_threats: dict[int, tuple[float, tuple[str, ...]]],
) -> DiscardRisk:
    if visible_counts[tile] >= 4:
        return DiscardRisk(tile=tile, score=0.0, dangerous_opponents=(), reasons=("all_copies_visible",))

    values: list[tuple[int, float, tuple[str, ...]]] = []
    for opponent, (threat, reasons) in opponent_threats.items():
        if tile in opponent_discards.get(opponent, set()):
            values.append((opponent, threat * 0.15, reasons + ("opponent_discard",)))
        else:
            values.append((opponent, threat, reasons))

    if not values:
        return DiscardRisk(tile=tile, score=0.0, dangerous_opponents=(), reasons=("no_opponent_data",))
    # 对每家独立近似，再合并为“至少一名对手危险”的概率；这样一名对手的
    # 现物安全牌会降低总风险，但不会错误地把其余对手也当成安全。
    combined = 1.0
    for _, value, _ in values:
        combined *= 1.0 - value
    combined = 1.0 - combined
    highest = max(value for _, value, _ in values)
    dangerous = tuple(opponent for opponent, value, _ in values if value == highest)
    # 保留所有影响过该候选牌的解释项；最危险对手仍单独由 dangerous_opponents
    # 给出，调用方可以同时看到“某一家曾打过此牌”的降险原因。
    reason_set = {reason for _, _, reasons in values for reason in reasons}
    return DiscardRisk(
        tile=tile,
        score=round(min(1.0, combined), 4),
        dangerous_opponents=dangerous,
        reasons=tuple(sorted(reason_set)),
    )


def _opponent_threats(
    data: dict,
    acting_position: int,
    *,
    opponent_positions: Iterable[int] | None = None,
) -> dict[int, tuple[float, tuple[str, ...]]]:
    player_count = _player_count(data)
    selected_opponents = (
        {position for position in opponent_positions if 0 <= position < player_count and position != acting_position}
        if opponent_positions is not None
        else set(range(player_count)) - {acting_position}
    )
    turn_count = sum(1 for action in data.get("action_seq") or [] if _is_action(action, ACTION_DISCARD))
    wall_size = len(data["remain_card_stack"]) if isinstance(data.get("remain_card_stack"), list) else None
    ting_players = {
        action[0]
        for action in data.get("action_seq") or []
        if _is_action(action, ACTION_TING) and isinstance(action[0], int)
    }
    meld_counts = _meld_counts(data, player_count)

    threats: dict[int, tuple[float, tuple[str, ...]]] = {}
    for opponent in sorted(selected_opponents):
        score = 0.18 + min(turn_count, 18) * 0.015 + min(meld_counts[opponent], 4) * 0.09
        reasons: list[str] = []
        if turn_count:
            reasons.append("round_progress")
        if meld_counts[opponent]:
            reasons.append("open_meld")
        if opponent in ting_players:
            score += 0.35
            reasons.append("opponent_ting")
        if wall_size is not None and wall_size <= 20:
            score += 0.12
            reasons.append("late_wall")
        if wall_size is not None and wall_size <= 10:
            score += 0.10
        threats[opponent] = (min(0.95, score), tuple(reasons))
    return threats


def _visible_discards(data: dict) -> tuple[Counter[int], dict[int, set[int]]]:
    visible = Counter()
    by_player: dict[int, set[int]] = {}
    played_cards = data.get("played_cards") or []
    if any(played_cards):
        for player, cards in enumerate(played_cards):
            for tile in cards or []:
                if tile in JIUJIANG_TILE_SET:
                    visible[tile] += 1
                    by_player.setdefault(player, set()).add(tile)
        return visible, by_player

    for action in data.get("action_seq") or []:
        if not _is_action(action, ACTION_DISCARD) or len(action) < 3:
            continue
        player, _, tile = action[:3]
        if isinstance(player, int) and tile in JIUJIANG_TILE_SET:
            visible[tile] += 1
            by_player.setdefault(player, set()).add(tile)
    return visible, by_player


def _meld_counts(data: dict, player_count: int) -> list[int]:
    result = [0] * player_count
    for field in (
        "player_peng_cards",
        "player_gang_cards",
        "player_bugang_cards",
        "player_angang_cards",
        "player_bu_cards",
    ):
        groups_by_player = data.get(field) or []
        for player, groups in enumerate(groups_by_player[:player_count]):
            result[player] += len(groups or [])
    return result


def _player_count(data: dict) -> int:
    for field in ("player_hand_cards", "played_cards", "player_peng_cards"):
        values = data.get(field)
        if isinstance(values, list) and values:
            return max(4, len(values))
    return 4


def _is_action(action: object, action_type: int) -> bool:
    return isinstance(action, (list, tuple)) and len(action) >= 2 and action[1] == action_type
