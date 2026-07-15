from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

from .hand_split import analyze_hand
from .ting import winning_tile_counts
from .tiles import HONGZHONG, JIUJIANG_TILE_CODES, remaining_tile_counts, validate_hand


@dataclass(frozen=True)
class DiscardDecision:
    discard: int
    score: float
    shanten_after_discard: int
    effective_count: int
    winning_tiles: dict[int, int]
    safety_score: int


def score_discards(
    hand: list[int],
    candidate_cards: list[list[int]],
    fixed_melds: int = 0,
    visible_discards: Mapping[int, int] | None = None,
    remaining_counts: Mapping[int, int] | None = None,
    deadline: float | None = None,
) -> dict[int, DiscardDecision]:
    validate_hand(hand)
    visible_discards = visible_discards or {}
    scores: dict[int, DiscardDecision] = {}
    for card_group in candidate_cards:
        _check_deadline(deadline)
        if not card_group:
            continue
        discard = card_group[0]
        # 平台通常会按实体牌返回候选，同一种牌可能出现多次；按牌值评分一次即可。
        if discard not in hand or discard in scores:
            continue
        after = list(hand)
        after.remove(discard)
        analysis = analyze_hand(after, fixed_melds=fixed_melds)
        # 如果打完后已经听牌，优先使用真实胡牌张数；否则再用向听下降近似有效进张。
        winning_tiles = winning_tile_counts(
            after,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
        )
        _check_deadline(deadline)
        effective_count = (
            sum(winning_tiles.values())
            if winning_tiles
            else _effective_draw_count(
                after,
                analysis.shanten,
                fixed_melds=fixed_melds,
                remaining_counts=remaining_counts,
                deadline=deadline,
            )
        )
        score = _score_analysis(analysis, effective_count)
        scores[discard] = DiscardDecision(
            discard=discard,
            score=score,
            shanten_after_discard=analysis.shanten,
            effective_count=effective_count,
            winning_tiles=winning_tiles,
            safety_score=int(visible_discards.get(discard, 0)),
        )
    return scores


def choose_discard(
    hand: list[int],
    candidate_cards: list[list[int]],
    fixed_melds: int = 0,
    visible_discards: Mapping[int, int] | None = None,
    remaining_counts: Mapping[int, int] | None = None,
    deadline: float | None = None,
) -> DiscardDecision:
    try:
        scores = score_discards(
            hand,
            candidate_cards,
            fixed_melds=fixed_melds,
            visible_discards=visible_discards,
            remaining_counts=remaining_counts,
            deadline=deadline,
        )
    except _DecisionDeadlineExceeded:
        return choose_fast_discard(
            hand,
            candidate_cards,
            fixed_melds=fixed_melds,
            visible_discards=visible_discards,
        )
    if not scores:
        raise ValueError("no valid discard candidates")
    return max(scores.values(), key=_discard_sort_key)


def choose_fast_discard(
    hand: list[int],
    candidate_cards: list[list[int]],
    fixed_melds: int = 0,
    visible_discards: Mapping[int, int] | None = None,
) -> DiscardDecision:
    """并发繁忙时使用的常数级近似策略，不进入递归手牌分析。"""
    validate_hand(hand)
    visible_discards = visible_discards or {}
    decisions: dict[int, DiscardDecision] = {}
    for card_group in candidate_cards:
        if not card_group:
            continue
        discard = card_group[0]
        if discard not in hand or discard in decisions:
            continue
        after = list(hand)
        after.remove(discard)
        decisions[discard] = DiscardDecision(
            discard=discard,
            score=_fast_shape_score(after),
            shanten_after_discard=0,
            effective_count=0,
            winning_tiles={},
            safety_score=int(visible_discards.get(discard, 0)),
        )
    if not decisions:
        raise ValueError("no valid discard candidates")
    return max(decisions.values(), key=_discard_sort_key)


def _fast_shape_score(hand: list[int]) -> float:
    """用对子、刻子和同门邻张近似保留价值，保证繁忙回退在毫秒内完成。"""
    counts = Counter(hand)
    score = counts[HONGZHONG] * 8.0
    for tile, count in counts.items():
        if tile == HONGZHONG:
            continue
        score += max(0, count - 1) * 3.0 + max(0, count - 2) * 2.0
        rank = tile & 0x0F
        if rank <= 8:
            score += min(count, counts.get(tile + 1, 0)) * 1.6
        if rank <= 7:
            score += min(count, counts.get(tile + 2, 0)) * 0.8
    return score


def hand_value(
    hand: list[int],
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
) -> float:
    analysis = analyze_hand(hand, fixed_melds=fixed_melds)
    return _score_analysis(
        analysis,
        _effective_draw_count(hand, analysis.shanten, fixed_melds=fixed_melds, remaining_counts=remaining_counts),
    )


def _discard_sort_key(decision: DiscardDecision) -> tuple:
    # 第一层：真实能进听的出牌优先，避免被向听近似分误导。
    if decision.winning_tiles:
        return (
            1,
            decision.effective_count,
            decision.score,
            decision.safety_score,
            decision.discard != HONGZHONG,
            -decision.shanten_after_discard,
            -decision.discard,
        )

    # 未进听时仍沿用综合分，红中只作为同分附近的保留倾向。
    return (
        0,
        decision.score,
        decision.effective_count,
        decision.safety_score,
        decision.discard != HONGZHONG,
        -decision.shanten_after_discard,
        -decision.discard,
    )


def _effective_draw_count(
    hand: list[int],
    current_shanten: int,
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
    deadline: float | None = None,
) -> int:
    remaining = remaining_counts or remaining_tile_counts(hand)
    count = 0
    for tile in JIUJIANG_TILE_CODES:
        _check_deadline(deadline)
        if remaining.get(tile, 0) <= 0:
            continue
        next_hand = sorted([*hand, tile])
        if analyze_hand(next_hand, fixed_melds=fixed_melds).shanten < current_shanten:
            count += remaining.get(tile, 0)
    return count


class _DecisionDeadlineExceeded(RuntimeError):
    pass


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise _DecisionDeadlineExceeded


def _score_analysis(analysis, effective_count: int) -> float:
    return (
        100
        - 30 * analysis.shanten
        + 4 * analysis.melds
        + 1.5 * analysis.taatsu
        + analysis.pairs
        + 0.8 * effective_count
        - 0.5 * analysis.leftovers
        - 0.3 * analysis.hongzhong_used
    )
