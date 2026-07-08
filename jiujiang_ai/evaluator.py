from __future__ import annotations

from dataclasses import dataclass

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


def score_discards(hand: list[int], candidate_cards: list[list[int]]) -> dict[int, DiscardDecision]:
    validate_hand(hand)
    scores: dict[int, DiscardDecision] = {}
    for card_group in candidate_cards:
        if not card_group:
            continue
        discard = card_group[0]
        if discard not in hand:
            continue
        after = list(hand)
        after.remove(discard)
        analysis = analyze_hand(after)
        # 如果打完后已经听牌，优先使用真实胡牌张数；否则再用向听下降近似有效进张。
        winning_tiles = winning_tile_counts(after)
        effective_count = sum(winning_tiles.values()) if winning_tiles else _effective_draw_count(after, analysis.shanten)
        score = _score_analysis(analysis, effective_count)
        scores[discard] = DiscardDecision(
            discard=discard,
            score=score,
            shanten_after_discard=analysis.shanten,
            effective_count=effective_count,
            winning_tiles=winning_tiles,
        )
    return scores


def choose_discard(hand: list[int], candidate_cards: list[list[int]]) -> DiscardDecision:
    scores = score_discards(hand, candidate_cards)
    if not scores:
        raise ValueError("no valid discard candidates")
    return max(scores.values(), key=_discard_sort_key)


def hand_value(hand: list[int]) -> float:
    analysis = analyze_hand(hand)
    return _score_analysis(analysis, _effective_draw_count(hand, analysis.shanten))


def _discard_sort_key(decision: DiscardDecision) -> tuple:
    # 第一层：真实能进听的出牌优先，避免被向听近似分误导。
    if decision.winning_tiles:
        return (
            1,
            decision.effective_count,
            decision.discard != HONGZHONG,
            decision.score,
            -decision.shanten_after_discard,
            -decision.discard,
        )

    # 未进听时仍沿用综合分，红中只作为同分附近的保留倾向。
    return (
        0,
        decision.score,
        decision.effective_count,
        decision.discard != HONGZHONG,
        -decision.shanten_after_discard,
        -decision.discard,
    )


def _effective_draw_count(hand: list[int], current_shanten: int) -> int:
    remaining = remaining_tile_counts(hand)
    count = 0
    for tile in JIUJIANG_TILE_CODES:
        if remaining[tile] <= 0:
            continue
        next_hand = sorted([*hand, tile])
        if analyze_hand(next_hand).shanten < current_shanten:
            count += remaining[tile]
    return count


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
