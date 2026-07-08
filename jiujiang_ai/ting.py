from __future__ import annotations

from dataclasses import dataclass

from .hu import HuOptions, can_hu
from .tiles import JIUJIANG_TILE_CODES, remaining_tile_counts, validate_hand


@dataclass(frozen=True)
class TingDiscard:
    discard: int
    winning_tiles: dict[int, int]

    @property
    def effective_count(self) -> int:
        # 有效进张总数等于所有可胡牌剩余张数之和。
        return sum(self.winning_tiles.values())


def winning_tile_counts(hand: list[int] | tuple[int, ...], options: HuOptions | None = None) -> dict[int, int]:
    """返回当前 13 张手牌摸哪些牌可以胡，以及这些牌还剩多少张。"""
    validate_hand(hand)
    options = options or HuOptions()
    remaining = remaining_tile_counts(hand)
    result: dict[int, int] = {}

    for tile in JIUJIANG_TILE_CODES:
        if remaining[tile] <= 0:
            continue
        # 逐张模拟摸牌，并调用真实胡牌判断，避免只用向听数近似。
        if can_hu([*hand, tile], options):
            result[tile] = remaining[tile]

    return result


def is_ting(hand: list[int] | tuple[int, ...], options: HuOptions | None = None) -> bool:
    """判断当前 13 张手牌是否已经听牌。"""
    return bool(winning_tile_counts(hand, options))


def ting_discards(
    hand: list[int] | tuple[int, ...],
    candidate_cards: list[list[int]] | None = None,
    options: HuOptions | None = None,
) -> dict[int, TingDiscard]:
    """枚举 14 张手牌打出哪些候选牌后可以进入听牌。"""
    validate_hand(hand)
    options = options or HuOptions()
    candidates = candidate_cards or [[tile] for tile in sorted(set(hand))]
    results: dict[int, TingDiscard] = {}

    for card_group in candidates:
        if not card_group:
            continue
        discard = card_group[0]
        if discard not in hand:
            continue
        after_discard = list(hand)
        after_discard.remove(discard)
        winning_tiles = winning_tile_counts(after_discard, options)
        if winning_tiles:
            results[discard] = TingDiscard(discard=discard, winning_tiles=winning_tiles)

    return results
