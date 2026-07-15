from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .tiles import HONGZHONG, SUITED_TILE_CODES, validate_hand


@dataclass(frozen=True)
class HandAnalysis:
    shanten: int
    melds: int
    pairs: int
    taatsu: int
    leftovers: int
    hongzhong_count: int
    hongzhong_used: int


@dataclass(frozen=True)
class _Partial:
    melds: int = 0
    pairs: int = 0
    taatsu: int = 0
    leftovers: int = 0


def is_four_hongzhong(tiles: list[int] | tuple[int, ...]) -> bool:
    return list(tiles).count(HONGZHONG) >= 4


def analyze_hand(hand: list[int] | tuple[int, ...], fixed_melds: int = 0) -> HandAnalysis:
    validate_hand(hand)
    hongzhong_count = list(hand).count(HONGZHONG)
    ordinary = tuple(sorted(tile for tile in hand if tile != HONGZHONG))
    return _analyze_counts(_counts_tuple(ordinary), hongzhong_count, fixed_melds)


@lru_cache(maxsize=100_000)
def _analyze_counts(
    counts: tuple[int, ...],
    hongzhong_count: int,
    fixed_melds: int,
) -> HandAnalysis:
    """分析规范化牌计数；相同子局面跨候选、跨请求直接复用。"""
    partials = _split_counts(counts)
    return min((_apply_hongzhong(partial, hongzhong_count, fixed_melds) for partial in partials), key=_analysis_sort_key)


def _counts_tuple(tiles: tuple[int, ...]) -> tuple[int, ...]:
    counts = Counter(tiles)
    return tuple(counts[tile] for tile in SUITED_TILE_CODES)


@lru_cache(maxsize=50_000)
def _split_counts(counts: tuple[int, ...]) -> tuple[_Partial, ...]:
    """分别拆分三门数牌后合并摘要，避免跨花色递归的组合爆炸。"""
    partials = (_Partial(),)
    for start in range(0, len(SUITED_TILE_CODES), 9):
        partials = _combine_partials(partials, _split_suit_counts(counts[start : start + 9]))
    return partials


@lru_cache(maxsize=20_000)
def _split_suit_counts(counts: tuple[int, ...]) -> tuple[_Partial, ...]:
    """拆分单门1到9；顺子和搭子不会跨花色，因此可独立缓存。"""
    try:
        index = next(i for i, count in enumerate(counts) if count)
    except StopIteration:
        return (_Partial(),)

    results: list[_Partial] = []
    seen: set[_Partial] = set()

    def add_branch(
        new_counts: tuple[int, ...],
        melds: int = 0,
        pairs: int = 0,
        taatsu: int = 0,
        leftovers: int = 0,
    ) -> None:
        for child in _split_suit_counts(new_counts):
            result = _Partial(
                melds=child.melds + melds,
                pairs=child.pairs + pairs,
                taatsu=child.taatsu + taatsu,
                leftovers=child.leftovers + leftovers,
            )
            if result not in seen:
                seen.add(result)
                results.append(result)

    if counts[index] >= 3:
        add_branch(_remove_indexes(counts, (index, index, index)), melds=1)

    if index <= 6 and counts[index + 1] and counts[index + 2]:
        add_branch(_remove_indexes(counts, (index, index + 1, index + 2)), melds=1)

    if counts[index] >= 2:
        add_branch(_remove_indexes(counts, (index, index)), pairs=1)

    if index <= 7 and counts[index + 1]:
        add_branch(_remove_indexes(counts, (index, index + 1)), taatsu=1)
    if index <= 6 and counts[index + 2]:
        add_branch(_remove_indexes(counts, (index, index + 2)), taatsu=1)

    add_branch(_remove_indexes(counts, (index,)), leftovers=1)
    # 不同递归路径经常得到完全相同的结构摘要；add_branch 已按首次出现
    # 的顺序去重，避免先堆积大量重复对象再统一清理。
    return tuple(results)


def _combine_partials(
    left_partials: tuple[_Partial, ...],
    right_partials: tuple[_Partial, ...],
) -> tuple[_Partial, ...]:
    results: list[_Partial] = []
    seen: set[_Partial] = set()
    for left in left_partials:
        for right in right_partials:
            result = _Partial(
                melds=left.melds + right.melds,
                pairs=left.pairs + right.pairs,
                taatsu=left.taatsu + right.taatsu,
                leftovers=left.leftovers + right.leftovers,
            )
            if result not in seen:
                seen.add(result)
                results.append(result)
    return tuple(results)


def _remove_indexes(counts: tuple[int, ...], indexes: tuple[int, ...]) -> tuple[int, ...]:
    new_counts = list(counts)
    for index in indexes:
        new_counts[index] -= 1
    return tuple(new_counts)


def _apply_hongzhong(partial: _Partial, hongzhong_count: int, fixed_melds: int) -> HandAnalysis:
    best: HandAnalysis | None = None
    for pair_mode in range(3):
        for meld_red_count in range(hongzhong_count + 1):
            analysis = _try_hongzhong_plan(partial, hongzhong_count, pair_mode, meld_red_count, fixed_melds)
            if analysis is None:
                continue
            if best is None or _analysis_sort_key(analysis) < _analysis_sort_key(best):
                best = analysis
    assert best is not None
    return best


def _try_hongzhong_plan(
    partial: _Partial,
    hongzhong_count: int,
    pair_mode: int,
    meld_red_count: int,
    fixed_melds: int,
) -> HandAnalysis | None:
    pairs = partial.pairs
    melds = partial.melds
    taatsu = partial.taatsu
    leftovers = partial.leftovers
    used = 0

    if pair_mode == 1:
        if pairs > 0 or leftovers <= 0 or hongzhong_count < 1:
            return None
        pairs = 1
        leftovers -= 1
        used += 1
    elif pair_mode == 2:
        if pairs > 0 or hongzhong_count < 2:
            return None
        pairs = 1
        used += 2

    if used + meld_red_count > hongzhong_count:
        return None

    promoted_taatsu = min(taatsu, meld_red_count)
    melds += promoted_taatsu
    taatsu -= promoted_taatsu
    used += promoted_taatsu

    remaining_red = hongzhong_count - used
    promoted_leftovers = min(leftovers, remaining_red // 2)
    melds += promoted_leftovers
    leftovers -= promoted_leftovers
    used += promoted_leftovers * 2

    return _calculate_analysis(melds, pairs, taatsu, leftovers, hongzhong_count, used, fixed_melds)


def _calculate_analysis(
    melds: int,
    pairs: int,
    taatsu: int,
    leftovers: int,
    hongzhong_count: int,
    hongzhong_used: int,
    fixed_melds: int,
) -> HandAnalysis:
    capped_melds = min(4, fixed_melds + melds)
    has_pair = pairs > 0
    extra_pairs_as_taatsu = max(0, pairs - (1 if has_pair else 0))
    incomplete = extra_pairs_as_taatsu + taatsu
    useful_incomplete = min(incomplete, max(0, 4 - capped_melds))
    shanten = 9 - 2 * capped_melds - useful_incomplete - (1 if has_pair else 0)
    return HandAnalysis(
        shanten=max(0, shanten),
        melds=capped_melds,
        pairs=pairs,
        taatsu=taatsu,
        leftovers=leftovers,
        hongzhong_count=hongzhong_count,
        hongzhong_used=hongzhong_used,
    )


def _analysis_sort_key(analysis: HandAnalysis) -> tuple[int, int, int, int, int]:
    return (
        analysis.shanten,
        -analysis.melds,
        -(analysis.pairs + analysis.taatsu),
        analysis.leftovers,
        analysis.hongzhong_used,
    )
