from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .tiles import HONGZHONG, SUITED_TILE_CODES, can_start_sequence, is_suited, tile_rank, validate_hand


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
    partials = _split_counts(_counts_tuple(ordinary))
    return min((_apply_hongzhong(partial, hongzhong_count, fixed_melds) for partial in partials), key=_analysis_sort_key)


def _counts_tuple(tiles: tuple[int, ...]) -> tuple[int, ...]:
    counts = Counter(tiles)
    return tuple(counts[tile] for tile in SUITED_TILE_CODES)


@lru_cache(maxsize=None)
def _split_counts(counts: tuple[int, ...]) -> tuple[_Partial, ...]:
    try:
        index = next(i for i, count in enumerate(counts) if count)
    except StopIteration:
        return (_Partial(),)

    tile = SUITED_TILE_CODES[index]
    results: list[_Partial] = []

    def add_branch(
        new_counts: tuple[int, ...],
        melds: int = 0,
        pairs: int = 0,
        taatsu: int = 0,
        leftovers: int = 0,
    ) -> None:
        for child in _split_counts(new_counts):
            results.append(
                _Partial(
                    melds=child.melds + melds,
                    pairs=child.pairs + pairs,
                    taatsu=child.taatsu + taatsu,
                    leftovers=child.leftovers + leftovers,
                )
            )

    if counts[index] >= 3:
        add_branch(_remove_tiles(counts, (tile, tile, tile)), melds=1)

    if can_start_sequence(tile):
        seq = (tile, tile + 1, tile + 2)
        if _has_tiles(counts, seq):
            add_branch(_remove_tiles(counts, seq), melds=1)

    if counts[index] >= 2:
        add_branch(_remove_tiles(counts, (tile, tile)), pairs=1)

    if is_suited(tile):
        if tile_rank(tile) <= 8 and _has_tiles(counts, (tile, tile + 1)):
            add_branch(_remove_tiles(counts, (tile, tile + 1)), taatsu=1)
        if tile_rank(tile) <= 7 and _has_tiles(counts, (tile, tile + 2)):
            add_branch(_remove_tiles(counts, (tile, tile + 2)), taatsu=1)

    add_branch(_remove_tiles(counts, (tile,)), leftovers=1)
    return tuple(results)


def _has_tiles(counts: tuple[int, ...], tiles: tuple[int, ...]) -> bool:
    needed = Counter(tiles)
    return all(counts[SUITED_TILE_CODES.index(tile)] >= amount for tile, amount in needed.items())


def _remove_tiles(counts: tuple[int, ...], tiles: tuple[int, ...]) -> tuple[int, ...]:
    new_counts = list(counts)
    for tile in tiles:
        new_counts[SUITED_TILE_CODES.index(tile)] -= 1
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
