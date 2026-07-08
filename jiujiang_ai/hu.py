from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .hand_split import is_four_hongzhong
from .tiles import HONGZHONG, SUITED_TILE_CODES, can_start_sequence, tile_rank, tile_suit, validate_hand

_TILE_INDEX = {tile: index for index, tile in enumerate(SUITED_TILE_CODES)}


@dataclass(frozen=True)
class HuOptions:
    # 七对是九江红中房间选项，默认不开启。
    allow_qidui: bool = False


def can_hu(hand: list[int] | tuple[int, ...], options: HuOptions | None = None) -> bool:
    """判断一副手牌在九江红中规则下是否已经胡牌。"""
    validate_hand(hand)
    options = options or HuOptions()

    # 四红中是特殊规则，不要求走普通 3n+2 牌型判断。
    if is_four_hongzhong(hand):
        return True

    # 普通平胡和七对都要求完整手牌张数满足 3n+2。
    if len(hand) % 3 != 2:
        return False

    hongzhong_count = list(hand).count(HONGZHONG)
    ordinary_counts = _counts_tuple(tile for tile in hand if tile != HONGZHONG)

    if options.allow_qidui and _can_qidui(ordinary_counts, hongzhong_count, len(hand)):
        return True

    return _can_standard_hu(ordinary_counts, hongzhong_count)


def _counts_tuple(tiles) -> tuple[int, ...]:
    counts = Counter(tiles)
    return tuple(counts[tile] for tile in SUITED_TILE_CODES)


def _can_qidui(counts: tuple[int, ...], hongzhong_count: int, hand_size: int) -> bool:
    # 七对固定 14 张；四张相同牌可按两对处理。
    if hand_size != 14:
        return False

    pair_count = 0
    single_count = 0
    for count in counts:
        pair_count += count // 2
        single_count += count % 2

    # 红中优先补孤张成对子，剩余红中每两张还能组成一对。
    red_for_singles = min(hongzhong_count, single_count)
    pair_count += red_for_singles
    remaining_red = hongzhong_count - red_for_singles
    pair_count += remaining_red // 2
    return pair_count >= 7


def _can_standard_hu(counts: tuple[int, ...], hongzhong_count: int) -> bool:
    # 先枚举将牌，再判断剩余牌能否拆成四组面子。
    for pair_counts, remaining_red in _pair_choices(counts, hongzhong_count):
        if _can_form_melds(pair_counts, remaining_red):
            return True
    return False


def _pair_choices(counts: tuple[int, ...], hongzhong_count: int):
    for index, count in enumerate(counts):
        if count >= 2:
            new_counts = _remove_by_index(counts, index, 2)
            yield new_counts, hongzhong_count
        if count >= 1 and hongzhong_count >= 1:
            new_counts = _remove_by_index(counts, index, 1)
            yield new_counts, hongzhong_count - 1

    if hongzhong_count >= 2:
        yield counts, hongzhong_count - 2


@lru_cache(maxsize=None)
def _can_form_melds(counts: tuple[int, ...], hongzhong_count: int) -> bool:
    index = _first_tile_index(counts)
    if index is None:
        # 剩余红中可以每三张组成一组万能面子。
        return hongzhong_count % 3 == 0

    tile = SUITED_TILE_CODES[index]

    # 分支一：当前最小牌作为刻子，缺几张就用几张红中补。
    max_same_tiles = min(3, counts[index])
    for same_tiles in range(1, max_same_tiles + 1):
        red_needed = 3 - same_tiles
        if red_needed <= hongzhong_count:
            new_counts = _remove_by_index(counts, index, same_tiles)
            if _can_form_melds(new_counts, hongzhong_count - red_needed):
                return True

    # 分支二：当前牌作为顺子的一部分，红中可以补顺子两侧缺张。
    for sequence in _sequences_containing(tile):
        for new_counts, red_needed in _sequence_choices(counts, tile, sequence):
            if red_needed <= hongzhong_count and _can_form_melds(new_counts, hongzhong_count - red_needed):
                return True

    return False


def _first_tile_index(counts: tuple[int, ...]) -> int | None:
    for index, count in enumerate(counts):
        if count:
            return index
    return None


def _sequences_containing(tile: int) -> tuple[tuple[int, int, int], ...]:
    sequences: list[tuple[int, int, int]] = []
    suit = tile_suit(tile)
    rank = tile_rank(tile)
    for start_rank in range(rank - 2, rank + 1):
        if not 1 <= start_rank <= 7:
            continue
        start_tile = (tile & 0xF0) + start_rank
        if tile_suit(start_tile) == suit and can_start_sequence(start_tile):
            sequences.append((start_tile, start_tile + 1, start_tile + 2))
    return tuple(sequences)


def _sequence_choices(counts: tuple[int, ...], current_tile: int, sequence: tuple[int, int, int]):
    choices: list[tuple[tuple[int, ...], int]] = [(counts, 0)]
    for tile in sequence:
        next_choices: list[tuple[tuple[int, ...], int]] = []
        tile_index = _TILE_INDEX[tile]
        for current_counts, red_needed in choices:
            if tile == current_tile:
                # 当前最小牌必须真实消耗掉，否则递归会卡在同一张牌。
                if current_counts[tile_index] > 0:
                    next_choices.append((_remove_by_index(current_counts, tile_index, 1), red_needed))
                continue

            if current_counts[tile_index] > 0:
                next_choices.append((_remove_by_index(current_counts, tile_index, 1), red_needed))
            # 即使有真实牌，也允许用红中替代，给其它面子保留真实牌。
            next_choices.append((current_counts, red_needed + 1))
        choices = next_choices
    return choices


def _remove_by_index(counts: tuple[int, ...], index: int, amount: int) -> tuple[int, ...]:
    new_counts = list(counts)
    new_counts[index] -= amount
    return tuple(new_counts)
