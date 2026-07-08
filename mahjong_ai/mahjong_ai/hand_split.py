from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .tiles import TILE_CODES, can_start_sequence, is_suited, tile_rank, validate_hand


@dataclass(frozen=True)
class HandCombination:
    """一次完整的手牌拆分结果。

    这个结构正好对应任务书要求的输出信息：
    刻子集合、顺子集合、搭子集合、向听数、剩余牌。
    """

    triplets: tuple[tuple[int, int, int], ...] = ()
    sequences: tuple[tuple[int, int, int], ...] = ()
    pairs: tuple[tuple[int, int], ...] = ()
    taatsu: tuple[tuple[int, int], ...] = ()
    leftovers: tuple[int, ...] = ()
    shanten: int = 8

    @property
    def pair(self) -> tuple[int, int] | None:
        """返回第一个对子；平胡中它可以作为将牌候选。"""
        return self.pairs[0] if self.pairs else None

    def all_tiles(self) -> list[int]:
        """把本次拆分里的所有牌重新合并，主要用于测试校验。"""
        tiles: list[int] = []
        for group in self.triplets + self.sequences + self.pairs + self.taatsu:
            tiles.extend(group)
        tiles.extend(self.leftovers)
        return tiles

    def as_task_output(self) -> dict[str, object]:
        """按任务书里的中文字段组织输出。"""
        return {
            "刻子集合": [list(group) for group in self.triplets],
            "顺子集合": [list(group) for group in self.sequences],
            "搭子集合": [list(group) for group in self.pairs + self.taatsu],
            "向听数": self.shanten,
            "剩余牌": list(self.leftovers),
        }


@dataclass(frozen=True)
class _PartialCombination:
    """递归搜索过程中的临时拆分结果。

    这里不带向听数，因为递归还没结束；等所有牌都处理完后，
    再统一计算向听数并转成 HandCombination。
    """

    triplets: tuple[tuple[int, int, int], ...] = ()
    sequences: tuple[tuple[int, int, int], ...] = ()
    pairs: tuple[tuple[int, int], ...] = ()
    taatsu: tuple[tuple[int, int], ...] = ()
    leftovers: tuple[int, ...] = ()


def split_hand(hand: list[int] | tuple[int, ...], limit: int | None = None) -> list[HandCombination]:
    """计算一副手牌的所有平胡拆分组合。

    参数 limit 用于只返回前 N 个较优组合，演示或搜索时可以减少输出量。
    """
    validate_hand(hand)
    counts = _counts_tuple(hand)
    partials = _split_counts(counts)
    combinations = [_finish_combination(partial) for partial in partials]
    combinations = _dedupe_combinations(combinations)
    combinations.sort(key=_combination_sort_key)
    return combinations[:limit] if limit else combinations


def best_shanten(hand: list[int] | tuple[int, ...]) -> int:
    """返回这副牌在所有拆分中的最小向听数。"""
    return split_hand(hand, limit=1)[0].shanten


def effective_tiles(hand: list[int] | tuple[int, ...]) -> list[int]:
    """计算能让当前向听数变小的有效进张。"""
    validate_hand(hand)
    current = best_shanten(hand)
    counts = Counter(hand)
    effective: list[int] = []
    for tile in TILE_CODES:
        if counts[tile] >= 4:
            continue
        if best_shanten(tuple(sorted([*hand, tile]))) < current:
            effective.append(tile)
    return effective


def _counts_tuple(hand: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """把手牌转成固定长度的计数元组，便于递归缓存。"""
    counts = Counter(hand)
    return tuple(counts[tile] for tile in TILE_CODES)


@lru_cache(maxsize=None)
def _split_counts(counts: tuple[int, ...]) -> tuple[_PartialCombination, ...]:
    """递归枚举所有可能拆法。

    核心思路：
    1. 找到当前剩余牌里最小的一张 tile。
    2. 尝试把它作为刻子、顺子、对子、搭子或孤张。
    3. 每尝试一种，就从计数里移除对应牌，再继续递归。
    """
    try:
        index = next(i for i, count in enumerate(counts) if count)
    except StopIteration:
        return (_PartialCombination(),)

    tile = TILE_CODES[index]
    results: list[_PartialCombination] = []

    def add_branch(new_counts: tuple[int, ...], **added: tuple[int, ...] | int) -> None:
        """把当前选择接到后续递归结果前面。"""
        for child in _split_counts(new_counts):
            results.append(_prepend(child, **added))

    # 1. 尝试刻子 AAA。
    if counts[index] >= 3:
        add_branch(_remove_tiles(counts, (tile, tile, tile)), triplet=(tile, tile, tile))

    # 2. 尝试顺子 ABC。只有万、条、筒可以组成顺子。
    if can_start_sequence(tile):
        seq = (tile, tile + 1, tile + 2)
        if _has_tiles(counts, seq):
            add_branch(_remove_tiles(counts, seq), sequence=seq)

    # 3. 尝试对子 AA。它可能是将牌，也可能只是一个搭子。
    if counts[index] >= 2:
        add_branch(_remove_tiles(counts, (tile, tile)), pair=(tile, tile))

    # 4. 尝试两张搭子：AB 连张，AC 嵌张。
    if is_suited(tile):
        if tile_rank(tile) <= 8 and _has_tiles(counts, (tile, tile + 1)):
            add_branch(_remove_tiles(counts, (tile, tile + 1)), taatsu=(tile, tile + 1))
        if tile_rank(tile) <= 7 and _has_tiles(counts, (tile, tile + 2)):
            add_branch(_remove_tiles(counts, (tile, tile + 2)), taatsu=(tile, tile + 2))

    # 5. 如果上面的组合都不选，也可以把这张牌当作剩余孤张。
    add_branch(_remove_tiles(counts, (tile,)), leftover=tile)
    return tuple(results)


def _has_tiles(counts: tuple[int, ...], tiles: tuple[int, ...]) -> bool:
    """判断当前计数里是否包含指定的一组牌。"""
    needed = Counter(tiles)
    return all(counts[TILE_CODES.index(tile)] >= amount for tile, amount in needed.items())


def _remove_tiles(counts: tuple[int, ...], tiles: tuple[int, ...]) -> tuple[int, ...]:
    """从计数元组中移除一组牌，并返回新的计数元组。"""
    new_counts = list(counts)
    for tile in tiles:
        new_counts[TILE_CODES.index(tile)] -= 1
    return tuple(new_counts)


def _prepend(child: _PartialCombination, **added: tuple[int, ...] | int) -> _PartialCombination:
    """把当前递归层选出的组合放到子结果前面。"""
    if "triplet" in added:
        return _PartialCombination(
            triplets=(added["triplet"], *child.triplets),  # type: ignore[arg-type]
            sequences=child.sequences,
            pairs=child.pairs,
            taatsu=child.taatsu,
            leftovers=child.leftovers,
        )
    if "sequence" in added:
        return _PartialCombination(
            triplets=child.triplets,
            sequences=(added["sequence"], *child.sequences),  # type: ignore[arg-type]
            pairs=child.pairs,
            taatsu=child.taatsu,
            leftovers=child.leftovers,
        )
    if "pair" in added:
        return _PartialCombination(
            triplets=child.triplets,
            sequences=child.sequences,
            pairs=(added["pair"], *child.pairs),  # type: ignore[arg-type]
            taatsu=child.taatsu,
            leftovers=child.leftovers,
        )
    if "taatsu" in added:
        return _PartialCombination(
            triplets=child.triplets,
            sequences=child.sequences,
            pairs=child.pairs,
            taatsu=(added["taatsu"], *child.taatsu),  # type: ignore[arg-type]
            leftovers=child.leftovers,
        )
    return _PartialCombination(
        triplets=child.triplets,
        sequences=child.sequences,
        pairs=child.pairs,
        taatsu=child.taatsu,
        leftovers=(added["leftover"], *child.leftovers),  # type: ignore[arg-type]
    )


def _finish_combination(partial: _PartialCombination) -> HandCombination:
    """递归结束后，整理排序并计算向听数。"""
    return HandCombination(
        triplets=tuple(sorted(partial.triplets)),
        sequences=tuple(sorted(partial.sequences)),
        pairs=tuple(sorted(partial.pairs)),
        taatsu=tuple(sorted(partial.taatsu)),
        leftovers=tuple(sorted(partial.leftovers)),
        shanten=_task_shanten(partial),
    )


def _task_shanten(partial: _PartialCombination) -> int:
    """计算任务书语境下的平胡向听数。

    本项目采用任务书写法：胡牌向听数为 0，听牌向听数为 1。
    平胡目标是 4 个面子 + 1 个将牌，所以公式围绕缺多少面子和将牌计算。
    """
    melds = min(4, len(partial.triplets) + len(partial.sequences))
    has_pair = bool(partial.pairs)
    pair_count_as_taatsu = max(0, len(partial.pairs) - (1 if has_pair else 0))
    incomplete = pair_count_as_taatsu + len(partial.taatsu)
    useful_incomplete = min(incomplete, max(0, 4 - melds))
    shanten = 9 - 2 * melds - useful_incomplete - (1 if has_pair else 0)
    return max(0, shanten)


def _dedupe_combinations(combinations: list[HandCombination]) -> list[HandCombination]:
    """去掉完全重复的拆分结果。"""
    seen: set[tuple[object, ...]] = set()
    unique: list[HandCombination] = []
    for combo in combinations:
        key = (combo.triplets, combo.sequences, combo.pairs, combo.taatsu, combo.leftovers)
        if key in seen:
            continue
        seen.add(key)
        unique.append(combo)
    return unique


def _combination_sort_key(combo: HandCombination) -> tuple[int, int, int, int, int]:
    """排序规则：向听数越小越靠前，其次面子越多、搭子越多越靠前。"""
    melds = len(combo.triplets) + len(combo.sequences)
    incomplete = len(combo.pairs) + len(combo.taatsu)
    return (combo.shanten, -melds, -incomplete, len(combo.leftovers), len(combo.all_tiles()))
