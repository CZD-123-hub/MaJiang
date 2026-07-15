"""保留多种合理拆法的九江红中手牌路线生成器。

本模块刻意不在递归阶段作“顺子优先”或“无效搭子”剪枝。那些规则只适合
作为后续候选保留策略，不能在生成阶段抹掉向听稍高但进张更宽的路线。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from itertools import combinations

from .tiles import HONGZHONG, SUITED_TILE_CODES, can_start_sequence, tile_rank, validate_hand


class MeldKind(StrEnum):
    SEQUENCE = "sequence"
    TRIPLET = "triplet"


class TaatsuKind(StrEnum):
    RYANMEN = "ryanmen"
    KANCHAN = "kanchan"
    PENCHAN = "penchan"


@dataclass(frozen=True, order=True)
class Meld:
    kind: MeldKind
    tiles: tuple[int, int, int]


@dataclass(frozen=True, order=True)
class Taatsu:
    kind: TaatsuKind
    tiles: tuple[int, int]


@dataclass(frozen=True)
class HandRoute:
    """一条可解释的平胡拆分路线。

    ``taatsu`` 保留普通牌天然搭子；其中已被红中补成面子的搭子数量记录在
    ``red_completed_melds`` 中，避免丢失它原来的形状信息。
    """

    melds: tuple[Meld, ...]
    pairs: tuple[tuple[int, int], ...]
    taatsu: tuple[Taatsu, ...]
    isolated: tuple[int, ...]
    fixed_melds: int
    hongzhong_count: int
    hongzhong_used: int
    hongzhong_assignments: tuple[str, ...]
    red_completed_taatsu_indexes: tuple[int, ...]
    red_completed_melds: int
    shanten: int

    @property
    def meld_count(self) -> int:
        return min(4, self.fixed_melds + len(self.melds) + self.red_completed_melds)

    @property
    def remaining_taatsu_count(self) -> int:
        return len(self.remaining_taatsu)

    @property
    def remaining_taatsu(self) -> tuple[Taatsu, ...]:
        completed = set(self.red_completed_taatsu_indexes)
        return tuple(taatsu for index, taatsu in enumerate(self.taatsu) if index not in completed)

    @property
    def signature(self) -> tuple[object, ...]:
        """用于生成完成后的统一去重，而非递归中的提前过滤。"""
        return (
            self.melds,
            self.pairs,
            self.taatsu,
            self.isolated,
            self.fixed_melds,
            self.hongzhong_count,
            self.hongzhong_used,
            self.hongzhong_assignments,
            self.red_completed_taatsu_indexes,
            self.red_completed_melds,
            self.shanten,
        )


@dataclass(frozen=True)
class _NaturalRoute:
    melds: tuple[Meld, ...] = ()
    pairs: tuple[tuple[int, int], ...] = ()
    taatsu: tuple[Taatsu, ...] = ()
    isolated: tuple[int, ...] = ()

    def add(
        self,
        *,
        meld: Meld | None = None,
        pair: tuple[int, int] | None = None,
        taatsu: Taatsu | None = None,
        isolated: int | None = None,
    ) -> "_NaturalRoute":
        return _NaturalRoute(
            melds=self.melds + ((meld,) if meld else ()),
            pairs=self.pairs + ((pair,) if pair else ()),
            taatsu=self.taatsu + ((taatsu,) if taatsu else ()),
            isolated=self.isolated + ((isolated,) if isolated is not None else ()),
        )


def enumerate_hand_routes(hand: list[int] | tuple[int, ...], fixed_melds: int = 0) -> tuple[HandRoute, ...]:
    """枚举手牌的合理拆分，并在全部生成后按规范签名去重。

    不做候选数量截断；调用方应在已计算真实有效进张后使用 ``retain_routes``
    做 Pareto 保留，避免拆分层擅自以向听数淘汰高潜力路线。
    """
    validate_hand(hand)
    if not 0 <= fixed_melds <= 4:
        raise ValueError("fixed_melds must be between 0 and 4")

    hongzhong_count = list(hand).count(HONGZHONG)
    counts = tuple(list(hand).count(tile) for tile in SUITED_TILE_CODES)
    generated: list[HandRoute] = []
    for natural in _split_counts(counts):
        generated.extend(_apply_hongzhong_plans(natural, hongzhong_count, fixed_melds))

    unique = {route.signature: route for route in generated}
    return tuple(sorted(unique.values(), key=_route_sort_key))


@lru_cache(maxsize=None)
def _split_counts(counts: tuple[int, ...]) -> tuple[_NaturalRoute, ...]:
    try:
        index = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        return (_NaturalRoute(),)

    tile = SUITED_TILE_CODES[index]
    results: list[_NaturalRoute] = []

    def branch(next_counts: tuple[int, ...], **addition: object) -> None:
        for child in _split_counts(next_counts):
            results.append(child.add(**addition))

    if counts[index] >= 3:
        branch(
            _remove(counts, (tile, tile, tile)),
            meld=Meld(MeldKind.TRIPLET, (tile, tile, tile)),
        )
    if can_start_sequence(tile) and _has(counts, (tile, tile + 1, tile + 2)):
        branch(
            _remove(counts, (tile, tile + 1, tile + 2)),
            meld=Meld(MeldKind.SEQUENCE, (tile, tile + 1, tile + 2)),
        )
    if counts[index] >= 2:
        branch(_remove(counts, (tile, tile)), pair=(tile, tile))
    if tile_rank(tile) <= 8 and _has(counts, (tile, tile + 1)):
        kind = TaatsuKind.PENCHAN if tile_rank(tile) in {1, 8} else TaatsuKind.RYANMEN
        branch(_remove(counts, (tile, tile + 1)), taatsu=Taatsu(kind, (tile, tile + 1)))
    if tile_rank(tile) <= 7 and _has(counts, (tile, tile + 2)):
        branch(_remove(counts, (tile, tile + 2)), taatsu=Taatsu(TaatsuKind.KANCHAN, (tile, tile + 2)))
    branch(_remove(counts, (tile,)), isolated=tile)
    return tuple(results)


def _apply_hongzhong_plans(
    natural: _NaturalRoute,
    hongzhong_count: int,
    fixed_melds: int,
) -> tuple[HandRoute, ...]:
    routes: list[HandRoute] = []
    # 第三个元素记录“红中补将”已经占用的普通单张，防止同一张牌又被
    # 当作两红补刻的原料，造成不合法的双重使用。
    pair_options: list[tuple[int, tuple[str, ...], frozenset[int]]] = [(0, (), frozenset())]
    if not natural.pairs:
        if hongzhong_count >= 1:
            pair_options.extend(
                (1, (f"pair_with_single:{tile:02x}",), frozenset({index}))
                for index, tile in enumerate(natural.isolated)
            )
        if hongzhong_count >= 2:
            pair_options.append((2, ("pair_with_two_hongzhong",), frozenset()))

    for pair_red_count, pair_assignments, pair_consumed_isolated in pair_options:
        if pair_red_count > hongzhong_count:
            continue
        remaining_red = hongzhong_count - pair_red_count
        max_taatsu = min(len(natural.taatsu), remaining_red)
        for taatsu_count in range(max_taatsu + 1):
            for taatsu_indexes in combinations(range(len(natural.taatsu)), taatsu_count):
                red_after_taatsu = remaining_red - taatsu_count
                available_isolated_indexes = tuple(
                    index for index in range(len(natural.isolated)) if index not in pair_consumed_isolated
                )
                max_isolated = min(len(available_isolated_indexes), red_after_taatsu // 2)
                for isolated_count in range(max_isolated + 1):
                    for isolated_indexes in combinations(available_isolated_indexes, isolated_count):
                        assignments = pair_assignments + tuple(
                            f"taatsu_to_meld:{natural.taatsu[index].kind}:{_tiles_key(natural.taatsu[index].tiles)}"
                            for index in taatsu_indexes
                        ) + tuple(
                            f"single_to_meld:{natural.isolated[index]:02x}" for index in isolated_indexes
                        )
                        used = pair_red_count + taatsu_count + isolated_count * 2
                        routes.append(
                            _make_route(
                                natural=natural,
                                fixed_melds=fixed_melds,
                                hongzhong_count=hongzhong_count,
                                hongzhong_used=used,
                                assignments=assignments,
                                red_completed_taatsu_indexes=taatsu_indexes,
                                red_completed_melds=taatsu_count + isolated_count,
                            )
                        )
    return tuple(routes)


def _make_route(
    *,
    natural: _NaturalRoute,
    fixed_melds: int,
    hongzhong_count: int,
    hongzhong_used: int,
    assignments: tuple[str, ...],
    red_completed_taatsu_indexes: tuple[int, ...],
    red_completed_melds: int,
) -> HandRoute:
    meld_count = min(4, fixed_melds + len(natural.melds) + red_completed_melds)
    has_pair = bool(natural.pairs) or any(item.startswith("pair_") for item in assignments)
    extra_pairs = max(0, len(natural.pairs) - (1 if natural.pairs else 0))
    incomplete = extra_pairs + max(0, len(natural.taatsu) - red_completed_melds)
    useful_incomplete = min(incomplete, max(0, 4 - meld_count))
    shanten = max(0, 9 - 2 * meld_count - useful_incomplete - int(has_pair))
    return HandRoute(
        melds=natural.melds,
        pairs=natural.pairs,
        taatsu=natural.taatsu,
        isolated=natural.isolated,
        fixed_melds=fixed_melds,
        hongzhong_count=hongzhong_count,
        hongzhong_used=hongzhong_used,
        hongzhong_assignments=assignments,
        red_completed_taatsu_indexes=red_completed_taatsu_indexes,
        red_completed_melds=red_completed_melds,
        shanten=shanten,
    )


def _route_sort_key(route: HandRoute) -> tuple[int, int, int, int, int]:
    return (
        route.shanten,
        -route.meld_count,
        -route.remaining_taatsu_count,
        len(route.isolated),
        route.hongzhong_used,
    )


def _has(counts: tuple[int, ...], tiles: tuple[int, ...]) -> bool:
    needed = {tile: tiles.count(tile) for tile in set(tiles)}
    return all(counts[SUITED_TILE_CODES.index(tile)] >= amount for tile, amount in needed.items())


def _remove(counts: tuple[int, ...], tiles: tuple[int, ...]) -> tuple[int, ...]:
    result = list(counts)
    for tile in tiles:
        result[SUITED_TILE_CODES.index(tile)] -= 1
    return tuple(result)


def _tiles_key(tiles: tuple[int, ...]) -> str:
    return "-".join(f"{tile:02x}" for tile in tiles)
