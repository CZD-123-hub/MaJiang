"""多拆分路线的进张与灵活度指标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .hand_routes import HandRoute, TaatsuKind
from .tiles import HONGZHONG, JIUJIANG_TILE_CODES, remaining_tile_counts


@dataclass(frozen=True)
class RouteMetrics:
    route: HandRoute
    effective_tiles: dict[int, int]
    effective_count: int
    effective_tile_kinds: int
    flexibility: float
    ryanmen_count: int
    replacement_count: int


def measure_routes(
    hand: list[int] | tuple[int, ...],
    routes: tuple[HandRoute, ...] | list[HandRoute],
    remaining_counts: Mapping[int, int] | None = None,
) -> dict[tuple[object, ...], RouteMetrics]:
    """按路线计算可见牌墙下的有效进张和灵活度。

    此处的有效进张是“能直接完成当前路线中一个未完成结构”的牌，和全局最优
    向听下降并不混为一谈；后者仍由搜索树在决策层计算。
    """
    remaining = dict(remaining_counts or remaining_tile_counts(hand))
    return {route.signature: _measure_route(route, remaining) for route in routes}


def retain_routes(
    routes: tuple[HandRoute, ...] | list[HandRoute],
    metrics_by_signature: Mapping[tuple[object, ...], RouteMetrics],
    *,
    shanten_window: int = 1,
    max_routes: int = 12,
) -> tuple[HandRoute, ...]:
    """按向听数和真实有效进张做 Pareto 保留。

    保留最优向听以及高一向听中的非支配路线：因此“多一向听但进张更广”的
    路线不会被旧式的最小向听过滤直接删掉。
    """
    if shanten_window < 0 or max_routes <= 0:
        raise ValueError("shanten_window must be non-negative and max_routes must be positive")
    if not routes:
        return ()
    minimum_shanten = min(route.shanten for route in routes)
    eligible = [route for route in routes if route.shanten <= minimum_shanten + shanten_window]
    non_dominated = [
        route
        for route in eligible
        if not any(_dominates(other, route, metrics_by_signature) for other in eligible if other is not route)
    ]
    ranked = sorted(non_dominated, key=lambda route: _retention_sort_key(route, metrics_by_signature))
    return tuple(ranked[:max_routes])


def _measure_route(route: HandRoute, remaining: Mapping[int, int]) -> RouteMetrics:
    effective_kinds = _effective_tile_kinds(route)
    effective_tiles = {
        tile: max(0, int(remaining.get(tile, 0)))
        for tile in sorted(effective_kinds)
        if remaining.get(tile, 0) > 0
    }
    ryanmen_count = sum(taatsu.kind == TaatsuKind.RYANMEN for taatsu in route.taatsu)
    replacement_tiles = _replacement_tiles(route)
    replacement_count = sum(1 for tile in replacement_tiles if remaining.get(tile, 0) > 0)
    uncommitted_hongzhong = route.hongzhong_count - route.hongzhong_used
    flexibility = (
        len(effective_tiles)
        + 0.75 * ryanmen_count
        + 0.40 * sum(taatsu.kind == TaatsuKind.KANCHAN for taatsu in route.taatsu)
        + 0.20 * sum(taatsu.kind == TaatsuKind.PENCHAN for taatsu in route.taatsu)
        + 0.15 * replacement_count
        + 0.50 * max(0, uncommitted_hongzhong)
    )
    return RouteMetrics(
        route=route,
        effective_tiles=effective_tiles,
        effective_count=sum(effective_tiles.values()),
        effective_tile_kinds=len(effective_tiles),
        flexibility=flexibility,
        ryanmen_count=ryanmen_count,
        replacement_count=replacement_count,
    )


def _effective_tile_kinds(route: HandRoute) -> set[int]:
    tiles: set[int] = set()
    for taatsu in route.remaining_taatsu:
        first, second = taatsu.tiles
        if taatsu.kind == TaatsuKind.KANCHAN:
            tiles.add(first + 1)
        elif taatsu.kind == TaatsuKind.PENCHAN:
            tiles.add(second + 1 if first & 0x0F == 1 else first - 1)
        else:
            tiles.update((first - 1, second + 1))
    if route.meld_count < 4:
        tiles.update(pair[0] for pair in route.pairs)
    if route.hongzhong_count > route.hongzhong_used and (route.taatsu or route.isolated):
        tiles.add(HONGZHONG)
    return {tile for tile in tiles if tile in JIUJIANG_TILE_CODES}


def _replacement_tiles(route: HandRoute) -> set[int]:
    tiles: set[int] = set()
    for tile in route.isolated:
        rank = tile & 0x0F
        if rank > 1:
            tiles.add(tile - 1)
        if rank < 9:
            tiles.add(tile + 1)
        if 1 < rank < 9:
            tiles.add(tile)
    return {tile for tile in tiles if tile in JIUJIANG_TILE_CODES}


def _dominates(
    left: HandRoute,
    right: HandRoute,
    metrics_by_signature: Mapping[tuple[object, ...], RouteMetrics],
) -> bool:
    left_metrics = metrics_by_signature[left.signature]
    right_metrics = metrics_by_signature[right.signature]
    no_worse = (
        left.shanten <= right.shanten
        and left_metrics.effective_count >= right_metrics.effective_count
        and left_metrics.flexibility >= right_metrics.flexibility
    )
    strictly_better = (
        left.shanten < right.shanten
        or left_metrics.effective_count > right_metrics.effective_count
        or left_metrics.flexibility > right_metrics.flexibility
    )
    return no_worse and strictly_better


def _retention_sort_key(
    route: HandRoute,
    metrics_by_signature: Mapping[tuple[object, ...], RouteMetrics],
) -> tuple[float, ...]:
    metrics = metrics_by_signature[route.signature]
    return (
        route.shanten,
        -metrics.effective_count,
        -metrics.flexibility,
        -metrics.effective_tile_kinds,
        route.hongzhong_used,
    )
