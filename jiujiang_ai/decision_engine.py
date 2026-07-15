"""多路线进攻/防守综合弃牌决策。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, overload

from .expected_value import ExpectedWinValue, estimate_win_value
from .hand_routes import HandRoute, enumerate_hand_routes
from .risk import DiscardRisk, evaluate_discard_risks
from .route_metrics import RouteMetrics, measure_routes, retain_routes
from .ting import winning_tile_counts
from .tiles import remaining_tile_counts, validate_hand


@dataclass(frozen=True)
class MultiRouteDiscardDecision:
    discard: int
    score: float
    progress_probability: float
    expected_win_value: float
    expected_value: ExpectedWinValue
    risk_score: float
    flexibility: float
    effective_tiles: dict[int, int]
    route_count: int
    retained_route_count: int
    retained_routes: tuple[HandRoute, ...]
    risk: DiscardRisk


@overload
def choose_discard(
    hand: list[int] | tuple[int, ...],
    candidate_cards: list[list[int]],
    *,
    data: dict,
    acting_position: int,
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
    return_all: bool = False,
) -> MultiRouteDiscardDecision: ...


@overload
def choose_discard(
    hand: list[int] | tuple[int, ...],
    candidate_cards: list[list[int]],
    *,
    data: dict,
    acting_position: int,
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
    return_all: bool = True,
) -> dict[int, MultiRouteDiscardDecision]: ...


def choose_discard(
    hand: list[int] | tuple[int, ...],
    candidate_cards: list[list[int]],
    *,
    data: dict,
    acting_position: int,
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
    return_all: bool = False,
) -> MultiRouteDiscardDecision | dict[int, MultiRouteDiscardDecision]:
    """按 ``P(进张) × E(收益) - 风险 + 灵活度`` 选择候选弃牌。"""
    validate_hand(hand)
    candidates = tuple(sorted({cards[0] for cards in candidate_cards if cards and cards[0] in hand}))
    if not candidates:
        raise ValueError("no valid discard candidates")
    risks = evaluate_discard_risks(data, acting_position=acting_position, candidates=candidates)
    decisions = {
        discard: _evaluate_discard(
            hand=list(hand),
            discard=discard,
            data=data,
            acting_position=acting_position,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
            risk=risks[discard],
        )
        for discard in candidates
    }
    if return_all:
        return decisions
    return max(
        decisions.values(),
        key=lambda item: (
            item.score,
            item.progress_probability,
            item.expected_win_value,
            item.flexibility,
            -item.discard,
        ),
    )


def _evaluate_discard(
    *,
    hand: list[int],
    discard: int,
    data: dict,
    acting_position: int,
    fixed_melds: int,
    remaining_counts: Mapping[int, int] | None,
    risk: DiscardRisk,
) -> MultiRouteDiscardDecision:
    after = list(hand)
    after.remove(discard)
    remaining = dict(remaining_counts or remaining_tile_counts(after))
    routes = enumerate_hand_routes(after, fixed_melds=fixed_melds)
    metrics = measure_routes(after, routes, remaining)
    retained = retain_routes(routes, metrics)
    effective_tiles = _combined_effective_tiles(retained, metrics)
    winning_tiles = winning_tile_counts(after, fixed_melds=fixed_melds, remaining_counts=remaining)
    if winning_tiles:
        effective_tiles = winning_tiles

    wall_total = sum(max(0, count) for count in remaining.values())
    progress_probability = sum(effective_tiles.values()) / wall_total if wall_total else 0.0
    flexibility = max((metrics[route.signature].flexibility for route in retained), default=0.0)
    normalized_flexibility = min(1.0, flexibility / 8.0)
    expected_value = estimate_win_value(data, winner=acting_position, pending_discard=discard)
    expected_win_value = expected_value.total

    # 风险和灵活度为归一化项。向听数没有被硬编码为最终收益，而是通过保留
    # 路线和真实进张概率影响结果；更高向听但宽进张的路线仍可以胜出。
    score = progress_probability * expected_win_value - 0.65 * risk.score + 0.30 * normalized_flexibility
    return MultiRouteDiscardDecision(
        discard=discard,
        score=score,
        progress_probability=progress_probability,
        expected_win_value=expected_win_value,
        expected_value=expected_value,
        risk_score=risk.score,
        flexibility=flexibility,
        effective_tiles=effective_tiles,
        route_count=len(routes),
        retained_route_count=len(retained),
        retained_routes=retained,
        risk=risk,
    )


def _combined_effective_tiles(
    routes: tuple[HandRoute, ...],
    metrics_by_signature: Mapping[tuple[object, ...], RouteMetrics],
) -> dict[int, int]:
    combined: dict[int, int] = {}
    for route in routes:
        for tile, count in metrics_by_signature[route.signature].effective_tiles.items():
            combined[tile] = max(combined.get(tile, 0), count)
    return combined

