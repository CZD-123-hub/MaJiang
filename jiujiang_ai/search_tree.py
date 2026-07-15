from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .expected_value import estimate_win_value
from .hand_split import HandAnalysis, analyze_hand
from .hand_routes import HandRoute, enumerate_hand_routes
from .hu import HuOptions, can_hu
from .route_metrics import RouteMetrics, measure_routes, retain_routes
from .tiles import JIUJIANG_TILE_CODES, remaining_tile_counts, validate_hand
from .ting import ting_discards, winning_tile_counts


@dataclass(frozen=True)
class CombinationSummary:
    """组合状态摘要：体现当前手牌的组合驱动信息，向 v5 的组合状态树靠近。

    该摘要直接来自 analyze_hand 的结果，把“纯评估型节点”里隐含的组合信息
    显式挂到树节点上，方便后续做组合驱动的递归扩展。字段含义：
    - melds: 已成面子数（含固定副露，上限 4）
    - pairs: 将牌对子数（0/1，红中充当将牌时也算 1）
    - taatsu: 搭子数（两面/边张/嵌张等未完成面子）
    - leftovers: 孤张数（无法归入面子/将/搭的单张）
    - hongzhong_count: 当前手牌里的红中总数
    - hongzhong_used: 已被用于充当面子/将牌的红中数
    - shanten: 向听数（0 即听牌）
    """

    melds: int
    pairs: int
    taatsu: int
    leftovers: int
    hongzhong_count: int
    hongzhong_used: int
    shanten: int


@dataclass(frozen=True)
class RouteSummary:
    """多候选拆分在一个树节点上的保留结果与综合路线价值。"""

    route_count: int
    retained_route_count: int
    effective_count: int
    flexibility: float
    progress_probability: float
    expected_win_value: float
    route_value: float
    retained_routes: tuple[HandRoute, ...]


_EMPTY_ROUTE_SUMMARY = RouteSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0, ())


@dataclass(frozen=True)
class TakingPath:
    """记录一条搜索路径需要的进张、弃牌顺序及其局部权重。"""

    taking_tiles: tuple[int, ...]
    taking_weights: tuple[float, ...]
    discard_path: tuple[int, ...]
    path_weight: float
    hu_value: float
    ting_value: float
    improvement_value: float
    total_value: float


@dataclass(frozen=True)
class DrawNode:
    """弃牌后的摸牌子节点。"""

    draw: int
    remaining: int
    shanten: int
    is_hu: bool
    hu_value: float
    ting_value: float
    improvement_value: float
    best_follow_up_discard: int | None
    follow_up_shanten: int
    follow_up_effective_count: int
    follow_up_winning_tiles: dict[int, int]
    path_record: TakingPath
    follow_up_nodes: tuple["FollowUpNode", ...]
    follow_up_path_records: tuple[TakingPath, ...]
    follow_up_expected_hu_value: float
    follow_up_expected_ting_value: float
    follow_up_expected_improvement_value: float
    follow_up_expected_next_draw_hu_value: float
    follow_up_expected_next_draw_ting_value: float
    follow_up_expected_next_draw_improvement_value: float
    follow_up_expected_next_draw_bonus: float
    follow_up_score_bonus: float
    path_value: float
    # 组合状态摘要：摸到该张牌后手牌的组合驱动信息，向 v5 靠近
    combination: CombinationSummary
    route_summary: RouteSummary


@dataclass(frozen=True)
class DiscardDecision:
    """某一张候选弃牌的一层树搜索结果。"""

    discard: int
    score: float
    shanten_after_discard: int
    effective_count: int
    winning_tiles: dict[int, int]
    expected_path_value: float
    expected_hu_value: float
    expected_ting_value: float
    expected_improvement_value: float
    expected_follow_up_hu_value: float
    expected_follow_up_ting_value: float
    expected_follow_up_improvement_value: float
    expected_third_draw_hu_value: float
    expected_third_draw_ting_value: float
    expected_third_draw_improvement_value: float
    expected_third_draw_bonus: float
    expected_follow_up_shanten: float
    expected_follow_up_effective_count: float
    hu_child_count: int
    hu_total_remaining: int
    improving_child_count: int
    ting_child_count: int
    children: tuple[DrawNode, ...]
    route_summary: RouteSummary


@dataclass(frozen=True)
class FollowUpNode:
    """第二层显式展开的“摸后再弃”节点。"""

    discard: int
    shanten: int
    effective_count: int
    winning_tiles: dict[int, int]
    hu_value: float
    ting_value: float
    improvement_value: float
    path_record: TakingPath
    next_draw_nodes: tuple["ThirdDrawNode", ...]
    next_draw_expected_hu_value: float
    next_draw_expected_ting_value: float
    next_draw_expected_improvement_value: float
    next_draw_score_bonus: float
    score: float
    # 组合状态摘要：弃掉该张牌后手牌的组合驱动信息，向 v5 靠近
    combination: CombinationSummary
    route_summary: RouteSummary


@dataclass(frozen=True)
class ThirdDrawNode:
    """受控第三层的“再摸牌”节点。"""

    draw: int
    remaining: int
    shanten: int
    is_hu: bool
    hu_value: float
    ting_value: float
    improvement_value: float
    path_record: TakingPath
    score: float
    # 组合状态摘要：第三层再摸牌后手牌的组合驱动信息，向 v5 靠近
    combination: CombinationSummary
    route_summary: RouteSummary


@dataclass(frozen=True)
class SearchTreeResult:
    """整棵一层显式搜索树的汇总结果。"""

    best_discard: int
    discard_scores: dict[int, DiscardDecision]
    node_count: int


def _combination_from_analysis(analysis: HandAnalysis) -> CombinationSummary:
    """把 analyze_hand 的结果转成组合状态摘要，供树节点挂载使用。

    这是当前树从“纯评估型节点”往“组合驱动型节点”过渡的最小桥梁：
    直接复用 hand_split 的拆分结果，不重新做组合枚举，避免破坏现有评估链路。
    """
    return CombinationSummary(
        melds=analysis.melds,
        pairs=analysis.pairs,
        taatsu=analysis.taatsu,
        leftovers=analysis.leftovers,
        hongzhong_count=analysis.hongzhong_count,
        hongzhong_used=analysis.hongzhong_used,
        shanten=analysis.shanten,
    )


def _summarize_routes(
    hand: list[int],
    *,
    fixed_melds: int,
    remaining_counts: Mapping[int, int],
    decision_data: dict | None,
    acting_position: int,
    pending_discard: int | None = None,
) -> RouteSummary:
    """为树节点计算多拆分保留、进张概率和可比较的路线收益。"""
    routes = enumerate_hand_routes(hand, fixed_melds=fixed_melds)
    metrics = measure_routes(hand, routes, remaining_counts)
    retained = retain_routes(routes, metrics)
    effective_tiles = _combined_route_effective_tiles(retained, metrics)
    wall_total = sum(max(0, count) for count in remaining_counts.values())
    progress_probability = sum(effective_tiles.values()) / wall_total if wall_total else 0.0
    flexibility = max((metrics[route.signature].flexibility for route in retained), default=0.0)
    expected_win_value = estimate_win_value(
        decision_data or {},
        winner=acting_position,
        pending_discard=pending_discard,
    ).total
    route_value = progress_probability * expected_win_value + 0.30 * min(1.0, flexibility / 8.0)
    return RouteSummary(
        route_count=len(routes),
        retained_route_count=len(retained),
        effective_count=sum(effective_tiles.values()),
        flexibility=flexibility,
        progress_probability=progress_probability,
        expected_win_value=expected_win_value,
        route_value=route_value,
        retained_routes=retained,
    )


def _combined_route_effective_tiles(
    routes: tuple[HandRoute, ...],
    metrics_by_signature: Mapping[tuple[object, ...], RouteMetrics],
) -> dict[int, int]:
    combined: dict[int, int] = {}
    for route in routes:
        for tile, count in metrics_by_signature[route.signature].effective_tiles.items():
            combined[tile] = max(combined.get(tile, 0), count)
    return combined


def expand_discard_tree(
    hand: list[int] | tuple[int, ...],
    candidate_cards: list[list[int]] | None = None,
    options: HuOptions | None = None,
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
    max_draws: int | None = None,
    use_multi_route: bool = False,
    decision_data: dict | None = None,
    acting_position: int = 0,
) -> SearchTreeResult:
    """构建第一阶段“弃牌 -> 摸牌”的显式树搜索结果。"""
    validate_hand(hand)
    if len(hand) % 3 != 2:
        raise ValueError("弃牌搜索需要 3n+2 张手牌，通常是 14 张。")

    options = options or HuOptions()
    candidates = candidate_cards or [[tile] for tile in sorted(set(hand))]
    decisions: dict[int, DiscardDecision] = {}

    for card_group in candidates:
        if not card_group:
            continue
        discard = card_group[0]
        if discard not in hand or discard in decisions:
            continue
        after_discard = list(hand)
        after_discard.remove(discard)
        decisions[discard] = _evaluate_discard(
            discard=discard,
            after_discard=after_discard,
            options=options,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
            max_draws=max_draws,
            use_multi_route=use_multi_route,
            decision_data=decision_data,
            acting_position=acting_position,
        )

    if not decisions:
        raise ValueError("no valid discard candidates")

    best = max(decisions.values(), key=_decision_sort_key)
    node_count = len(decisions) + sum(len(decision.children) for decision in decisions.values())
    return SearchTreeResult(best_discard=best.discard, discard_scores=decisions, node_count=node_count)


def choose_discard(
    hand: list[int] | tuple[int, ...],
    candidate_cards: list[list[int]] | None = None,
    options: HuOptions | None = None,
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
    max_draws: int | None = None,
    use_multi_route: bool = False,
    decision_data: dict | None = None,
    acting_position: int = 0,
) -> DiscardDecision:
    """只返回推荐弃牌对应的一层树搜索结果。"""
    result = expand_discard_tree(
        hand=hand,
        candidate_cards=candidate_cards,
        options=options,
        fixed_melds=fixed_melds,
        remaining_counts=remaining_counts,
        max_draws=max_draws,
        use_multi_route=use_multi_route,
        decision_data=decision_data,
        acting_position=acting_position,
    )
    return result.discard_scores[result.best_discard]


def _evaluate_discard(
    discard: int,
    after_discard: list[int],
    options: HuOptions,
    fixed_melds: int,
    remaining_counts: Mapping[int, int] | None,
    max_draws: int | None,
    use_multi_route: bool,
    decision_data: dict | None,
    acting_position: int,
) -> DiscardDecision:
    """评估打出某张牌之后的一层摸牌树。"""
    analysis = analyze_hand(after_discard, fixed_melds=fixed_melds)
    remaining = dict(remaining_counts or remaining_tile_counts(after_discard))
    winning_tiles = winning_tile_counts(
        after_discard,
        options=options,
        fixed_melds=fixed_melds,
        remaining_counts=remaining,
    )
    effective_count = sum(winning_tiles.values())
    route_summary = _summarize_routes(
        after_discard,
        fixed_melds=fixed_melds,
        remaining_counts=remaining,
        decision_data=decision_data,
        acting_position=acting_position,
        pending_discard=discard,
    ) if use_multi_route else _EMPTY_ROUTE_SUMMARY

    children = [
        _build_draw_node(
            after_discard=after_discard,
            root_discard=discard,
            draw=tile,
            remaining=remaining.get(tile, 0),
            options=options,
            fixed_melds=fixed_melds,
            current_shanten=analysis.shanten,
            remaining_counts=remaining,
            use_multi_route=use_multi_route,
            decision_data=decision_data,
            acting_position=acting_position,
        )
        for tile in JIUJIANG_TILE_CODES
        if remaining.get(tile, 0) > 0
    ]
    children.sort(key=lambda child: (not child.is_hu, child.shanten, -child.remaining, child.draw))
    if max_draws is not None:
        children = children[:max_draws]

    expected_path_value = _expected_path_value(children)
    expected_hu_value = _expected_component_value(children, "hu_value")
    expected_ting_value = _expected_component_value(children, "ting_value")
    expected_improvement_value = _expected_component_value(children, "improvement_value")
    expected_follow_up_hu_value = _expected_component_value(children, "follow_up_expected_hu_value")
    expected_follow_up_ting_value = _expected_component_value(children, "follow_up_expected_ting_value")
    expected_follow_up_improvement_value = _expected_component_value(children, "follow_up_expected_improvement_value")
    expected_third_draw_hu_value = _expected_component_value(children, "follow_up_expected_next_draw_hu_value")
    expected_third_draw_ting_value = _expected_component_value(children, "follow_up_expected_next_draw_ting_value")
    expected_third_draw_improvement_value = _expected_component_value(
        children,
        "follow_up_expected_next_draw_improvement_value",
    )
    expected_third_draw_bonus = _expected_component_value(children, "follow_up_expected_next_draw_bonus")
    expected_follow_up_shanten = _expected_follow_up_shanten(children)
    expected_follow_up_effective_count = _expected_follow_up_effective_count(children)
    hu_children = [child for child in children if child.is_hu]
    improving_children = [child for child in children if child.shanten < analysis.shanten]
    ting_children = [child for child in children if child.ting_value > 0]
    score = _score_decision(
        shanten_after_discard=analysis.shanten,
        effective_count=effective_count,
        winning_tiles=winning_tiles,
        expected_path_value=expected_path_value,
        expected_hu_value=expected_hu_value,
        expected_ting_value=expected_ting_value,
        expected_improvement_value=expected_improvement_value,
        expected_follow_up_hu_value=expected_follow_up_hu_value,
        expected_follow_up_ting_value=expected_follow_up_ting_value,
        expected_follow_up_improvement_value=expected_follow_up_improvement_value,
        expected_third_draw_hu_value=expected_third_draw_hu_value,
        expected_third_draw_ting_value=expected_third_draw_ting_value,
        expected_third_draw_improvement_value=expected_third_draw_improvement_value,
        expected_third_draw_bonus=expected_third_draw_bonus,
        expected_follow_up_shanten=expected_follow_up_shanten,
        expected_follow_up_effective_count=expected_follow_up_effective_count,
        hu_total_remaining=sum(child.remaining for child in hu_children),
        improving_total_remaining=sum(child.remaining for child in improving_children),
        route_value=route_summary.route_value,
    )
    return DiscardDecision(
        discard=discard,
        score=score,
        shanten_after_discard=analysis.shanten,
        effective_count=effective_count,
        winning_tiles=winning_tiles,
        expected_path_value=expected_path_value,
        expected_hu_value=expected_hu_value,
        expected_ting_value=expected_ting_value,
        expected_improvement_value=expected_improvement_value,
        expected_follow_up_hu_value=expected_follow_up_hu_value,
        expected_follow_up_ting_value=expected_follow_up_ting_value,
        expected_follow_up_improvement_value=expected_follow_up_improvement_value,
        expected_third_draw_hu_value=expected_third_draw_hu_value,
        expected_third_draw_ting_value=expected_third_draw_ting_value,
        expected_third_draw_improvement_value=expected_third_draw_improvement_value,
        expected_third_draw_bonus=expected_third_draw_bonus,
        expected_follow_up_shanten=expected_follow_up_shanten,
        expected_follow_up_effective_count=expected_follow_up_effective_count,
        hu_child_count=len(hu_children),
        hu_total_remaining=sum(child.remaining for child in hu_children),
        improving_child_count=len(improving_children),
        ting_child_count=len(ting_children),
        children=tuple(children),
        route_summary=route_summary,
    )


def _build_draw_node(
    after_discard: list[int],
    root_discard: int,
    draw: int,
    remaining: int,
    options: HuOptions,
    fixed_melds: int,
    current_shanten: int,
    remaining_counts: Mapping[int, int],
    use_multi_route: bool,
    decision_data: dict | None,
    acting_position: int,
) -> DrawNode:
    next_hand = sorted([*after_discard, draw])
    is_hu = can_hu(next_hand, options, fixed_melds=fixed_melds)
    next_analysis = analyze_hand(next_hand, fixed_melds=fixed_melds)
    next_remaining_counts = dict(remaining_counts)
    if draw in next_remaining_counts:
        next_remaining_counts[draw] = max(0, next_remaining_counts[draw] - 1)
    else:
        next_remaining_counts = remaining_tile_counts(next_hand)
    hu_value = _hu_path_value(is_hu)
    follow_up_nodes = _build_follow_up_nodes(
        next_hand=next_hand,
        root_discard=root_discard,
        taken_tile=draw,
        taken_weight=float(remaining),
        options=options,
        fixed_melds=fixed_melds,
        remaining_counts=next_remaining_counts,
        use_multi_route=use_multi_route,
        decision_data=decision_data,
        acting_position=acting_position,
    )
    best_follow_up_node = follow_up_nodes[0] if follow_up_nodes else None
    # 摸牌节点的下一次实际决策发生在“摸后再弃”阶段，因此复用最优后继的
    # 路线摘要，避免在全部摸牌分支上重复完整枚举拆法。
    route_summary = best_follow_up_node.route_summary if best_follow_up_node is not None else _EMPTY_ROUTE_SUMMARY
    follow_up_expected_hu_value = _expected_follow_up_component_value(follow_up_nodes, "hu_value")
    follow_up_expected_ting_value = _expected_follow_up_component_value(follow_up_nodes, "ting_value")
    follow_up_expected_improvement_value = _expected_follow_up_component_value(
        follow_up_nodes,
        "improvement_value",
    )
    follow_up_expected_next_draw_hu_value = _expected_follow_up_component_value(
        follow_up_nodes,
        "next_draw_expected_hu_value",
    )
    follow_up_expected_next_draw_ting_value = _expected_follow_up_component_value(
        follow_up_nodes,
        "next_draw_expected_ting_value",
    )
    follow_up_expected_next_draw_improvement_value = _expected_follow_up_component_value(
        follow_up_nodes,
        "next_draw_expected_improvement_value",
    )
    follow_up_expected_next_draw_bonus = _expected_follow_up_component_value(
        follow_up_nodes,
        "next_draw_score_bonus",
    )
    ting_value = _ting_path_value(
        next_hand=next_hand,
        is_hu=is_hu,
        options=options,
        fixed_melds=fixed_melds,
        remaining_counts=next_remaining_counts,
    )
    improvement_value = _improvement_path_value(
        is_hu=is_hu,
        current_shanten=current_shanten,
        follow_up_shanten=best_follow_up_node.shanten if best_follow_up_node is not None else current_shanten,
    )
    follow_up_score_bonus = _follow_up_score_bonus(follow_up_nodes)
    path_record = TakingPath(
        taking_tiles=(draw,),
        taking_weights=(float(remaining),),
        discard_path=(root_discard,),
        path_weight=float(remaining),
        hu_value=hu_value,
        ting_value=ting_value,
        improvement_value=improvement_value,
        total_value=hu_value + ting_value + improvement_value + follow_up_score_bonus,
    )
    return DrawNode(
        draw=draw,
        remaining=remaining,
        shanten=0 if is_hu else next_analysis.shanten,
        is_hu=is_hu,
        hu_value=hu_value,
        ting_value=ting_value,
        improvement_value=improvement_value,
        best_follow_up_discard=best_follow_up_node.discard if best_follow_up_node is not None else None,
        follow_up_shanten=best_follow_up_node.shanten if best_follow_up_node is not None else current_shanten,
        follow_up_effective_count=best_follow_up_node.effective_count if best_follow_up_node is not None else 0,
        follow_up_winning_tiles=best_follow_up_node.winning_tiles if best_follow_up_node is not None else {},
        path_record=path_record,
        follow_up_nodes=tuple(follow_up_nodes),
        follow_up_path_records=tuple(node.path_record for node in follow_up_nodes),
        follow_up_expected_hu_value=follow_up_expected_hu_value,
        follow_up_expected_ting_value=follow_up_expected_ting_value,
        follow_up_expected_improvement_value=follow_up_expected_improvement_value,
        follow_up_expected_next_draw_hu_value=follow_up_expected_next_draw_hu_value,
        follow_up_expected_next_draw_ting_value=follow_up_expected_next_draw_ting_value,
        follow_up_expected_next_draw_improvement_value=follow_up_expected_next_draw_improvement_value,
        follow_up_expected_next_draw_bonus=follow_up_expected_next_draw_bonus,
        follow_up_score_bonus=follow_up_score_bonus,
        path_value=hu_value + ting_value + improvement_value + follow_up_score_bonus,
        # 组合状态摘要：摸到该张牌后手牌的组合驱动信息
        combination=_combination_from_analysis(next_analysis),
        route_summary=route_summary,
    )


def _score_decision(
    shanten_after_discard: int,
    effective_count: int,
    winning_tiles: Mapping[int, int],
    expected_path_value: float,
    expected_hu_value: float,
    expected_ting_value: float,
    expected_improvement_value: float,
    expected_follow_up_hu_value: float,
    expected_follow_up_ting_value: float,
    expected_follow_up_improvement_value: float,
    expected_third_draw_hu_value: float,
    expected_third_draw_ting_value: float,
    expected_third_draw_improvement_value: float,
    expected_third_draw_bonus: float,
    expected_follow_up_shanten: float,
    expected_follow_up_effective_count: float,
    hu_total_remaining: int,
    improving_total_remaining: int,
    route_value: float,
) -> float:
    # 第二阶段先补路径收益第一版：
    # 1. 真实胡牌张仍然是核心收益；
    # 2. 子节点的加权路径期望作为显式树价值；
    # 3. 直胡子节点和改良子节点的总张数再做额外放大。
    return (
        (100 - 30 * shanten_after_discard)
        + 3 * effective_count
        + 2 * len(winning_tiles)
        + expected_path_value
        + 0.4 * expected_hu_value
        + 0.2 * expected_ting_value
        + 0.2 * expected_improvement_value
        + 0.1 * expected_follow_up_hu_value
        + 0.15 * expected_follow_up_ting_value
        + 0.15 * expected_follow_up_improvement_value
        + 0.08 * expected_third_draw_hu_value
        + 0.08 * expected_third_draw_ting_value
        + 0.08 * expected_third_draw_improvement_value
        + 0.2 * expected_third_draw_bonus
        - 5.0 * expected_follow_up_shanten
        + 0.8 * expected_follow_up_effective_count
        + 5 * hu_total_remaining
        + 1.5 * improving_total_remaining
        + 12.0 * route_value
    )


def _expected_path_value(children: list[DrawNode]) -> float:
    if not children:
        return 0.0
    total_remaining = sum(child.remaining for child in children)
    if total_remaining <= 0:
        return 0.0
    return sum(child.path_value * child.remaining for child in children) / total_remaining


def _expected_component_value(children: list[DrawNode], field: str) -> float:
    if not children:
        return 0.0
    total_remaining = sum(child.remaining for child in children)
    if total_remaining <= 0:
        return 0.0
    return sum(getattr(child, field) * child.remaining for child in children) / total_remaining


def _expected_follow_up_shanten(children: list[DrawNode]) -> float:
    if not children:
        return 0.0
    total_remaining = sum(child.remaining for child in children)
    if total_remaining <= 0:
        return 0.0
    return sum(child.follow_up_shanten * child.remaining for child in children) / total_remaining


def _expected_follow_up_effective_count(children: list[DrawNode]) -> float:
    if not children:
        return 0.0
    total_remaining = sum(child.remaining for child in children)
    if total_remaining <= 0:
        return 0.0
    return sum(child.follow_up_effective_count * child.remaining for child in children) / total_remaining


def _expected_follow_up_component_value(nodes: list[FollowUpNode], field: str) -> float:
    if not nodes:
        return 0.0
    weights = (0.6, 0.3, 0.1)
    weighted_total = 0.0
    weight_sum = 0.0
    for index, node in enumerate(nodes[: len(weights)]):
        weight = weights[index]
        weighted_total += getattr(node, field) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return 0.0
    return weighted_total / weight_sum


def _hu_path_value(is_hu: bool) -> float:
    if is_hu:
        return 200.0
    return 0.0


def _ting_path_value(
    next_hand: list[int],
    is_hu: bool,
    options: HuOptions,
    fixed_melds: int,
    remaining_counts: Mapping[int, int],
) -> float:
    if is_hu:
        return 0.0
    ting_candidates = ting_discards(
        next_hand,
        options=options,
        fixed_melds=fixed_melds,
        remaining_counts=remaining_counts,
    )
    if not ting_candidates:
        return 0.0
    best_effective_count = max(candidate.effective_count for candidate in ting_candidates.values())
    return 60.0 + 2.0 * best_effective_count


def _improvement_path_value(
    is_hu: bool,
    current_shanten: int,
    follow_up_shanten: int,
) -> float:
    if is_hu:
        return 0.0
    improvement = current_shanten - follow_up_shanten
    if improvement <= 0:
        return 0.0
    return max(0.0, (100 - 30 * follow_up_shanten) + 12.0 * improvement)


def _build_follow_up_nodes(
    next_hand: list[int],
    root_discard: int,
    taken_tile: int,
    taken_weight: float,
    options: HuOptions,
    fixed_melds: int,
    remaining_counts: Mapping[int, int],
    use_multi_route: bool,
    decision_data: dict | None,
    acting_position: int,
) -> list[FollowUpNode]:
    candidate_infos: list[tuple[int, int]] = []
    best_shanten: int | None = None
    for discard in sorted(set(next_hand)):
        after_discard = list(next_hand)
        after_discard.remove(discard)
        shanten = analyze_hand(after_discard, fixed_melds=fixed_melds).shanten
        candidate_infos.append((discard, shanten))
        if best_shanten is None or shanten < best_shanten:
            best_shanten = shanten

    if best_shanten is None:
        return []

    base_nodes: list[tuple[FollowUpNode, list[int]]] = []
    for discard, shanten in candidate_infos:
        # 开启多路线后允许高一向听的后继弃牌进入比较，是否保留由真实进张、
        # 灵活度和收益决定；默认路径保持原来的最小向听收束。
        if shanten > best_shanten + (1 if use_multi_route else 0):
            continue
        after_discard = list(next_hand)
        after_discard.remove(discard)
        winning_tiles = winning_tile_counts(
            after_discard,
            options=options,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
        )
        effective_count = (
            sum(winning_tiles.values())
            if winning_tiles
            else _count_improving_draws(
                hand=after_discard,
                current_shanten=shanten,
                fixed_melds=fixed_melds,
                remaining_counts=remaining_counts,
            )
        )
        base_hu_value = _follow_up_hu_value(shanten=shanten, winning_tiles=winning_tiles)
        base_ting_value = _follow_up_ting_value(effective_count=effective_count, winning_tiles=winning_tiles)
        base_improvement_value = _follow_up_improvement_value(
            shanten=shanten,
            effective_count=effective_count,
            winning_tiles=winning_tiles,
        )
        base_score = base_hu_value + base_ting_value + base_improvement_value
        # 组合状态摘要：弃掉该张牌后手牌的组合驱动信息
        follow_up_combination = _combination_from_analysis(
            analyze_hand(after_discard, fixed_melds=fixed_melds)
        )
        base_node = FollowUpNode(
            discard=discard,
            shanten=shanten,
            effective_count=effective_count,
            winning_tiles=winning_tiles,
            hu_value=base_hu_value,
            ting_value=base_ting_value,
            improvement_value=base_improvement_value,
            path_record=_build_follow_up_path_record(
                root_discard=root_discard,
                taken_tile=taken_tile,
                taken_weight=taken_weight,
                follow_up_discard=discard,
                shanten=shanten,
                effective_count=effective_count,
                winning_tiles=winning_tiles,
                total_value=base_score,
            ),
            next_draw_nodes=(),
            next_draw_expected_hu_value=0.0,
            next_draw_expected_ting_value=0.0,
            next_draw_expected_improvement_value=0.0,
            next_draw_score_bonus=0.0,
            score=base_score,
            combination=follow_up_combination,
            route_summary=_EMPTY_ROUTE_SUMMARY,
        )
        base_nodes.append((base_node, after_discard))

    if not base_nodes:
        return []

    base_nodes.sort(key=lambda item: _follow_up_sort_key(item[0]), reverse=True)
    # 先用原有廉价指标截断，再只对三个最强后继运行完整多路线拆分；这是
    # 搜索树的受控剪枝点，避免每层所有弃牌路径都指数级展开。
    base_nodes = base_nodes[:3]
    if use_multi_route:
        base_nodes = [
            (
                replace(
                    node,
                    route_summary=_summarize_routes(
                        after_discard,
                        fixed_melds=fixed_melds,
                        remaining_counts=remaining_counts,
                        decision_data=decision_data,
                        acting_position=acting_position,
                        pending_discard=node.discard,
                    ),
                ),
                after_discard,
            )
            for node, after_discard in base_nodes
        ]
        base_nodes.sort(key=lambda item: _follow_up_sort_key(item[0]), reverse=True)

    enriched_nodes: list[FollowUpNode] = []
    for index, (node, after_discard) in enumerate(base_nodes):
        if index >= 1:
            enriched_nodes.append(node)
            continue
        next_draw_nodes = _build_third_draw_nodes(
            hand_after_follow_up=after_discard,
            root_discard=root_discard,
            first_taken_tile=taken_tile,
            first_taken_weight=taken_weight,
            follow_up_discard=node.discard,
            current_shanten=node.shanten,
            options=options,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
            use_multi_route=use_multi_route,
            decision_data=decision_data,
            acting_position=acting_position,
        )
        next_draw_expected_hu_value = _expected_third_draw_component_value(next_draw_nodes, "hu_value")
        next_draw_expected_ting_value = _expected_third_draw_component_value(next_draw_nodes, "ting_value")
        next_draw_expected_improvement_value = _expected_third_draw_component_value(
            next_draw_nodes,
            "improvement_value",
        )
        next_draw_score_bonus = _third_draw_score_bonus(next_draw_nodes)
        enriched_nodes.append(
            replace(
                node,
                path_record=_build_follow_up_path_record(
                    root_discard=root_discard,
                    taken_tile=taken_tile,
                    taken_weight=taken_weight,
                    follow_up_discard=node.discard,
                    shanten=node.shanten,
                    effective_count=node.effective_count,
                    winning_tiles=node.winning_tiles,
                    total_value=node.score + next_draw_score_bonus,
                ),
                next_draw_nodes=tuple(next_draw_nodes),
                next_draw_expected_hu_value=next_draw_expected_hu_value,
                next_draw_expected_ting_value=next_draw_expected_ting_value,
                next_draw_expected_improvement_value=next_draw_expected_improvement_value,
                next_draw_score_bonus=next_draw_score_bonus,
                score=node.score + next_draw_score_bonus,
            )
        )

    enriched_nodes.sort(key=_follow_up_sort_key, reverse=True)
    return enriched_nodes[:3]


def _count_improving_draws(
    hand: list[int],
    current_shanten: int,
    fixed_melds: int,
    remaining_counts: Mapping[int, int],
) -> int:
    count = 0
    for tile in JIUJIANG_TILE_CODES:
        if remaining_counts.get(tile, 0) <= 0:
            continue
        next_hand = sorted([*hand, tile])
        if analyze_hand(next_hand, fixed_melds=fixed_melds).shanten < current_shanten:
            count += remaining_counts.get(tile, 0)
    return count


def _follow_up_sort_key(node: FollowUpNode) -> tuple:
    return (
        node.route_summary.route_value,
        node.score,
        node.next_draw_score_bonus,
        bool(node.winning_tiles),
        -node.shanten,
        node.effective_count,
        -node.discard,
    )


def _build_third_draw_nodes(
    hand_after_follow_up: list[int],
    root_discard: int,
    first_taken_tile: int,
    first_taken_weight: float,
    follow_up_discard: int,
    current_shanten: int,
    options: HuOptions,
    fixed_melds: int,
    remaining_counts: Mapping[int, int],
    use_multi_route: bool,
    decision_data: dict | None,
    acting_position: int,
) -> list[ThirdDrawNode]:
    candidates: list[ThirdDrawNode] = []
    for tile in _select_third_draw_candidates(hand_after_follow_up, current_shanten, fixed_melds, remaining_counts):
        remaining = remaining_counts.get(tile, 0)
        if remaining <= 0:
            continue
        next_hand = sorted([*hand_after_follow_up, tile])
        is_hu = can_hu(next_hand, options, fixed_melds=fixed_melds)
        next_analysis = analyze_hand(next_hand, fixed_melds=fixed_melds)
        route_summary = _summarize_routes(
            next_hand,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
            decision_data=decision_data,
            acting_position=acting_position,
        ) if use_multi_route else _EMPTY_ROUTE_SUMMARY
        hu_value = 180.0 if is_hu else 0.0
        ting_value = 0.0
        if not is_hu:
            ting_candidates = ting_discards(
                next_hand,
                options=options,
                fixed_melds=fixed_melds,
                remaining_counts=remaining_counts,
            )
            if ting_candidates:
                best_effective_count = max(candidate.effective_count for candidate in ting_candidates.values())
                ting_value = 45.0 + 1.5 * best_effective_count
        improvement_value = 0.0
        if not is_hu and next_analysis.shanten < current_shanten:
            improvement = current_shanten - next_analysis.shanten
            improvement_value = max(0.0, (100 - 30 * next_analysis.shanten) + 10.0 * improvement)
        score = hu_value + ting_value + improvement_value
        path_weight = float(first_taken_weight) * float(remaining)
        path_record = TakingPath(
            taking_tiles=(first_taken_tile, tile),
            taking_weights=(float(first_taken_weight), float(remaining)),
            discard_path=(root_discard, follow_up_discard),
            path_weight=path_weight,
            hu_value=hu_value,
            ting_value=ting_value,
            improvement_value=improvement_value,
            total_value=score,
        )
        candidates.append(
            ThirdDrawNode(
                draw=tile,
                remaining=remaining,
                shanten=0 if is_hu else next_analysis.shanten,
                is_hu=is_hu,
                hu_value=hu_value,
                ting_value=ting_value,
                improvement_value=improvement_value,
                path_record=path_record,
                score=score,
                # 组合状态摘要：第三层再摸牌后手牌的组合驱动信息
                combination=_combination_from_analysis(next_analysis),
                route_summary=route_summary,
            )
        )
    candidates.sort(key=_third_draw_sort_key, reverse=True)
    return candidates[:3]


def _select_third_draw_candidates(
    hand_after_follow_up: list[int],
    current_shanten: int,
    fixed_melds: int,
    remaining_counts: Mapping[int, int],
) -> list[int]:
    winning_tiles = winning_tile_counts(
        hand_after_follow_up,
        fixed_melds=fixed_melds,
        remaining_counts=remaining_counts,
    )
    ranked_tiles: list[int] = []
    ranked_tiles.extend(sorted(winning_tiles, key=lambda tile: (-remaining_counts.get(tile, 0), tile)))

    improving_tiles: list[tuple[int, int]] = []
    for tile in JIUJIANG_TILE_CODES:
        if tile in winning_tiles or remaining_counts.get(tile, 0) <= 0:
            continue
        next_hand = sorted([*hand_after_follow_up, tile])
        shanten = analyze_hand(next_hand, fixed_melds=fixed_melds).shanten
        if shanten < current_shanten:
            improving_tiles.append((tile, shanten))
    improving_tiles.sort(key=lambda item: (item[1], -remaining_counts.get(item[0], 0), item[0]))
    ranked_tiles.extend(tile for tile, _ in improving_tiles)

    fallback_tiles = [
        tile for tile in JIUJIANG_TILE_CODES if tile not in ranked_tiles and remaining_counts.get(tile, 0) > 0
    ]
    fallback_tiles.sort(key=lambda tile: (-remaining_counts.get(tile, 0), tile))
    ranked_tiles.extend(fallback_tiles)

    deduped: list[int] = []
    seen: set[int] = set()
    for tile in ranked_tiles:
        if tile in seen:
            continue
        seen.add(tile)
        deduped.append(tile)
        if len(deduped) >= 3:
            break
    return deduped


def _third_draw_sort_key(node: ThirdDrawNode) -> tuple:
    return (
        node.score,
        node.is_hu,
        -node.shanten,
        node.remaining,
        -node.draw,
    )


def _expected_third_draw_component_value(nodes: list[ThirdDrawNode], field: str) -> float:
    if not nodes:
        return 0.0
    total_remaining = sum(node.remaining for node in nodes)
    if total_remaining <= 0:
        return 0.0
    return sum(getattr(node, field) * node.remaining for node in nodes) / total_remaining


def _third_draw_score_bonus(nodes: list[ThirdDrawNode]) -> float:
    if not nodes:
        return 0.0
    weights = (0.12, 0.06, 0.03)
    bonus = 0.0
    for index, node in enumerate(nodes[: len(weights)]):
        bonus += node.score * weights[index]
    return bonus


def _build_follow_up_path_record(
    root_discard: int,
    taken_tile: int,
    taken_weight: float,
    follow_up_discard: int,
    shanten: int,
    effective_count: int,
    winning_tiles: Mapping[int, int],
    total_value: float | None = None,
) -> TakingPath:
    hu_value = _follow_up_hu_value(shanten=shanten, winning_tiles=winning_tiles)
    ting_value = _follow_up_ting_value(effective_count=effective_count, winning_tiles=winning_tiles)
    improvement_value = _follow_up_improvement_value(
        shanten=shanten,
        effective_count=effective_count,
        winning_tiles=winning_tiles,
    )
    path_weight = float(taken_weight) * max(1.0, float(effective_count or len(winning_tiles) or 1))
    total_value = total_value if total_value is not None else hu_value + ting_value + improvement_value
    return TakingPath(
        taking_tiles=(taken_tile,),
        taking_weights=(float(taken_weight),),
        discard_path=(root_discard, follow_up_discard),
        path_weight=path_weight,
        hu_value=hu_value,
        ting_value=ting_value,
        improvement_value=improvement_value,
        total_value=total_value,
    )


def _follow_up_hu_value(shanten: int, winning_tiles: Mapping[int, int]) -> float:
    if shanten == 0 and winning_tiles:
        return 40.0 + 2.0 * len(winning_tiles)
    return 0.0


def _follow_up_ting_value(effective_count: int, winning_tiles: Mapping[int, int]) -> float:
    if not winning_tiles:
        return 0.0
    return 50.0 + 2.0 * effective_count + 3.0 * len(winning_tiles)


def _follow_up_improvement_value(
    shanten: int,
    effective_count: int,
    winning_tiles: Mapping[int, int],
) -> float:
    base_value = max(0.0, (100 - 30 * shanten) + 1.2 * effective_count)
    if winning_tiles:
        return base_value * 0.6
    return base_value


def _follow_up_node_score(shanten: int, effective_count: int, winning_tiles: Mapping[int, int]) -> float:
    hu_value = _follow_up_hu_value(shanten=shanten, winning_tiles=winning_tiles)
    ting_value = _follow_up_ting_value(effective_count=effective_count, winning_tiles=winning_tiles)
    improvement_value = _follow_up_improvement_value(
        shanten=shanten,
        effective_count=effective_count,
        winning_tiles=winning_tiles,
    )
    return hu_value + ting_value + improvement_value


def _follow_up_score_bonus(follow_up_nodes: list[FollowUpNode]) -> float:
    if not follow_up_nodes:
        return 0.0
    # 只取前两个最优后继节点，并做衰减累加，避免第二层把当前层完全淹没。
    weights = (0.15, 0.08)
    bonus = 0.0
    for index, node in enumerate(follow_up_nodes[: len(weights)]):
        bonus += node.score * weights[index]
    return bonus


def _decision_sort_key(decision: DiscardDecision) -> tuple:
    return (
        decision.score,
        decision.expected_path_value,
        decision.expected_hu_value,
        decision.hu_total_remaining,
        decision.effective_count,
        -decision.shanten_after_discard,
        -decision.discard,
    )
