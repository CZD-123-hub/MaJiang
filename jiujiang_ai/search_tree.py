from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .hand_split import analyze_hand
from .hu import HuOptions, can_hu
from .tiles import JIUJIANG_TILE_CODES, remaining_tile_counts, validate_hand
from .ting import ting_discards, winning_tile_counts


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
    follow_up_nodes: tuple["FollowUpNode", ...]
    follow_up_score_bonus: float
    path_value: float


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
    expected_follow_up_shanten: float
    expected_follow_up_effective_count: float
    hu_child_count: int
    hu_total_remaining: int
    improving_child_count: int
    ting_child_count: int
    children: tuple[DrawNode, ...]


@dataclass(frozen=True)
class FollowUpNode:
    """第二层显式展开的“摸后再弃”节点。"""

    discard: int
    shanten: int
    effective_count: int
    winning_tiles: dict[int, int]
    score: float


@dataclass(frozen=True)
class SearchTreeResult:
    """整棵一层显式搜索树的汇总结果。"""

    best_discard: int
    discard_scores: dict[int, DiscardDecision]
    node_count: int


def expand_discard_tree(
    hand: list[int] | tuple[int, ...],
    candidate_cards: list[list[int]] | None = None,
    options: HuOptions | None = None,
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
    max_draws: int | None = None,
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
) -> DiscardDecision:
    """只返回推荐弃牌对应的一层树搜索结果。"""
    result = expand_discard_tree(
        hand=hand,
        candidate_cards=candidate_cards,
        options=options,
        fixed_melds=fixed_melds,
        remaining_counts=remaining_counts,
        max_draws=max_draws,
    )
    return result.discard_scores[result.best_discard]


def _evaluate_discard(
    discard: int,
    after_discard: list[int],
    options: HuOptions,
    fixed_melds: int,
    remaining_counts: Mapping[int, int] | None,
    max_draws: int | None,
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

    children = [
        _build_draw_node(
            after_discard=after_discard,
            draw=tile,
            remaining=remaining.get(tile, 0),
            options=options,
            fixed_melds=fixed_melds,
            current_shanten=analysis.shanten,
            remaining_counts=remaining,
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
        expected_follow_up_shanten=expected_follow_up_shanten,
        expected_follow_up_effective_count=expected_follow_up_effective_count,
        hu_total_remaining=sum(child.remaining for child in hu_children),
        improving_total_remaining=sum(child.remaining for child in improving_children),
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
        expected_follow_up_shanten=expected_follow_up_shanten,
        expected_follow_up_effective_count=expected_follow_up_effective_count,
        hu_child_count=len(hu_children),
        hu_total_remaining=sum(child.remaining for child in hu_children),
        improving_child_count=len(improving_children),
        ting_child_count=len(ting_children),
        children=tuple(children),
    )


def _build_draw_node(
    after_discard: list[int],
    draw: int,
    remaining: int,
    options: HuOptions,
    fixed_melds: int,
    current_shanten: int,
    remaining_counts: Mapping[int, int],
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
        options=options,
        fixed_melds=fixed_melds,
        remaining_counts=next_remaining_counts,
    )
    best_follow_up_node = follow_up_nodes[0] if follow_up_nodes else None
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
        follow_up_nodes=tuple(follow_up_nodes),
        follow_up_score_bonus=follow_up_score_bonus,
        path_value=hu_value + ting_value + improvement_value + follow_up_score_bonus,
    )


def _score_decision(
    shanten_after_discard: int,
    effective_count: int,
    winning_tiles: Mapping[int, int],
    expected_path_value: float,
    expected_hu_value: float,
    expected_ting_value: float,
    expected_improvement_value: float,
    expected_follow_up_shanten: float,
    expected_follow_up_effective_count: float,
    hu_total_remaining: int,
    improving_total_remaining: int,
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
        - 5.0 * expected_follow_up_shanten
        + 0.8 * expected_follow_up_effective_count
        + 5 * hu_total_remaining
        + 1.5 * improving_total_remaining
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
    options: HuOptions,
    fixed_melds: int,
    remaining_counts: Mapping[int, int],
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

    nodes: list[FollowUpNode] = []
    for discard, shanten in candidate_infos:
        if shanten != best_shanten:
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
        node = FollowUpNode(
            discard=discard,
            shanten=shanten,
            effective_count=effective_count,
            winning_tiles=winning_tiles,
            score=_follow_up_node_score(shanten=shanten, effective_count=effective_count, winning_tiles=winning_tiles),
        )
        nodes.append(node)
    nodes.sort(key=_follow_up_sort_key, reverse=True)
    return nodes[:3]


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
        node.score,
        bool(node.winning_tiles),
        -node.shanten,
        node.effective_count,
        -node.discard,
    )


def _follow_up_node_score(shanten: int, effective_count: int, winning_tiles: Mapping[int, int]) -> float:
    return (100 - 30 * shanten) + 2.0 * effective_count + 3.0 * len(winning_tiles)


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
