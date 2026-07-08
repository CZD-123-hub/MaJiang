from __future__ import annotations

from dataclasses import dataclass

from .hand_split import best_shanten, split_hand
from .tiles import TILE_CODES, remaining_tile_counts, validate_hand


@dataclass(frozen=True)
class DrawNode:
    """摸牌子节点。

    draw 是这一步摸到的牌，remaining 表示这张牌还剩几张，
    shanten 表示摸到它之后重新计算出的向听数。
    """

    draw: int
    remaining: int
    shanten: int


@dataclass(frozen=True)
class DiscardDecision:
    """某一张候选弃牌的评估结果。"""

    discard: int
    score: float
    shanten_after_discard: int
    effective_count: int
    children: tuple[DrawNode, ...]


@dataclass(frozen=True)
class SearchTreeResult:
    """整棵一层搜索树的结果。"""

    best_discard: int
    discard_scores: dict[int, DiscardDecision]
    node_count: int


def expand_discard_tree(hand: list[int] | tuple[int, ...], max_draws: int | None = None) -> SearchTreeResult:
    """构建一层“弃牌 -> 摸牌”的博弈树，并给每种弃牌评分。

    这里做的是任务二里的简化搜索树：
    根节点是当前 14 张手牌，第一层是每一种可弃牌，
    第二层是弃牌后可能摸到的牌。
    """
    validate_hand(hand)
    if len(hand) % 3 != 2:
        raise ValueError("弃牌搜索需要 3n+2 张手牌，通常是 14 张。")

    decisions: dict[int, DiscardDecision] = {}
    for discard in sorted(set(hand)):
        # 同一种牌有多张时，弃其中任意一张效果一样，所以只枚举 set(hand)。
        after_discard = list(hand)
        after_discard.remove(discard)
        decisions[discard] = _evaluate_discard(discard, after_discard, max_draws=max_draws)

    # 分数最高的弃牌作为推荐出牌。
    best = max(decisions.values(), key=lambda decision: (decision.score, -decision.shanten_after_discard, -decision.discard))
    node_count = len(decisions) + sum(len(decision.children) for decision in decisions.values())
    return SearchTreeResult(best_discard=best.discard, discard_scores=decisions, node_count=node_count)


def choose_discard(hand: list[int] | tuple[int, ...]) -> DiscardDecision:
    """只返回推荐弃牌对应的评估结果。"""
    result = expand_discard_tree(hand)
    return result.discard_scores[result.best_discard]


def _evaluate_discard(discard: int, after_discard: list[int], max_draws: int | None) -> DiscardDecision:
    """评估打出某张牌之后的局面。"""
    shanten_after_discard = best_shanten(after_discard)
    remaining = remaining_tile_counts(after_discard)

    # 对所有还没被看完的牌，假设下一张有可能摸到它。
    draws = [tile for tile in TILE_CODES if remaining[tile] > 0]
    children = [
        DrawNode(
            draw=tile,
            remaining=remaining[tile],
            shanten=best_shanten(tuple(sorted([*after_discard, tile]))),
        )
        for tile in draws
    ]

    # 摸牌子节点排序：向听数越低越好，剩余张数越多越好。
    children.sort(key=lambda child: (child.shanten, -child.remaining, child.draw))
    if max_draws:
        children = children[:max_draws]

    # 有效进张：摸到后能让向听数下降的牌。
    effective = [child for child in children if child.shanten < shanten_after_discard]
    score = _score_decision(shanten_after_discard, effective, children, after_discard)
    return DiscardDecision(
        discard=discard,
        score=score,
        shanten_after_discard=shanten_after_discard,
        effective_count=sum(child.remaining for child in effective),
        children=tuple(children),
    )


def _score_decision(
    shanten_after_discard: int,
    effective: list[DrawNode],
    children: list[DrawNode],
    after_discard: list[int],
) -> float:
    """给一个弃牌选择打分。

    评分不是麻将最终强 AI，只是用于任务二演示“节点评估”的规则函数。
    主要目标：
    1. 弃牌后向听数越小越好。
    2. 能改善向听数的有效进张越多越好。
    3. 已有面子、搭子越多越好。
    4. 剩余孤张越少越好。
    """
    best_combo = split_hand(after_discard, limit=1)[0]
    meld_bonus = 2.0 * (len(best_combo.triplets) + len(best_combo.sequences))
    taatsu_bonus = 0.6 * (len(best_combo.pairs) + len(best_combo.taatsu))
    isolated_penalty = 0.25 * len(best_combo.leftovers)
    draw_value = sum(child.remaining * (shanten_after_discard - child.shanten + 1) for child in effective)
    average_future = sum((8 - child.shanten) * child.remaining for child in children) / max(
        1, sum(child.remaining for child in children)
    )
    # 向听数是第一优先级；有效进张只在进度接近的候选弃牌之间做细分。
    return (100 - 30 * shanten_after_discard) + draw_value + average_future + meld_bonus + taatsu_bonus - isolated_penalty
