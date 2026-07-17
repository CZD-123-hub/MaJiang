from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

from .hand_split import analyze_hand, analyze_normalized_counts
from .hu import HuOptions
from .ting import winning_tile_counts
from .tiles import HONGZHONG, JIUJIANG_TILE_CODES, SUITED_TILE_CODES, remaining_tile_counts, validate_hand


_SUITED_TILE_INDEX = {tile: index for index, tile in enumerate(SUITED_TILE_CODES)}


@dataclass(frozen=True)
class DiscardDecision:
    discard: int
    score: float
    shanten_after_discard: int
    effective_count: int
    winning_tiles: dict[int, int]
    safety_score: int


@dataclass(frozen=True)
class TwoPlyDiscardDecision(DiscardDecision):
    """A bounded two-ply extension of :class:`DiscardDecision`.

    ``expected_path_value`` is the probability-weighted shape value after an
    improving draw and the best following discard.  It is deliberately kept
    separate from the first-ply score so decision logs can explain a changed
    recommendation without changing the public, lightweight decision shape.
    """

    expected_path_value: float
    explored_draw_types: int


@dataclass(frozen=True)
class _QiduiAnalysis:
    """七对路线的轻量摘要，只在房间显式开启七对时参与弃牌比较。"""

    shanten: int
    pairs: int
    singles: int
    hongzhong_used: int


def score_discards(
    hand: list[int],
    candidate_cards: list[list[int]],
    fixed_melds: int = 0,
    visible_discards: Mapping[int, int] | None = None,
    remaining_counts: Mapping[int, int] | None = None,
    options: HuOptions | None = None,
    win_multiplier_by_discard: Mapping[int, int | float] | None = None,
    deadline: float | None = None,
) -> dict[int, DiscardDecision]:
    validate_hand(hand)
    visible_discards = visible_discards or {}
    options = options or HuOptions()
    win_multiplier_by_discard = win_multiplier_by_discard or {}
    hand_counts = Counter(hand)
    ordinary_counts = tuple(hand_counts[tile] for tile in SUITED_TILE_CODES)
    hongzhong_count = hand_counts[HONGZHONG]
    # First build every candidate's cheap, structural preview.  This pass is
    # deliberately completed before the deadline-sensitive effective-draw
    # work: when a busy machine runs out of time, it can still choose from all
    # legal discards using the same hand-split quality rather than falling
    # back to a crude shape heuristic.
    previews = []
    for card_group in candidate_cards:
        if not card_group:
            continue
        discard = card_group[0]
        # 平台通常会按实体牌返回候选，同一种牌可能出现多次；按牌值评分一次即可。
        if discard not in hand or any(item[0] == discard for item in previews):
            continue
        after = list(hand)
        after.remove(discard)
        after_counts, after_hongzhong_count = _counts_after_discard(
            ordinary_counts,
            hongzhong_count,
            discard,
        )
        analysis = analyze_normalized_counts(
            after_counts,
            after_hongzhong_count,
            fixed_melds=fixed_melds,
        )
        qidui = _qidui_analysis(
            after_counts,
            after_hongzhong_count,
            fixed_melds=fixed_melds,
            options=options,
        )
        shanten_after_discard = _best_shanten(analysis.shanten, qidui)
        previews.append(
            (
                discard,
                after,
                after_counts,
                after_hongzhong_count,
                analysis,
                qidui,
                shanten_after_discard,
            )
        )

    scores: dict[int, DiscardDecision] = {}
    for discard, after, after_counts, after_hongzhong_count, analysis, qidui, shanten_after_discard in previews:
        # Exact waits are comparatively cheap and are the highest-priority
        # information in the production sort key, so collect them for every
        # candidate before spending budget on non-tenpai effective draws.
        winning_tiles = winning_tile_counts(
            after,
            options=options,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
        )
        effective_count = sum(winning_tiles.values()) if winning_tiles else 0
        multiplier = _positive_multiplier(win_multiplier_by_discard.get(discard, 1.0))
        weighted_effective_count = effective_count * multiplier
        score = max(
            _score_analysis(analysis, weighted_effective_count),
            _score_qidui_analysis(qidui, weighted_effective_count),
        )
        scores[discard] = DiscardDecision(
            discard=discard,
            score=score,
            shanten_after_discard=shanten_after_discard,
            effective_count=effective_count,
            winning_tiles=winning_tiles,
            safety_score=int(visible_discards.get(discard, 0)),
        )

    # Non-tenpai effective draws are the expensive part.  Calculate the
    # strongest structural candidates first.  A deadline only stops this
    # refinement pass; the complete preview table above remains available.
    ordered_non_tenpai = sorted(
        (decision for decision in scores.values() if not decision.winning_tiles),
        key=_discard_sort_key,
        reverse=True,
    )
    preview_by_discard = {preview[0]: preview for preview in previews}
    for initial in ordered_non_tenpai:
        try:
            _check_deadline(deadline)
            (
                discard,
                after,
                after_counts,
                after_hongzhong_count,
                analysis,
                qidui,
                shanten_after_discard,
            ) = preview_by_discard[initial.discard]
            effective_count = _effective_draw_count(
                after,
                shanten_after_discard,
                fixed_melds=fixed_melds,
                remaining_counts=remaining_counts,
                deadline=deadline,
                ordinary_counts=after_counts,
                hongzhong_count=after_hongzhong_count,
                options=options,
            )
        except _DecisionDeadlineExceeded:
            break
        multiplier = _positive_multiplier(win_multiplier_by_discard.get(discard, 1.0))
        weighted_effective_count = effective_count * multiplier
        scores[discard] = DiscardDecision(
            discard=discard,
            score=max(
                _score_analysis(analysis, weighted_effective_count),
                _score_qidui_analysis(qidui, weighted_effective_count),
            ),
            shanten_after_discard=shanten_after_discard,
            effective_count=effective_count,
            winning_tiles={},
            safety_score=int(visible_discards.get(discard, 0)),
        )
    return scores


def choose_discard(
    hand: list[int],
    candidate_cards: list[list[int]],
    fixed_melds: int = 0,
    visible_discards: Mapping[int, int] | None = None,
    remaining_counts: Mapping[int, int] | None = None,
    options: HuOptions | None = None,
    win_multiplier_by_discard: Mapping[int, int | float] | None = None,
    deadline: float | None = None,
) -> DiscardDecision:
    if deadline is not None and perf_counter() >= deadline:
        return choose_fast_discard(
            hand,
            candidate_cards,
            fixed_melds=fixed_melds,
            visible_discards=visible_discards,
        )
    try:
        scores = score_discards(
            hand,
            candidate_cards,
            fixed_melds=fixed_melds,
            visible_discards=visible_discards,
            remaining_counts=remaining_counts,
            options=options,
            win_multiplier_by_discard=win_multiplier_by_discard,
            deadline=deadline,
        )
    except _DecisionDeadlineExceeded:
        return choose_fast_discard(
            hand,
            candidate_cards,
            fixed_melds=fixed_melds,
            visible_discards=visible_discards,
        )
    if not scores:
        raise ValueError("no valid discard candidates")
    return max(scores.values(), key=_discard_sort_key)


def choose_two_ply_discard(
    hand: list[int],
    candidate_cards: list[list[int]],
    fixed_melds: int = 0,
    visible_discards: Mapping[int, int] | None = None,
    remaining_counts: Mapping[int, int] | None = None,
    options: HuOptions | None = None,
    win_multiplier_by_discard: Mapping[int, int | float] | None = None,
    deadline: float | None = None,
    *,
    root_limit: int = 3,
    draw_limit: int = 6,
) -> DiscardDecision:
    """Choose a discard with a time-bounded, two-ply continuation search.

    The first ply uses the production heuristic for every legal discard.  We
    then expand only its best few *non-tenpai* roots through a small number of
    shanten-preserving/improving draws, and choose the best lightweight second
    discard at each child.  The expansion is an adjustment to the robust
    first-ply score, not a replacement for it.  If the shared request deadline
    is reached while expanding, the already-calculated first-ply result is
    returned unchanged.
    """
    if deadline is not None and perf_counter() >= deadline:
        return choose_fast_discard(
            hand,
            candidate_cards,
            fixed_melds=fixed_melds,
            visible_discards=visible_discards,
        )
    try:
        scores = score_discards(
            hand,
            candidate_cards,
            fixed_melds=fixed_melds,
            visible_discards=visible_discards,
            remaining_counts=remaining_counts,
            options=options,
            win_multiplier_by_discard=win_multiplier_by_discard,
            deadline=deadline,
        )
    except _DecisionDeadlineExceeded:
        return choose_fast_discard(
            hand,
            candidate_cards,
            fixed_melds=fixed_melds,
            visible_discards=visible_discards,
        )
    if not scores:
        raise ValueError("no valid discard candidates")

    baseline = max(scores.values(), key=_discard_sort_key)
    roots = sorted(scores.values(), key=_discard_sort_key, reverse=True)[:max(1, root_limit)]
    adjusted: dict[int, DiscardDecision] = dict(scores)
    # Keep a small response margin for framework serialization/network work.
    # The extension is also capped independently, so a costly first ply never
    # turns into an expensive first-plus-second ply.
    extension_deadline = (
        min(deadline - 0.015, perf_counter() + 0.060)
        if deadline is not None
        else None
    )
    try:
        for root in roots:
            _check_deadline(extension_deadline)
            # Actual tenpai waits are already exact and have priority in the
            # production sort key.  Do not dilute that information with an
            # approximate future-discard simulation.
            if root.winning_tiles:
                continue
            after = list(hand)
            after.remove(root.discard)
            remaining = remaining_counts or remaining_tile_counts(after)
            draw_nodes = _two_ply_draw_nodes(
                after,
                root.shanten_after_discard,
                fixed_melds=fixed_melds,
                options=options,
                remaining_counts=remaining,
                draw_limit=draw_limit,
                deadline=extension_deadline,
            )
            if not draw_nodes:
                continue

            weighted_value = 0.0
            total_weight = 0
            for draw, copies in draw_nodes:
                _check_deadline(extension_deadline)
                leaf_value = _best_light_followup_value(
                    [*after, draw],
                    fixed_melds=fixed_melds,
                    options=options,
                    deadline=extension_deadline,
                )
                weighted_value += copies * leaf_value
                total_weight += copies
            if not total_weight:
                continue

            expected_path_value = weighted_value / total_weight
            current_shape_value = _light_hand_value(
                after,
                fixed_melds=fixed_melds,
                options=options,
            )
            wall_total = sum(max(0, count) for count in remaining.values())
            progress_probability = total_weight / wall_total if wall_total else 0.0
            # A continuation should influence only the fraction of draws it
            # represents.  This keeps the known-good first-ply score dominant
            # while rewarding roots whose useful draws retain a better shape.
            continuation_bonus = 0.55 * progress_probability * (expected_path_value - current_shape_value)
            adjusted[root.discard] = TwoPlyDiscardDecision(
                discard=root.discard,
                score=root.score + continuation_bonus,
                shanten_after_discard=root.shanten_after_discard,
                effective_count=root.effective_count,
                winning_tiles=root.winning_tiles,
                safety_score=root.safety_score,
                expected_path_value=expected_path_value,
                explored_draw_types=len(draw_nodes),
            )
    except _DecisionDeadlineExceeded:
        return baseline
    return max(adjusted.values(), key=_discard_sort_key)


def choose_fast_discard(
    hand: list[int],
    candidate_cards: list[list[int]],
    fixed_melds: int = 0,
    visible_discards: Mapping[int, int] | None = None,
) -> DiscardDecision:
    """并发繁忙时使用的常数级近似策略，不进入递归手牌分析。"""
    validate_hand(hand)
    visible_discards = visible_discards or {}
    decisions: dict[int, DiscardDecision] = {}
    for card_group in candidate_cards:
        if not card_group:
            continue
        discard = card_group[0]
        if discard not in hand or discard in decisions:
            continue
        after = list(hand)
        after.remove(discard)
        decisions[discard] = DiscardDecision(
            discard=discard,
            score=_fast_shape_score(after),
            shanten_after_discard=0,
            effective_count=0,
            winning_tiles={},
            safety_score=int(visible_discards.get(discard, 0)),
        )
    if not decisions:
        raise ValueError("no valid discard candidates")
    return max(decisions.values(), key=_discard_sort_key)


def _fast_shape_score(hand: list[int]) -> float:
    """用对子、刻子和同门邻张近似保留价值，保证繁忙回退在毫秒内完成。"""
    counts = Counter(hand)
    score = counts[HONGZHONG] * 8.0
    for tile, count in counts.items():
        if tile == HONGZHONG:
            continue
        score += max(0, count - 1) * 3.0 + max(0, count - 2) * 2.0
        rank = tile & 0x0F
        if rank <= 8:
            score += min(count, counts.get(tile + 1, 0)) * 1.6
        if rank <= 7:
            score += min(count, counts.get(tile + 2, 0)) * 0.8
    return score


def hand_value(
    hand: list[int],
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
    options: HuOptions | None = None,
    deadline: float | None = None,
) -> float:
    analysis = analyze_hand(hand, fixed_melds=fixed_melds)
    options = options or HuOptions()
    qidui = _qidui_analysis_from_hand(hand, fixed_melds=fixed_melds, options=options)
    shanten = _best_shanten(analysis.shanten, qidui)
    try:
        effective_count = _effective_draw_count(
            hand,
            shanten,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
            deadline=deadline,
            options=options,
        )
    except _DecisionDeadlineExceeded:
        # Shape analysis is already available.  Returning that deterministic
        # preview is preferable to spending an unbounded amount of time only
        # to estimate non-tenpai effective draws.
        effective_count = 0
    return max(_score_analysis(analysis, effective_count), _score_qidui_analysis(qidui, effective_count))


def _discard_sort_key(decision: DiscardDecision) -> tuple:
    # 第一层：真实能进听的出牌优先，避免被向听近似分误导。
    if decision.winning_tiles:
        return (
            1,
            decision.effective_count,
            decision.score,
            decision.safety_score,
            decision.discard != HONGZHONG,
            -decision.shanten_after_discard,
            -decision.discard,
        )

    # 未进听时仍沿用综合分，红中只作为同分附近的保留倾向。
    return (
        0,
        decision.score,
        decision.effective_count,
        decision.safety_score,
        decision.discard != HONGZHONG,
        -decision.shanten_after_discard,
        -decision.discard,
    )


def _effective_draw_count(
    hand: list[int],
    current_shanten: int,
    fixed_melds: int = 0,
    remaining_counts: Mapping[int, int] | None = None,
    deadline: float | None = None,
    ordinary_counts: tuple[int, ...] | None = None,
    hongzhong_count: int | None = None,
    options: HuOptions | None = None,
) -> int:
    remaining = remaining_counts or remaining_tile_counts(hand)
    count = 0
    for tile in JIUJIANG_TILE_CODES:
        _check_deadline(deadline)
        if remaining.get(tile, 0) <= 0:
            continue
        if ordinary_counts is None or hongzhong_count is None:
            next_analysis = analyze_hand([*hand, tile], fixed_melds=fixed_melds)
        else:
            next_counts, next_hongzhong_count = _counts_after_draw(
                ordinary_counts,
                hongzhong_count,
                tile,
            )
            next_analysis = analyze_normalized_counts(
                next_counts,
                next_hongzhong_count,
                fixed_melds=fixed_melds,
            )
        next_qidui = (
            _qidui_analysis_from_hand([*hand, tile], fixed_melds=fixed_melds, options=options)
            if ordinary_counts is None or hongzhong_count is None
            else _qidui_analysis(next_counts, next_hongzhong_count, fixed_melds=fixed_melds, options=options)
        )
        if _best_shanten(next_analysis.shanten, next_qidui) < current_shanten:
            count += remaining.get(tile, 0)
    return count


def _two_ply_draw_nodes(
    hand: list[int],
    current_shanten: int,
    *,
    fixed_melds: int,
    options: HuOptions | None,
    remaining_counts: Mapping[int, int],
    draw_limit: int,
    deadline: float | None,
) -> list[tuple[int, int]]:
    """Return only the best bounded set of useful draw nodes.

    This is intentionally a normalized-count calculation: no full hu checks
    happen while pruning children.  Exact waits are handled by the first ply
    before this helper is called.
    """
    counts = Counter(hand)
    ordinary_counts = tuple(counts[tile] for tile in SUITED_TILE_CODES)
    hongzhong_count = counts[HONGZHONG]
    nodes: list[tuple[int, float, int, int]] = []
    for draw in JIUJIANG_TILE_CODES:
        _check_deadline(deadline)
        copies = int(remaining_counts.get(draw, 0))
        if copies <= 0:
            continue
        next_counts, next_hongzhong = _counts_after_draw(ordinary_counts, hongzhong_count, draw)
        analysis = analyze_normalized_counts(next_counts, next_hongzhong, fixed_melds=fixed_melds)
        qidui = _qidui_analysis(
            next_counts,
            next_hongzhong,
            fixed_melds=fixed_melds,
            options=options,
        )
        shanten = _best_shanten(analysis.shanten, qidui)
        # A draw which cannot retain the current route can always be thrown
        # back; excluding it both bounds the tree and avoids double-counting
        # no-progress paths already represented by the first-ply heuristic.
        if shanten > current_shanten:
            continue
        shape = max(_score_analysis(analysis, 0), _score_qidui_analysis(qidui, 0))
        nodes.append((shanten, -shape, -copies, draw))
    nodes.sort()
    return [(draw, -negative_copies) for _, _, negative_copies, draw in nodes[:max(1, draw_limit)]]


def _best_light_followup_value(
    hand: list[int],
    *,
    fixed_melds: int,
    options: HuOptions | None,
    deadline: float | None,
) -> float:
    """Score the best second discard without starting a third-ply search."""
    counts = Counter(hand)
    ordinary_counts = tuple(counts[tile] for tile in SUITED_TILE_CODES)
    hongzhong_count = counts[HONGZHONG]
    best = float("-inf")
    for discard in sorted(counts):
        _check_deadline(deadline)
        after_counts, after_hongzhong = _counts_after_discard(ordinary_counts, hongzhong_count, discard)
        analysis = analyze_normalized_counts(after_counts, after_hongzhong, fixed_melds=fixed_melds)
        qidui = _qidui_analysis(
            after_counts,
            after_hongzhong,
            fixed_melds=fixed_melds,
            options=options,
        )
        best = max(best, _score_analysis(analysis, 0), _score_qidui_analysis(qidui, 0))
    return best


def _light_hand_value(
    hand: list[int],
    *,
    fixed_melds: int,
    options: HuOptions | None,
) -> float:
    counts = Counter(hand)
    ordinary_counts = tuple(counts[tile] for tile in SUITED_TILE_CODES)
    hongzhong_count = counts[HONGZHONG]
    analysis = analyze_normalized_counts(ordinary_counts, hongzhong_count, fixed_melds=fixed_melds)
    qidui = _qidui_analysis(
        ordinary_counts,
        hongzhong_count,
        fixed_melds=fixed_melds,
        options=options,
    )
    return max(_score_analysis(analysis, 0), _score_qidui_analysis(qidui, 0))


def _counts_after_discard(
    ordinary_counts: tuple[int, ...],
    hongzhong_count: int,
    discard: int,
) -> tuple[tuple[int, ...], int]:
    if discard == HONGZHONG:
        return ordinary_counts, hongzhong_count - 1
    index = _SUITED_TILE_INDEX[discard]
    return (
        ordinary_counts[:index]
        + (ordinary_counts[index] - 1,)
        + ordinary_counts[index + 1 :],
        hongzhong_count,
    )


def _counts_after_draw(
    ordinary_counts: tuple[int, ...],
    hongzhong_count: int,
    draw: int,
) -> tuple[tuple[int, ...], int]:
    if draw == HONGZHONG:
        return ordinary_counts, hongzhong_count + 1
    index = _SUITED_TILE_INDEX[draw]
    return (
        ordinary_counts[:index]
        + (ordinary_counts[index] + 1,)
        + ordinary_counts[index + 1 :],
        hongzhong_count,
    )


class _DecisionDeadlineExceeded(RuntimeError):
    pass


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise _DecisionDeadlineExceeded


def _score_analysis(analysis, effective_count: float) -> float:
    return (
        100
        - 30 * analysis.shanten
        + 4 * analysis.melds
        + 1.5 * analysis.taatsu
        + analysis.pairs
        + 0.8 * effective_count
        - 0.5 * analysis.leftovers
        - 0.3 * analysis.hongzhong_used
    )


def _score_qidui_analysis(analysis: _QiduiAnalysis | None, effective_count: float) -> float:
    if analysis is None:
        return float("-inf")
    return (
        100
        - 30 * analysis.shanten
        + 2.5 * analysis.pairs
        + 0.8 * effective_count
        - 0.5 * analysis.singles
        - 0.3 * analysis.hongzhong_used
    )


def _qidui_analysis_from_hand(
    hand: list[int],
    *,
    fixed_melds: int,
    options: HuOptions | None,
) -> _QiduiAnalysis | None:
    if not options or not options.allow_qidui or fixed_melds:
        return None
    counts = Counter(hand)
    ordinary_counts = tuple(counts[tile] for tile in SUITED_TILE_CODES)
    return _qidui_analysis(
        ordinary_counts,
        counts[HONGZHONG],
        fixed_melds=fixed_melds,
        options=options,
    )


def _qidui_analysis(
    ordinary_counts: tuple[int, ...],
    hongzhong_count: int,
    *,
    fixed_melds: int,
    options: HuOptions | None,
) -> _QiduiAnalysis | None:
    if not options or not options.allow_qidui or fixed_melds:
        return None
    pairs = sum(count // 2 for count in ordinary_counts)
    singles = sum(count % 2 for count in ordinary_counts)
    used_to_pair_single = min(hongzhong_count, singles)
    pairs += used_to_pair_single
    singles -= used_to_pair_single
    remaining_hongzhong = hongzhong_count - used_to_pair_single
    red_pairs = remaining_hongzhong // 2
    pairs += red_pairs
    used = used_to_pair_single + red_pairs * 2
    singles += remaining_hongzhong % 2
    return _QiduiAnalysis(
        shanten=max(0, 7 - min(7, pairs)),
        pairs=min(7, pairs),
        singles=singles,
        hongzhong_used=used,
    )


def _best_shanten(normal_shanten: int, qidui: _QiduiAnalysis | None) -> int:
    return min(normal_shanten, qidui.shanten) if qidui is not None else normal_shanten


def _positive_multiplier(value: int | float | object) -> float:
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return 1.0
