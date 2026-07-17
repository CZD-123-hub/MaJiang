from __future__ import annotations

from collections import Counter
from threading import BoundedSemaphore
from time import perf_counter

from .context import remaining_counts_from_data
from .decision_engine import choose_discard as choose_multi_route_discard
from .decision_log import append_decision_log
from .evaluator import choose_discard, choose_fast_discard, choose_two_ply_discard, hand_value
from .hand_split import analyze_hand
from .hu import HuOptions, can_hu
from .search_tree import choose_discard as choose_tree_discard
from .risk import evaluate_discard_risks
from .settlement import calculate_run_hongzhong_multiplier, calculate_total_score
from .rules import (
    ACTION_ANGANG,
    ACTION_BUGANG,
    ACTION_DISCARD,
    ACTION_GANG,
    ACTION_HU,
    ACTION_PASS,
    ACTION_PENG,
    ACTION_TING,
    GANG_ACTIONS,
    is_legal_operation,
    normalize_action_cards,
)
from .stats import record_round_end
from .ting import winning_tile_counts
from .tiles import HONGZHONG, JIUJIANG_TILE_SET
from .win_context import detect_win_context


# 两个测试房间并行时允许各自完成一次精确评估；更高并发仍快速回退，避免
# CPU 堆积导致测试服收不到动作。该上限必须经并发压测后再提高。
_DISCARD_EVALUATION_SLOTS = BoundedSemaphore(2)
_DISCARD_DECISION_BUDGET_SECONDS = 0.24
# 杠分是真实即时收益，但仍保守地只折算一半，避免为了单次杠分破坏成和速度。
_GANG_IMMEDIATE_VALUE_WEIGHT = 0.5
# The bounded two-ply search is the current production strategy.  Callers can
# still turn it off explicitly for A/B replay by sending
# ``two_ply_search_enabled: false`` in their room options.
_TWO_PLY_SEARCH_DEFAULT_ENABLED = True
_BASELINE_SHANTEN_PENALTY = 30.0
_CANDIDATE_SHANTEN_PENALTY = 35.0


def get_action(data: dict) -> tuple[int, list[int]]:
    decision_started_at = perf_counter()
    action_cards = normalize_action_cards(data.get("action_cards", {}))

    # 外部裁判已经明确给出可胡时，胡牌优先级最高。
    if ACTION_HU in action_cards:
        return ACTION_HU, []

    hand = _acting_hand(data)
    hand = _jiujiang_only(hand)
    hu_options = _hu_options(data)
    fixed_melds = _fixed_meld_count(data)
    remaining_counts = remaining_counts_from_data(data, hand) if hand else None

    # 本地兜底胡牌判断要把副露数量一起带上，避免副露后误判不能胡。
    if hand and can_hu(hand, hu_options, fixed_melds=fixed_melds):
        return ACTION_HU, []

    # 只有“手牌 + 固定副露”合计为 13 张时才可能处于等待摸牌状态。
    # 自己摸牌后的 14 张状态无需再遍历全部牌种做听牌判断。
    is_ting = bool(
        hand
        and len(hand) + fixed_melds * 3 == 13
        and winning_tile_counts(
            hand,
            hu_options,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
        )
    )

    # 杠牌保留原始动作类型：明杠=3、暗杠=5、补杠=6。
    if not is_ting:
        best_gang = _best_gang(action_cards, hand, data, hu_options)
        if best_gang is not None:
            return best_gang

    # 已听牌时先不主动碰，尽量保持当前听口。
    best_peng = None if is_ting else _best_peng(action_cards, hand, data, hu_options)
    if best_peng is not None:
        return ACTION_PENG, best_peng

    if ACTION_DISCARD in action_cards and hand:
        discard_candidates = _legal_discard_candidates(action_cards[ACTION_DISCARD], hand, data)
        # 未开启跑红中翻倍时不能出红中；如果候选只剩红中，也不能崩溃。
        if not discard_candidates:
            return ACTION_PASS, []
        decision = _choose_responsive_discard(
            hand,
            discard_candidates,
            data,
            fixed_melds,
            remaining_counts,
            hu_options,
            deadline=decision_started_at + _DISCARD_DECISION_BUDGET_SECONDS,
        )
        result = [decision.discard]
        _record_discard_decision(data, decision, result)
        return ACTION_DISCARD, result

    # 没有更高优先级动作时，若服务端提示可听，则选择听牌。
    if ACTION_TING in action_cards:
        return ACTION_TING, []

    return ACTION_PASS, []


def round_end(data: dict) -> dict[str, object]:
    # 对局结束后，除了累计统计，也同步返回这一局的完整结算结果。
    normalized_data = _normalize_round_end_data(data)
    stats = record_round_end(normalized_data)
    settlement = calculate_total_score(normalized_data)
    return {
        "status": "ok",
        "received": True,
        "data": normalized_data,
        "stats": stats,
        "settlement": settlement,
        "win_context": detect_win_context(normalized_data).to_dict(),
    }


def _normalize_round_end_data(data: dict) -> dict:
    """兼容对战平台的结算字段名，统一为项目内部统计口径。"""
    normalized = dict(data)
    aliases = (
        ("dealer", "banker_position"),
        ("scores", "total_score"),
    )
    for target, source in aliases:
        if normalized.get(target) is None and data.get(source) is not None:
            normalized[target] = data[source]

    platform_winner = data.get("win_player_position")
    is_draw = data.get("end_type") == 2 or (
        platform_winner is not None and not _valid_player_position(platform_winner)
    )
    if is_draw:
        # -1 是测试服的流局哨兵值，不能当作赢家或自摸。
        normalized.pop("winner", None)
        normalized.pop("dianpao_player", None)
        normalized.pop("win_type", None)
        normalized["is_draw"] = True
        return normalized

    if normalized.get("winner") is None and _valid_player_position(platform_winner):
        normalized["winner"] = platform_winner

    winner = normalized.get("winner")
    platform_dianpao = data.get("dp_player_position")
    if (
        normalized.get("dianpao_player") is None
        and _valid_player_position(platform_dianpao)
        and platform_dianpao != winner
    ):
        normalized["dianpao_player"] = platform_dianpao

    if normalized.get("win_type") is None and winner is not None:
        # 平台将自摸时的 dp_player_position 置为赢家本人；点炮时则为另一座位。
        normalized["win_type"] = "dianpao" if normalized.get("dianpao_player") is not None else "zimo"
    return normalized


def _valid_player_position(value: object) -> bool:
    try:
        return 0 <= int(value) <= 3
    except (TypeError, ValueError):
        return False


def _acting_hand(data: dict) -> list[int]:
    # acting_do_player_position 表示当前真正需要执行动作的玩家座位。
    hands = data.get("player_hand_cards") or []
    position = int(data.get("acting_do_player_position", 0))
    if 0 <= position < len(hands):
        return list(hands[position])
    return []


def _jiujiang_only(cards: list[int]) -> list[int]:
    # 输入说明里的通用样例可能带有非九江牌，这里在 API 层先过滤掉。
    return [card for card in cards if card in JIUJIANG_TILE_SET]


def _legal_discard_candidates(candidate_cards: list[list[int]], hand: list[int], data: dict) -> list[list[int]]:
    # 未开启跑红中翻倍时，红中不能作为弃牌候选。
    hand_set = set(hand)
    run_hongzhong_enabled = _run_hongzhong_enabled(data)
    return [
        cards
        for cards in candidate_cards
        if cards
        and cards[0] in JIUJIANG_TILE_SET
        and cards[0] in hand_set
        and (run_hongzhong_enabled or cards[0] != HONGZHONG)
    ]


def _run_hongzhong_enabled(data: dict) -> bool:
    # 兼容任务书里列出的几种字段名，未配置时默认不开启。
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = (
        "run_hongzhong_double",
        "allow_discard_hongzhong",
        "hongzhong_double",
        "跑红中翻倍",
        "出红中翻倍",
    )
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _hu_options(data: dict) -> HuOptions:
    # 兼容不同调用方使用的房间配置字段名；没有配置时默认不开七对。
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    allow_qidui_keys = ("allow_qidui", "can_hu_qidui", "allow_seven_pairs", "可胡七对")
    allow_qidui = any(_truthy(source.get(key)) for source in option_sources for key in allow_qidui_keys)
    return HuOptions(allow_qidui=allow_qidui)


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y", "是", "开启"}
    return bool(value)


def _search_tree_enabled(data: dict) -> bool:
    # 博弈树先开关式接入，默认保持原启发式路径，便于逐步联调。
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = (
        "search_tree_enabled",
        "use_search_tree",
        "enable_search_tree",
        "use_tree_search",
        "启用博弈树搜索",
    )
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _multi_route_enabled(data: dict) -> bool:
    """多拆分综合评分开关，默认关闭以便与既有策略做回放对照。"""
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = (
        "multi_route_enabled",
        "use_multi_route",
        "enable_multi_route",
        "use_composite_decision",
        "启用多路线决策",
    )
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _multi_route_tree_enabled(data: dict) -> bool:
    """多路线搜索树开关；它比单层多路线评分更慢，单独灰度启用。"""
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = (
        "multi_route_tree_enabled",
        "use_multi_route_tree",
        "enable_multi_route_tree",
        "启用多路线搜索树",
    )
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _two_ply_search_enabled(data: dict) -> bool:
    """Enable the bounded production-safe two-ply discard search."""
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = (
        "two_ply_search_enabled",
        "bounded_two_ply_enabled",
        "use_two_ply_search",
        "enable_two_ply_search",
    )
    explicit_values = [
        source[key]
        for source in option_sources
        for key in keys
        if key in source
    ]
    return any(_truthy(value) for value in explicit_values) if explicit_values else _TWO_PLY_SEARCH_DEFAULT_ENABLED


def _team_value_search_enabled(data: dict) -> bool:
    """Opt in to team-aware risk adjustments for discard decisions."""
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = ("team_value_search_enabled", "team_risk_enabled")
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _two_ply_leaf_waits_enabled(data: dict) -> bool:
    """Opt in to exact waiting-tile checks at two-ply leaf nodes."""
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = ("two_ply_leaf_waits_enabled", "two_ply_exact_waits_enabled")
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _decision_logging_enabled(data: dict) -> bool:
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
        data.get("strategy_options") or {},
    ]
    keys = ("decision_log_enabled", "enable_decision_log", "记录决策日志")
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _decision_log_path(data: dict) -> str | None:
    for source in (
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
        data.get("strategy_options") or {},
    ):
        value = source.get("decision_log_path")
        if isinstance(value, str) and value.strip():
            return value
    return None


def strategy_variant(data: dict) -> str:
    """Return the deterministic live A/B strategy label for this room."""
    if _multi_route_tree_enabled(data):
        return "multi_route_tree"
    if _multi_route_enabled(data):
        return "multi_route"
    if _search_tree_enabled(data):
        return "search_tree"
    if _two_ply_search_enabled(data):
        penalty = _shanten_penalty_for_room(data)
        return f"two_ply_shanten{int(penalty)}"
    return "heuristic"


def _strategy_name(data: dict) -> str:
    return strategy_variant(data)


def _shanten_penalty_for_room(data: dict) -> float:
    """Use baseline 30 for every live room while candidate 35 is paused."""
    return _BASELINE_SHANTEN_PENALTY


def _record_discard_decision(data: dict, decision: object, action_card: list[int]) -> None:
    if not _decision_logging_enabled(data):
        return
    try:
        append_decision_log(
            data,
            action_type=ACTION_DISCARD,
            action_card=action_card,
            strategy=_strategy_name(data),
            decision=decision,
            log_path=_decision_log_path(data),
        )
    except OSError:
        # 回放日志属于观测能力，不能因磁盘问题影响对局动作。
        pass


def _choose_discard_decision(
    hand: list[int],
    discard_candidates: list[list[int]],
    data: dict,
    fixed_melds: int,
    remaining_counts: dict[int, int] | None,
    hu_options: HuOptions,
    deadline: float | None = None,
):
    # 开启搜索树时优先用显式树搜索；树搜索异常时自动回退到旧启发式评估器。
    visible_discards = _visible_discard_counts(data)
    score_adjustments = (
        _team_value_score_adjustments(data, discard_candidates)
        if _team_value_search_enabled(data)
        else None
    )
    if _multi_route_tree_enabled(data):
        try:
            return choose_tree_discard(
                hand,
                discard_candidates,
                options=hu_options,
                fixed_melds=fixed_melds,
                remaining_counts=remaining_counts,
                use_multi_route=True,
                decision_data=data,
                acting_position=int(data.get("acting_do_player_position", 0)),
            )
        except Exception:
            pass
    if _multi_route_enabled(data):
        try:
            return choose_multi_route_discard(
                hand,
                discard_candidates,
                data=data,
                acting_position=int(data.get("acting_do_player_position", 0)),
                fixed_melds=fixed_melds,
                remaining_counts=remaining_counts,
            )
        except Exception:
            # 灰度期保证接口稳定；后续接入结构化日志后再收紧异常边界。
            pass
    if _search_tree_enabled(data):
        try:
            return choose_tree_discard(
                hand,
                discard_candidates,
                options=hu_options,
                fixed_melds=fixed_melds,
                remaining_counts=remaining_counts,
            )
        except Exception:
            # An explicit legacy-tree experiment must preserve its historical
            # fallback for reproducible A/B replay; it should not silently
            # switch to the production two-ply strategy.
            return choose_discard(
                hand,
                discard_candidates,
                fixed_melds=fixed_melds,
                visible_discards=visible_discards,
                remaining_counts=remaining_counts,
                options=hu_options,
                win_multiplier_by_discard=_run_hongzhong_multipliers(data, discard_candidates),
                score_adjustments=score_adjustments,
                shanten_penalty=_shanten_penalty_for_room(data),
                deadline=deadline,
            )
    if _two_ply_search_enabled(data):
        return choose_two_ply_discard(
            hand,
            discard_candidates,
            fixed_melds=fixed_melds,
            visible_discards=visible_discards,
            remaining_counts=remaining_counts,
            options=hu_options,
            win_multiplier_by_discard=_run_hongzhong_multipliers(data, discard_candidates),
            score_adjustments=score_adjustments,
            deadline=deadline,
            leaf_exact_waits=_two_ply_leaf_waits_enabled(data),
            continuation_weight=0.55,
            shanten_penalty=_shanten_penalty_for_room(data),
        )
    return choose_discard(
        hand,
        discard_candidates,
        fixed_melds=fixed_melds,
        visible_discards=visible_discards,
        remaining_counts=remaining_counts,
        options=hu_options,
        win_multiplier_by_discard=_run_hongzhong_multipliers(data, discard_candidates),
        score_adjustments=score_adjustments,
        shanten_penalty=_shanten_penalty_for_room(data),
        deadline=deadline,
    )


def _choose_responsive_discard(
    hand: list[int],
    discard_candidates: list[list[int]],
    data: dict,
    fixed_melds: int,
    remaining_counts: dict[int, int] | None,
    hu_options: HuOptions,
    *,
    deadline: float | None = None,
):
    """最多两个重型弃牌评估并行；额外请求立即走轻量合法回退。"""
    acquired = _DISCARD_EVALUATION_SLOTS.acquire(blocking=False)
    if not acquired:
        return choose_fast_discard(
            hand,
            discard_candidates,
            fixed_melds=fixed_melds,
            visible_discards=_visible_discard_counts(data),
        )
    try:
        return _choose_discard_decision(
            hand,
            discard_candidates,
            data,
            fixed_melds,
            remaining_counts,
            hu_options,
            deadline,
        )
    finally:
        _DISCARD_EVALUATION_SLOTS.release()


def _best_peng(
    action_cards: dict[int, list[list[int]]],
    hand: list[int],
    data: dict,
    hu_options: HuOptions,
) -> list[int] | None:
    candidates = [
        cards
        for cards in action_cards.get(ACTION_PENG, [])
        if is_legal_operation(ACTION_PENG, cards) and not _is_peng_blocked_by_pass(data, cards[0])
    ]
    if not candidates or not hand:
        return None

    best_cards: list[int] | None = None
    fixed_melds = _fixed_meld_count(data)
    best_value = hand_value(hand, fixed_melds=fixed_melds, options=hu_options)
    before_shanten = analyze_hand(hand, fixed_melds=fixed_melds).shanten
    for cards in candidates:
        # 碰牌会消耗手中的两张同牌，这里用碰后的手牌价值来决定是否碰。
        simulated = list(hand)
        for tile in cards[:2]:
            if tile not in simulated:
                simulated = []
                break
            simulated.remove(tile)
        if not simulated:
            continue
        if _best_peng_post_discard_shanten(simulated, fixed_melds + 1) > before_shanten:
            continue
        value = hand_value(simulated, fixed_melds=fixed_melds + 1, options=hu_options)
        if value > best_value:
            best_value = value
            best_cards = cards
    return best_cards


def _best_peng_post_discard_shanten(hand: list[int], fixed_melds: int) -> int:
    """Return the best normal-hand shanten after Peng's forced discard.

    This is intentionally structural and cheap: it prevents only a Peng that
    is unambiguously worse even after its best immediate discard, while
    retaining the established immediate-value policy for all other calls.
    """
    if not hand:
        return 99
    return min(
        analyze_hand([card for index, card in enumerate(hand) if index != discard_index], fixed_melds=fixed_melds).shanten
        for discard_index in range(len(hand))
    )


def _best_gang(
    action_cards: dict[int, list[list[int]]],
    hand: list[int],
    data: dict,
    hu_options: HuOptions,
) -> tuple[int, list[int]] | None:
    candidates = [
        (gang_action, cards)
        for gang_action in sorted(GANG_ACTIONS)
        for cards in action_cards.get(gang_action, [])
        if is_legal_operation(gang_action, cards)
    ]
    if not candidates:
        return None

    # 没有手牌上下文时保留旧行为：只要合法就杠，避免接口提示场景直接失效。
    if not hand:
        return candidates[0]

    fixed_melds = _fixed_meld_count(data)
    before_value = hand_value(hand, fixed_melds=fixed_melds, options=hu_options)
    before_analysis = analyze_hand(hand, fixed_melds=fixed_melds)

    best_action: int | None = None
    best_cards: list[int] | None = None
    best_value = float("-inf")
    for gang_action, cards in candidates:
        after_hand, after_fixed_melds = _simulate_gang_hand(hand, fixed_melds, gang_action, cards)
        after_analysis = analyze_hand(after_hand, fixed_melds=after_fixed_melds)
        after_value = hand_value(after_hand, fixed_melds=after_fixed_melds, options=hu_options)
        gang_value = after_value + _gang_immediate_gain(gang_action) * _GANG_IMMEDIATE_VALUE_WEIGHT

        # 第一版收益判断保持保守：杠后不能明显退步。
        if after_analysis.shanten > before_analysis.shanten:
            continue
        if after_analysis.shanten == before_analysis.shanten and gang_value + 1.0 < before_value:
            continue
        if gang_value > best_value:
            best_value = gang_value
            best_action = gang_action
            best_cards = cards
    if best_action is None or best_cards is None:
        return None
    return best_action, best_cards


def _gang_immediate_gain(gang_action: int) -> float:
    """按当前九江结算规则估算杠者个人立即获得的分数。"""
    if gang_action == ACTION_ANGANG:
        return 6.0
    if gang_action in {ACTION_GANG, ACTION_BUGANG}:
        return 3.0
    return 0.0


def _run_hongzhong_multipliers(
    data: dict,
    discard_candidates: list[list[int]],
) -> dict[int, int]:
    """返回每种候选弃牌对应的未来跑红中胡分倍率。"""
    position = int(data.get("acting_do_player_position", 0))
    # 大部分候选都不是红中，先只扫描一次历史牌河；只有红中候选才额外模拟一次。
    base_multiplier = calculate_run_hongzhong_multiplier(data, winner=position)
    has_hongzhong_candidate = any(cards and cards[0] == HONGZHONG for cards in discard_candidates)
    hongzhong_multiplier = (
        calculate_run_hongzhong_multiplier(data, winner=position, extra_hongzhong_discards=1)
        if has_hongzhong_candidate
        else base_multiplier
    )
    multipliers: dict[int, int] = {}
    for cards in discard_candidates:
        if not cards:
            continue
        discard = cards[0]
        if discard in multipliers:
            continue
        multipliers[discard] = hongzhong_multiplier if discard == HONGZHONG else base_multiplier
    return multipliers


def _team_value_score_adjustments(data: dict, discard_candidates: list[list[int]]) -> dict[int, float]:
    """Return conditional team-aware danger penalties for candidate discards.

    The penalty is deliberately inactive in quiet early hands.  It becomes
    relevant only when opponents have opened melds, declared Ting, or the
    wall is late; otherwise this feature must not trade early attacking speed
    for generic caution.
    """
    position = int(data.get("acting_do_player_position", 0))
    player_count = _player_count_from_data(data)
    team_positions = _team_positions(data, position, player_count)
    opponents = set(range(player_count)) - team_positions
    if not opponents:
        return {}

    opponent_melds = sum(_meld_count_for_player(data, opponent) for opponent in opponents)
    ting_players = {
        action[0]
        for action in data.get("action_seq") or []
        if isinstance(action, (list, tuple)) and len(action) >= 2 and action[1] == ACTION_TING
    }
    remaining = data.get("remain_card_count")
    late_wall = isinstance(remaining, int) and remaining <= 20
    if not opponent_melds and not (ting_players & opponents) and not late_wall:
        return {}

    weight = 2.5 + min(opponent_melds, 4) * 0.9
    if ting_players & opponents:
        weight += 2.5
    if late_wall:
        weight += 1.5
    tiles = [cards[0] for cards in discard_candidates if cards]
    risks = evaluate_discard_risks(
        data,
        acting_position=position,
        candidates=tiles,
        opponent_positions=opponents,
    )
    return {tile: -weight * risk.score for tile, risk in risks.items()}


def _team_positions(data: dict, position: int, player_count: int) -> set[int]:
    for source in (data, data.get("room_options") or {}, data.get("game_options") or {}, data.get("options") or {}):
        for key in ("team_positions", "ally_positions"):
            value = source.get(key)
            if isinstance(value, (list, tuple, set)):
                positions = {item for item in value if isinstance(item, int) and 0 <= item < player_count}
                if position in positions:
                    return positions
    # The test server uses the opposite seat as teammate.  Keeping this as a
    # fallback preserves correct behavior when room options omit team metadata.
    return {position, (position + 2) % player_count} if player_count >= 4 else {position}


def _player_count_from_data(data: dict) -> int:
    for field in ("player_hand_cards", "played_cards", "player_peng_cards"):
        value = data.get(field)
        if isinstance(value, list) and value:
            return max(4, len(value))
    return 4


def _meld_count_for_player(data: dict, position: int) -> int:
    total = 0
    for field in ("player_chi_cards", "player_peng_cards", "player_gang_cards", "player_bugang_cards", "player_angang_cards"):
        groups = data.get(field) or []
        if 0 <= position < len(groups):
            total += len(groups[position] or [])
    return total


def _fixed_meld_count(data: dict) -> int:
    # 统计当前玩家已存在的副露组数，供胡牌/听牌/出牌评估统一使用。
    position = int(data.get("acting_do_player_position", 0))
    meld_fields = (
        "player_chi_cards",
        "player_peng_cards",
        "player_gang_cards",
        "player_bugang_cards",
        "player_angang_cards",
    )
    total = 0
    for field in meld_fields:
        groups = data.get(field) or []
        if 0 <= position < len(groups):
            total += len(groups[position] or [])
    return total


def _simulate_gang_hand(
    hand: list[int],
    fixed_melds: int,
    gang_action: int,
    cards: list[int],
) -> tuple[list[int], int]:
    # 明杠通常消耗手中三张，暗杠消耗四张，补杠只在原碰牌基础上再补一张。
    tile = cards[0]
    required = {
        ACTION_GANG: 3,
        ACTION_ANGANG: 4,
        ACTION_BUGANG: 1,
    }.get(gang_action, 4)
    remove_count = min(hand.count(tile), required)
    after_hand = list(hand)
    for _ in range(remove_count):
        after_hand.remove(tile)

    added_melds = 0 if gang_action == ACTION_BUGANG else 1
    return after_hand, fixed_melds + added_melds


def _visible_discard_counts(data: dict) -> dict[int, int]:
    # 安全性第一版只统计场上已经出现过几次同牌，出现越多默认越相对安全。
    counts: Counter[int] = Counter()
    played_cards = data.get("played_cards") or []
    if any(played_cards):
        for player_cards in played_cards:
            for tile in player_cards or []:
                if tile in JIUJIANG_TILE_SET:
                    counts[tile] += 1
        return dict(counts)

    action_seq = data.get("action_seq") or []
    for action in action_seq:
        if not isinstance(action, (list, tuple)) or len(action) < 3:
            continue
        if action[1] != ACTION_DISCARD:
            continue
        tile = action[2]
        if tile in JIUJIANG_TILE_SET:
            counts[tile] += 1
    return dict(counts)


def _is_peng_blocked_by_pass(data: dict, tile: int) -> bool:
    # 过碰过圈：玩家选择不碰某张牌后，在自己下次出牌前不能再碰同一张牌。
    position = int(data.get("acting_do_player_position", 0))
    action_seq = data.get("action_seq") or []
    blocked_tiles: set[int] = set()
    last_discard_tile: int | None = None

    for action in action_seq:
        if not isinstance(action, (list, tuple)) or len(action) < 2:
            continue
        actor = action[0]
        action_type = action[1]

        # 自己出过牌后，上一轮过碰限制重置。
        if actor == position and action_type == ACTION_DISCARD:
            blocked_tiles.clear()

        if action_type == ACTION_DISCARD and len(action) >= 3:
            last_discard_tile = action[2]
            continue

        if actor == position and action_type == ACTION_PASS and last_discard_tile == tile:
            blocked_tiles.add(tile)

    return tile in blocked_tiles
