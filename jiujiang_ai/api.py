from __future__ import annotations

from collections import Counter

from .context import remaining_counts_from_data
from .evaluator import choose_discard, hand_value
from .hand_split import analyze_hand
from .hu import HuOptions, can_hu
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


def get_action(data: dict) -> tuple[int, list[int]]:
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

    is_ting = bool(
        hand
        and winning_tile_counts(
            hand,
            hu_options,
            fixed_melds=fixed_melds,
            remaining_counts=remaining_counts,
        )
    )

    # 杠牌保留原始动作类型：明杠=3、暗杠=5、补杠=6。
    if not is_ting:
        best_gang = _best_gang(action_cards, hand, data)
        if best_gang is not None:
            return best_gang

    # 已听牌时先不主动碰，尽量保持当前听口。
    best_peng = None if is_ting else _best_peng(action_cards, hand, data)
    if best_peng is not None:
        return ACTION_PENG, best_peng

    if ACTION_DISCARD in action_cards and hand:
        discard_candidates = _legal_discard_candidates(action_cards[ACTION_DISCARD], hand, data)
        # 未开启跑红中翻倍时不能出红中；如果候选只剩红中，也不能崩溃。
        if not discard_candidates:
            return ACTION_PASS, []
        decision = choose_discard(
            hand,
            discard_candidates,
            fixed_melds=fixed_melds,
            visible_discards=_visible_discard_counts(data),
            remaining_counts=remaining_counts,
        )
        return ACTION_DISCARD, [decision.discard]

    # 没有更高优先级动作时，若服务端提示可听，则选择听牌。
    if ACTION_TING in action_cards:
        return ACTION_TING, []

    return ACTION_PASS, []


def round_end(data: dict) -> dict[str, object]:
    # 对局结束后累计统计结果，方便后续自博弈和与其他 AI 对打时汇总。
    stats = record_round_end(data)
    return {
        "status": "ok",
        "received": True,
        "data": data,
        "stats": stats,
        "win_context": detect_win_context(data).to_dict(),
    }


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


def _best_peng(action_cards: dict[int, list[list[int]]], hand: list[int], data: dict) -> list[int] | None:
    best_cards: list[int] | None = None
    fixed_melds = _fixed_meld_count(data)
    best_value = hand_value(hand, fixed_melds=fixed_melds) if hand else 0
    for cards in action_cards.get(ACTION_PENG, []):
        if not is_legal_operation(ACTION_PENG, cards):
            continue
        if _is_peng_blocked_by_pass(data, cards[0]):
            continue

        # 碰牌会消耗手中的两张同牌，这里用碰后的手牌价值来决定是否碰。
        simulated = list(hand)
        for tile in cards[:2]:
            if tile in simulated:
                simulated.remove(tile)
        value = hand_value(simulated, fixed_melds=fixed_melds + 1) if simulated else 0
        if value > best_value:
            best_value = value
            best_cards = cards
    return best_cards


def _best_gang(
    action_cards: dict[int, list[list[int]]],
    hand: list[int],
    data: dict,
) -> tuple[int, list[int]] | None:
    # 没有手牌上下文时保留旧行为：只要合法就杠，避免接口提示场景直接失效。
    if not hand:
        for gang_action in sorted(GANG_ACTIONS):
            for cards in action_cards.get(gang_action, []):
                if is_legal_operation(gang_action, cards):
                    return gang_action, cards
        return None

    fixed_melds = _fixed_meld_count(data)
    before_value = hand_value(hand, fixed_melds=fixed_melds)
    before_analysis = analyze_hand(hand, fixed_melds=fixed_melds)

    best_action: int | None = None
    best_cards: list[int] | None = None
    best_value = float("-inf")
    for gang_action in sorted(GANG_ACTIONS):
        for cards in action_cards.get(gang_action, []):
            if not is_legal_operation(gang_action, cards):
                continue
            after_hand, after_fixed_melds = _simulate_gang_hand(hand, fixed_melds, gang_action, cards)
            after_analysis = analyze_hand(after_hand, fixed_melds=after_fixed_melds)
            after_value = hand_value(after_hand, fixed_melds=after_fixed_melds)

            # 第一版收益判断保持保守：杠后不能明显退步。
            if after_analysis.shanten > before_analysis.shanten:
                continue
            if after_analysis.shanten == before_analysis.shanten and after_value + 1.0 < before_value:
                continue
            if after_value > best_value:
                best_value = after_value
                best_action = gang_action
                best_cards = cards
    if best_action is None or best_cards is None:
        return None
    return best_action, best_cards


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
