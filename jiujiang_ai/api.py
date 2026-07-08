from __future__ import annotations

from .evaluator import choose_discard, hand_value
from .hu import HuOptions, can_hu
from .rules import (
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
from .ting import winning_tile_counts
from .tiles import JIUJIANG_TILE_SET

def get_action(data: dict) -> tuple[int, list[int]]:
    action_cards = normalize_action_cards(data.get("action_cards", {}))

    # 外部裁判已经提示可胡时，胡牌优先级最高。
    if ACTION_HU in action_cards:
        return ACTION_HU, []

    hand = _acting_hand(data)
    hand = _jiujiang_only(hand)
    hu_options = _hu_options(data)
    # 本地兜底胡牌判断：覆盖平胡、红中万能牌、四红中和可选七对。
    if hand and can_hu(hand, hu_options):
        return ACTION_HU, []

    is_ting = bool(hand and winning_tile_counts(hand, hu_options))

    # 明杠、暗杠、补杠都按杠牌动作处理，但红中杠会在规则层被过滤。
    if not is_ting:
        for gang_action in sorted(GANG_ACTIONS):
            for cards in action_cards.get(gang_action, []):
                if is_legal_operation(gang_action, cards):
                    return ACTION_GANG, cards

    # 已经听牌时先不碰，避免改变手牌结构导致失去当前听口。
    best_peng = None if is_ting else _best_peng(action_cards, hand, data)
    if best_peng is not None:
        return ACTION_PENG, best_peng

    if ACTION_DISCARD in action_cards and hand:
        decision = choose_discard(hand, _legal_discard_candidates(action_cards[ACTION_DISCARD], hand))
        return ACTION_DISCARD, [decision.discard]

    # 没有更高优先级操作时，若服务端提示可听，则选择听牌。
    if ACTION_TING in action_cards:
        return ACTION_TING, []

    return ACTION_PASS, []


def round_end(data: dict) -> dict[str, object]:
    return {"status": "ok", "received": True, "data": data}


def _acting_hand(data: dict) -> list[int]:
    # acting_do_player_position 表示当前真正需要执行动作的玩家座位。
    hands = data.get("player_hand_cards") or []
    position = int(data.get("acting_do_player_position", 0))
    if 0 <= position < len(hands):
        return list(hands[position])
    return []


def _jiujiang_only(cards: list[int]) -> list[int]:
    # 接口说明里的通用样例可能带有非九江牌；API 层过滤掉，核心规则模块仍保持严格校验。
    return [card for card in cards if card in JIUJIANG_TILE_SET]


def _legal_discard_candidates(candidate_cards: list[list[int]], hand: list[int]) -> list[list[int]]:
    # 弃牌候选也只保留九江合法牌，并且必须在当前过滤后的手牌中。
    hand_set = set(hand)
    return [
        cards
        for cards in candidate_cards
        if cards and cards[0] in JIUJIANG_TILE_SET and cards[0] in hand_set
    ]


def _hu_options(data: dict) -> HuOptions:
    # 兼容不同调用方可能使用的房间配置字段名；没有配置时默认不开七对。
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
    best_value = hand_value(hand) if hand else 0
    for cards in action_cards.get(ACTION_PENG, []):
        if not is_legal_operation(ACTION_PENG, cards):
            continue
        if _is_peng_blocked_by_pass(data, cards[0]):
            continue
        # 碰牌会消耗手中的两张同牌，这里用模拟后的手牌价值决定是否碰。
        simulated = list(hand)
        for tile in cards[:2]:
            if tile in simulated:
                simulated.remove(tile)
        value = hand_value(simulated) if simulated else 0
        if value > best_value:
            best_value = value
            best_cards = cards
    return best_cards


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
