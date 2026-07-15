"""九江红中麻将四人同策略自博弈裁判。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable

from .api import get_action, round_end
from .hu import can_hu
from .rules import (
    ACTION_ANGANG,
    ACTION_BUGANG,
    ACTION_DISCARD,
    ACTION_GANG,
    ACTION_HU,
    ACTION_PASS,
    ACTION_PENG,
    ACTION_TING,
)
from .settlement import calculate_total_score
from .tiles import HONGZHONG, JIUJIANG_TILE_CODES
from .ting import ting_discards

PLAYER_COUNT = 4
DecisionFunction = Callable[[dict[str, Any]], tuple[int, list[int]]]
EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class _RoundState:
    deck: list[int]
    hands: list[list[int]]
    dealer: int
    played_cards: list[list[int]] = field(default_factory=lambda: [[] for _ in range(PLAYER_COUNT)])
    peng_cards: list[list[list[int]]] = field(default_factory=lambda: [[] for _ in range(PLAYER_COUNT)])
    gang_cards: list[list[list[int]]] = field(default_factory=lambda: [[] for _ in range(PLAYER_COUNT)])
    bugang_cards: list[list[list[int]]] = field(default_factory=lambda: [[] for _ in range(PLAYER_COUNT)])
    angang_cards: list[list[list[int]]] = field(default_factory=lambda: [[] for _ in range(PLAYER_COUNT)])
    action_seq: list[list[int]] = field(default_factory=list)
    event_callback: EventCallback | None = None


def play_round(
    *,
    seed: int | None = None,
    dealer: int = 0,
    decision_fn: DecisionFunction = get_action,
    event_callback: EventCallback | None = None,
) -> dict[str, Any]:
    """运行一局四人同策略自博弈，并返回局面、结算与简要结果。"""
    if not 0 <= dealer < PLAYER_COUNT:
        raise ValueError("dealer must be between 0 and 3")

    state = _new_round(Random(seed), dealer, event_callback)
    current = dealer
    draw_required = True
    winner: int | None = None
    dianpao_player: int | None = None
    win_type: str | None = None
    turns = 0

    while turns < 600:
        turns += 1
        if draw_required:
            if not state.deck:
                break
            tile = state.deck.pop()
            state.hands[current].append(tile)
            _emit(state, {"event": "draw", "player": current, "tile": tile, "wall_remaining": len(state.deck)})
            draw_required = False

        own_action = _decide_own_turn(state, current, decision_fn)
        if own_action[0] == "hu":
            winner, win_type = current, "zimo"
            break
        if own_action[0] == "gang":
            draw_required = True
            continue

        discarded_tile = own_action[1]
        resolution = _resolve_discard(state, current, discarded_tile, decision_fn)
        if resolution[0] == "hu":
            winner, dianpao_player, win_type = resolution[1], current, "dianpao"
            break
        if resolution[0] == "peng":
            current, draw_required = resolution[1], False
            continue
        if resolution[0] == "gang":
            current, draw_required = resolution[1], True
            continue

        current = (current + 1) % PLAYER_COUNT
        draw_required = True

    is_draw = winner is None
    round_data = _round_data(state, winner, dianpao_player, win_type, is_draw)
    score_delta = calculate_total_score(round_data)["score_by_player"]
    round_data["scores"] = score_delta
    result = round_end(round_data)
    summary = {
        "dealer": dealer,
        "winner": winner,
        "dianpao_player": dianpao_player,
        "win_type": win_type or "draw",
        "is_draw": is_draw,
        "turns": turns,
        "action_count": len(state.action_seq),
        "score_delta": score_delta,
    }
    _emit(state, {"event": "round_end", **summary})
    return {"summary": summary, "round_data": round_data, "settlement": result["settlement"]}


def run_self_play(
    *,
    rounds: int,
    seed: int | None = None,
    decision_fn: DecisionFunction = get_action,
    event_callback: EventCallback | None = None,
) -> dict[str, Any]:
    """连续运行指定局数，并给出独立于进程全局统计的汇总。"""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    random = Random(seed)
    results = []
    for index in range(rounds):
        callback = None
        if event_callback is not None:
            callback = lambda event, round_number=index + 1: event_callback({"round": round_number, **event})
        results.append(
            play_round(
                seed=random.randrange(2**63),
                dealer=index % PLAYER_COUNT,
                decision_fn=decision_fn,
                event_callback=callback,
            )
        )
    summaries = [result["summary"] for result in results]
    wins_by_player = {str(player): 0 for player in range(PLAYER_COUNT)}
    total_score_by_player = {str(player): 0.0 for player in range(PLAYER_COUNT)}
    for summary in summaries:
        winner = summary["winner"]
        if winner is not None:
            wins_by_player[str(winner)] += 1
        for player, score in enumerate(summary["score_delta"]):
            total_score_by_player[str(player)] += float(score)

    wins = sum(wins_by_player.values())
    return {
        "rounds": rounds,
        "wins": wins,
        "draws": rounds - wins,
        "wins_by_player": wins_by_player,
        "total_score_by_player": total_score_by_player,
        "round_results": summaries,
    }


def _new_round(random: Random, dealer: int, event_callback: EventCallback | None) -> _RoundState:
    deck = [tile for tile in JIUJIANG_TILE_CODES for _ in range(4)]
    random.shuffle(deck)
    hands = [[deck.pop() for _ in range(13)] for _ in range(PLAYER_COUNT)]
    return _RoundState(deck=deck, hands=hands, dealer=dealer, event_callback=event_callback)


def _decide_own_turn(state: _RoundState, position: int, decision_fn: DecisionFunction) -> tuple[str, int | None]:
    hand = state.hands[position]
    cards = _own_action_cards(state, position)
    action_type, action_card = _call_decision(state, position, position, cards, decision_fn)
    if action_type == ACTION_HU and "4" in cards:
        _record_action(state, [position, ACTION_HU])
        return "hu", None
    if action_type == ACTION_ANGANG and action_card in cards.get("5", []):
        tile = action_card[0]
        _remove_tiles(hand, tile, 4)
        state.angang_cards[position].append(list(action_card))
        _record_action(state, [position, ACTION_ANGANG, *action_card])
        return "gang", None
    if action_type == ACTION_BUGANG and action_card in cards.get("6", []):
        tile = action_card[0]
        _remove_tiles(hand, tile, 1)
        _promote_peng_to_bugang(state, position, tile, action_card)
        _record_action(state, [position, ACTION_BUGANG, *action_card])
        return "gang", None

    if action_type == ACTION_TING and "8" in cards:
        _record_action(state, [position, ACTION_TING])

    legal_discards = cards["7"]
    chosen = action_card if action_type == ACTION_DISCARD and action_card in legal_discards else legal_discards[0]
    tile = chosen[0]
    hand.remove(tile)
    state.played_cards[position].append(tile)
    _record_action(state, [position, ACTION_DISCARD, tile])
    return "discard", tile


def _resolve_discard(
    state: _RoundState,
    discard_player: int,
    tile: int | None,
    decision_fn: DecisionFunction,
) -> tuple[str, int | None]:
    if tile is None:
        return "pass", None
    responders = [((discard_player + offset) % PLAYER_COUNT) for offset in range(1, PLAYER_COUNT)]

    # 胡牌有最高优先级；本自博弈版本按最近座位处理一炮一响。
    for position in responders:
        cards = _response_action_cards(state, position, tile)
        if "4" not in cards:
            continue
        action_type, _ = _call_decision(state, discard_player, position, cards, decision_fn)
        if action_type == ACTION_HU:
            _record_action(state, [position, ACTION_HU])
            return "hu", position

    for position in responders:
        cards = _response_action_cards(state, position, tile)
        if not ({"2", "3"} & set(cards)):
            continue
        action_type, action_card = _call_decision(state, discard_player, position, cards, decision_fn)
        if action_type == ACTION_GANG and action_card in cards.get("3", []):
            _remove_tiles(state.hands[position], tile, 3)
            state.gang_cards[position].append(list(action_card))
            _record_action(state, [position, ACTION_GANG, *action_card])
            return "gang", position
        if action_type == ACTION_PENG and action_card in cards.get("2", []):
            _remove_tiles(state.hands[position], tile, 2)
            state.peng_cards[position].append(list(action_card))
            _record_action(state, [position, ACTION_PENG, *action_card])
            return "peng", position
        _record_action(state, [position, ACTION_PASS])
    return "pass", None


def _own_action_cards(state: _RoundState, position: int) -> dict[str, list[list[int]]]:
    hand = state.hands[position]
    cards: dict[str, list[list[int]]] = {}
    if can_hu(hand, fixed_melds=_fixed_meld_count(state, position)):
        cards["4"] = []
    counts = Counter(hand)
    angang = [[tile] * 4 for tile, count in sorted(counts.items()) if count >= 4 and tile != HONGZHONG]
    if angang:
        cards["5"] = angang
    bugang = []
    for group in state.peng_cards[position]:
        tile = group[0]
        if counts[tile] and tile != HONGZHONG:
            bugang.append([tile] * 4)
    if bugang:
        cards["6"] = bugang
    cards["7"] = [[tile] for tile in sorted(set(hand))]
    if ting_discards(hand, cards["7"], fixed_melds=_fixed_meld_count(state, position)):
        cards["8"] = []
    return cards


def _response_action_cards(state: _RoundState, position: int, tile: int) -> dict[str, list[list[int]]]:
    hand = state.hands[position]
    cards: dict[str, list[list[int]]] = {"0": []}
    if can_hu([*hand, tile], fixed_melds=_fixed_meld_count(state, position)):
        cards["4"] = []
    if tile != HONGZHONG:
        count = hand.count(tile)
        if count >= 3:
            cards["3"] = [[tile] * 4]
        if count >= 2:
            cards["2"] = [[tile] * 3]
    return cards


def _call_decision(
    state: _RoundState,
    acting_player: int,
    position: int,
    action_cards: dict[str, list[list[int]]],
    decision_fn: DecisionFunction,
) -> tuple[int, list[int]]:
    payload = _decision_payload(state, acting_player, position, action_cards)
    try:
        action_type, action_card = decision_fn(payload)
        return int(action_type), list(action_card)
    except (TypeError, ValueError, KeyError):
        return ACTION_PASS, []


def _decision_payload(
    state: _RoundState,
    acting_player: int,
    position: int,
    action_cards: dict[str, list[list[int]]],
) -> dict[str, Any]:
    # 裁判保存完整手牌，但 AI 只能看到当前自己的手牌。
    visible_hands = [[] for _ in range(PLAYER_COUNT)]
    visible_hands[position] = list(state.hands[position])
    return {
        "room_id": 0,
        "game_area_id": 10021,
        "acting_player_position": acting_player,
        "acting_do_player_position": position,
        "dealer": state.dealer,
        "played_cards": [list(cards) for cards in state.played_cards],
        "player_hand_cards": visible_hands,
        "action_seq": [list(action) for action in state.action_seq],
        "last_action": list(state.action_seq[-1]) if state.action_seq else [],
        "player_chi_cards": [[] for _ in range(PLAYER_COUNT)],
        "player_peng_cards": [[list(group) for group in groups] for groups in state.peng_cards],
        "player_gang_cards": [[list(group) for group in groups] for groups in state.gang_cards],
        "player_bugang_cards": [[list(group) for group in groups] for groups in state.bugang_cards],
        "player_angang_cards": [[list(group) for group in groups] for groups in state.angang_cards],
        "player_bu_cards": [[] for _ in range(PLAYER_COUNT)],
        "action_cards": action_cards,
        "remain_card_stack": list(state.deck),
    }


def _round_data(
    state: _RoundState,
    winner: int | None,
    dianpao_player: int | None,
    win_type: str | None,
    is_draw: bool,
) -> dict[str, Any]:
    return {
        "dealer": state.dealer,
        "winner": winner,
        "dianpao_player": dianpao_player,
        "win_type": win_type,
        "huangzhuang": is_draw,
        "played_cards": state.played_cards,
        "player_hand_cards": state.hands,
        "action_seq": state.action_seq,
        "player_chi_cards": [[] for _ in range(PLAYER_COUNT)],
        "player_peng_cards": state.peng_cards,
        "player_gang_cards": state.gang_cards,
        "player_bugang_cards": state.bugang_cards,
        "player_angang_cards": state.angang_cards,
        "player_bu_cards": [[] for _ in range(PLAYER_COUNT)],
        "remain_card_stack": state.deck,
    }


def _fixed_meld_count(state: _RoundState, position: int) -> int:
    return sum(
        len(groups[position])
        for groups in (state.peng_cards, state.gang_cards, state.bugang_cards, state.angang_cards)
    )


def _remove_tiles(hand: list[int], tile: int, count: int) -> None:
    for _ in range(count):
        hand.remove(tile)


def _promote_peng_to_bugang(state: _RoundState, position: int, tile: int, group: list[int]) -> None:
    for index, peng in enumerate(state.peng_cards[position]):
        if peng and peng[0] == tile:
            state.peng_cards[position].pop(index)
            break
    state.bugang_cards[position].append(list(group))


def _record_action(state: _RoundState, action: list[int]) -> None:
    state.action_seq.append(action)
    _emit(
        state,
        {
            "event": "action",
            "player": action[0],
            "action_type": action[1],
            "action_card": action[2:],
            "wall_remaining": len(state.deck),
        },
    )


def _emit(state: _RoundState, event: dict[str, Any]) -> None:
    if state.event_callback is not None:
        state.event_callback(event)
