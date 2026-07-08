from __future__ import annotations

from .tiles import HONGZHONG

ACTION_PASS = 0
ACTION_CHI = 1
ACTION_PENG = 2
ACTION_GANG = 3
ACTION_HU = 4
ACTION_ANGANG = 5
ACTION_BUGANG = 6
ACTION_DISCARD = 7
ACTION_TING = 8

GANG_ACTIONS = {ACTION_GANG, ACTION_ANGANG, ACTION_BUGANG}


def normalize_action_cards(action_cards: dict[str | int, list[list[int]]]) -> dict[int, list[list[int]]]:
    normalized: dict[int, list[list[int]]] = {}
    for action_type, cards in action_cards.items():
        normalized[int(action_type)] = cards
    return normalized


def is_legal_operation(action_type: int, cards: list[int]) -> bool:
    if action_type == ACTION_CHI:
        return False
    if action_type == ACTION_PENG:
        return bool(cards) and HONGZHONG not in cards
    if action_type in GANG_ACTIONS:
        return bool(cards) and HONGZHONG not in cards
    if action_type in {ACTION_PASS, ACTION_HU, ACTION_DISCARD, ACTION_TING}:
        return True
    return False
