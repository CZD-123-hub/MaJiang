from __future__ import annotations

from collections import Counter

from .tiles import JIUJIANG_TILE_CODES, JIUJIANG_TILE_SET


def remaining_counts_from_data(data: dict, hand: list[int] | tuple[int, ...]) -> dict[int, int]:
    """根据局面数据估算当前还能摸到的九江牌数量。"""
    remain_card_stack = data.get("remain_card_stack")
    if isinstance(remain_card_stack, list):
        # 有真实牌墙时直接统计，优先级最高。
        counts = Counter(tile for tile in remain_card_stack if tile in JIUJIANG_TILE_SET)
        return {tile: counts.get(tile, 0) for tile in JIUJIANG_TILE_CODES}

    # 没有真实牌墙时，退回到“自己手牌 + 场上可见牌 + 副露”的已知信息估算。
    known_tiles = [tile for tile in hand if tile in JIUJIANG_TILE_SET]

    played_cards = data.get("played_cards") or []
    if any(played_cards):
        for player_cards in played_cards:
            known_tiles.extend(tile for tile in (player_cards or []) if tile in JIUJIANG_TILE_SET)
    else:
        action_seq = data.get("action_seq") or []
        for action in action_seq:
            if not isinstance(action, (list, tuple)) or len(action) < 3:
                continue
            if action[1] != 7:
                continue
            tile = action[2]
            if tile in JIUJIANG_TILE_SET:
                known_tiles.append(tile)

    meld_fields = (
        "player_chi_cards",
        "player_peng_cards",
        "player_gang_cards",
        "player_bugang_cards",
        "player_angang_cards",
        "player_bu_cards",
    )
    for field in meld_fields:
        players = data.get(field) or []
        for groups in players:
            for group in groups or []:
                if isinstance(group, (list, tuple)):
                    known_tiles.extend(tile for tile in group if tile in JIUJIANG_TILE_SET)

    known_counts = Counter(known_tiles)
    return {tile: max(0, 4 - known_counts[tile]) for tile in JIUJIANG_TILE_CODES}
