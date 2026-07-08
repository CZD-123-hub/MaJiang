from __future__ import annotations

from collections import Counter

WAN_CODES = tuple(range(0x01, 0x0A))
TIAO_CODES = tuple(range(0x11, 0x1A))
TONG_CODES = tuple(range(0x21, 0x2A))
HONGZHONG = 0x35

SUITED_TILE_CODES = WAN_CODES + TIAO_CODES + TONG_CODES
JIUJIANG_TILE_CODES = SUITED_TILE_CODES + (HONGZHONG,)
JIUJIANG_TILE_SET = set(JIUJIANG_TILE_CODES)


def validate_tile(tile: int) -> None:
    if tile not in JIUJIANG_TILE_SET:
        raise ValueError(f"invalid Jiujiang tile: {tile!r}")


def validate_hand(tiles: list[int] | tuple[int, ...]) -> None:
    invalid = [tile for tile in tiles if tile not in JIUJIANG_TILE_SET]
    if invalid:
        raise ValueError(f"hand contains invalid Jiujiang tiles: {invalid!r}")
    over_limit = [tile for tile, count in Counter(tiles).items() if count > 4]
    if over_limit:
        raise ValueError(f"tile count exceeds four copies: {over_limit!r}")


def is_hongzhong(tile: int) -> bool:
    return tile == HONGZHONG


def is_suited(tile: int) -> bool:
    return tile in set(SUITED_TILE_CODES)


def tile_rank(tile: int) -> int:
    validate_tile(tile)
    return tile & 0x0F


def tile_suit(tile: int) -> str:
    validate_tile(tile)
    if tile == HONGZHONG:
        return "Z"
    prefix = tile & 0xF0
    if prefix == 0x00:
        return "W"
    if prefix == 0x10:
        return "T"
    return "B"


def can_start_sequence(tile: int) -> bool:
    return is_suited(tile) and tile_rank(tile) <= 7


def tile_name(tile: int) -> str:
    validate_tile(tile)
    if tile == HONGZHONG:
        return "RED"
    return f"{tile_rank(tile)}{tile_suit(tile)}"


def format_tiles(tiles: list[int] | tuple[int, ...]) -> list[str]:
    return [tile_name(tile) for tile in tiles]


def remaining_tile_counts(known_tiles: list[int] | tuple[int, ...]) -> dict[int, int]:
    validate_hand(known_tiles)
    known = Counter(known_tiles)
    return {tile: 4 - known[tile] for tile in JIUJIANG_TILE_CODES}
