from __future__ import annotations

from collections import Counter

# 三种数字牌的花色基准值。
# 这里沿用上饶麻将 v5_rh 的编码方式：
# 万子 0x01-0x09，条子 0x11-0x19，筒子 0x21-0x29。
SUITED_BASES = {"W": 0x00, "T": 0x10, "B": 0x20}

# 字牌编码：东南西北中发白。
HONOR_CODES = {
    "EAST": 0x31,
    "SOUTH": 0x32,
    "WEST": 0x33,
    "NORTH": 0x34,
    "RED": 0x35,
    "GREEN": 0x36,
    "WHITE": 0x37,
}

# 允许用户输入中文牌名，内部统一转成上面的英文 token。
# 例如“一万”会先转成“1W”，再转成编码 0x01。
CHINESE_TO_TOKEN = {
    "一万": "1W",
    "二万": "2W",
    "三万": "3W",
    "四万": "4W",
    "五万": "5W",
    "六万": "6W",
    "七万": "7W",
    "八万": "8W",
    "九万": "9W",
    "一条": "1T",
    "二条": "2T",
    "三条": "3T",
    "四条": "4T",
    "五条": "5T",
    "六条": "6T",
    "七条": "7T",
    "八条": "8T",
    "九条": "9T",
    "一筒": "1B",
    "二筒": "2B",
    "三筒": "3B",
    "四筒": "4B",
    "五筒": "5B",
    "六筒": "6B",
    "七筒": "7B",
    "八筒": "8B",
    "九筒": "9B",
    "东": "EAST",
    "南": "SOUTH",
    "西": "WEST",
    "北": "NORTH",
    "中": "RED",
    "发": "GREEN",
    "白": "WHITE",
}

# 所有合法牌编码，共 34 种牌。
TILE_CODES = tuple(
    list(range(0x01, 0x0A))
    + list(range(0x11, 0x1A))
    + list(range(0x21, 0x2A))
    + list(range(0x31, 0x38))
)
TILE_SET = set(TILE_CODES)


def tile_code(token: str | int) -> int:
    """把用户输入的一张牌转换成内部编码。"""
    if isinstance(token, int):
        validate_tile(token)
        return token

    token = token.strip().upper()

    # 先兼容中文输入，再处理英文缩写输入。
    token = CHINESE_TO_TOKEN.get(token, token)
    if token in HONOR_CODES:
        return HONOR_CODES[token]

    if len(token) >= 2 and token[0].isdigit():
        rank = int(token[:-1])
        suit = token[-1]
        if suit in SUITED_BASES and 1 <= rank <= 9:
            # 数字牌编码 = 花色基准值 + 点数。
            # 例如 5T = 0x10 + 5 = 0x15。
            return SUITED_BASES[suit] + rank

    raise ValueError(f"无法识别的麻将牌输入：{token!r}")


def tile_name(tile: int) -> str:
    """把内部编码转回便于阅读的牌名。"""
    validate_tile(tile)
    for name, code in HONOR_CODES.items():
        if tile == code:
            return name
    suit = tile_suit(tile)
    return f"{tile_rank(tile)}{suit}"


def parse_tiles(text: str) -> list[int]:
    """把一整副手牌字符串解析成编码列表。"""
    normalized = text.replace(",", " ").replace("，", " ")
    return [tile_code(part) for part in normalized.split() if part.strip()]


def format_tiles(tiles: list[int] | tuple[int, ...]) -> list[str]:
    """把编码列表转成牌名列表，主要用于打印结果。"""
    return [tile_name(tile) for tile in tiles]


def validate_tile(tile: int) -> None:
    """检查单张牌编码是否合法。"""
    if tile not in TILE_SET:
        raise ValueError(f"非法麻将牌编码：{tile!r}")


def validate_hand(tiles: list[int] | tuple[int, ...]) -> None:
    """检查一副手牌是否由合法牌组成，并且同一张牌不超过 4 张。"""
    counts = Counter(tiles)
    invalid = [tile for tile in tiles if tile not in TILE_SET]
    if invalid:
        raise ValueError(f"手牌中存在非法麻将牌编码：{invalid!r}")
    over_limit = [tile_name(tile) for tile, count in counts.items() if count > 4]
    if over_limit:
        raise ValueError(f"同一种牌不能超过 4 张：{over_limit}")


def tile_suit(tile: int) -> str:
    """返回牌的花色：W=万，T=条，B=筒，Z=字牌。"""
    validate_tile(tile)
    prefix = tile & 0xF0
    if prefix == 0x00:
        return "W"
    if prefix == 0x10:
        return "T"
    if prefix == 0x20:
        return "B"
    return "Z"


def tile_rank(tile: int) -> int:
    """返回牌的点数，字牌返回其低 4 位编号。"""
    validate_tile(tile)
    return tile & 0x0F


def is_suited(tile: int) -> bool:
    """判断是否是万、条、筒这类可以组成顺子的数字牌。"""
    return tile_suit(tile) in {"W", "T", "B"}


def can_start_sequence(tile: int) -> bool:
    """判断这张牌能不能作为顺子的第一张牌。"""
    return is_suited(tile) and tile_rank(tile) <= 7


def remaining_tile_counts(known_tiles: list[int] | tuple[int, ...]) -> dict[int, int]:
    """根据已知牌计算每种牌理论上还剩几张。"""
    validate_hand(known_tiles)
    known = Counter(known_tiles)
    return {tile: 4 - known[tile] for tile in TILE_CODES}
