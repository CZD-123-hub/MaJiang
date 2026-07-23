from enum import Enum
from typing import Optional, List
from mortal_part.tile import Tile


class Sutehai:
    def __init__(self, tile: Tile, is_tedashi: bool = False):
        self.tile = tile
        self.is_tedashi = is_tedashi  # 是否是手切

    def __str__(self) -> str:
        return (f"{self.tile}"
                f"{'' if self.is_tedashi else '^'}")

    def __repr__(self) -> str:
        return f"Sutehai(tile={self.tile}, is_tedashi={self.is_tedashi}"


class ChiPon:
    def __init__(self, consumed: [Tile, Tile], target_tile: Tile):
        self.consumed = consumed
        self.target_tile = target_tile

    def __str__(self) -> str:
        return f"({self.consumed[0]}{self.consumed[1]}+{self.target_tile})"

    def __repr__(self) -> str:
        return f"ChiPon(consumed={self.consumed}, target_tile={self.target_tile})"


class KawaItem:
    def __init__(self, chi_pon: ChiPon = None,
                 kan: Optional[List[Tile]] = None,
                 sutehai: Optional[Sutehai] = None):
        self.chi_pon = chi_pon
        self.kan = kan if kan is not None else []
        self.sutehai = sutehai

    def __str__(self) -> str:
        result = []

        # 处理杠
        if self.kan:
            kan_str = "{" + "".join(str(tile) for tile in self.kan) + "}"
            result.append(kan_str)

        # 处理吃碰
        if self.chi_pon:
            result.append(str(self.chi_pon))

        # 处理舍牌
        if self.sutehai:
            result.append(str(self.sutehai))

        return "".join(result)

    def __repr__(self) -> str:
        return (f"KawaItem(pon={self.chi_pon!r}, "
                f"kan={self.kan!r}, sutehai={self.sutehai!r})")


class MoveType(Enum):
    """表示游戏中不同的牌移动类型"""
    TSUMO = "tsumo"      # 摸牌
    DISCARD = "discard"  # 打牌
    FULU_CONSUME = "fulu_consume"  # 用于副露的牌
