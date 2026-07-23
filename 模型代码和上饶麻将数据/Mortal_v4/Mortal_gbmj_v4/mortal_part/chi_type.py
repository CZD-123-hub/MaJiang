from __future__ import annotations
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tile import Tile   # 仅类型检查用，按你的实际模块路径调整


class ChiType(IntEnum):
    """
    吃牌时，所打的那枚牌在顺子中的相对位置：
    Low  – 最小的一张（123 中的 1）
    Mid  – 中间的一张（123 中的 2）
    High – 最大的一张（123 中的 3）
    """
    Low = 0
    Mid = 1
    High = 2

    @classmethod
    def from_tiles(cls, consumed: tuple[Tile, Tile], tile: Tile) -> ChiType:
        """
        根据吃牌时消耗的两张手牌 `consumed` 与从别家打出的那张牌 `tile`，
        返回该牌在顺子中的位置枚举。
        """
        a = consumed[0]
        b = consumed[1]
        min_ = min(a.id, b.id)
        max_ = max(a.id, b.id)
        tile_id = tile.id

        if tile_id < min_:
            return ChiType.Low
        elif tile_id < max_:
            return ChiType.Mid
        else:
            return ChiType.High
