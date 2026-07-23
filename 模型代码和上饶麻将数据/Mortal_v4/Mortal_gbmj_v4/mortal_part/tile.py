from enum import Enum
from typing import Dict, Optional

MJAI_PAI_STRINGS_LEN = 9 * 3 + 7 + 1

# 麻将牌的字符串表示
MJAI_PAI_STRINGS = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",  # m
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",  # p
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",  # s
    "E", "S", "W", "N", "C", "F", "P",  # 东南西北中发白
    "?",  # unknown
    # 00, 01, 02, 03, 04, 05, 06, 07, 08,
    # 09, 10, 11, 12, 13, 14, 15, 16, 17,
    # 18, 19, 20, 21, 22, 23, 24, 25, 26,
    # 27, 28, 29, 30, 31, 32, 33,
]

# 丢牌时的优先级
DISCARD_PRIORITIES = [
    6, 5, 4, 3, 2, 3, 4, 5, 6,  # m
    6, 5, 4, 3, 2, 3, 4, 5, 6,  # p
    6, 5, 4, 3, 2, 3, 4, 5, 6,  # s
    7, 7, 7, 7, 7, 7, 7,
    0,  # unknown
]

# 构造字典以麻将字符串为键，所引为值
MJAI_PAI_STRINGS_MAP = {s: i for i, s in enumerate(MJAI_PAI_STRINGS)}


class InvalidTile(Exception):
    """
    无效牌异常类
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"not a valid tile: {self.value}"


class Tile:
    def __init__(self, tile_id: int):
        if tile_id >= len(MJAI_PAI_STRINGS):
            raise InvalidTile(tile_id)
        self.id = tile_id

    @classmethod
    def from_str(cls, s: str):
        """
        通过麻将牌的字符串形式，创建一个相应的Tile类
        @param s:
        @return:
        """
        if s not in MJAI_PAI_STRINGS_MAP:
            raise InvalidTile(s)
        return cls(MJAI_PAI_STRINGS_MAP[s])

    def __repr__(self):
        return MJAI_PAI_STRINGS[self.id]

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def is_yaokyuu(self):
        """
        是否是幺九牌
        @return:true or false
        """
        if self.is_jihai():
            return True
        return self.id in [0, 8, 9, 17, 18, 26]

    def is_jihai(self):
        """
        是否是字牌
        @return:
        """
        return 27 <= self.id <= 33

    def is_unknown(self):
        """
        是否是未知牌
        @return: true or false
        """
        return self.id == 34

    def next(self):
        """
        返回同花色的下一张牌
        @return: Tile类
        """
        if self.is_unknown():
            return self
        kind = self.id // 9  # 获取花色
        num = self.id % 9  # 获取数字
        if kind < 3:
            return Tile(kind * 9 + (num + 1) % 9)
        else:
            if num < 4:
                return Tile(27 + (num + 1) % 4)
            else:
                return Tile(31 + (num - 3) % 3)

    def prev(self):
        """
        返回前一张牌
        @return: Tile类
        """
        if self.is_unknown():
            return self
        kind = self.id // 9
        num = self.id % 9
        if kind < 3:
            return Tile(kind * 9 + (num + 8) % 9)
        else:
            if num < 4:
                return Tile(27 + (num + 3) % 4)
            else:
                return Tile(31 + (num - 2) % 3)

    def augment(self):
        """
        增强处理，没看太懂有什么用，就交换一下位置吗
        @return:Tile
        """
        if self.is_unknown():
            return self
        tid = tile.id
        kind = tid // 9
        ret = tile
        if kind == 0:
            ret = Tile(tid + 9)
        elif kind == 1:
            ret = Tile(tid - 9)
        return ret

    def cmp_discard_priority(self, other):
        """
        比较弃牌的优先级，如果两张牌的优先级相同，则按照它们的 id 进行反向比较（即 id 值较小的牌会有更高的优先级）。
        @param other:要比较的Tile
        @return:大于、等于、小于
        """
        r = other.id
        if DISCARD_PRIORITIES[self.id] == DISCARD_PRIORITIES[r]:
            return r - self.id
        return DISCARD_PRIORITIES[self.id] - DISCARD_PRIORITIES[r]

    def to_dict(self):
        return self.__repr__()


# Test cases
if __name__ == "__main__":
    # Convert
    print(Tile.from_str("5m").id)
    print(Tile.from_str("5p").id)
    print(Tile.from_str("5s").id)
    t1 = Tile(6)
    t2 = Tile(16)
    t3 = Tile(26)
    print(t1.cmp_discard_priority(t3))

    try:
        Tile.from_str("")
    except InvalidTile as e:
        print(e)

    try:
        Tile.from_str("0s")
    except InvalidTile as e:
        print(e)

    try:
        Tile.from_str("!")
    except InvalidTile as e:
        print(e)

    try:
        Tile(29)
    except InvalidTile as e:
        print(e)

    try:
        Tile(255)
    except InvalidTile as e:
        print(e)

    # Next and prev
    for s in MJAI_PAI_STRINGS[:-1]:
        tile = Tile.from_str(s)
        assert tile.prev().next() == tile
        assert tile.next().prev() == tile
