"""上饶麻将最小规则适配器。

这里故意不依赖 ``mortal_cpp.so``，以便先在 Windows 本机做规则单测，
再由 ``PlayerState`` 在训练/回放时调用。当前覆盖：

* 精牌作为万能牌的普通四面子一雀头和牌；
* 无副露七对和牌；
* 对应的待牌和近似向听数。

十三烂、九幺、精牌特殊限制及最终计分仍需用牌谱样本继续扩展；当这些
规则出现时，转换/回放测试会明确报告，而不会回退到国标八番判断。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence


TILE_KIND_COUNT = 34
YAOJIU_TILE_IDS = frozenset((0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33))


def _validate_hand(hand34: Sequence[int]) -> tuple[int, ...]:
    if len(hand34) != TILE_KIND_COUNT:
        raise ValueError(f"hand34 应有 {TILE_KIND_COUNT} 项，实际为 {len(hand34)}")
    hand = tuple(int(count) for count in hand34)
    if any(count < 0 or count > 4 for count in hand):
        raise ValueError(f"hand34 的牌数必须在 0–4，实际为 {hand!r}")
    return hand


def _is_suited(tile_id: int) -> bool:
    return 0 <= tile_id < 27


def _regular_shanten_without_king(hand: tuple[int, ...], open_melds: int) -> int:
    """计算不含精牌时的普通牌型向听数。

    经典公式为 ``8 - 2 * 面子 - 搭子 - 雀头``。深度优先搜索枚举刻子、
    顺子、对子和搭子；开放副露已经是完整面子，所以作为初始值传入。
    """

    best = 8
    counts = list(hand)

    def search(start: int, melds: int, taatsu: int, pair: int) -> None:
        nonlocal best
        while start < TILE_KIND_COUNT and counts[start] == 0:
            start += 1
        if start >= TILE_KIND_COUNT:
            usable_taatsu = min(taatsu, max(0, 4 - melds))
            best = min(best, 8 - 2 * melds - usable_taatsu - pair)
            return

        # 三张相同牌组成刻子。
        if counts[start] >= 3:
            counts[start] -= 3
            search(start, melds + 1, taatsu, pair)
            counts[start] += 3

        # 同花色相邻三张组成顺子。
        if _is_suited(start) and start % 9 <= 6 and counts[start + 1] and counts[start + 2]:
            counts[start] -= 1
            counts[start + 1] -= 1
            counts[start + 2] -= 1
            search(start, melds + 1, taatsu, pair)
            counts[start] += 1
            counts[start + 1] += 1
            counts[start + 2] += 1

        # 两张相同牌既可以作雀头，也可以作一个搭子；两种分支都要保留。
        if counts[start] >= 2:
            counts[start] -= 2
            if not pair:
                search(start, melds, taatsu, 1)
            search(start, melds, taatsu + 1, pair)
            counts[start] += 2

        # 两面/边张/嵌张搭子。
        if _is_suited(start) and start % 9 <= 7 and counts[start + 1]:
            counts[start] -= 1
            counts[start + 1] -= 1
            search(start, melds, taatsu + 1, pair)
            counts[start] += 1
            counts[start + 1] += 1
        if _is_suited(start) and start % 9 <= 6 and counts[start + 2]:
            counts[start] -= 1
            counts[start + 2] -= 1
            search(start, melds, taatsu + 1, pair)
            counts[start] += 1
            counts[start + 2] += 1

        # 将当前牌视为孤张。
        counts[start] -= 1
        search(start, melds, taatsu, pair)
        counts[start] += 1

    search(0, int(open_melds), 0, 0)
    return best


def _seven_pairs_can_hu(hand: tuple[int, ...], wildcards: int, open_melds: int) -> bool:
    """检查无副露七对；四张同牌按两对处理。"""
    if open_melds:
        return False
    if sum(hand) + wildcards != 14:
        return False

    pairs = sum(count // 2 for count in hand)
    singles = sum(count % 2 for count in hand)
    use_for_singles = min(singles, wildcards)
    pairs += use_for_singles
    wildcards -= use_for_singles
    pairs += wildcards // 2
    return pairs >= 7


def _thirteen_lan_can_hu(hand: tuple[int, ...], wildcards: int, open_melds: int) -> bool:
    """检查上饶十三烂：无副露、字牌不重复、同花色任意两张至少相隔三位。

    这与 ``v5_rh/Node_SSL.py`` 的 `ssl_two_table` / `ssl_three_table`
    一致。精牌通过回溯补成任意合法位置；规则本身不要求四面子一雀头。
    """
    if open_melds or sum(hand) + wildcards != 14 or any(count > 1 for count in hand):
        return False

    def can_place(counts: tuple[int, ...], tile: int) -> bool:
        if counts[tile]:
            return False
        if tile >= 27:
            return True
        suit_start = tile // 9 * 9
        rank = tile % 9
        return all(
            counts[other] == 0 or abs((other % 9) - rank) >= 3
            for other in range(suit_start, suit_start + 9)
        )

    @lru_cache(maxsize=None)
    def fill_with_wildcards(counts: tuple[int, ...], remaining: int) -> bool:
        if remaining == 0:
            return True
        for tile in range(TILE_KIND_COUNT):
            if can_place(counts, tile):
                next_counts = list(counts)
                next_counts[tile] = 1
                if fill_with_wildcards(tuple(next_counts), remaining - 1):
                    return True
        return False

    return fill_with_wildcards(hand, wildcards)


def _jiuyao_can_hu(hand: tuple[int, ...], wildcards: int, open_melds_are_yaojiu: bool) -> bool:
    """检查 `v5_rh/Node_JY.py` 所使用的九幺牌型。

    该项目的九幺搜索把所有幺九与字牌视为有效牌，并允许已有的幺九副露；
    它不是国标国士无双的“十三种各一张”限定。精牌可补作幺九/字牌。
    """
    return open_melds_are_yaojiu and all(
        count == 0 or tile in YAOJIU_TILE_IDS for tile, count in enumerate(hand)
    ) and sum(hand) + wildcards > 0


def _regular_can_hu(hand: tuple[int, ...], wildcards: int, open_melds: int) -> bool:
    """用记忆化搜索判断“面子 + 雀头”是否可由精牌补齐。"""
    sets_needed = 4 - int(open_melds)
    if sets_needed < 0 or sum(hand) + wildcards != sets_needed * 3 + 2:
        return False

    @lru_cache(maxsize=None)
    def complete_melds(counts: tuple[int, ...], remaining_wildcards: int, remaining_sets: int) -> bool:
        if remaining_sets == 0:
            return not any(counts) and remaining_wildcards == 0
        try:
            tile = next(index for index, count in enumerate(counts) if count)
        except StopIteration:
            return remaining_wildcards == remaining_sets * 3

        count = counts[tile]

        # 刻子可以保留 1/2 张真牌、用精牌补其余张；必须枚举而不能只取
        # 3 张真牌，否则会遗漏“留一张去组顺子”的情况。
        for real_count in range(1, min(3, count) + 1):
            needed_wildcards = 3 - real_count
            if needed_wildcards > remaining_wildcards:
                continue
            next_counts = list(counts)
            next_counts[tile] -= real_count
            if complete_melds(tuple(next_counts), remaining_wildcards - needed_wildcards, remaining_sets - 1):
                return True

        # 顺子不一定从当前最小真牌开始：例如手里只有 8、9，精牌可以补
        # 7 组成 7-8-9。因此枚举所有“包含当前牌”的三个可能顺子起点。
        if _is_suited(tile):
            suit_start = tile // 9 * 9
            suit_end = suit_start + 9
            for sequence_start in (tile - 2, tile - 1, tile):
                if not (suit_start <= sequence_start and sequence_start + 2 < suit_end):
                    continue
                next_counts = list(counts)
                next_counts[tile] -= 1
                needed_wildcards = 0
                for sequence_tile in range(sequence_start, sequence_start + 3):
                    if sequence_tile == tile:
                        continue
                    if next_counts[sequence_tile]:
                        next_counts[sequence_tile] -= 1
                    else:
                        needed_wildcards += 1
                if needed_wildcards <= remaining_wildcards and complete_melds(
                    tuple(next_counts), remaining_wildcards - needed_wildcards, remaining_sets - 1
                ):
                    return True
        return False

    def try_pair(real_tile: int | None, real_count: int, pair_wildcards: int) -> bool:
        if pair_wildcards > wildcards:
            return False
        counts = list(hand)
        if real_tile is not None:
            counts[real_tile] -= real_count
        return complete_melds(tuple(counts), wildcards - pair_wildcards, sets_needed)

    # 两张精牌作雀头。
    if try_pair(None, 0, 2):
        return True
    for tile, count in enumerate(hand):
        if not count:
            continue
        # 一张真牌 + 一张精牌作雀头。
        if try_pair(tile, 1, 1):
            return True
        # 两张真牌作雀头。
        if count >= 2 and try_pair(tile, 2, 0):
            return True
    return False


class ShangraoRuleAdapter:
    """为状态机提供精牌相关的向听、待牌与胡牌接口。"""

    def shanten(self, hand34: Sequence[int], open_melds: int, king_tile_id: int | None) -> int:
        """返回 0–6 的特征用向听数。

        `v5_rh/lib_MJ.py:cal_xts()` 的精牌处理同样以“一张精牌至少降低一
        向听”为基线。这里先用标准 DFS 计算普通牌型，再扣除精牌数；真正
        的可胡判断仍由 :meth:`can_hu` 的精确搜索负责。
        """
        hand = _validate_hand(hand34)
        wildcards = 0
        visible_hand = list(hand)
        if king_tile_id is not None:
            if not 0 <= king_tile_id < TILE_KIND_COUNT:
                raise ValueError(f"非法精牌 tile id: {king_tile_id}")
            wildcards = visible_hand[king_tile_id]
            visible_hand[king_tile_id] = 0

        normal = _regular_shanten_without_king(tuple(visible_hand), open_melds) - wildcards

        # 无副露七对的近似向听。精牌优先补孤张，再两两组成对子。
        if not open_melds:
            pairs = sum(count // 2 for count in visible_hand)
            singles = sum(count % 2 for count in visible_hand)
            spare_wildcards = max(0, wildcards - singles)
            seven_pairs = 6 - pairs - min(singles, wildcards) - spare_wildcards // 2
            normal = min(normal, seven_pairs)

        return max(0, min(6, int(normal)))

    def can_hu(
        self,
        hand34: Sequence[int],
        open_melds: int,
        king_tile_id: int | None,
        winning_tile_id: int | None,
        *,
        is_ron: bool,
        open_melds_are_yaojiu: bool = True,
    ) -> bool:
        """判断当前手牌是否能胡。

        自摸调用时，`hand34` 已包含摸到的牌；荣和调用时，`winning_tile_id`
        仍在桌面上，因此需要先加进手牌。该函数不处理上饶番种/结算，只回答
        牌形是否成立，后续可在这里补充十三烂、九幺和精牌限制。
        """
        hand = list(_validate_hand(hand34))
        if is_ron:
            if winning_tile_id is None or not 0 <= winning_tile_id < TILE_KIND_COUNT:
                return False
            if hand[winning_tile_id] >= 4:
                return False
            hand[winning_tile_id] += 1

        wildcards = 0
        if king_tile_id is not None:
            if not 0 <= king_tile_id < TILE_KIND_COUNT:
                return False
            wildcards = hand[king_tile_id]
            hand[king_tile_id] = 0
        concealed = tuple(hand)
        return (
            _regular_can_hu(concealed, wildcards, open_melds)
            or _seven_pairs_can_hu(concealed, wildcards, open_melds)
            or _thirteen_lan_can_hu(concealed, wildcards, open_melds)
            or _jiuyao_can_hu(concealed, wildcards, open_melds_are_yaojiu)
        )

    def waits(
        self,
        hand34: Sequence[int],
        open_melds: int,
        king_tile_id: int | None,
        visible34: Sequence[int],
        open_melds_are_yaojiu: bool = True,
    ) -> list[bool]:
        """枚举当前 3n+1 手牌的有效待牌，不把已见四张的牌标为待牌。"""
        hand = _validate_hand(hand34)
        if len(visible34) != TILE_KIND_COUNT:
            raise ValueError(f"visible34 应有 {TILE_KIND_COUNT} 项，实际为 {len(visible34)}")
        return [
            int(visible34[tile]) < 4
            and self.can_hu(
                hand,
                open_melds,
                king_tile_id,
                tile,
                is_ron=True,
                open_melds_are_yaojiu=open_melds_are_yaojiu,
            )
            for tile in range(TILE_KIND_COUNT)
        ]
