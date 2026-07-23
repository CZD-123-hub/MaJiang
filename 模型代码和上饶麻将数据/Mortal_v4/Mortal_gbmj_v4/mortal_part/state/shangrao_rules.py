"""上饶麻将规则计算的最小适配层。

本模块刻意只封装已验证可调用的“精牌平胡结构向听”计算，不把 v5_rh 的
启发式推荐分数当作规则，也不假装已经完成胡牌判断。后续扩展 waits/can_hu
时应继续保持接口可单测。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence


class ShangraoRuleError(ValueError):
    """上饶规则输入或 v5_rh 后端不可用时抛出的异常。"""


def tile_id_to_raw(tile_id: int) -> int:
    """把 Mortal 的 0–33 tile id 转为 v5_rh 使用的十六进制牌码整数。"""
    if 0 <= tile_id <= 8:
        return tile_id + 1
    if 9 <= tile_id <= 17:
        return tile_id + 8
    if 18 <= tile_id <= 26:
        return tile_id + 15
    if 27 <= tile_id <= 33:
        return tile_id + 22
    raise ShangraoRuleError(f"tile id 超出 0–33 范围：{tile_id}")


class ShangraoRuleEngine:
    """面向 PlayerState 的上饶规则后端。

    ``v5_rh`` 的 ``PingHu.pinghu_CS2`` 会将精牌从手牌中移出、以 ``kingNum``
    修正向听数，因此可先安全替代国标 ``ShantenCalculator`` 的结构向听部分。
    它返回的值最低为 0，不区分“已和”和“听牌”，故不能直接实现 ``can_hu``。
    """

    def __init__(self, king_tile_id: Optional[int] = None, v5_module_root: Optional[Path] = None):
        self.king_tile_id = king_tile_id
        self.v5_module_root = v5_module_root
        if king_tile_id is not None:
            tile_id_to_raw(king_tile_id)

    @staticmethod
    def _default_v5_module_root() -> Path:
        # 文件位于 .../Mortal_gbmj_v4/mortal_part/state/，工作区根目录是 parents[5]。
        return Path(__file__).resolve().parents[5] / "v5_rh" / "v5_rh"

    def _load_v5_backend(self):
        module_root = self.v5_module_root or self._default_v5_module_root()
        if not module_root.is_dir():
            raise ShangraoRuleError(f"未找到 v5_rh 规则目录：{module_root}")

        workspace_root = module_root.parents[1]
        # GameConfig.py 内部仍使用绝对导入 `import lib_MJ`，需要同时暴露两个目录。
        for path in (str(module_root), str(workspace_root)):
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            from v5_rh.v5_rh.GameConfig import GameConfig
            from v5_rh.v5_rh.Node_PH import PingHu
        except ImportError as exc:
            raise ShangraoRuleError("无法导入 v5_rh 的 PingHu 向听后端") from exc
        return GameConfig, PingHu

    @staticmethod
    def _expand_hand(hand_counts: Sequence[int]) -> list[int]:
        if len(hand_counts) != 34:
            raise ShangraoRuleError(f"hand_counts 长度应为 34，实际为 {len(hand_counts)}")
        cards: list[int] = []
        for tile_id, count in enumerate(hand_counts):
            if not isinstance(count, int) or count < 0 or count > 4:
                raise ShangraoRuleError(f"tile id {tile_id} 的张数非法：{count!r}")
            cards.extend([tile_id_to_raw(tile_id)] * count)
        return cards

    @staticmethod
    def _convert_melds(melds: Sequence[Sequence[int]]) -> list[list[int]]:
        raw_melds: list[list[int]] = []
        for meld_index, meld in enumerate(melds):
            if len(meld) not in (3, 4):
                raise ShangraoRuleError(f"副露 {meld_index} 的长度应为 3 或 4，实际为 {len(meld)}")
            raw_melds.append([tile_id_to_raw(tile_id) for tile_id in meld])
        return raw_melds

    def shanten(self, hand_counts: Sequence[int], melds: Sequence[Sequence[int]] = ()) -> int:
        """计算精牌参与下的平胡结构向听数。

        这是独立于可见牌数的结构向听，故将 v5_rh 的 ``LEFT_NUM`` 临时设为
        全部可用，避免“场上已见完”这一牌效剪枝改变向听定义。
        """
        GameConfig, PingHu = self._load_v5_backend()
        GameConfig().LEFT_NUM = [4] * 34
        cards = self._expand_hand(hand_counts)
        raw_melds = self._convert_melds(melds)
        king_card = tile_id_to_raw(self.king_tile_id) if self.king_tile_id is not None else None

        combinations = PingHu(cards=cards, suits=raw_melds, kingCard=king_card, fei_king=0).pinghu_CS2()
        if not combinations:
            raise ShangraoRuleError("v5_rh 未返回任何平胡组合")
        return max(0, int(combinations[0][-2]))
