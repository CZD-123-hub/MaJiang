import unittest

from jiujiang_ai.tiles import HONGZHONG
from jiujiang_ai.ting import is_ting, ting_discards, winning_tile_counts


class JiujiangTingTests(unittest.TestCase):
    def test_winning_tile_counts_uses_real_can_hu(self):
        # 这副 13 张已经有四组面子，只缺一张将牌。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09]

        result = winning_tile_counts(hand)

        self.assertEqual(result[0x09], 3)
        self.assertEqual(result[HONGZHONG], 4)

    def test_is_ting_returns_true_when_any_draw_can_hu(self):
        # 摸 9 万或红中都可以补成将牌，所以当前是听牌。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09]

        self.assertTrue(is_ting(hand))

    def test_winning_tile_counts_returns_empty_for_non_ting_hand(self):
        # 结构过散时，任意摸一张都不能立刻胡。
        hand = [0x01, 0x01, 0x02, 0x04, 0x06, 0x08, 0x11, 0x13, 0x15, 0x17, 0x21, 0x24, 0x27]

        self.assertEqual(winning_tile_counts(hand), {})

    def test_ting_discards_reports_winning_tiles_after_discard(self):
        # 14 张手牌中打掉 8 条后，剩余 13 张听 9 万和红中。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x18, 0x21, 0x22, 0x23, 0x09]

        result = ting_discards(hand, [[0x18], [0x01]])

        self.assertIn(0x18, result)
        self.assertEqual(result[0x18].winning_tiles[0x09], 3)
        self.assertEqual(result[0x18].winning_tiles[HONGZHONG], 4)
        self.assertNotIn(0x01, result)


if __name__ == "__main__":
    unittest.main()
