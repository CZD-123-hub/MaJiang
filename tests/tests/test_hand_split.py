import unittest

from mahjong_ai.hand_split import best_shanten, split_hand
from mahjong_ai.tiles import parse_tiles, tile_name


class HandSplitTests(unittest.TestCase):
    def test_tile_encoding_matches_shangrao_v5_style(self):
        # 验证本项目编码方式和 v5_rh 的 16 进制编码一致。
        tiles = parse_tiles("1W 9W 1T 9T 1B 9B EAST RED")

        self.assertEqual(tiles, [0x01, 0x09, 0x11, 0x19, 0x21, 0x29, 0x31, 0x35])
        self.assertEqual(tile_name(0x21), "1B")
        self.assertEqual(tile_name(0x35), "RED")

    def test_split_hand_returns_winning_standard_hand(self):
        # 4 个面子 + 1 个对子，应该被识别为胡牌，向听数为 0。
        hand = parse_tiles("1W 2W 3W 2T 3T 4T 5B 6B 7B EAST EAST EAST RED RED")

        combinations = split_hand(hand)
        winning = [combo for combo in combinations if combo.shanten == 0]

        self.assertTrue(winning)
        self.assertTrue(any(len(combo.triplets) + len(combo.sequences) == 4 and combo.pair for combo in winning))
        self.assertEqual(best_shanten(hand), 0)

    def test_split_hand_reports_tenpai_as_one_shanten(self):
        # 已有 4 个面子，但没有对子，还差一张将牌，所以是听牌，向听数为 1。
        hand = parse_tiles("1W 2W 3W 2T 3T 4T 5B 6B 7B EAST EAST EAST RED")

        self.assertEqual(best_shanten(hand), 1)

    def test_split_hand_keeps_taatsu_and_leftovers(self):
        # 复杂散牌要能保留对子、搭子和剩余孤张信息。
        hand = parse_tiles("1W 2W 4W 5W 7W 7W 2T 3T 5B 7B EAST SOUTH RED")

        combinations = split_hand(hand)
        best = min(combinations, key=lambda combo: combo.shanten)

        self.assertLessEqual(best.shanten, 4)
        self.assertTrue(best.pairs or best.taatsu)
        self.assertEqual(sorted(best.all_tiles()), sorted(hand))


if __name__ == "__main__":
    unittest.main()
