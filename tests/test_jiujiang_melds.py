import unittest

from jiujiang_ai.hu import HuOptions, can_hu
from jiujiang_ai.ting import is_ting, winning_tile_counts


class JiujiangMeldTests(unittest.TestCase):
    def test_standard_hu_without_fixed_melds_still_works(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x09]

        self.assertTrue(can_hu(hand, fixed_melds=0))

    def test_can_hu_with_one_fixed_meld(self):
        # 已有 1 组副露后，手牌只需要再组成 3 面子 + 1 将。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x09, 0x09]

        self.assertTrue(can_hu(hand, fixed_melds=1))

    def test_can_hu_with_two_fixed_melds(self):
        # 已有 2 组副露后，手牌只需要再组成 2 面子 + 1 将。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x09, 0x09]

        self.assertTrue(can_hu(hand, fixed_melds=2))

    def test_fixed_melds_do_not_create_false_positive(self):
        hand = [0x01, 0x01, 0x02, 0x04, 0x05, 0x07, 0x09, 0x09]

        self.assertFalse(can_hu(hand, fixed_melds=2))

    def test_qidui_requires_no_fixed_melds(self):
        hand = [0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x05, 0x05, 0x06, 0x06, 0x07, 0x07]

        self.assertFalse(can_hu(hand, HuOptions(allow_qidui=True), fixed_melds=1))

    def test_is_ting_with_fixed_melds(self):
        # 已有 1 组副露，当前手牌听 9 万。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x09]

        self.assertTrue(is_ting(hand, fixed_melds=1))

    def test_winning_tile_counts_with_fixed_melds(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x09]

        result = winning_tile_counts(hand, fixed_melds=1)

        self.assertEqual(result[0x09], 3)


if __name__ == "__main__":
    unittest.main()
