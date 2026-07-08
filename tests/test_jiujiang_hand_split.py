import unittest

from jiujiang_ai.hand_split import analyze_hand, is_four_hongzhong
from jiujiang_ai.tiles import HONGZHONG


class JiujiangHandSplitTests(unittest.TestCase):
    def test_complete_pinghu_without_hongzhong(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x09]

        result = analyze_hand(hand)

        self.assertEqual(result.shanten, 0)
        self.assertEqual(result.hongzhong_used, 0)

    def test_one_hongzhong_completes_pair(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, HONGZHONG]

        result = analyze_hand(hand)

        self.assertEqual(result.shanten, 0)
        self.assertGreaterEqual(result.hongzhong_used, 1)

    def test_hongzhong_improves_incomplete_hand(self):
        no_red = analyze_hand([0x01, 0x02, 0x03, 0x04, 0x05, 0x11, 0x12, 0x21, 0x22, 0x09, 0x09, 0x18, 0x19])
        with_red = analyze_hand([0x01, 0x02, 0x03, 0x04, 0x05, 0x11, 0x12, 0x21, 0x22, 0x09, 0x09, 0x18, HONGZHONG])

        self.assertLessEqual(with_red.shanten, no_red.shanten)

    def test_four_hongzhong_detection(self):
        self.assertTrue(is_four_hongzhong([HONGZHONG, HONGZHONG, HONGZHONG, HONGZHONG]))
        self.assertFalse(is_four_hongzhong([HONGZHONG, HONGZHONG, HONGZHONG, 0x01]))


if __name__ == "__main__":
    unittest.main()
