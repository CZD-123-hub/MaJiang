import unittest

from jiujiang_ai.context import remaining_counts_from_data
from jiujiang_ai.evaluator import choose_discard
from jiujiang_ai.tiles import HONGZHONG
from jiujiang_ai.ting import winning_tile_counts


class JiujiangRemainingTests(unittest.TestCase):
    def test_remain_card_stack_filters_out_empty_winning_tile(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09]
        data = {"remain_card_stack": [0x09, 0x09]}

        remaining_counts = remaining_counts_from_data(data, hand)
        result = winning_tile_counts(hand, remaining_counts=remaining_counts)

        self.assertEqual(result, {0x09: 2})

    def test_remain_card_stack_zero_hongzhong_removes_red_as_out(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09]
        data = {"remain_card_stack": [0x09, 0x09, 0x09]}

        remaining_counts = remaining_counts_from_data(data, hand)
        result = winning_tile_counts(hand, remaining_counts=remaining_counts)

        self.assertNotIn(HONGZHONG, result)
        self.assertEqual(result[0x09], 3)

    def test_choose_discard_uses_real_remaining_counts(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x18]
        candidates = [[0x09], [0x18]]
        data = {"remain_card_stack": [0x09, 0x09, HONGZHONG, HONGZHONG, HONGZHONG, HONGZHONG]}

        remaining_counts = remaining_counts_from_data(data, hand)
        decision = choose_discard(hand, candidates, remaining_counts=remaining_counts)

        self.assertEqual(decision.discard, 0x18)
        self.assertEqual(decision.effective_count, 6)

    def test_remaining_counts_falls_back_without_remain_card_stack(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09]
        data = {
            "played_cards": [[0x09], [], [], []],
            "player_peng_cards": [[], [[0x21, 0x21, 0x21]], [], []],
        }

        remaining_counts = remaining_counts_from_data(data, hand)

        self.assertEqual(remaining_counts[0x09], 2)
        self.assertEqual(remaining_counts[0x21], 0)


if __name__ == "__main__":
    unittest.main()
