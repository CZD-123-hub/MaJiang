import unittest

from jiujiang_ai.tiles import HONGZHONG
from jiujiang_ai.zama import calculate_zama_score


class JiujiangZamaTests(unittest.TestCase):
    def test_four_hongzhong_hand_gets_extra_four_zama_points(self):
        result = calculate_zama_score(
            {
                "winner": 0,
                "room_options": {"zama_count": 2},
                "zama_cards": [0x01, 0x02],
                "player_hand_cards": [[HONGZHONG, HONGZHONG, HONGZHONG, HONGZHONG], [], [], []],
            }
        )

        self.assertEqual(result["awarded_cards"], [0x01, 0x02])
        self.assertEqual(result["extra_for_four_hongzhong"], 4)
        self.assertEqual(result["zama_score"], 7)

    def test_zama_can_follow_base_score_multiplier(self):
        result = calculate_zama_score(
            {
                "winner": 1,
                "room_options": {
                    "zama_count": 2,
                    "zama_follow_base_score": True,
                    "base_score": 2,
                },
                "zama_cards": [0x01, HONGZHONG],
                "player_hand_cards": [[], [HONGZHONG], [], []],
            }
        )

        self.assertEqual(result["raw_zama_score"], 11)
        self.assertEqual(result["base_score_multiplier"], 2)
        self.assertEqual(result["zama_score"], 22)

    def test_yimaquanzhong_scores_each_tile_by_rank(self):
        result = calculate_zama_score(
            {
                "winner": 0,
                "room_options": {"zama_count": 3},
                "zama_cards": [0x01, 0x19, 0x29],
                "player_hand_cards": [[HONGZHONG], [], [], []],
            }
        )

        self.assertEqual(result["awarded_cards"], [0x01, 0x19, 0x29])
        self.assertEqual(result["zama_score"], 19)

    def test_hongzhong_counts_as_ten_points(self):
        result = calculate_zama_score(
            {
                "winner": 1,
                "room_options": {"zama_count": 2},
                "zama_cards": [HONGZHONG, 0x02],
                "player_hand_cards": [[], [HONGZHONG], [], []],
            }
        )

        self.assertEqual(result["awarded_cards"], [HONGZHONG, 0x02])
        self.assertEqual(result["zama_score"], 12)

    def test_winner_without_hongzhong_gets_one_extra_zama_card(self):
        result = calculate_zama_score(
            {
                "winner": 2,
                "room_options": {"zama_count": 2},
                "zama_cards": [0x03, 0x04, 0x05],
                "player_hand_cards": [[], [], [0x11, 0x12, 0x13], []],
            }
        )

        self.assertEqual(result["awarded_cards"], [0x03, 0x04, 0x05])
        self.assertEqual(result["zama_score"], 12)

    def test_winner_with_hongzhong_does_not_get_extra_zama_card(self):
        result = calculate_zama_score(
            {
                "winner": 3,
                "room_options": {"zama_count": 2},
                "zama_cards": [0x03, 0x04, 0x05],
                "player_hand_cards": [[], [], [], [HONGZHONG, 0x21, 0x22]],
            }
        )

        self.assertEqual(result["awarded_cards"], [0x03, 0x04])
        self.assertEqual(result["zama_score"], 7)


if __name__ == "__main__":
    unittest.main()
