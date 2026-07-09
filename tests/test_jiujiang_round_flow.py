import unittest

from jiujiang_ai.round_flow import detect_round_flow, resolve_next_dealer
from jiujiang_ai.rules import ACTION_DISCARD, ACTION_GANG
from jiujiang_ai.settlement import calculate_gang_score


class JiujiangRoundFlowTests(unittest.TestCase):
    def test_single_winner_becomes_next_dealer(self):
        result = resolve_next_dealer({"winner": 2, "dealer": 0})

        self.assertFalse(result["is_huangzhuang"])
        self.assertEqual(result["next_dealer"], 2)
        self.assertEqual(result["reason"], "winner_is_dealer")

    def test_multi_win_uses_dianpao_player_as_next_dealer(self):
        result = resolve_next_dealer(
            {
                "winners": [1, 3],
                "dianpao_player": 2,
                "dealer": 0,
            }
        )

        self.assertFalse(result["is_huangzhuang"])
        self.assertEqual(result["next_dealer"], 2)
        self.assertEqual(result["reason"], "dianpao_player_is_dealer")

    def test_empty_wall_without_winner_is_huangzhuang(self):
        flow = detect_round_flow(
            {
                "dealer": 1,
                "remain_card_stack": [],
            }
        )

        self.assertTrue(flow["is_huangzhuang"])
        self.assertTrue(flow["is_draw_round"])
        self.assertFalse(flow["has_winner"])

    def test_huangzhuang_moves_dealer_to_next_player(self):
        result = resolve_next_dealer(
            {
                "dealer": 3,
                "remain_card_stack": [],
                "player_hand_cards": [[], [], [], []],
            }
        )

        self.assertTrue(result["is_huangzhuang"])
        self.assertEqual(result["next_dealer"], 0)
        self.assertEqual(result["reason"], "draw_next_player")

    def test_huangzhuang_round_does_not_count_gang_score(self):
        result = calculate_gang_score(
            {
                "dealer": 0,
                "remain_card_stack": [],
                "player_hand_cards": [[], [], [], []],
                "action_seq": [
                    [2, ACTION_DISCARD, 0x11],
                    [1, ACTION_GANG, 0x11, 0x11, 0x11, 0x11],
                ],
            }
        )

        self.assertEqual(result["score_by_player"], [0, 0, 0, 0])
        self.assertEqual(result["events"], [])


if __name__ == "__main__":
    unittest.main()
