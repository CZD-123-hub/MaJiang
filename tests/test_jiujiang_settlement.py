import unittest

from jiujiang_ai.rules import ACTION_ANGANG, ACTION_BUGANG, ACTION_DISCARD, ACTION_GANG
from jiujiang_ai.settlement import (
    calculate_buy_score,
    calculate_gang_score,
    calculate_hu_score,
    calculate_run_hongzhong_multiplier,
    calculate_total_score,
)
from jiujiang_ai.tiles import HONGZHONG


class JiujiangSettlementTests(unittest.TestCase):
    def test_total_score_combines_zimo_hu_buy_and_zama(self):
        result = calculate_total_score(
            {
                "winner": 0,
                "win_type": "zimo",
                "room_options": {
                    "enable_buy_score": True,
                    "zama_count": 1,
                },
                "player_buy_scores": [1, 0, 0, 0],
                "zama_cards": [0x01],
                "player_hand_cards": [[HONGZHONG, 0x11, 0x12], [], [], []],
            }
        )

        self.assertEqual(result["score_by_player"], [12, -4, -4, -4])
        self.assertEqual(result["components"]["hu"]["score_by_player"], [6, -2, -2, -2])
        self.assertEqual(result["components"]["buy"]["score_by_player"], [3, -1, -1, -1])
        self.assertEqual(result["components"]["zama"]["score_by_player"], [3, -1, -1, -1])

    def test_total_score_combines_dianpao_hu_buy_gang_and_zama(self):
        result = calculate_total_score(
            {
                "winner": 1,
                "dianpao_player": 3,
                "room_options": {
                    "enable_buy_score": True,
                    "zama_count": 1,
                },
                "player_buy_scores": [0, 2, 1, 0],
                "zama_cards": [HONGZHONG],
                "player_hand_cards": [[], [HONGZHONG, 0x21, 0x22], [], []],
                "action_seq": [
                    [0, ACTION_DISCARD, 0x11],
                    [2, ACTION_GANG, 0x11, 0x11, 0x11, 0x11],
                ],
            }
        )

        self.assertEqual(result["score_by_player"], [-5, 18, 0, -13])
        self.assertEqual(result["components"]["hu"]["score_by_player"], [0, 1, 0, -1])
        self.assertEqual(result["components"]["buy"]["score_by_player"], [-2, 7, -3, -2])
        self.assertEqual(result["components"]["gang"]["score_by_player"], [-3, 0, 3, 0])
        self.assertEqual(result["components"]["zama"]["score_by_player"], [0, 10, 0, -10])

    def test_buy_score_is_zero_when_option_disabled(self):
        result = calculate_buy_score(
            {
                "winner": 1,
                "room_options": {"enable_buy_score": False},
                "player_buy_scores": [1, 2, 0, 3],
            }
        )

        self.assertEqual(result["score_by_player"], [0, 0, 0, 0])
        self.assertEqual(result["total_buy_score"], 0)

    def test_buy_score_uses_player_specific_scores(self):
        result = calculate_buy_score(
            {
                "winner": 1,
                "room_options": {"enable_buy_score": True},
                "player_buy_scores": [1, 2, 0, 3],
            }
        )

        self.assertEqual(result["winner"], 1)
        self.assertEqual(result["score_by_player"], [-3, 10, -2, -5])
        self.assertEqual(result["total_buy_score"], 10)

    def test_buy_score_falls_back_to_uniform_room_buy_score(self):
        result = calculate_buy_score(
            {
                "winner": 0,
                "room_options": {"enable_buy_score": True, "buy_score": 1},
            }
        )

        self.assertEqual(result["score_by_player"], [6, -2, -2, -2])
        self.assertEqual(result["buy_scores"], [1, 1, 1, 1])

    def test_zhigang_charges_single_payer_three_points(self):
        result = calculate_gang_score(
            {
                "action_seq": [
                    [2, ACTION_DISCARD, 0x11],
                    [1, ACTION_GANG, 0x11, 0x11, 0x11, 0x11],
                ]
            }
        )

        self.assertEqual(result["score_by_player"], [0, 3, -3, 0])
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["gang_type"], "zhigang")
        self.assertEqual(result["events"][0]["player"], 1)
        self.assertEqual(result["events"][0]["payer"], 2)

    def test_bugang_charges_other_three_players_one_point_each(self):
        result = calculate_gang_score(
            {
                "action_seq": [
                    [1, ACTION_BUGANG, 0x19, 0x19, 0x19, 0x19],
                ]
            }
        )

        self.assertEqual(result["score_by_player"], [-1, 3, -1, -1])
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["gang_type"], "bugang")
        self.assertEqual(result["events"][0]["player"], 1)
        self.assertEqual(result["events"][0]["payers"], [0, 2, 3])

    def test_angang_charges_other_three_players_two_points_each(self):
        result = calculate_gang_score(
            {
                "action_seq": [
                    [3, ACTION_ANGANG, 0x21, 0x21, 0x21, 0x21],
                ]
            }
        )

        self.assertEqual(result["score_by_player"], [-2, -2, -2, 6])
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["gang_type"], "angang")
        self.assertEqual(result["events"][0]["player"], 3)
        self.assertEqual(result["events"][0]["payers"], [0, 1, 2])

    def test_run_hongzhong_multiplier_is_one_when_option_disabled(self):
        result = calculate_hu_score(
            {
                "winner": 0,
                "win_type": "zimo",
                "room_options": {"run_hongzhong_double": False},
                "played_cards": [[HONGZHONG, HONGZHONG], [], [], []],
            }
        )

        self.assertEqual(result["run_hongzhong_multiplier"], 1)
        self.assertEqual(result["hu_score"], 2)

    def test_run_hongzhong_multiplier_uses_power_of_two_for_winner_discards(self):
        self.assertEqual(
            calculate_run_hongzhong_multiplier(
                {
                    "winner": 0,
                    "room_options": {"run_hongzhong_double": True},
                    "played_cards": [[0x01, HONGZHONG, HONGZHONG], [], [], []],
                }
            ),
            4,
        )

    def test_run_hongzhong_multiplier_falls_back_to_action_seq(self):
        self.assertEqual(
            calculate_run_hongzhong_multiplier(
                {
                    "winner": 2,
                    "room_options": {"run_hongzhong_double": True},
                    "action_seq": [[2, 7, HONGZHONG], [1, 7, 0x01], [2, 7, HONGZHONG]],
                }
            ),
            4,
        )

    def test_dianpao_hu_score_uses_base_multiplier_one(self):
        result = calculate_hu_score({"winner": 2, "dianpao_player": 1})

        self.assertEqual(result["winner"], 2)
        self.assertEqual(result["win_type"], "dianpao")
        self.assertEqual(result["base_hu_score"], 1)
        self.assertEqual(result["win_type_multiplier"], 1)
        self.assertEqual(result["hu_score"], 1)

    def test_zimo_hu_score_uses_multiplier_two(self):
        result = calculate_hu_score({"winner": 0, "win_type": "zimo"})

        self.assertEqual(result["winner"], 0)
        self.assertEqual(result["win_type"], "zimo")
        self.assertEqual(result["win_type_multiplier"], 2)
        self.assertEqual(result["hu_score"], 2)

    def test_gangkai_hu_score_uses_multiplier_two(self):
        result = calculate_hu_score({"winner": 1, "win_type": "gangkai"})

        self.assertEqual(result["win_type"], "gangkai")
        self.assertEqual(result["win_type_multiplier"], 2)
        self.assertEqual(result["hu_score"], 2)

    def test_qianggang_counts_as_self_draw_for_basic_hu_score(self):
        result = calculate_hu_score({"winner": 3, "win_type": "qianggang", "pao_player": 1})

        self.assertEqual(result["win_type"], "qianggang")
        self.assertEqual(result["win_type_multiplier"], 2)
        self.assertEqual(result["hu_score"], 2)

    def test_unknown_win_type_falls_back_to_base_score(self):
        result = calculate_hu_score({"winner": 0})

        self.assertEqual(result["win_type"], "unknown")
        self.assertEqual(result["win_type_multiplier"], 1)
        self.assertEqual(result["hu_score"], 1)

    def test_hu_score_includes_run_hongzhong_multiplier(self):
        result = calculate_hu_score(
            {
                "winner": 0,
                "win_type": "zimo",
                "room_options": {"run_hongzhong_double": True},
                "played_cards": [[HONGZHONG, HONGZHONG], [], [], []],
            }
        )

        self.assertEqual(result["win_type_multiplier"], 2)
        self.assertEqual(result["run_hongzhong_multiplier"], 4)
        self.assertEqual(result["hu_score"], 8)


if __name__ == "__main__":
    unittest.main()
