import unittest

from jiujiang_ai.expected_value import estimate_win_value
from jiujiang_ai.tiles import HONGZHONG


class JiujiangExpectedValueTests(unittest.TestCase):
    def test_estimates_self_draw_hu_run_hongzhong_and_buy_income(self):
        data = {
            "player_hand_cards": [[], [], [], []],
            "played_cards": [[HONGZHONG], [], [], []],
            "room_options": {
                "run_hongzhong_double": True,
                "enable_buy_score": True,
            },
            "player_buy_scores": [1, 0, 0, 0],
            "strategy_options": {"expected_win_type": "zimo"},
        }

        value = estimate_win_value(data, winner=0)

        # 胡分：1 × 自摸 2 × 跑一红中 2 = 4，三家各付 4；
        # 加买再为赢家带来 3 分。
        self.assertEqual(value.expected_hu_gain, 12.0)
        self.assertEqual(value.expected_buy_gain, 3.0)
        self.assertEqual(value.total, 15.0)
        self.assertEqual(value.mode_weights, {"zimo": 1.0})

    def test_default_mix_is_explicit_about_zimo_and_dianpao_assumption(self):
        value = estimate_win_value({"player_hand_cards": [[], [], [], []]}, winner=0)

        # 默认等权：自摸净胡分为 6，点炮净胡分为 1，期望为 3.5。
        self.assertEqual(value.expected_hu_gain, 3.5)
        self.assertEqual(value.expected_buy_gain, 0.0)
        self.assertEqual(value.total, 3.5)
        self.assertEqual(value.mode_weights, {"dianpao": 0.5, "zimo": 0.5})

    def test_pending_hongzhong_discard_is_included_in_run_multiplier(self):
        data = {
            "player_hand_cards": [[], [], [], []],
            "played_cards": [[], [], [], []],
            "room_options": {"run_hongzhong_double": True},
            "strategy_options": {"expected_win_type": "dianpao"},
        }

        normal = estimate_win_value(data, winner=0)
        after_red_discard = estimate_win_value(data, winner=0, pending_discard=HONGZHONG)

        self.assertEqual(after_red_discard.total, normal.total * 2)


if __name__ == "__main__":
    unittest.main()
