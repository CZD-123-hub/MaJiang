import unittest

from jiujiang_ai.decision_engine import choose_discard


class JiujiangDecisionEngineTests(unittest.TestCase):
    def test_exposes_probability_risk_and_flexibility_breakdown(self):
        hand = [
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x11,
            0x12,
            0x13,
            0x21,
            0x22,
            0x23,
            0x09,
            0x18,
        ]

        decision = choose_discard(hand, [[0x09], [0x18]], data={"action_seq": []}, acting_position=0)

        self.assertIn(decision.discard, {0x09, 0x18})
        self.assertGreaterEqual(decision.progress_probability, 0.0)
        self.assertGreaterEqual(decision.risk_score, 0.0)
        self.assertGreater(decision.flexibility, 0.0)
        self.assertGreaterEqual(decision.route_count, decision.retained_route_count)

    def test_visible_safe_tile_reduces_composite_risk(self):
        hand = [
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x11,
            0x12,
            0x13,
            0x21,
            0x22,
            0x23,
            0x09,
            0x18,
        ]
        data = {"action_seq": [[1, 7, 0x18], [2, 7, 0x18], [3, 7, 0x18]]}

        decisions = choose_discard(
            hand,
            [[0x09], [0x18]],
            data=data,
            acting_position=0,
            return_all=True,
        )

        self.assertLess(decisions[0x18].risk_score, decisions[0x09].risk_score)

    def test_composite_uses_settlement_based_expected_win_value(self):
        hand = [
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x11,
            0x12,
            0x13,
            0x21,
            0x22,
            0x23,
            0x09,
            0x18,
        ]
        data = {
            "player_hand_cards": [hand, [], [], []],
            "played_cards": [[0x35], [], [], []],
            "room_options": {"run_hongzhong_double": True, "enable_buy_score": True},
            "player_buy_scores": [1, 0, 0, 0],
            "strategy_options": {"expected_win_type": "zimo"},
        }

        decision = choose_discard(hand, [[0x18]], data=data, acting_position=0)

        self.assertEqual(decision.expected_win_value, 15.0)


if __name__ == "__main__":
    unittest.main()
