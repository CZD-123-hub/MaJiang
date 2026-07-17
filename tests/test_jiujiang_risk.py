import unittest

from jiujiang_ai.risk import evaluate_discard_risks
from jiujiang_ai.rules import ACTION_DISCARD, ACTION_TING


class JiujiangRiskTests(unittest.TestCase):
    def test_tile_discarded_by_an_opponent_is_relatively_safer(self):
        data = {
            "action_seq": [[1, ACTION_DISCARD, 0x18]],
            "played_cards": [[], [], [], []],
        }

        risks = evaluate_discard_risks(data, acting_position=0, candidates=(0x09, 0x18))

        self.assertLess(risks[0x18].score, risks[0x09].score)
        self.assertIn("opponent_discard", risks[0x18].reasons)

    def test_declared_ting_and_open_melds_raise_risk(self):
        quiet = {"action_seq": [], "player_peng_cards": [[], [], [], []]}
        threatening = {
            "action_seq": [[1, ACTION_TING]],
            "player_peng_cards": [[], [[0x05, 0x05, 0x05], [0x11, 0x11, 0x11]], [], []],
            "remain_card_stack": [0x01] * 10,
        }

        quiet_risk = evaluate_discard_risks(quiet, acting_position=0, candidates=(0x09,))[0x09]
        threat_risk = evaluate_discard_risks(threatening, acting_position=0, candidates=(0x09,))[0x09]

        self.assertGreater(threat_risk.score, quiet_risk.score)
        self.assertIn("opponent_ting", threat_risk.reasons)

    def test_teammate_discard_is_not_treated_as_opponent_safety(self):
        data = {
            "action_seq": [[3, ACTION_DISCARD, 0x18]],
            "played_cards": [[], [], [], []],
        }

        risks = evaluate_discard_risks(
            data,
            acting_position=1,
            candidates=(0x18,),
            opponent_positions=(0, 2),
        )

        self.assertNotIn("opponent_discard", risks[0x18].reasons)


if __name__ == "__main__":
    unittest.main()
