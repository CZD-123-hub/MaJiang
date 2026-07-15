import unittest

from jiujiang_ai.strategy_replay import compare_strategy_snapshots


class JiujiangStrategyReplayTests(unittest.TestCase):
    def test_compares_requested_strategies_for_each_snapshot(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        snapshots = [
            {
                "snapshot_id": "r1-t8",
                "action_cards": {"7": [[0x01], [0x08]]},
                "player_hand_cards": [hand, [], [], []],
                "acting_do_player_position": 0,
                "action_seq": [],
            }
        ]

        report = compare_strategy_snapshots(snapshots, strategies=("heuristic", "multi_route"))

        self.assertEqual(report["summary"]["snapshot_count"], 1)
        self.assertEqual(report["comparisons"][0]["snapshot_id"], "r1-t8")
        self.assertEqual(set(report["comparisons"][0]["actions"]), {"heuristic", "multi_route"})
        self.assertEqual(report["comparisons"][0]["actions"]["heuristic"][0], 7)
        self.assertIn(report["comparisons"][0]["actions"]["multi_route"][1][0], {0x01, 0x08})

    def test_summary_counts_agreement(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        snapshots = [
            {
                "action_cards": {"7": [[0x08]]},
                "player_hand_cards": [hand, [], [], []],
                "acting_do_player_position": 0,
            }
        ]

        report = compare_strategy_snapshots(snapshots, strategies=("heuristic", "multi_route"))

        self.assertEqual(report["summary"]["agreement_count"], 1)
        self.assertEqual(report["summary"]["agreement_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
