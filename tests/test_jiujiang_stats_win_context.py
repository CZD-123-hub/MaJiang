import unittest

from jiujiang_ai.api import round_end
from jiujiang_ai.stats import get_stats, reset_stats


class JiujiangStatsWinContextTests(unittest.TestCase):
    def setUp(self):
        reset_stats()

    def test_round_end_accumulates_win_type_and_multi_win_stats(self):
        round_end({"winner": 0, "win_type": "zimo", "scores": [3, -1, -1, -1]})
        round_end({"winners": [1, 2], "dianpao_player": 3, "scores": [-2, 1, 1, 0]})

        stats = get_stats()

        self.assertEqual(stats["win_type_count"], {"zimo": 1, "dianpao": 1})
        self.assertEqual(stats["multi_win_rounds"], 1)
        self.assertEqual(stats["multi_win_winner_count"], 2)
        self.assertEqual(stats["multi_win_by_dianpao_player"], {"3": 1})

    def test_round_end_response_includes_win_context(self):
        result = round_end({"winner": 2, "dianpao_player": 1})

        self.assertEqual(result["win_context"]["win_type"], "dianpao")
        self.assertEqual(result["win_context"]["winners"], [2])
        self.assertEqual(result["win_context"]["dianpao_player"], 1)
        self.assertFalse(result["win_context"]["is_multi_win"])

    def test_multi_win_team_summary_counts_shared_discard_round(self):
        round_end({"winners": [0, 2], "dianpao_player": 1, "scores": [2, -4, 2, 0]})

        stats = get_stats()

        self.assertEqual(stats["discard_win_count"], 1)
        self.assertEqual(stats["win_count"], 2)


if __name__ == "__main__":
    unittest.main()
