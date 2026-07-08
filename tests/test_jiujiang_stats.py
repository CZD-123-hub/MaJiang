import unittest

from jiujiang_ai.api import round_end
from jiujiang_ai.stats import get_stats, reset_stats, summarize_match_report, summarize_rounds


class JiujiangStatsTests(unittest.TestCase):
    def setUp(self):
        # 每个测试都从空统计开始，避免不同测试之间互相影响。
        reset_stats()

    def test_round_end_accumulates_wins_scores_and_win_types(self):
        round_end({"winner": 0, "scores": [3, -1, -1, -1], "win_type": "zimo"})
        round_end(
            {
                "winner": 2,
                "score_delta": {"0": 0, "1": -2, "2": 2, "3": 0},
                "dianpao_player": 1,
            }
        )

        stats = get_stats()

        self.assertEqual(stats["total_rounds"], 2)
        self.assertEqual(stats["win_count"], 2)
        self.assertEqual(stats["self_draw_count"], 1)
        self.assertEqual(stats["discard_win_count"], 1)
        self.assertEqual(stats["wins_by_player"], {"0": 1, "2": 1})
        self.assertEqual(stats["dianpao_by_player"], {"1": 1})
        self.assertEqual(stats["total_score_by_player"], {"0": 3.0, "1": -3.0, "2": 1.0, "3": -1.0})

    def test_round_end_response_includes_current_stats_snapshot(self):
        result = round_end({"winner": 3, "scores": [0, 0, -1, 1]})

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["received"])
        self.assertEqual(result["stats"]["total_rounds"], 1)
        self.assertEqual(result["stats"]["wins_by_player"], {"3": 1})

    def test_summarize_rounds_returns_pure_batch_stats(self):
        rounds = [
            {"winner": 0, "scores": [3, -1, -1, -1], "win_type": "zimo"},
            {"winner": 2, "score_delta": {"0": 0, "1": -2, "2": 2, "3": 0}, "dianpao_player": 1},
        ]

        stats = summarize_rounds(rounds)

        self.assertEqual(stats["total_rounds"], 2)
        self.assertEqual(stats["win_count"], 2)
        self.assertEqual(stats["self_draw_count"], 1)
        self.assertEqual(stats["discard_win_count"], 1)
        self.assertEqual(stats["wins_by_player"], {"0": 1, "2": 1})
        self.assertEqual(stats["dianpao_by_player"], {"1": 1})
        self.assertEqual(stats["total_score_by_player"], {"0": 3.0, "1": -3.0, "2": 1.0, "3": -1.0})

    def test_summarize_match_report_supports_our_players_view(self):
        rounds = [
            {"winner": 0, "scores": [3, -1, -1, -1], "win_type": "zimo"},
            {"winner": 2, "score_delta": {"0": 0, "1": -2, "2": 2, "3": 0}, "dianpao_player": 1},
            {"winner": 3, "scores": [-2, 0, 0, 2], "dianpao_player": 0},
        ]

        report = summarize_match_report(rounds, our_players=[0, 2])

        self.assertEqual(report["overall"]["total_rounds"], 3)
        self.assertEqual(report["team"]["players"], ["0", "2"])
        self.assertEqual(report["team"]["win_rounds"], 2)
        self.assertEqual(report["team"]["self_draw_rounds"], 1)
        self.assertEqual(report["team"]["discard_win_rounds"], 1)
        self.assertEqual(report["team"]["dianpao_rounds"], 1)
        self.assertEqual(report["team"]["total_score"], 2.0)
        self.assertAlmostEqual(report["team"]["average_round_score"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
