import tempfile
import unittest
from pathlib import Path
import json
from unittest.mock import patch

from jiujiang_ai.api import round_end
from jiujiang_ai.stats import (
    append_round_log,
    get_stats,
    load_round_logs,
    record_round_end,
    reset_stats,
    summarize_match_report,
    summarize_rounds,
)


class JiujiangStatsTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._log_path_patch = patch(
            "jiujiang_ai.stats.DEFAULT_ROUND_LOG_PATH",
            Path(self._temp_dir.name) / "round_end.jsonl",
        )
        self._log_path_patch.start()
        self.addCleanup(self._log_path_patch.stop)
        self.addCleanup(self._temp_dir.cleanup)
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

    def test_summarize_match_report_counts_multi_win_rounds_for_team(self):
        rounds = [
            {"winners": [0, 2], "dianpao_player": 1, "scores": [2, -4, 2, 0]},
            {"winner": 3, "scores": [-2, 0, 0, 2], "dianpao_player": 0},
        ]

        report = summarize_match_report(rounds, our_players=[0, 2])

        self.assertEqual(report["overall"]["multi_win_rounds"], 1)
        self.assertEqual(report["overall"]["multi_win_winner_count"], 2)
        self.assertEqual(report["overall"]["multi_win_by_dianpao_player"], {"1": 1})
        self.assertEqual(report["team"]["multi_win_rounds"], 1)
        self.assertEqual(report["team"]["multi_win_wins"], 2)

    def test_record_round_end_appends_jsonl_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "round_end.jsonl"

            stats = record_round_end({"winner": 0, "scores": [3, -1, -1, -1]}, log_path=log_path)

            self.assertEqual(stats["total_rounds"], 1)
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertIn("timestamp", payload)
            self.assertTrue(payload["timestamp"].endswith("+08:00"))
            self.assertEqual(payload["data"]["winner"], 0)
            self.assertEqual(payload["stats"]["total_rounds"], 1)

    def test_load_round_logs_reads_jsonl_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "round_end.jsonl"
            append_round_log({"winner": 0, "scores": [1, -1, 0, 0]}, {"total_rounds": 1}, log_path=log_path)
            append_round_log({"winner": 2, "scores": [0, -1, 1, 0]}, {"total_rounds": 2}, log_path=log_path)

            rounds = load_round_logs(log_path)

            self.assertEqual(len(rounds), 2)
            self.assertEqual(rounds[0]["winner"], 0)
            self.assertEqual(rounds[1]["winner"], 2)

    def test_load_round_logs_accepts_legacy_plain_round_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "round_end.jsonl"
            log_path.write_text('{"winner": 0, "scores": [1, -1, 0, 0]}\n', encoding="utf-8")

            rounds = load_round_logs(log_path)

            self.assertEqual(rounds, [{"winner": 0, "scores": [1, -1, 0, 0]}])


if __name__ == "__main__":
    unittest.main()
