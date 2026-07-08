import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from examples.jiujiang_match_report import generate_match_report, load_rounds


class JiujiangMatchReportTests(unittest.TestCase):
    def test_load_rounds_reads_json_array_file(self):
        rounds = [{"winner": 0, "scores": [1, -1, 0, 0]}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rounds.json"
            path.write_text(json.dumps(rounds, ensure_ascii=False), encoding="utf-8")

            loaded = load_rounds(path)

        self.assertEqual(loaded, rounds)

    def test_load_rounds_reads_jsonl_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rounds.jsonl"
            path.write_text(
                '{"winner": 0, "scores": [1, -1, 0, 0]}\n{"winner": 2, "scores": [0, -1, 1, 0]}\n',
                encoding="utf-8",
            )

            loaded = load_rounds(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[1]["winner"], 2)

    def test_generate_match_report_builds_team_summary(self):
        rounds = [
            {"winner": 0, "scores": [3, -1, -1, -1], "win_type": "zimo"},
            {"winner": 3, "scores": [-2, 0, 0, 2], "dianpao_player": 0},
        ]

        report = generate_match_report(rounds, our_players=[0])

        self.assertEqual(report["overall"]["total_rounds"], 2)
        self.assertEqual(report["team"]["players"], ["0"])
        self.assertEqual(report["team"]["win_rounds"], 1)
        self.assertEqual(report["team"]["dianpao_rounds"], 1)
        self.assertEqual(report["team"]["total_score"], 1.0)

    def test_cli_sample_outputs_json_report(self):
        script = Path(__file__).resolve().parents[1] / "examples" / "jiujiang_match_report.py"

        result = subprocess.run(
            [sys.executable, str(script), "--sample", "--our-players", "0,2"],
            capture_output=True,
            text=True,
            check=True,
        )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["overall"]["total_rounds"], 3)
        self.assertEqual(payload["team"]["players"], ["0", "2"])


if __name__ == "__main__":
    unittest.main()
