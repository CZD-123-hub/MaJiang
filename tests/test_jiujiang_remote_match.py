import unittest
from unittest.mock import patch

from jiujiang_ai.remote_match import build_remote_report, start_sequential_remote_robot_matches


class JiujiangRemoteMatchTests(unittest.TestCase):
    def test_build_report_summarizes_each_robot_room_result(self):
        rounds = [
            {"room_id": 101, "winner": 0, "win_type": "zimo", "scores": [6, -2, -2, -2]},
            {"room_id": 102, "winner": 2, "dianpao_player": 1, "win_type": "dianpao", "scores": [0, -2, 2, 0]},
        ]

        report = build_remote_report(
            place_id=1103800002,
            requested_rooms=2,
            created_rooms=[101, 102],
            failed_rooms=0,
            rounds=rounds,
            timed_out=False,
        )

        self.assertEqual(report["completed_rounds"], 2)
        self.assertFalse(report["timed_out"])
        self.assertEqual(report["overall"]["wins_by_player"], {"0": 1, "2": 1})
        self.assertEqual(report["overall"]["total_score_by_player"], {"0": 6.0, "1": -4.0, "2": 0.0, "3": -2.0})

    @patch("jiujiang_ai.remote_match.wait_for_room_results")
    @patch("jiujiang_ai.remote_match._open_robot_rooms")
    def test_sequential_runner_opens_one_room_after_each_result(self, open_rooms, wait_for_results):
        open_rooms.side_effect = [
            {"errno": 0, "data": {"room_list": [101], "fail_count": 0}},
            {"errno": 0, "data": {"room_list": [102], "fail_count": 0}},
        ]
        wait_for_results.side_effect = [
            ([{"room_id": 101, "winner": 0, "scores": [1, -1, 0, 0]}], False),
            ([{"room_id": 102, "winner": 1, "scores": [-1, 1, 0, 0]}], False),
        ]

        result = start_sequential_remote_robot_matches(place_id=1, rounds=2, log_path="unused.jsonl")

        self.assertEqual(open_rooms.call_count, 2)
        self.assertTrue(all(call.args[2] == 1 for call in open_rooms.call_args_list))
        self.assertEqual(wait_for_results.call_count, 2)
        self.assertEqual(result["completed_rounds"], 2)


if __name__ == "__main__":
    unittest.main()
