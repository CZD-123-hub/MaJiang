import json
import tempfile
import unittest
from pathlib import Path

from jiujiang_ai.decision_log import append_decision_log, load_decision_logs
from jiujiang_ai.tiles import HONGZHONG


class _Decision:
    discard = 0x18
    score = 0.72
    progress_probability = 0.1
    expected_win_value = 3.5
    risk_score = 0.2
    flexibility = 4.0
    route_count = 6
    retained_route_count = 3
    effective_tiles = {0x09: 3, HONGZHONG: 4}


class JiujiangDecisionLogTests(unittest.TestCase):
    def test_append_log_records_public_context_and_score_breakdown(self):
        data = {
            "room_id": 42,
            "acting_do_player_position": 0,
            "player_hand_cards": [[0x01, 0x02, 0x03, 0x18], [], [], []],
            "action_cards": {"7": [[0x01], [0x18]]},
            "action_seq": [[1, 7, 0x09]],
            "remain_card_stack": [0x01, 0x02],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "decisions.jsonl"

            append_decision_log(data, action_type=7, action_card=[0x18], strategy="multi_route", decision=_Decision(), log_path=path)

            records = load_decision_logs(path)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["strategy"], "multi_route")
        self.assertEqual(record["context"]["hand"], [0x01, 0x02, 0x03, 0x18])
        self.assertEqual(record["context"]["turn"], 1)
        self.assertEqual(record["decision"]["risk_score"], 0.2)
        self.assertEqual(record["decision"]["effective_tiles"][str(HONGZHONG)], 4)

    def test_load_accepts_jsonl_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "decisions.jsonl"
            path.write_text(json.dumps({"strategy": "heuristic"}) + "\n", encoding="utf-8")

            records = load_decision_logs(path)

        self.assertEqual(records, [{"strategy": "heuristic"}])


if __name__ == "__main__":
    unittest.main()
