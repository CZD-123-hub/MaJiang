import json
import threading
import unittest
import urllib.request

from jiujiang_ai.api import get_action
from jiujiang_ai.rules import ACTION_DISCARD
from jiujiang_ai.server import create_server


SAMPLE_DATA = {
    "room_id": 184776,
    "game_area_id": 10021,
    "acting_player_position": 1,
    "acting_do_player_position": 1,
    "played_cards": [
        [1, 33, 36, 53, 54, 39, 52, 2],
        [33, 1, 50, 52, 40, 17, 4],
        [25, 51, 52, 40, 53, 55, 24, 9],
        [35, 1, 4, 51, 8, 53, 3, 55],
    ],
    "player_hand_cards": [
        [3, 5, 6, 8, 8, 18, 18, 19, 21, 21, 34, 34, 34],
        [2, 3, 4, 6, 7, 9, 21, 22, 25, 25, 37, 38, 39, 54],
        [7, 7, 20, 21, 22, 36, 37, 38, 49, 49],
        [17, 20, 20, 22, 24, 33, 35, 36, 37, 38, 38, 40, 41],
    ],
    "action_seq": [
        [2, 7, 25],
        [1, 7],
        [3, 7, 35],
        [0, 7, 1],
        [1, 7, 33],
        [2, 7, 51],
        [3, 7, 1],
        [0, 7, 33],
        [1, 7, 1],
        [2, 7, 52],
        [3, 7, 4],
        [0, 7, 36],
        [1, 7, 50],
        [2, 7, 40],
        [3, 7, 51],
        [0, 7, 53],
        [1, 7, 52],
        [2, 7, 53],
        [3, 7, 8],
        [0, 7, 54],
        [1, 7, 40],
        [2, 7, 55],
        [3, 7, 53],
        [0, 7, 39],
        [1, 7, 17],
        [2, 2, 17, 17],
        [2, 7, 24],
        [3, 7, 3],
        [0, 7, 52],
        [1, 7, 4],
        [2, 7, 9],
        [3, 7, 55],
        [0, 7, 2],
    ],
    "last_action": [0, 7, 2],
    "player_chi_cards": [[], [], [], []],
    "player_peng_cards": [[], [], [[17, 17, 17]], []],
    "player_gang_cards": [[], [], [], []],
    "player_bugang_cards": [[], [], [], []],
    "player_angang_cards": [[], [], [], []],
    "player_bu_cards": [[], [], [], []],
    "action_cards": {"7": [[2], [3], [4], [6], [7], [9], [21], [22], [25], [25], [37], [38], [39], [54]]},
    "remain_card_stack": [
        39,
        1,
        8,
        7,
        9,
        39,
        19,
        51,
        40,
        20,
        5,
        2,
        5,
        18,
        52,
        50,
        4,
        23,
        36,
        34,
        50,
        2,
        41,
        23,
        55,
        41,
        33,
        51,
        25,
        37,
        54,
        49,
        23,
        6,
        18,
        54,
        9,
        53,
        22,
        35,
        5,
        55,
        24,
        49,
        3,
        6,
        19,
        19,
        24,
        23,
        50,
        41,
        35,
    ],
}


class JiujiangReplayTests(unittest.TestCase):
    def test_get_action_replays_document_sample_without_crashing(self):
        # 输入输出说明里的样例包含完整真实字段结构，AI 至少要能稳定返回合法动作。
        action_type, action_card = get_action(SAMPLE_DATA)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertIn(action_card, SAMPLE_DATA["action_cards"]["7"])

    def test_http_get_action_replays_wrapped_document_sample(self):
        # 学长远程调用时可能把局面包在 data 字段里，HTTP 层也要能回放样例。
        server = create_server("127.0.0.1", 0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://{host}:{port}/get_action",
                data=json.dumps({"data": SAMPLE_DATA}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))

            self.assertEqual(payload[0], ACTION_DISCARD)
            self.assertIn(payload[1], SAMPLE_DATA["action_cards"]["7"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
