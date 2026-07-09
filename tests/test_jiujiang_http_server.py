import json
import threading
import unittest
import urllib.error
import urllib.request

from jiujiang_ai.rules import ACTION_HU
from jiujiang_ai.server import create_server


class JiujiangHttpServerTests(unittest.TestCase):
    def test_post_get_action_returns_json_action_pair(self):
        # /get_action 需要按接口文档返回 [action_type, action_card]。
        server, base_url, thread = self._start_server()
        try:
            response = self._post_json(
                f"{base_url}/get_action",
                {
                    "action_cards": {"4": []},
                    "player_hand_cards": [[], [], [], []],
                    "acting_do_player_position": 0,
                },
            )

            self.assertEqual(response, [ACTION_HU, []])
        finally:
            self._stop_server(server, thread)

    def test_post_get_action_accepts_wrapped_data_payload(self):
        # 远程服务可能把真实局面包在 data 字段里，这里验证兼容性。
        server, base_url, thread = self._start_server()
        try:
            response = self._post_json(
                f"{base_url}/get_action",
                {"data": {"action_cards": {"4": []}}},
            )

            self.assertEqual(response, [ACTION_HU, []])
        finally:
            self._stop_server(server, thread)

    def test_post_round_end_acknowledges_payload(self):
        # /round_end 除了返回 stats，也要把单局结算结果一起返回，方便联调和对打验证。
        server, base_url, thread = self._start_server()
        try:
            response = self._post_json(
                f"{base_url}/round_end",
                {
                    "winner": 0,
                    "win_type": "zimo",
                    "player_hand_cards": [[1, 17, 18], [], [], []],
                    "room_options": {"zama_count": 1},
                    "zama_cards": [1],
                },
            )

            self.assertEqual(response["status"], "ok")
            self.assertTrue(response["received"])
            self.assertIn("settlement", response)
            self.assertEqual(response["settlement"]["score_by_player"], [9, -3, -3, -3])
        finally:
            self._stop_server(server, thread)

    def test_unknown_path_returns_404(self):
        # 未支持的路径应该明确返回 404，避免调用方误以为请求成功。
        server, base_url, thread = self._start_server()
        try:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self._post_json(f"{base_url}/unknown", {})

            self.assertEqual(raised.exception.code, 404)
        finally:
            self._stop_server(server, thread)

    def _start_server(self):
        # 端口传 0 让系统自动分配空闲端口，避免测试之间端口冲突。
        server = create_server("127.0.0.1", 0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://{host}:{port}", thread

    def _stop_server(self, server, thread):
        # shutdown 后等待线程退出，避免测试进程残留后台服务。
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def _post_json(self, url, payload):
        # 用标准库发 JSON 请求，测试环境里不需要额外安装 requests。
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
