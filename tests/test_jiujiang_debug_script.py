import importlib.util
import threading
import unittest
from pathlib import Path

from jiujiang_ai.rules import ACTION_DISCARD
from jiujiang_ai.server import create_server


class JiujiangDebugScriptTests(unittest.TestCase):
    def test_debug_script_builds_valid_sample_data(self):
        # 调试脚本应能构造一份可直接传给 get_action 的九江红中样例。
        module = self._load_debug_script()

        data = module.build_sample_data()

        self.assertIn("action_cards", data)
        self.assertIn("player_hand_cards", data)
        self.assertEqual(data["acting_do_player_position"], 0)

    def test_debug_script_can_call_local_http_server(self):
        # 调试脚本的 HTTP 客户端函数应能请求本地 /get_action 服务。
        module = self._load_debug_script()
        server = create_server("127.0.0.1", 0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = module.post_get_action(f"http://{host}:{port}/get_action", module.build_sample_data())

            self.assertEqual(result["action_type"], ACTION_DISCARD)
            self.assertIn(result["action_card"], module.build_sample_data()["action_cards"]["7"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def _load_debug_script(self):
        script_path = Path(__file__).resolve().parents[1] / "examples" / "jiujiang_http_debug.py"
        spec = importlib.util.spec_from_file_location("jiujiang_http_debug", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
