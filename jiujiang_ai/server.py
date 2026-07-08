from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .api import get_action, round_end


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    # 使用标准库 HTTPServer，避免为了两个接口额外引入 Flask/FastAPI 依赖。
    return ThreadingHTTPServer((host, port), JiujiangRequestHandler)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    # 命令行启动入口：python -m jiujiang_ai.server
    server = create_server(host, port)
    try:
        print(f"jiujiang_ai server listening on http://{host}:{port}")
        server.serve_forever()
    finally:
        server.server_close()


class JiujiangRequestHandler(BaseHTTPRequestHandler):
    server_version = "JiujiangAI/0.1"

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            data = _unwrap_data(payload)
            if self.path == "/get_action":
                # 学长调用 /get_action 时，接口返回 [action_type, action_card] 列表。
                action_type, action_card = get_action(data)
                self._write_json([action_type, action_card])
                return
            if self.path == "/round_end":
                # /round_end 只接收对局结束信息，后续可在 round_end 内扩展统计逻辑。
                self._write_json(round_end(data))
                return
            self.send_error(404, "not found")
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
        except ValueError as exc:
            self.send_error(400, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        # 单元测试和本地对打时不输出默认 HTTP 访问日志，避免刷屏。
        return

    def _read_json(self) -> dict[str, Any]:
        # 从 POST body 读取 JSON；空 body 按空对象处理，方便 round_end 空调用。
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        if not body:
            return {}
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("json payload must be an object")
        return payload

    def _write_json(self, payload: object, status: int = 200) -> None:
        # ensure_ascii=False 让后续中文日志或错误信息也能按 UTF-8 正常返回。
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _unwrap_data(payload: dict[str, Any]) -> dict[str, Any]:
    # 兼容两种调用格式：直接传 data 对象，或传 {"data": {...}} 外层包裹。
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    return data


if __name__ == "__main__":
    run_server()
