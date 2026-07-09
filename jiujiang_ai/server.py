from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from werkzeug.serving import WSGIRequestHandler, make_server

from .api import get_action, round_end


class QuietRequestHandler(WSGIRequestHandler):
    """静音 Flask/Werkzeug 默认访问日志，避免联调时刷屏。"""

    def log(self, type: str, message: str, *args: Any) -> None:
        return


def create_app() -> Flask:
    """创建 Flask 应用，供本地启动和服务器部署复用。"""
    app = Flask(__name__)

    @app.post("/get_action")
    def get_action_route():
        payload = _read_json()
        data = _unwrap_data(payload)
        # 学长调用 /get_action 时，接口返回 [action_type, action_card] 列表。
        action_type, action_card = get_action(data)
        return jsonify([action_type, action_card])

    @app.post("/round_end")
    def round_end_route():
        payload = _read_json()
        data = _unwrap_data(payload)
        # /round_end 接收对局结束信息，并返回统计和结算结果。
        return jsonify(round_end(data))

    @app.errorhandler(404)
    def not_found(_: Exception):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(ValueError)
    def value_error(exc: ValueError):
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(400)
    def bad_request(_: Exception):
        return jsonify({"error": "invalid request"}), 400

    return app


def create_server(host: str = "127.0.0.1", port: int = 8000):
    """创建 Flask 对应的 WSGI 服务对象，保持和旧测试兼容。"""
    app = create_app()
    return make_server(host, port, app, threaded=True, request_handler=QuietRequestHandler)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """命令行启动入口：python -m jiujiang_ai.server"""
    server = create_server(host, port)
    try:
        print(f"jiujiang_ai flask server listening on http://{host}:{port}")
        server.serve_forever()
    finally:
        server.server_close()


def _read_json() -> dict[str, Any]:
    # 兼容空 body；如果 body 非法则抛 ValueError 统一转成 400。
    payload = request.get_json(silent=True)
    if payload is None:
        raw = request.get_data(cache=False, as_text=False)
        if not raw:
            return {}
        raise ValueError("invalid json")
    if not isinstance(payload, dict):
        raise ValueError("json payload must be an object")
    return payload


def _unwrap_data(payload: dict[str, Any]) -> dict[str, Any]:
    # 兼容两种调用格式：直接传 data 对象，或传 {"data": {...}} 外层包裹。
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    return data


if __name__ == "__main__":
    run_server()
