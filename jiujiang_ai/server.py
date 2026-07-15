from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.serving import WSGIRequestHandler, make_server

from .api import get_action, round_end
from .dashboard import dashboard_payload
from .stats import append_action_log


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "jiujiang_http.log"
LOGGER = logging.getLogger("jiujiang_ai.http")


def _configure_logger() -> None:
    """将联调请求摘要同时输出到控制台和本地日志文件。"""
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
    except OSError:
        LOGGER.warning("HTTP log file is unavailable: %s", LOG_PATH)


class QuietRequestHandler(WSGIRequestHandler):
    """静音 Flask/Werkzeug 默认访问日志，避免联调时刷屏。"""

    def log(self, type: str, message: str, *args: Any) -> None:
        return


def create_app() -> Flask:
    """创建 Flask 应用，供本地启动和服务器部署复用。"""
    _configure_logger()
    app = Flask(__name__)

    @app.get("/dashboard")
    def dashboard_page():
        return app.send_static_file("dashboard.html")

    @app.get("/api/dashboard/events")
    def dashboard_events():
        return jsonify(dashboard_payload())

    @app.post("/get_action")
    def get_action_route():
        started_at = time.perf_counter()
        payload = _read_json()
        data = _unwrap_data(payload)
        position = _decision_position(data)
        LOGGER.info(
            "get_action request client=%s room_id=%s player_id=%s acting=%s decision_player=%s turn=%s last_action=%s hand_size=%s wall_remaining=%s action_types=%s",
            request.remote_addr,
            data.get("room_id"),
            _player_identity(data, position),
            data.get("acting_player_position"),
            position,
            len(data.get("action_seq") or []),
            data.get("last_action") or [],
            _acting_hand_size(data),
            len(data.get("remain_card_stack") or []),
            sorted(str(key) for key in (data.get("action_cards") or {})),
        )
        action_type, action_card = get_action(data)
        response = {"action_card": action_card, "action_type": action_type}
        try:
            append_action_log(
                data,
                action_type=action_type,
                action_card=action_card,
                client=request.remote_addr,
            )
        except OSError:
            # 观测日志写入失败不能影响实际对局响应。
            pass
        LOGGER.info(
            "get_action response room_id=%s player_id=%s action_type=%s action_card=%s elapsed_ms=%.1f",
            data.get("room_id"),
            _player_identity(data, position),
            action_type,
            action_card,
            (time.perf_counter() - started_at) * 1000,
        )
        return jsonify(response)

    @app.post("/round_end")
    def round_end_route():
        payload = _read_json()
        data = _unwrap_data(payload)
        # /round_end 接收对局结束信息，并返回统计和结算结果。
        LOGGER.info(
            "round_end request client=%s winner=%s platform_winner=%s win_type=%s score=%s",
            request.remote_addr,
            data.get("winner", data.get("winners")),
            data.get("win_player_position"),
            data.get("win_type"),
            data.get("scores", data.get("total_score")),
        )
        result = round_end(data)
        LOGGER.info(
            "round_end response status=%s winner=%s win_type=%s score=%s",
            result["status"],
            result["data"].get("winner", result["data"].get("winners")),
            result["win_context"]["win_type"],
            result["data"].get("scores"),
        )
        return jsonify(result)

    @app.errorhandler(404)
    def not_found(_: Exception):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(ValueError)
    def value_error(exc: ValueError):
        LOGGER.warning("invalid request client=%s error=%s", request.remote_addr, exc)
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(400)
    def bad_request(_: Exception):
        LOGGER.warning("bad request client=%s", request.remote_addr)
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


def _acting_hand_size(data: dict[str, Any]) -> int:
    hands = data.get("player_hand_cards") or []
    position = _decision_position(data)
    try:
        return len(hands[position] or [])
    except (IndexError, TypeError, ValueError):
        return 0


def _decision_position(data: dict[str, Any]) -> int:
    try:
        return int(data.get("acting_do_player_position", 0))
    except (TypeError, ValueError):
        return 0


def _player_identity(data: dict[str, Any], position: int) -> Any:
    """优先记录平台提供的用户标识；没有时以座位号作为 player_id。"""
    for key in ("player_id", "user_id", "uid", "acting_player_id", "acting_do_player_id"):
        if data.get(key) is not None:
            return data[key]
    for key in ("player_ids", "user_ids", "uids"):
        values = data.get(key)
        if isinstance(values, list) and 0 <= position < len(values):
            return values[position]
    return position


if __name__ == "__main__":
    run_server()
