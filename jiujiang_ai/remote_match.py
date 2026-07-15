"""测试服机器人房间的顺序对局、持续对局与结果汇总。"""

from __future__ import annotations

import ast
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .stats import DEFAULT_ACTION_LOG_PATH, DEFAULT_ROUND_LOG_PATH, summarize_rounds

DEFAULT_ROBOT_ROOM_URL = "http://cardapisrv.test.xq5.com/cardapisrv/common/robot-pression-with-room"
ProgressCallback = Callable[[dict[str, Any]], None]
ReportCallback = Callable[[dict[str, Any]], None]

ACTION_NAMES = {
    2: "peng",
    3: "ming_gang",
    4: "hu",
    5: "an_gang",
    6: "bu_gang",
}


def start_sequential_remote_robot_matches(
    *,
    place_id: int,
    rooms: int | None = None,
    rounds: int | None = None,
    url: str = DEFAULT_ROBOT_ROOM_URL,
    log_path: str | Path = DEFAULT_ROUND_LOG_PATH,
    action_log_path: str | Path = DEFAULT_ACTION_LOG_PATH,
    timeout_seconds: int = 1800,
    poll_seconds: float = 2.0,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """创建固定局数的机器人房间；每次只创建一局，结算后才开下一局。"""
    if rooms is None:
        rooms = rounds
    elif rounds is not None and rooms != rounds:
        raise ValueError("rooms and rounds must match when both are supplied")
    if rooms is None or rooms < 1:
        raise ValueError("rooms must be at least 1")

    target_log = Path(log_path)
    action_start_line = _line_count(Path(action_log_path))
    created_rooms: list[int] = []
    completed_rounds: list[dict[str, Any]] = []
    failed_rooms = 0
    timed_out_room_ids: list[int] = []

    for round_number in range(1, rooms + 1):
        _notify(progress_callback, {"event": "opening_round", "round": round_number, "total": rooms})
        start_line_count = _line_count(target_log)
        response = _open_robot_rooms(url, place_id, 1)
        if int(response.get("errno", -1)) != 0:
            raise RuntimeError(f"robot room request failed: {response.get('errmsg', response)}")

        data = response.get("data") or {}
        opened_rooms = [int(room) for room in data.get("room_list") or []]
        created_rooms.extend(opened_rooms)
        failed_rooms += int(data.get("fail_count") or 0)
        _notify(
            progress_callback,
            {
                "event": "room_created",
                "round": round_number,
                "total": rooms,
                "created_count": len(opened_rooms),
                "failed_count": int(data.get("fail_count") or 0),
                "room_ids": opened_rooms,
            },
        )
        if not opened_rooms:
            continue

        current_results, current_timed_out = wait_for_room_results(
            room_ids=opened_rooms,
            log_path=target_log,
            start_line_count=start_line_count,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            progress_callback=progress_callback,
        )
        completed_rounds.extend(current_results)
        if current_timed_out:
            timed_out_room_ids.extend(room_id for room_id in opened_rooms if room_id not in {row.get("room_id") for row in current_results})
            break

    action_events = _load_new_action_events(Path(action_log_path), action_start_line, set(created_rooms))
    return build_remote_report(
        place_id=place_id,
        requested_rooms=rooms,
        created_rooms=created_rooms,
        failed_rooms=failed_rooms,
        rounds=completed_rounds,
        timed_out=bool(timed_out_room_ids),
        timed_out_room_ids=timed_out_room_ids,
        action_events=action_events,
    )


def start_continuous_remote_robot_matches(
    *,
    place_id: int,
    url: str = DEFAULT_ROBOT_ROOM_URL,
    log_path: str | Path = DEFAULT_ROUND_LOG_PATH,
    action_log_path: str | Path = DEFAULT_ACTION_LOG_PATH,
    timeout_seconds: int = 1800,
    poll_seconds: float = 2.0,
    retry_seconds: float = 2.0,
    progress_callback: ProgressCallback | None = None,
    report_callback: ReportCallback | None = None,
) -> dict[str, Any]:
    """持续执行“开一房、等结算、再开一房”，直到 Ctrl+C。"""
    target_log = Path(log_path)
    target_action_log = Path(action_log_path)
    action_start_line = _line_count(target_action_log)
    created_rooms: list[int] = []
    completed_rounds: list[dict[str, Any]] = []
    timed_out_room_ids: list[int] = []
    failed_rooms = 0
    attempts = 0
    stopped_by_user = False

    try:
        while True:
            attempts += 1
            _notify(progress_callback, {"event": "opening_round", "round": attempts, "total": None})
            start_line_count = _line_count(target_log)
            try:
                response = _open_robot_rooms(url, place_id, 1)
            except Exception as exc:
                failed_rooms += 1
                _notify(progress_callback, {"event": "room_error", "round": attempts, "error": str(exc)})
                time.sleep(retry_seconds)
                continue

            if int(response.get("errno", -1)) != 0:
                failed_rooms += 1
                _notify(progress_callback, {"event": "room_error", "round": attempts, "error": response.get("errmsg", response)})
                time.sleep(retry_seconds)
                continue

            data = response.get("data") or {}
            opened_rooms = [int(room) for room in data.get("room_list") or []]
            created_rooms.extend(opened_rooms)
            failed_rooms += int(data.get("fail_count") or 0)
            _notify(
                progress_callback,
                {
                    "event": "room_created",
                    "round": attempts,
                    "total": None,
                    "created_count": len(opened_rooms),
                    "failed_count": int(data.get("fail_count") or 0),
                    "room_ids": opened_rooms,
                },
            )
            if not opened_rooms:
                time.sleep(retry_seconds)
                continue

            current_results, current_timed_out = wait_for_room_results(
                room_ids=opened_rooms,
                log_path=target_log,
                start_line_count=start_line_count,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
                progress_callback=progress_callback,
            )
            completed_rounds.extend(current_results)
            if current_timed_out:
                timed_out_room_ids.extend(room_id for room_id in opened_rooms if room_id not in {row.get("room_id") for row in current_results})

            report = _build_live_report(
                place_id,
                attempts,
                created_rooms,
                failed_rooms,
                completed_rounds,
                timed_out_room_ids,
                target_action_log,
                action_start_line,
                stopped_by_user=False,
            )
            if report_callback is not None:
                report_callback(report)
    except KeyboardInterrupt:
        stopped_by_user = True

    return _build_live_report(
        place_id,
        attempts,
        created_rooms,
        failed_rooms,
        completed_rounds,
        timed_out_room_ids,
        target_action_log,
        action_start_line,
        stopped_by_user=stopped_by_user,
    )


def start_remote_robot_matches(**kwargs: Any) -> dict[str, Any]:
    """兼容旧调用名；默认仍使用固定局数模式。"""
    return start_sequential_remote_robot_matches(**kwargs)


def wait_for_room_results(
    *,
    room_ids: list[int],
    log_path: str | Path,
    start_line_count: int,
    timeout_seconds: int,
    poll_seconds: float,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """从开房后的新增 JSONL 中按房间号收集结算；同房间重复回调只保留首条。"""
    expected = set(room_ids)
    if not expected:
        return [], False

    deadline = time.monotonic() + timeout_seconds
    results_by_room: dict[int, dict[str, Any]] = {}
    last_completed = -1
    while time.monotonic() < deadline:
        for data in _load_new_rounds(Path(log_path), start_line_count):
            numeric_room_id = _numeric_room_id(data.get("room_id"))
            if numeric_room_id in expected:
                results_by_room.setdefault(numeric_room_id, data)
        if len(results_by_room) != last_completed:
            last_completed = len(results_by_room)
            _notify(progress_callback, {"event": "round_progress", "completed": last_completed, "expected": len(expected)})
        if expected.issubset(results_by_room):
            return [results_by_room[room_id] for room_id in room_ids], False
        time.sleep(poll_seconds)
    return [results_by_room[room_id] for room_id in room_ids if room_id in results_by_room], True


def build_remote_report(
    *,
    place_id: int | None,
    requested_rooms: int | None,
    created_rooms: list[int],
    failed_rooms: int,
    rounds: list[dict[str, Any]],
    timed_out: bool,
    timed_out_room_ids: list[int] | None = None,
    action_events: list[dict[str, Any]] | None = None,
    raw_round_end_callbacks: int | None = None,
    source: dict[str, Any] | None = None,
    stopped_by_user: bool = False,
) -> dict[str, Any]:
    """构造可保存为 JSON 的远程机器人对局报告，并按 room_id 去重。"""
    unique_rounds, duplicates = _deduplicate_rounds(rounds)
    completed_room_ids = {row.get("room_id") for row in unique_rounds}
    callback_count = raw_round_end_callbacks if raw_round_end_callbacks is not None else len(rounds)
    return {
        "place_id": place_id,
        "requested_rooms": requested_rooms,
        "created_rooms": list(dict.fromkeys(created_rooms)),
        "created_count": len(set(created_rooms)),
        "failed_rooms": failed_rooms,
        "raw_round_end_callbacks": callback_count,
        "duplicate_round_end_callbacks": callback_count - len(unique_rounds),
        "completed_rounds": len(unique_rounds),
        "pending_rooms": [room_id for room_id in created_rooms if room_id not in completed_room_ids],
        "timed_out": timed_out,
        "timed_out_room_ids": timed_out_room_ids or [],
        "stopped_by_user": stopped_by_user,
        "overall": summarize_rounds(unique_rounds),
        "round_end_summary": summarize_round_end_details(unique_rounds),
        "action_summary": summarize_action_events(action_events or []),
        "round_results": unique_rounds,
        "source": source or {},
    }


def build_remote_report_from_log(
    *,
    log_path: str | Path = DEFAULT_ROUND_LOG_PATH,
    start_line: int = 0,
    http_log_path: str | Path | None = None,
) -> dict[str, Any]:
    """离线读取指定行号后的 round_end 回调，生成一次任务的去重报告。"""
    payloads = _load_round_payloads(Path(log_path), start_line)
    rounds = [_normalize_historical_round(payload["data"]) for payload in payloads]
    start_timestamp = payloads[0].get("timestamp") if payloads else None
    end_timestamp = payloads[-1].get("timestamp") if payloads else None
    action_events = _load_http_response_actions(Path(http_log_path), start_timestamp, end_timestamp) if http_log_path else []
    source = {
        "round_end_log": str(log_path),
        "start_line": start_line + 1,
        "action_source": "http_response_log" if http_log_path else "none",
    }
    return build_remote_report(
        place_id=None,
        requested_rooms=None,
        created_rooms=list(dict.fromkeys(room_id for room_id in (_numeric_room_id(row.get("room_id")) for row in rounds) if room_id is not None)),
        failed_rooms=0,
        rounds=rounds,
        timed_out=False,
        raw_round_end_callbacks=len(rounds),
        action_events=action_events,
        source=source,
    )


def summarize_action_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 AI 实际返回的碰、杠、胡动作；保留总决策数供核对。"""
    counts = {name: 0 for name in ACTION_NAMES.values()}
    by_room: dict[str, dict[str, int]] = {}
    for event in events:
        try:
            action_type = int(event.get("action_type"))
        except (TypeError, ValueError):
            continue
        name = ACTION_NAMES.get(action_type)
        if name is None:
            continue
        counts[name] += 1
        room_id = event.get("room_id")
        if room_id is not None:
            room_key = str(room_id)
            by_room.setdefault(room_key, {key: 0 for key in ACTION_NAMES.values()})[name] += 1
    return {"total_decisions": len(events), "counts": counts, "by_room": by_room}


def summarize_round_end_details(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """保留测试服原始 hu_type 与杠分信息，方便后续拿规则表映射具体番型。"""
    hu_type_count: dict[str, int] = {}
    gang_score_rooms: list[int | str] = []
    for row in rounds:
        for seat_types in row.get("hu_type") or []:
            for code in seat_types or []:
                key = str(code)
                hu_type_count[key] = hu_type_count.get(key, 0) + 1
        if any(float(score) != 0 for score in row.get("gang_score") or []):
            gang_score_rooms.append(row.get("room_id"))
    return {
        "hu_type_count": hu_type_count,
        "gang_score_round_count": len(gang_score_rooms),
        "gang_score_room_ids": gang_score_rooms,
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _build_live_report(
    place_id: int,
    attempts: int,
    created_rooms: list[int],
    failed_rooms: int,
    completed_rounds: list[dict[str, Any]],
    timed_out_room_ids: list[int],
    action_log_path: Path,
    action_start_line: int,
    *,
    stopped_by_user: bool,
) -> dict[str, Any]:
    action_events = _load_new_action_events(action_log_path, action_start_line, set(created_rooms))
    return build_remote_report(
        place_id=place_id,
        requested_rooms=None,
        created_rooms=created_rooms,
        failed_rooms=failed_rooms,
        rounds=completed_rounds,
        timed_out=bool(timed_out_room_ids),
        timed_out_room_ids=timed_out_room_ids,
        action_events=action_events,
        source={"mode": "continuous", "open_attempts": attempts, "action_log": str(action_log_path)},
        stopped_by_user=stopped_by_user,
    )


def _open_robot_rooms(url: str, place_id: int, rooms: int) -> dict[str, Any]:
    payload = json.dumps({"place_id": place_id, "interval": rooms}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("robot room response must be a JSON object")
    return decoded


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def _load_new_rounds(path: Path, start_line_count: int) -> list[dict[str, Any]]:
    return [payload["data"] for payload in _load_round_payloads(path, start_line_count)]


def _load_round_payloads(path: Path, start_line: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[start_line:]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            payloads.append(payload)
    return payloads


def _load_new_action_events(path: Path, start_line: int, room_ids: set[int]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[start_line:]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and _numeric_room_id(event.get("room_id")) in room_ids:
            events.append(event)
    return events


def _load_http_response_actions(
    path: Path,
    start_timestamp: str | None,
    end_timestamp: str | None,
) -> list[dict[str, Any]]:
    """兼容动作日志上线前的 HTTP 文本日志；这类历史动作无法反查 room_id。"""
    if not path.exists() or start_timestamp is None:
        return []
    start = start_timestamp[:19].replace("T", " ")
    end = end_timestamp[:19].replace("T", " ") if end_timestamp else None
    pattern = re.compile(r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO get_action response .*?action_type=(?P<type>\d+) action_card=(?P<card>\[[^]]*\])")
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None or match.group("time") < start or (end is not None and match.group("time") > end):
            continue
        try:
            cards = ast.literal_eval(match.group("card"))
        except (SyntaxError, ValueError):
            cards = []
        events.append({"timestamp": match.group("time"), "action_type": int(match.group("type")), "action_card": cards})
    return events


def _deduplicate_rounds(rounds: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[int | str] = set()
    duplicates = 0
    for row in rounds:
        room_id = _numeric_room_id(row.get("room_id"))
        key: int | str = room_id if room_id is not None else f"no-room-{len(unique)}"
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, duplicates


def _numeric_room_id(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_historical_round(data: dict[str, Any]) -> dict[str, Any]:
    """纠正旧版本已写入日志的流局误标：-1 不是赢家，也不是自摸。"""
    normalized = dict(data)
    winner = normalized.get("win_player_position", normalized.get("winner"))
    is_invalid_winner = False
    try:
        is_invalid_winner = int(winner) < 0
    except (TypeError, ValueError):
        pass
    if normalized.get("end_type") == 2 or is_invalid_winner:
        normalized.pop("winner", None)
        normalized.pop("dianpao_player", None)
        normalized.pop("win_type", None)
        normalized["is_draw"] = True
    return normalized


def _notify(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback is not None:
        callback(event)
