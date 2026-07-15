from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .win_context import detect_win_context

_LOCK = Lock()
DEFAULT_ROUND_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "jiujiang_round_end.jsonl"
DEFAULT_ACTION_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "jiujiang_action.jsonl"
BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="CST")


def _empty_stats() -> dict[str, Any]:
    return {
        "total_rounds": 0,
        "draw_count": 0,
        "win_count": 0,
        "self_draw_count": 0,
        "discard_win_count": 0,
        "win_type_count": {},
        "multi_win_rounds": 0,
        "multi_win_winner_count": 0,
        "multi_win_by_dianpao_player": {},
        "wins_by_player": {},
        "dianpao_by_player": {},
        "total_score_by_player": {},
    }


_STATS = _empty_stats()
_RECORDED_ROOM_IDS: set[int | str] = set()


def reset_stats() -> None:
    """清空累计对局统计，主要用于本地测试和重新开始一轮评测。"""
    with _LOCK:
        _reset_stats_dict(_STATS)
        _RECORDED_ROOM_IDS.clear()


def get_stats() -> dict[str, Any]:
    """返回当前累计统计快照，避免调用方直接修改内部状态。"""
    with _LOCK:
        return deepcopy(_STATS)


def record_round_end(data: dict[str, Any], log_path: str | Path | None = None) -> dict[str, Any]:
    """记录一局结束数据，更新内存统计并追加写入本地日志。"""
    with _LOCK:
        duplicate = _is_duplicate_room_end(data)
        if not duplicate:
            _accumulate_round(_STATS, data)
        stats_snapshot = deepcopy(_STATS)
        try:
            append_round_log(data, stats_snapshot, log_path=log_path, duplicate=duplicate)
        except OSError:
            # 日志落盘失败时不阻断接口主流程，避免影响联调和对打。
            pass
        return stats_snapshot


def append_round_log(
    data: dict[str, Any],
    stats: dict[str, Any] | None = None,
    log_path: str | Path | None = None,
    duplicate: bool = False,
) -> Path:
    """把单局 round_end 数据按 JSONL 形式追加写入日志文件。"""
    target = Path(log_path) if log_path is not None else DEFAULT_ROUND_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(BEIJING_TIMEZONE).isoformat(),
        "data": data,
        "stats": stats or {},
        "duplicate": duplicate,
    }
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=str))
        file.write("\n")
    return target


def append_action_log(
    data: dict[str, Any],
    *,
    action_type: int,
    action_card: list[int],
    client: str | None = None,
    log_path: str | Path | None = None,
) -> Path:
    """追加一条 /get_action 的实际决策，供远程对局报告精确统计碰杠胡。"""
    target = Path(log_path) if log_path is not None else DEFAULT_ACTION_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    position = data.get("acting_do_player_position", data.get("acting_player_position"))
    payload = {
        "timestamp": datetime.now(BEIJING_TIMEZONE).isoformat(),
        "room_id": data.get("room_id"),
        "player_position": position,
        "action_type": action_type,
        "action_card": list(action_card),
        "client": client,
        "table_state": _action_table_state(data),
    }
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=str))
        file.write("\n")
    return target


def _action_table_state(data: dict[str, Any]) -> dict[str, Any]:
    """保留观察牌局所需的最小桌面快照，不影响决策逻辑。"""
    fields = (
        "player_hand_cards",
        "played_cards",
        "player_chi_cards",
        "player_peng_cards",
        "player_gang_cards",
        "player_bugang_cards",
        "player_angang_cards",
        "last_action",
    )
    state = {field: data.get(field) or [] for field in fields}
    state["acting_player_position"] = data.get("acting_player_position")
    state["acting_do_player_position"] = data.get("acting_do_player_position")
    state["remain_card_count"] = len(data.get("remain_card_stack") or [])
    return state


def load_round_logs(log_path: str | Path | None = None) -> list[dict[str, Any]]:
    """从 JSONL 日志文件读取所有对局结果。"""
    target = Path(log_path) if log_path is not None else DEFAULT_ROUND_LOG_PATH
    if not target.exists():
        return []
    rounds: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("round log line must decode to a JSON object")
        if "data" in payload and isinstance(payload["data"], dict):
            rounds.append(payload["data"])
        else:
            # 兼容旧版只写原始 round_end data 的日志格式。
            rounds.append(payload)
    return rounds


def summarize_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """对一批 round_end 数据做纯汇总，不修改进程内全局统计状态。"""
    stats = _empty_stats()
    for round_data in rounds:
        _accumulate_round(stats, round_data)
    return stats


def summarize_match_report(
    rounds: list[dict[str, Any]],
    our_players: list[int | str] | None = None,
) -> dict[str, Any]:
    """生成批量对打摘要，可选站在我方座位视角统计胜局、点炮和总分。"""
    overall = summarize_rounds(rounds)
    report: dict[str, Any] = {"overall": overall}
    if our_players:
        report["team"] = _summarize_team(rounds, our_players, overall)
    return report


def _reset_stats_dict(stats: dict[str, Any]) -> None:
    empty = _empty_stats()
    for key, value in empty.items():
        stats[key] = value


def _accumulate_round(stats: dict[str, Any], data: dict[str, Any]) -> None:
    # 单局累计逻辑集中在这里，保证实时 round_end 和离线批量汇总口径一致。
    stats["total_rounds"] += 1
    context = detect_win_context(data)
    winners = context.winners

    if _is_draw(data, winners):
        stats["draw_count"] += 1

    if winners:
        stats["win_count"] += len(winners)
        for winner in winners:
            _increment(stats["wins_by_player"], winner)

    if context.win_type != "unknown":
        _increment(stats["win_type_count"], context.win_type)

    if context.is_multi_win:
        stats["multi_win_rounds"] += 1
        stats["multi_win_winner_count"] += len(winners)
        if context.dianpao_player is not None:
            _increment(stats["multi_win_by_dianpao_player"], context.dianpao_player)

    if _is_self_draw(context.win_type):
        stats["self_draw_count"] += 1
    elif winners:
        # 一炮多响仍只算同一轮点炮胜局，但赢家数量会通过 win_count 体现。
        stats["discard_win_count"] += 1

    if context.dianpao_player is not None:
        _increment(stats["dianpao_by_player"], context.dianpao_player)

    for player, score in _extract_scores(data).items():
        key = str(player)
        stats["total_score_by_player"][key] = stats["total_score_by_player"].get(key, 0.0) + float(score)


def _summarize_team(
    rounds: list[dict[str, Any]],
    our_players: list[int | str],
    overall: dict[str, Any],
) -> dict[str, Any]:
    # 对打评测里通常关心“我方几个座位整体表现”，这里给一个简洁团队视角摘要。
    players = {str(player) for player in our_players}
    win_rounds = 0
    self_draw_rounds = 0
    discard_win_rounds = 0
    dianpao_rounds = 0
    multi_win_rounds = 0
    multi_win_wins = 0

    for round_data in rounds:
        context = detect_win_context(round_data)
        winners = {str(player) for player in context.winners}
        if winners & players:
            win_rounds += 1
            if _is_self_draw(context.win_type):
                self_draw_rounds += 1
            else:
                discard_win_rounds += 1
            if context.is_multi_win:
                multi_win_rounds += 1
                multi_win_wins += len(winners & players)

        if context.dianpao_player is not None and str(context.dianpao_player) in players:
            dianpao_rounds += 1

    total_score = sum(float(overall["total_score_by_player"].get(player, 0.0)) for player in players)
    total_rounds = overall["total_rounds"] or 1
    return {
        "players": sorted(players),
        "win_rounds": win_rounds,
        "self_draw_rounds": self_draw_rounds,
        "discard_win_rounds": discard_win_rounds,
        "dianpao_rounds": dianpao_rounds,
        "multi_win_rounds": multi_win_rounds,
        "multi_win_wins": multi_win_wins,
        "total_score": total_score,
        "average_round_score": total_score / total_rounds,
    }


def _is_self_draw(win_type: str) -> bool:
    # 杠开本质上也属于摸牌成胡，因此先并入自摸类统计。
    return win_type in {"zimo", "gangkai"}


def _is_draw(data: dict[str, Any], winners: list[int | str]) -> bool:
    """平台 end_type=2 或没有赢家且全员零分时，均按流局统计。"""
    if data.get("end_type") == 2:
        return True
    return bool(data.get("is_draw")) or (not winners and bool(data.get("liu_ju_score")))


def _is_duplicate_room_end(data: dict[str, Any]) -> bool:
    """测试服会对同一 room_id 重复回调；无 room_id 的本地调用仍逐条计数。"""
    room_id = data.get("room_id")
    if room_id is None:
        return False
    try:
        key: int | str = int(room_id)
    except (TypeError, ValueError):
        key = str(room_id)
    if key in _RECORDED_ROOM_IDS:
        return True
    _RECORDED_ROOM_IDS.add(key)
    return False


def _extract_scores(data: dict[str, Any]) -> dict[str, float]:
    for key in ("score_delta", "scores", "player_scores", "delta_scores"):
        if key not in data:
            continue
        scores = data[key]
        if isinstance(scores, dict):
            return {str(player): float(score) for player, score in scores.items()}
        if isinstance(scores, list):
            return {str(index): float(score) for index, score in enumerate(scores)}
    return {}


def _increment(counter: dict[str, int], player: int | str) -> None:
    key = str(player)
    counter[key] = counter.get(key, 0) + 1
