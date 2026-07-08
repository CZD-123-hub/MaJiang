from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any

_LOCK = Lock()


def _empty_stats() -> dict[str, Any]:
    return {
        "total_rounds": 0,
        "win_count": 0,
        "self_draw_count": 0,
        "discard_win_count": 0,
        "wins_by_player": {},
        "dianpao_by_player": {},
        "total_score_by_player": {},
    }


_STATS = _empty_stats()


def reset_stats() -> None:
    """清空累计对局统计，主要用于本地测试和重新开始一轮评测。"""
    with _LOCK:
        _reset_stats_dict(_STATS)


def get_stats() -> dict[str, Any]:
    """返回当前累计统计快照，避免调用方直接修改内部状态。"""
    with _LOCK:
        return deepcopy(_STATS)


def record_round_end(data: dict[str, Any]) -> dict[str, Any]:
    """记录一局结束数据，并返回更新后的统计快照。"""
    with _LOCK:
        _accumulate_round(_STATS, data)
        return deepcopy(_STATS)


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
    winners = _extract_winners(data)
    if winners:
        stats["win_count"] += len(winners)
        for winner in winners:
            _increment(stats["wins_by_player"], winner)

    if _is_self_draw(data):
        stats["self_draw_count"] += 1
    elif winners:
        stats["discard_win_count"] += 1

    dianpao_player = _extract_dianpao_player(data)
    if dianpao_player is not None:
        _increment(stats["dianpao_by_player"], dianpao_player)

    for player, score in _extract_scores(data).items():
        key = str(player)
        stats["total_score_by_player"][key] = stats["total_score_by_player"].get(key, 0.0) + float(score)


def _summarize_team(
    rounds: list[dict[str, Any]],
    our_players: list[int | str],
    overall: dict[str, Any],
) -> dict[str, Any]:
    # 自博弈和对打评测里，通常关心“我方几个座位整体打得怎么样”，这里给出一个简单团队视角摘要。
    players = {str(player) for player in our_players}
    win_rounds = 0
    self_draw_rounds = 0
    discard_win_rounds = 0
    dianpao_rounds = 0

    for round_data in rounds:
        winners = {str(player) for player in _extract_winners(round_data)}
        if winners & players:
            win_rounds += 1
            if _is_self_draw(round_data):
                self_draw_rounds += 1
            else:
                discard_win_rounds += 1

        dianpao_player = _extract_dianpao_player(round_data)
        if dianpao_player is not None and str(dianpao_player) in players:
            dianpao_rounds += 1

    total_score = sum(float(overall["total_score_by_player"].get(player, 0.0)) for player in players)
    total_rounds = overall["total_rounds"] or 1
    return {
        "players": sorted(players),
        "win_rounds": win_rounds,
        "self_draw_rounds": self_draw_rounds,
        "discard_win_rounds": discard_win_rounds,
        "dianpao_rounds": dianpao_rounds,
        "total_score": total_score,
        "average_round_score": total_score / total_rounds,
    }


def _extract_winners(data: dict[str, Any]) -> list[int | str]:
    # 兼容单赢家和一炮多响的多赢家字段。
    if "winners" in data and isinstance(data["winners"], list):
        return data["winners"]
    if "hu_players" in data and isinstance(data["hu_players"], list):
        return data["hu_players"]
    for key in ("winner", "hu_player", "win_player"):
        if key in data and data[key] is not None:
            return [data[key]]
    return []


def _is_self_draw(data: dict[str, Any]) -> bool:
    # 常见裁判字段可能用 zimo/self_draw 或 win_type 表示自摸。
    if bool(data.get("zimo")) or bool(data.get("self_draw")):
        return True
    win_type = str(data.get("win_type", "")).lower()
    return win_type in {"zimo", "self_draw", "自摸"}


def _extract_dianpao_player(data: dict[str, Any]) -> int | str | None:
    for key in ("dianpao_player", "pao_player", "discard_player", "loser"):
        if key in data and data[key] is not None:
            return data[key]
    return None


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
