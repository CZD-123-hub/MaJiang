from __future__ import annotations

from typing import Any

from .round_flow import detect_round_flow
from .rules import ACTION_ANGANG, ACTION_BUGANG, ACTION_DISCARD, ACTION_GANG
from .tiles import HONGZHONG
from .win_context import detect_win_context
from .zama import calculate_zama_score


def calculate_gang_score(data: dict[str, Any]) -> dict[str, Any]:
    """计算当前对局中的杠分明细。"""
    player_count = _player_count(data)
    round_flow = detect_round_flow(data)
    score_by_player = [0 for _ in range(player_count)]
    events: list[dict[str, Any]] = []

    # 规则要求“荒牌不算杠分”，黄庄局直接清空杠分。
    if round_flow["is_huangzhuang"]:
        return {
            "events": events,
            "score_by_player": score_by_player,
            "total_gang_score": 0,
        }

    for event in _extract_gang_events(data, player_count):
        actor = event["player"]
        gang_type = event["gang_type"]

        if gang_type == "zhigang":
            payer = event.get("payer")
            if payer is None:
                continue
            score_by_player[actor] += 3
            score_by_player[payer] -= 3
            events.append(
                {
                    "gang_type": "zhigang",
                    "player": actor,
                    "payer": payer,
                    "tile": event["tile"],
                    "score": 3,
                }
            )
            continue

        payers = [player for player in range(player_count) if player != actor]
        score_per_player = 1 if gang_type == "bugang" else 2
        for payer in payers:
            score_by_player[payer] -= score_per_player
        score_by_player[actor] += score_per_player * len(payers)
        events.append(
            {
                "gang_type": gang_type,
                "player": actor,
                "payers": payers,
                "tile": event["tile"],
                "score_per_player": score_per_player,
            }
        )

    return {
        "events": events,
        "score_by_player": score_by_player,
        "total_gang_score": sum(score for score in score_by_player if score > 0),
    }


def calculate_buy_score(
    data: dict[str, Any],
    winner: int | str | None = None,
) -> dict[str, Any]:
    """计算加买分。"""
    player_count = _player_count(data)
    score_by_player = [0 for _ in range(player_count)]
    buy_scores = _buy_scores_by_player(data, player_count)
    resolved_winner = _as_int_or_none(winner if winner is not None else _default_winner(data))

    if not _buy_score_enabled(data) or resolved_winner is None or not (0 <= resolved_winner < player_count):
        return {
            "winner": resolved_winner,
            "buy_scores": buy_scores,
            "score_by_player": score_by_player,
            "total_buy_score": 0,
        }

    winner_buy_score = buy_scores[resolved_winner]
    for player in range(player_count):
        if player == resolved_winner:
            continue
        payment = buy_scores[player] + winner_buy_score
        score_by_player[player] -= payment
        score_by_player[resolved_winner] += payment

    return {
        "winner": resolved_winner,
        "buy_scores": buy_scores,
        "score_by_player": score_by_player,
        "total_buy_score": score_by_player[resolved_winner],
    }


def calculate_total_score(data: dict[str, Any]) -> dict[str, Any]:
    """汇总胡牌分、加买分、杠分和码分，得到本局总分。"""
    player_count = _player_count(data)
    context = detect_win_context(data)
    winners = [winner for winner in context.winners if _as_int_or_none(winner) is not None]

    gang_component = calculate_gang_score(data)
    total_score_by_player = list(gang_component["score_by_player"])
    hu_results: list[dict[str, Any]] = []
    buy_results: list[dict[str, Any]] = []
    zama_results: list[dict[str, Any]] = []
    hu_score_by_player = [0.0 for _ in range(player_count)]
    buy_score_by_player = [0.0 for _ in range(player_count)]
    zama_score_by_player = [0.0 for _ in range(player_count)]

    for winner in winners:
        hu_result = calculate_hu_score(data, winner=winner)
        hu_result["score_by_player"] = _distribute_hu_like_score(data, hu_result["hu_score"], winner, context.win_type)
        hu_results.append(hu_result)
        _merge_scores(hu_score_by_player, hu_result["score_by_player"])
        _merge_scores(total_score_by_player, hu_result["score_by_player"])

        buy_result = calculate_buy_score(data, winner=winner)
        buy_results.append(buy_result)
        _merge_scores(buy_score_by_player, buy_result["score_by_player"])
        _merge_scores(total_score_by_player, buy_result["score_by_player"])

        zama_result = calculate_zama_score(data, winner=winner)
        zama_result["score_by_player"] = _distribute_zama_score(data, zama_result["zama_score"], winner, context.win_type)
        zama_results.append(zama_result)
        _merge_scores(zama_score_by_player, zama_result["score_by_player"])
        _merge_scores(total_score_by_player, zama_result["score_by_player"])

    return {
        "winners": winners,
        "win_type": context.win_type,
        "score_by_player": total_score_by_player,
        "components": {
            "hu": _component_summary(hu_results, hu_score_by_player),
            "buy": _component_summary(buy_results, buy_score_by_player),
            "gang": gang_component,
            "zama": _component_summary(zama_results, zama_score_by_player),
        },
    }


def calculate_hu_score(
    data: dict[str, Any],
    winner: int | str | None = None,
    base_hu_score: int | float = 1,
) -> dict[str, Any]:
    """计算当前阶段的胡牌分：胡型分 × 胡牌方式倍率 × 跑红中倍率。"""
    context = detect_win_context(data)
    resolved_winner = winner if winner is not None else (context.winners[0] if context.winners else None)
    win_type_multiplier = _win_type_multiplier(context.win_type)
    run_hongzhong_multiplier = calculate_run_hongzhong_multiplier(data, winner=resolved_winner)
    hu_score = float(base_hu_score) * win_type_multiplier * run_hongzhong_multiplier

    return {
        "winner": resolved_winner,
        "win_type": context.win_type,
        "dianpao_player": context.dianpao_player,
        "base_hu_score": float(base_hu_score),
        "win_type_multiplier": win_type_multiplier,
        "run_hongzhong_multiplier": run_hongzhong_multiplier,
        "hu_score": hu_score,
    }


def calculate_run_hongzhong_multiplier(
    data: dict[str, Any],
    winner: int | str | None = None,
) -> int:
    """计算跑红中倍率：未开启时为 1，开启后按 2^出红中次数。"""
    if not _run_hongzhong_enabled(data):
        return 1

    resolved_winner = winner if winner is not None else _default_winner(data)
    if resolved_winner is None:
        return 1

    discard_count = _count_hongzhong_discards(data, resolved_winner)
    return 2**discard_count


def _win_type_multiplier(win_type: str) -> int:
    # 抢杠胡当前先按任务书要求并入自摸类倍率处理。
    if win_type in {"zimo", "gangkai", "qianggang"}:
        return 2
    return 1


def _run_hongzhong_enabled(data: dict[str, Any]) -> bool:
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = (
        "run_hongzhong_double",
        "allow_discard_hongzhong",
        "hongzhong_double",
        "跑红中翻倍",
        "出红中翻倍",
    )
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _count_hongzhong_discards(data: dict[str, Any], winner: int | str) -> int:
    winner_index = _as_int_or_none(winner)
    played_cards = data.get("played_cards") or []
    if winner_index is not None and 0 <= winner_index < len(played_cards):
        return sum(1 for tile in (played_cards[winner_index] or []) if tile == HONGZHONG)

    action_seq = data.get("action_seq") or []
    count = 0
    for action in action_seq:
        if not isinstance(action, (list, tuple)) or len(action) < 3:
            continue
        if winner_index is None or action[0] != winner_index or action[1] != ACTION_DISCARD:
            continue
        if action[2] == HONGZHONG:
            count += 1
    return count


def _buy_score_enabled(data: dict[str, Any]) -> bool:
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    keys = ("enable_buy_score", "buy_score_enabled", "加买", "开启加买")
    return any(_truthy(source.get(key)) for source in option_sources for key in keys)


def _buy_scores_by_player(data: dict[str, Any], player_count: int) -> list[int]:
    # 优先读取每个玩家各自的买分；没有列表时，再回退为全员统一买分。
    for key in ("player_buy_scores", "buy_scores", "player_buy_score"):
        values = data.get(key)
        if isinstance(values, list) and values:
            scores = [_as_int_or_none(value) or 0 for value in values[:player_count]]
            if len(scores) < player_count:
                scores.extend([0] * (player_count - len(scores)))
            return scores

    uniform_score = 0
    option_sources = [
        data,
        data.get("room_options") or {},
        data.get("game_options") or {},
        data.get("options") or {},
    ]
    for source in option_sources:
        value = _as_int_or_none(source.get("buy_score"))
        if value is not None:
            uniform_score = value
            break
    return [uniform_score for _ in range(player_count)]


def _extract_gang_events(data: dict[str, Any], player_count: int) -> list[dict[str, int | str | None]]:
    # 当前优先从历史动作序列里提取杠事件，后续总分结算再继续扩展其他来源。
    action_seq = data.get("action_seq") or []
    events: list[dict[str, int | str | None]] = []
    last_discard: tuple[int, int] | None = None

    for action in action_seq:
        if not isinstance(action, (list, tuple)) or len(action) < 2:
            continue

        actor = _as_int_or_none(action[0])
        action_type = _as_int_or_none(action[1])
        if actor is None or action_type is None or not (0 <= actor < player_count):
            continue

        if action_type == ACTION_DISCARD and len(action) >= 3:
            tile = _as_int_or_none(action[2])
            if tile is not None:
                last_discard = (actor, tile)
            continue

        if action_type not in {ACTION_GANG, ACTION_BUGANG, ACTION_ANGANG}:
            continue

        tile = _as_int_or_none(action[2]) if len(action) >= 3 else None
        if tile is None:
            continue

        if action_type == ACTION_GANG:
            payer = None
            if last_discard is not None and last_discard[1] == tile and last_discard[0] != actor:
                payer = last_discard[0]
            events.append({"gang_type": "zhigang", "player": actor, "payer": payer, "tile": tile})
        elif action_type == ACTION_BUGANG:
            events.append({"gang_type": "bugang", "player": actor, "tile": tile})
        else:
            events.append({"gang_type": "angang", "player": actor, "tile": tile})

    return events


def _distribute_hu_like_score(
    data: dict[str, Any],
    score: int | float,
    winner: int | str | None,
    win_type: str,
) -> list[float]:
    player_count = _player_count(data)
    score_by_player = [0.0 for _ in range(player_count)]
    winner_index = _as_int_or_none(winner)
    if winner_index is None or not (0 <= winner_index < player_count) or score == 0:
        return score_by_player

    if win_type in {"zimo", "gangkai"}:
        for player in range(player_count):
            if player == winner_index:
                continue
            score_by_player[player] -= float(score)
            score_by_player[winner_index] += float(score)
        return score_by_player

    payer = _as_int_or_none(_extract_payer(data))
    if payer is None or not (0 <= payer < player_count) or payer == winner_index:
        return score_by_player

    multiplier = player_count - 1 if win_type == "qianggang" else 1
    payment = float(score) * multiplier
    score_by_player[payer] -= payment
    score_by_player[winner_index] += payment
    return score_by_player


def _distribute_zama_score(
    data: dict[str, Any],
    score: int | float,
    winner: int | str | None,
    win_type: str,
) -> list[float]:
    # 码分的支付方向与规则文档一致：自摸三家付，点炮一家付，抢杠胡包三家。
    return _distribute_hu_like_score(data, score, winner, win_type)


def _component_summary(results: list[dict[str, Any]], score_by_player: list[float]) -> dict[str, Any]:
    if not results:
        return {"results": [], "score_by_player": score_by_player}
    if len(results) == 1:
        return {**results[0], "results": results, "score_by_player": score_by_player}
    return {"results": results, "score_by_player": score_by_player}


def _merge_scores(target: list[int | float], source: list[int | float]) -> None:
    for index, value in enumerate(source):
        target[index] += value


def _extract_payer(data: dict[str, Any]) -> int | str | None:
    for key in ("dianpao_player", "pao_player", "discard_player", "loser"):
        if key in data and data[key] is not None:
            return data[key]
    return None


def _default_winner(data: dict[str, Any]) -> int | str | None:
    if "winner" in data and data["winner"] is not None:
        return data["winner"]
    winners = data.get("winners")
    if isinstance(winners, list) and winners:
        return winners[0]
    hu_players = data.get("hu_players")
    if isinstance(hu_players, list) and hu_players:
        return hu_players[0]
    return None


def _player_count(data: dict[str, Any]) -> int:
    # 九江麻将默认四人局；如果局面字段里能看出人数，则优先按真实人数计算。
    count_fields = (
        "player_hand_cards",
        "played_cards",
        "player_chi_cards",
        "player_peng_cards",
        "player_gang_cards",
        "player_bugang_cards",
        "player_angang_cards",
    )
    for field in count_fields:
        groups = data.get(field)
        if isinstance(groups, list) and groups:
            return len(groups)
    return 4


def _as_int_or_none(value: int | str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y", "是", "开启"}
    return bool(value)
