"""把已实现的九江结算规则转换为决策阶段可用的预期成和收益。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .settlement import calculate_buy_score, calculate_hu_score
from .tiles import HONGZHONG


@dataclass(frozen=True)
class ExpectedWinValue:
    """本家未来成和时的期望净收益。

    ``expected_zama_gain`` 仅在调用方明确提供预测值时计入；未知码牌不能从
    当前公开局面可靠推出，默认保守为零。
    """

    total: float
    expected_hu_gain: float
    expected_buy_gain: float
    expected_zama_gain: float
    mode_weights: dict[str, float]
    base_hu_score: float


def estimate_win_value(
    data: dict,
    *,
    winner: int,
    pending_discard: int | None = None,
) -> ExpectedWinValue:
    """估算本家在当前局面继续做成后的结算收益。

    可在 ``strategy_options`` 中设置 ``expected_win_type``（例如 ``zimo``、
    ``gangkai``）或 ``expected_zimo_probability``。未设置时使用明确、可调的
    自摸/点炮各 50% 基线，而非暗中假设某一种胡法。
    """
    mode_weights = _mode_weights(data)
    base_hu_score = _base_hu_score(data)
    scenario = _with_pending_discard(data, winner, pending_discard)
    player_count = _player_count(scenario)
    expected_hu_gain = 0.0

    for win_type, weight in mode_weights.items():
        mode_data = dict(scenario)
        mode_data.update(
            {
                "winner": winner,
                "winners": [winner],
                "win_type": win_type,
                "dianpao_player": _placeholder_payer(winner, player_count),
            }
        )
        hu_score = float(calculate_hu_score(mode_data, winner=winner, base_hu_score=base_hu_score)["hu_score"])
        expected_hu_gain += weight * hu_score * _winner_payment_count(win_type, player_count)

    buy_data = dict(scenario)
    buy_data["winner"] = winner
    expected_buy_gain = float(calculate_buy_score(buy_data, winner=winner)["total_buy_score"])
    expected_zama_gain = _expected_zama_gain(data)
    return ExpectedWinValue(
        total=expected_hu_gain + expected_buy_gain + expected_zama_gain,
        expected_hu_gain=expected_hu_gain,
        expected_buy_gain=expected_buy_gain,
        expected_zama_gain=expected_zama_gain,
        mode_weights=mode_weights,
        base_hu_score=base_hu_score,
    )


def _mode_weights(data: dict) -> dict[str, float]:
    options = _strategy_options(data)
    explicit_type = options.get("expected_win_type") or data.get("expected_win_type")
    if explicit_type:
        normalized = _normalize_win_type(explicit_type)
        if normalized:
            return {normalized: 1.0}

    configured_weights = options.get("win_type_weights") or data.get("win_type_weights")
    if isinstance(configured_weights, Mapping):
        parsed = {
            normalized: float(weight)
            for name, weight in configured_weights.items()
            if (normalized := _normalize_win_type(name)) is not None and _positive_number(weight)
        }
        total = sum(parsed.values())
        if total > 0:
            return {name: value / total for name, value in sorted(parsed.items())}

    zimo_probability = options.get("expected_zimo_probability", data.get("expected_zimo_probability", 0.5))
    try:
        zimo_probability = min(1.0, max(0.0, float(zimo_probability)))
    except (TypeError, ValueError):
        zimo_probability = 0.5
    return {"dianpao": 1.0 - zimo_probability, "zimo": zimo_probability}


def _with_pending_discard(data: dict, winner: int, pending_discard: int | None) -> dict:
    scenario = dict(data)
    if pending_discard != HONGZHONG:
        return scenario
    played_cards = [list(cards or []) for cards in (data.get("played_cards") or [])]
    player_count = max(_player_count(data), winner + 1)
    if len(played_cards) < player_count:
        played_cards.extend([] for _ in range(player_count - len(played_cards)))
    played_cards[winner].append(HONGZHONG)
    scenario["played_cards"] = played_cards
    return scenario


def _winner_payment_count(win_type: str, player_count: int) -> int:
    if win_type in {"zimo", "gangkai", "qianggang"}:
        return max(0, player_count - 1)
    return 1


def _placeholder_payer(winner: int, player_count: int) -> int:
    return (winner + 1) % max(2, player_count)


def _base_hu_score(data: dict) -> float:
    options = _strategy_options(data)
    for source in (options, data, data.get("room_options") or {}, data.get("game_options") or {}):
        value = source.get("expected_base_hu_score")
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return 1.0


def _expected_zama_gain(data: dict) -> float:
    value = _strategy_options(data).get("expected_zama_gain", data.get("expected_zama_gain", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _strategy_options(data: dict) -> Mapping[str, object]:
    options = data.get("strategy_options")
    return options if isinstance(options, Mapping) else {}


def _normalize_win_type(value: object) -> str | None:
    text = str(value).strip().lower()
    aliases = {
        "zimo": "zimo",
        "self_draw": "zimo",
        "自摸": "zimo",
        "dianpao": "dianpao",
        "点炮": "dianpao",
        "gangkai": "gangkai",
        "gang_kai": "gangkai",
        "杠开": "gangkai",
        "qianggang": "qianggang",
        "抢杠": "qianggang",
    }
    return aliases.get(text)


def _positive_number(value: object) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _player_count(data: dict) -> int:
    for field in ("player_hand_cards", "played_cards", "player_peng_cards"):
        players = data.get(field)
        if isinstance(players, list) and players:
            return len(players)
    return 4
