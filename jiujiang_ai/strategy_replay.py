"""对同一批局面离线重放多套决策策略，生成可比较的 A/B 结果。"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from .api import get_action


SUPPORTED_STRATEGIES = ("heuristic", "two_ply", "multi_route", "multi_route_tree")


def compare_strategy_snapshots(
    snapshots: Iterable[dict],
    *,
    strategies: tuple[str, ...] = SUPPORTED_STRATEGIES,
) -> dict:
    """对每个局面调用指定策略，并汇总策略一致率与各自推荐分布。"""
    _validate_strategies(strategies)
    comparisons: list[dict] = []
    for index, snapshot in enumerate(snapshots):
        actions = {
            strategy: list(get_action(_with_strategy(snapshot, strategy)))
            for strategy in strategies
        }
        comparisons.append(
            {
                "snapshot_id": snapshot.get("snapshot_id", str(index)),
                "actions": actions,
                "agreement": len({tuple(action[1]) for action in actions.values()}) == 1
                and len({action[0] for action in actions.values()}) == 1,
            }
        )
    return {
        "summary": _summary(comparisons, strategies),
        "comparisons": comparisons,
    }


def _with_strategy(snapshot: dict, strategy: str) -> dict:
    data = deepcopy(snapshot)
    room_options = dict(data.get("room_options") or {})
    room_options.update(
        {
            "search_tree_enabled": False,
            "multi_route_enabled": False,
            "multi_route_tree_enabled": False,
            "two_ply_search_enabled": False,
            "decision_log_enabled": False,
        }
    )
    if strategy == "two_ply":
        room_options["two_ply_search_enabled"] = True
    elif strategy == "multi_route":
        room_options["multi_route_enabled"] = True
    elif strategy == "multi_route_tree":
        room_options["multi_route_tree_enabled"] = True
    data["room_options"] = room_options
    return data


def _summary(comparisons: list[dict], strategies: tuple[str, ...]) -> dict:
    action_counts = {strategy: {} for strategy in strategies}
    for comparison in comparisons:
        for strategy, action in comparison["actions"].items():
            key = f"{action[0]}:{','.join(str(card) for card in action[1])}"
            action_counts[strategy][key] = action_counts[strategy].get(key, 0) + 1
    snapshot_count = len(comparisons)
    agreement_count = sum(bool(item["agreement"]) for item in comparisons)
    return {
        "snapshot_count": snapshot_count,
        "strategies": list(strategies),
        "agreement_count": agreement_count,
        "agreement_rate": agreement_count / snapshot_count if snapshot_count else 0.0,
        "action_counts": action_counts,
    }


def _validate_strategies(strategies: tuple[str, ...]) -> None:
    if not strategies:
        raise ValueError("at least one strategy is required")
    unsupported = set(strategies) - set(SUPPORTED_STRATEGIES)
    if unsupported:
        raise ValueError(f"unsupported strategies: {sorted(unsupported)!r}")
