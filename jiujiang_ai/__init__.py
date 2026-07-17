"""Jiujiang Hongzhong Mahjong AI package."""

from .api import get_action, round_end
from .decision_engine import MultiRouteDiscardDecision, choose_discard as choose_multi_route_discard
from .decision_log import DEFAULT_DECISION_LOG_PATH, append_decision_log, load_decision_logs
from .evaluator import DiscardDecision, TwoPlyDiscardDecision, choose_two_ply_discard
from .expected_value import ExpectedWinValue, estimate_win_value
from .hand_routes import HandRoute, TaatsuKind, enumerate_hand_routes
from .hu import HuOptions, can_hu
from .round_flow import detect_round_flow, resolve_next_dealer
from .search_tree import choose_discard as choose_tree_discard
from .search_tree import expand_discard_tree
from .settlement import calculate_buy_score, calculate_gang_score, calculate_hu_score, calculate_total_score
from .stats import (
    DEFAULT_ROUND_LOG_PATH,
    append_round_log,
    get_stats,
    load_round_logs,
    record_round_end,
    reset_stats,
    summarize_match_report,
    summarize_rounds,
)
from .strategy_replay import compare_strategy_snapshots
from .ting import is_ting, ting_discards, winning_tile_counts
from .tiles import HONGZHONG
from .win_context import WinContext, detect_win_context
from .zama import calculate_zama_score

__all__ = [
    "DEFAULT_ROUND_LOG_PATH",
    "DEFAULT_DECISION_LOG_PATH",
    "DiscardDecision",
    "ExpectedWinValue",
    "HONGZHONG",
    "HandRoute",
    "HuOptions",
    "MultiRouteDiscardDecision",
    "TwoPlyDiscardDecision",
    "TaatsuKind",
    "WinContext",
    "append_round_log",
    "append_decision_log",
    "can_hu",
    "calculate_buy_score",
    "calculate_gang_score",
    "calculate_hu_score",
    "calculate_total_score",
    "calculate_zama_score",
    "compare_strategy_snapshots",
    "choose_tree_discard",
    "choose_multi_route_discard",
    "choose_two_ply_discard",
    "detect_round_flow",
    "detect_win_context",
    "expand_discard_tree",
    "enumerate_hand_routes",
    "estimate_win_value",
    "get_stats",
    "get_action",
    "is_ting",
    "load_round_logs",
    "load_decision_logs",
    "record_round_end",
    "reset_stats",
    "resolve_next_dealer",
    "round_end",
    "summarize_match_report",
    "summarize_rounds",
    "ting_discards",
    "winning_tile_counts",
]
