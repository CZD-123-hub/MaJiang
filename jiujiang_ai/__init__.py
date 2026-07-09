"""Jiujiang Hongzhong Mahjong AI package."""

from .api import get_action, round_end
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
from .ting import is_ting, ting_discards, winning_tile_counts
from .tiles import HONGZHONG
from .win_context import WinContext, detect_win_context
from .zama import calculate_zama_score

__all__ = [
    "DEFAULT_ROUND_LOG_PATH",
    "HONGZHONG",
    "HuOptions",
    "WinContext",
    "append_round_log",
    "can_hu",
    "calculate_buy_score",
    "calculate_gang_score",
    "calculate_hu_score",
    "calculate_total_score",
    "calculate_zama_score",
    "choose_tree_discard",
    "detect_round_flow",
    "detect_win_context",
    "expand_discard_tree",
    "get_stats",
    "get_action",
    "is_ting",
    "load_round_logs",
    "record_round_end",
    "reset_stats",
    "resolve_next_dealer",
    "round_end",
    "summarize_match_report",
    "summarize_rounds",
    "ting_discards",
    "winning_tile_counts",
]
