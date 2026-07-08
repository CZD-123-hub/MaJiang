"""Jiujiang Hongzhong Mahjong AI package."""

from .api import get_action, round_end
from .hu import HuOptions, can_hu
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

__all__ = [
    "DEFAULT_ROUND_LOG_PATH",
    "HONGZHONG",
    "HuOptions",
    "append_round_log",
    "can_hu",
    "get_stats",
    "get_action",
    "is_ting",
    "load_round_logs",
    "record_round_end",
    "reset_stats",
    "round_end",
    "summarize_match_report",
    "summarize_rounds",
    "ting_discards",
    "winning_tile_counts",
]
