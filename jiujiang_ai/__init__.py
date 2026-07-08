"""Jiujiang Hongzhong Mahjong AI package."""

from .api import get_action, round_end
from .hu import HuOptions, can_hu
from .stats import get_stats, reset_stats, summarize_match_report, summarize_rounds
from .ting import is_ting, ting_discards, winning_tile_counts
from .tiles import HONGZHONG

__all__ = [
    "HONGZHONG",
    "HuOptions",
    "can_hu",
    "get_stats",
    "get_action",
    "is_ting",
    "reset_stats",
    "round_end",
    "summarize_match_report",
    "summarize_rounds",
    "ting_discards",
    "winning_tile_counts",
]
