"""Jiujiang Hongzhong Mahjong AI package."""

from .api import get_action, round_end
from .hu import HuOptions, can_hu
from .ting import is_ting, ting_discards, winning_tile_counts
from .tiles import HONGZHONG

__all__ = [
    "HONGZHONG",
    "HuOptions",
    "can_hu",
    "get_action",
    "is_ting",
    "round_end",
    "ting_discards",
    "winning_tile_counts",
]
