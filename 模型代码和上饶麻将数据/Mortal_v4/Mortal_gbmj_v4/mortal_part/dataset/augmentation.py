from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations

from mortal_part.mjai.event import Event
from mortal_part.tile import Tile


@dataclass(frozen=True)
class AugmentationSpec:
    suit_perm: tuple[int, int, int] = (0, 1, 2)
    mirror: bool = False
    name: str = "identity"


def build_augmentation_specs(factor: int):
    factor = int(factor)
    suit_perms = list(permutations((0, 1, 2)))

    if factor == 1:
        return [AugmentationSpec()]
    if factor == 2:
        return [
            AugmentationSpec((0, 1, 2), False, "identity"),
            AugmentationSpec((0, 1, 2), True, "mirror"),
        ]
    if factor == 6:
        return [
            AugmentationSpec(tuple(perm), False, f"suitperm_{idx}")
            for idx, perm in enumerate(suit_perms)
        ]
    if factor == 12:
        specs = []
        for idx, perm in enumerate(suit_perms):
            specs.append(AugmentationSpec(tuple(perm), False, f"suitperm_{idx}"))
            specs.append(AugmentationSpec(tuple(perm), True, f"suitperm_{idx}_mirror"))
        return specs
    raise ValueError(f"unsupported augmentation_factor={factor}; expected one of 1, 2, 6, 12")


def remap_tile_id(tile_id: int, spec: AugmentationSpec):
    if tile_id >= 27 or tile_id < 0:
        return tile_id
    suit = tile_id // 9
    rank = tile_id % 9
    new_suit = spec.suit_perm[suit]
    new_rank = 8 - rank if spec.mirror else rank
    return new_suit * 9 + new_rank


def remap_tile(tile: Tile, spec: AugmentationSpec):
    return Tile(remap_tile_id(tile.id, spec))


def _remap_tile_list(tiles, spec: AugmentationSpec):
    return [remap_tile(tile, spec) for tile in tiles]


def apply_augmentation(events, spec: AugmentationSpec):
    if spec.name == "identity":
        return events

    augmented = deepcopy(events)
    for wrapper in augmented:
        event = wrapper.event
        if isinstance(event, (Event.Tsumo, Event.Dahai)):
            event.tile = remap_tile(event.tile, spec)
        elif isinstance(event, Event.Chi):
            event.tile = remap_tile(event.tile, spec)
            event.consumed = _remap_tile_list(event.consumed, spec)
        elif isinstance(event, Event.Pon):
            event.tile = remap_tile(event.tile, spec)
            event.consumed = _remap_tile_list(event.consumed, spec)
        elif isinstance(event, Event.MinGang):
            event.tile = remap_tile(event.tile, spec)
            event.consumed = _remap_tile_list(event.consumed, spec)
        elif isinstance(event, Event.AnGang):
            event.consumed = _remap_tile_list(event.consumed, spec)
        elif isinstance(event, Event.BuGang):
            event.tile = remap_tile(event.tile, spec)
            event.consumed = _remap_tile_list(event.consumed, spec)
        elif isinstance(event, Event.Hu):
            event.tile = remap_tile(event.tile, spec)
        elif isinstance(event, Event.StartKyoku):
            event.tehais = [
                _remap_tile_list(hand, spec)
                for hand in event.tehais
            ]
        elif isinstance(event, Event.BotStartKyoku):
            event.tehais = _remap_tile_list(event.tehais, spec)
        elif isinstance(event, Event.BuHua):
            if hasattr(event, "replacement") and isinstance(event.replacement, Tile):
                event.replacement = remap_tile(event.replacement, spec)
    return augmented
