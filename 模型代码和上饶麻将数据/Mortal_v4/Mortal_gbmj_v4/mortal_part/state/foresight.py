import importlib
import math
import struct
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPP_EXTENSION_DIR = PROJECT_ROOT / "foresight_cpp"
CPP_TABLE_MAX_TARGETS = 12000
CPP_CACHE_SIZE = 50000
_CPP_MODULE = None
_CPP_IMPORT_TRIED = False
_CPP_DISABLED = False

FORESIGHT_ROUTE_PLANES = 7
FORESIGHT_TOP_DISCARD_PLANES = 4
FORESIGHT_PLANES = FORESIGHT_ROUTE_PLANES + FORESIGHT_TOP_DISCARD_PLANES
# [V4 RL closed-loop] In obs_repr the 11 foresight planes are inserted after
# the visible/progress core.  RL reward and critic read these exact planes so
# actor input, potential shaping, and critic features use one shared signal.
FORESIGHT_OBS_START_INDEX = 161
ROUTE_DISTANCE_ALPHA = 1.6
ROUTE_TOP_K = 4
ROUTE_TOPK_DECAY = 0.5
ROUTE_MAX_DISTANCE = 6.0
ROUTE_GROUP_WEIGHTS: Tuple[float, ...] = (6.0, 5.5, 7.0, 7.5, 7.5, 10.0, 8.0)


class FanRouteSpec:
    __slots__ = ("name", "fan", "table_file")

    def __init__(self, name: str, fan: int, table_file: str = ""):
        self.name = name
        self.fan = fan
        self.table_file = table_file


# [V4 foresight feature] The route set mirrors the high-frequency main-fan
# tables used by the RL potential reward.  The feature encoder groups them into
# seven semantic route planes so the actor sees direction without receiving a
# large, noisy per-fan vector.
FAN_ROUTE_SPECS: Tuple[FanRouteSpec, ...] = (
    FanRouteSpec("mixed_shifted_chows", 6, "gbmj_mixed_shifted_chows.bin"),
    FanRouteSpec("all_types", 6, "gbmj_all_types.bin"),
    FanRouteSpec("mixed_triple_chows", 8, "gbmj_mixed_triple_chows.bin"),
    FanRouteSpec("mixed_straight", 8, "gbmj_mixed_straight.bin"),
    FanRouteSpec("half_flush", 6, "gbmj_half_flush.bin"),
    FanRouteSpec("pure_straight", 16, "gbmj_pure_straight.bin"),
    FanRouteSpec("all_pungs", 6, "gbmj_all_pungs.bin"),
    FanRouteSpec("pure_shifted_chows", 16, "gbmj_pure_shifted_chows.bin"),
    FanRouteSpec("seven_pairs", 24, ""),
    FanRouteSpec("outside_hand", 4, "gbmj_outside_hand.bin"),
    FanRouteSpec("full_flush", 24, "gbmj_full_flush.bin"),
    FanRouteSpec("no_fan", 8, ""),
    FanRouteSpec("greater_than_five", 12, "gbmj_greater_than_five.bin"),
    FanRouteSpec("knitted_straight", 12, "gbmj_knitted_straight.bin"),
    FanRouteSpec("less_than_five", 12, "gbmj_less_than_five.bin"),
    FanRouteSpec("all_unrelated", 12, "gbmj_all_unrelated.bin"),
    FanRouteSpec("all_claimed", 6, ""),
)

FAN_BY_ROUTE = {spec.name: spec.fan for spec in FAN_ROUTE_SPECS}
WEIGHT_BY_ROUTE = {
    "mixed_shifted_chows": 10.0,
    "all_types": 8.0,
    "mixed_triple_chows": 8.0,
    "mixed_straight": 8.0,
    "half_flush": 7.5,
    "pure_straight": 7.5,
    "all_pungs": 7.0,
    "pure_shifted_chows": 6.5,
    "seven_pairs": 6.0,
    "outside_hand": 6.0,
    "full_flush": 5.5,
    "no_fan": 5.0,
    "greater_than_five": 5.5,
    "knitted_straight": 5.5,
    "less_than_five": 5.5,
    "all_unrelated": 5.3,
    "all_claimed": 5.0,
}

ROUTE_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("seven_pairs", ("seven_pairs",)),
    ("knitted_unrelated", ("knitted_straight", "all_unrelated")),
    ("pung_route", ("all_pungs", "all_claimed")),
    ("flush_route", ("half_flush", "full_flush")),
    ("pure_sequence", ("pure_straight", "pure_shifted_chows")),
    ("mixed_sequence", ("mixed_shifted_chows", "mixed_triple_chows", "mixed_straight")),
    ("terminal_value", ("all_types", "outside_hand", "greater_than_five", "less_than_five", "no_fan")),
)


def _clip_route_value(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def foresight_info_from_features(
    route_values: Sequence[float],
    top_discards: Sequence[int] = (),
) -> Dict:
    """[V4 RL closed-loop] Convert 7 route planes into one potential summary.

    The supervised actor receives route closeness as 1 / (distance + 1).  For
    PPO we reuse that exact representation and reconstruct a paper-style score:

        Score_group = sqrt(group_fan * route_closeness)

    This keeps reward shaping and critic features tied to the same information
    the actor actually observes, instead of a second Python-only estimator.
    """
    routes = [_clip_route_value(value) for value in list(route_values)[:FORESIGHT_ROUTE_PLANES]]
    if len(routes) < FORESIGHT_ROUTE_PLANES:
        routes += [0.0] * (FORESIGHT_ROUTE_PLANES - len(routes))

    values = []
    for weight, value in zip(ROUTE_GROUP_WEIGHTS, routes):
        if value > 0.0:
            values.append(weight * (value ** ROUTE_DISTANCE_ALPHA))
    values.sort(reverse=True)
    score = 0.0
    for idx, value in enumerate(values[:ROUTE_TOP_K]):
        score += (ROUTE_TOPK_DECAY ** idx) * value

    best_route = max(routes) if routes else 0.0
    best_distance = (1.0 / best_route - 1.0) if best_route > 1.0e-8 else 14.0
    top = [int(tile_id) for tile_id in list(top_discards)[:FORESIGHT_TOP_DISCARD_PLANES] if 0 <= int(tile_id) < 34]

    return {
        "potential_source": "foresight",
        "potential_score": float(score),
        "potential_best_distance": float(min(14.0, max(0.0, best_distance))),
        "potential_route_count": float(len(routes)),
        "potential_loaded_tables": 0.0,
        "foresight_route_values": tuple(float(value) for value in routes),
        "foresight_best_route": float(best_route),
        "foresight_mean_route": float(sum(routes) / max(1, len(routes))),
        "foresight_good_route_count": float(sum(1 for value in routes if value >= 0.25)),
        "foresight_top_discards": tuple(top),
    }


def foresight_info_from_obs(obs) -> Dict:
    """[V4 RL closed-loop] Read potential summary from already encoded obs."""
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < FORESIGHT_OBS_START_INDEX + FORESIGHT_PLANES:
        return {}

    route_values = [
        float(arr[FORESIGHT_OBS_START_INDEX + idx].mean())
        for idx in range(FORESIGHT_ROUTE_PLANES)
    ]

    top_discards: List[int] = []
    top_start = FORESIGHT_OBS_START_INDEX + FORESIGHT_ROUTE_PLANES
    for plane_idx in range(FORESIGHT_TOP_DISCARD_PLANES):
        plane = arr[top_start + plane_idx]
        positions = np.argwhere(plane > 0.5)
        if positions.size == 0:
            continue
        row, col = positions[0]
        tile_id = int(row) * 9 + int(col)
        if 0 <= tile_id < 34:
            top_discards.append(tile_id)

    return foresight_info_from_features(route_values, top_discards)


def _tile_id(tile_or_id) -> int:
    if hasattr(tile_or_id, "id"):
        return int(tile_or_id.id)
    return int(tile_or_id)


def _normalize_hand(hand: Sequence[int]) -> Tuple[int, ...]:
    values = list(hand)[:34]
    if len(values) < 34:
        values += [0] * (34 - len(values))
    return tuple(max(0, min(4, int(value))) for value in values)


def _safe_table_dir(value: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _sequence_requirement(suit: int, start: int) -> Dict[int, int]:
    base = suit * 9 + start
    return {base: 1, base + 1: 1, base + 2: 1}


def _merge_req(*reqs: Dict[int, int]) -> Dict[int, int]:
    merged: Dict[int, int] = {}
    for req in reqs:
        for tid, count in req.items():
            merged[tid] = merged.get(tid, 0) + count
    return merged


def _missing_for_req(hand: Tuple[int, ...], req: Dict[int, int]) -> int:
    return sum(max(0, need - hand[tid]) for tid, need in req.items())


def _open_meld_count(state) -> int:
    return (
        len(getattr(state, "chis", ()))
        + len(getattr(state, "pons", ()))
        + len(getattr(state, "minkans", ()))
        + len(getattr(state, "ankans", ()))
    )


def concealed_counts_from_state(state) -> Tuple[int, ...]:
    """[V4 foresight C++ batch] Count only concealed hand tiles."""
    counts = list(getattr(state, "handcards", [0] * 34))[:34]
    if len(counts) < 34:
        counts += [0] * (34 - len(counts))
    return _normalize_hand(counts)


def fixed_melds_from_state(state) -> Tuple[Tuple[int, ...], ...]:
    """[V4 foresight C++ batch] Encode own locked melds for exact route distance.

    The old Python fallback added meld tiles back into a flat 34-count hand,
    which was fast to write but loses shape information.  The C++ path receives
    fixed melds separately so routes such as PureStraight/AllPungs cannot
    regroup an already-open Chi/Pon/Kong into a different structure.
    """
    fixed: List[Tuple[int, ...]] = []

    for middle in getattr(state, "chis", []):
        mid = _tile_id(middle)
        if 0 < mid < 27 and mid % 9 not in (0, 8):
            suit_start = (mid // 9) * 9
            if suit_start <= mid - 1 and mid + 1 < suit_start + 9:
                fixed.append((mid - 1, mid, mid + 1))

    for tile in getattr(state, "pons", []):
        tid = _tile_id(tile)
        if 0 <= tid < 34:
            fixed.append((tid, tid, tid))

    for attr in ("minkans", "ankans"):
        for tile in getattr(state, attr, []):
            tid = _tile_id(tile)
            if 0 <= tid < 34:
                fixed.append((tid, tid, tid, tid))

    return tuple(fixed)


def route_counts_from_state(state) -> Tuple[int, ...]:
    """[V4 foresight feature] Count concealed hand plus own fixed melds.

    The route potential should describe the route already being built.  A
    consumed Chi/Pon/Kong no longer exists in handcards, so we add own meld
    tiles back before route evaluation.
    """
    counts = list(getattr(state, "handcards", [0] * 34))[:34]
    if len(counts) < 34:
        counts += [0] * (34 - len(counts))

    for middle in getattr(state, "chis", []):
        mid = _tile_id(middle)
        if 0 < mid < 27 and mid % 9 not in (0, 8):
            suit_start = (mid // 9) * 9
            if suit_start <= mid - 1 and mid + 1 < suit_start + 9:
                counts[mid - 1] += 1
                counts[mid] += 1
                counts[mid + 1] += 1

    for attr in ("pons", "minkans", "ankans"):
        for tile in getattr(state, attr, []):
            tid = _tile_id(tile)
            if 0 <= tid < 34:
                counts[tid] += 3

    return _normalize_hand(counts)


def _get_cpp_module():
    """[V4 foresight C++] Prefer the compiled extension when available."""
    global _CPP_MODULE, _CPP_IMPORT_TRIED
    if _CPP_DISABLED:
        return None
    if _CPP_IMPORT_TRIED:
        return _CPP_MODULE
    _CPP_IMPORT_TRIED = True
    if CPP_EXTENSION_DIR.exists():
        path = str(CPP_EXTENSION_DIR)
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        _CPP_MODULE = importlib.import_module("gbmj_foresight_cpp")
    except Exception:
        _CPP_MODULE = None
    return _CPP_MODULE


class GbmjForesightCalculator:
    """[V4 foresight feature] Bounded D=1 lookahead route evaluator.

    This is intentionally shallower than the old SPCalculator expectation
    search: every legal discard is scored by the resulting route potential
    only.  It is cheap enough for supervised data loading and gives the model
    the paper-like "local best discard" signal without a deep Python DFS.
    """

    def __init__(
        self,
        table_dir: str = "data",
        cache_size: int = 200000,
        table_max_targets: int = 12000,
    ):
        self.table_dir = _safe_table_dir(table_dir)
        self.cache_size = max(0, int(cache_size))
        self.table_max_targets = max(0, int(table_max_targets))
        self.cache: "OrderedDict[Tuple[Tuple[int, ...], bool], Tuple[float, Dict[str, float], float]]" = OrderedDict()
        self.tables: Dict[str, Tuple[int, np.ndarray]] = {}
        self._load_small_tables()

    def _load_small_tables(self) -> None:
        if self.table_max_targets <= 0:
            return
        for spec in FAN_ROUTE_SPECS:
            if not spec.table_file:
                continue
            path = self.table_dir / spec.table_file
            if not path.exists():
                continue
            with path.open("rb") as f:
                header = f.read(24)
                if len(header) != 24:
                    continue
                magic, version, fan, target_count, record_size = struct.unpack("<8sIIII", header)
                if magic != b"GBMJFT01" or version != 1 or record_size != 34:
                    continue
                if target_count > self.table_max_targets:
                    continue
                raw = f.read(target_count * record_size)
            if len(raw) != target_count * record_size:
                continue
            targets = np.frombuffer(raw, dtype=np.uint8).reshape(target_count, 34).astype(np.int16)
            self.tables[spec.name] = (int(fan), targets)

    def evaluate(self, hand: Sequence[int], has_open_meld: bool = False) -> Tuple[float, Dict[str, float], float]:
        key = (_normalize_hand(hand), bool(has_open_meld))
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached

        distances = self._fast_distances(key[0])
        distances.update(self._table_distances(key[0]))

        # Seven pairs is concealed-only; avoid letting an open hand get an
        # artificial seven-pairs direction from fixed triplets.
        if has_open_meld:
            distances["seven_pairs"] = 14

        route_values: List[float] = []
        best_distance = 99.0
        for spec in FAN_ROUTE_SPECS:
            if spec.name not in distances:
                continue
            distance = max(0.0, float(distances[spec.name]))
            if distance <= ROUTE_MAX_DISTANCE:
                weight = float(WEIGHT_BY_ROUTE.get(spec.name, spec.fan))
                route_values.append(weight / ((distance + 1.0) ** ROUTE_DISTANCE_ALPHA))
            best_distance = min(best_distance, distance)
        if best_distance == 99.0:
            best_distance = 14.0

        route_values.sort(reverse=True)
        total = sum((ROUTE_TOPK_DECAY ** idx) * value for idx, value in enumerate(route_values[:ROUTE_TOP_K]))

        result = (float(total), {name: float(value) for name, value in distances.items()}, float(best_distance))
        if self.cache_size > 0:
            self.cache[key] = result
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return result

    def route_closeness(self, hand: Sequence[int], has_open_meld: bool = False) -> List[float]:
        _, distances, _ = self.evaluate(hand, has_open_meld=has_open_meld)
        values: List[float] = []
        for _, route_names in ROUTE_GROUPS:
            valid = [distances[name] for name in route_names if name in distances]
            if not valid:
                values.append(0.0)
            else:
                values.append(1.0 / (min(valid) + 1.0))
        return values

    def top_discards(
        self,
        hand: Sequence[int],
        discard_candidates: Sequence[bool],
        has_open_meld: bool = False,
        k: int = FORESIGHT_TOP_DISCARD_PLANES,
    ) -> List[int]:
        base = list(_normalize_hand(hand))
        scored: List[Tuple[float, float, int, int]] = []
        for tid, flag in enumerate(list(discard_candidates)[:34]):
            if not flag or base[tid] <= 0:
                continue
            next_hand = list(base)
            next_hand[tid] -= 1
            score, _, best_distance = self.evaluate(next_hand, has_open_meld=has_open_meld)
            scored.append((score, -best_distance, -tid, tid))
        scored.sort(reverse=True)
        return [tid for _, _, _, tid in scored[:k]]

    def _table_distances(self, hand: Tuple[int, ...]) -> Dict[str, int]:
        if not self.tables:
            return {}
        hand_arr = np.asarray(hand, dtype=np.int16)
        result: Dict[str, int] = {}
        for name, (_, targets) in self.tables.items():
            result[name] = int(np.maximum(targets - hand_arr, 0).sum(axis=1).min())
        return result

    def _fast_distances(self, hand: Tuple[int, ...]) -> Dict[str, int]:
        hand_sum = sum(hand)
        suits = [range(0, 9), range(9, 18), range(18, 27)]
        suit_counts = [sum(hand[i] for i in suit) for suit in suits]
        honor_count = sum(hand[27:34])

        result: Dict[str, int] = {}

        pair_count = sum(1 for count in hand if count >= 2)
        kind_count = sum(1 for count in hand if count > 0)
        result["seven_pairs"] = max(0, 7 - pair_count) + max(0, 7 - kind_count)

        triplet_costs = [max(0, 3 - count) for count in hand]
        best_pungs = 14
        for pair_tid in range(34):
            pair_cost = max(0, 2 - hand[pair_tid])
            rest = sorted(triplet_costs[i] for i in range(34) if i != pair_tid)
            best_pungs = min(best_pungs, pair_cost + sum(rest[:4]))
        result["all_pungs"] = int(best_pungs)

        result["full_flush"] = int(max(0, hand_sum - max(suit_counts)))
        result["half_flush"] = int(max(0, hand_sum - max(count + honor_count for count in suit_counts)))

        low_ids = set(list(range(0, 4)) + list(range(9, 13)) + list(range(18, 22)))
        high_ids = set(list(range(5, 9)) + list(range(14, 18)) + list(range(23, 27)))
        result["less_than_five"] = int(sum(hand[i] for i in range(34) if i not in low_ids))
        result["greater_than_five"] = int(sum(hand[i] for i in range(34) if i not in high_ids))

        type_missing = 0
        if sum(hand[0:9]) <= 0:
            type_missing += 1
        if sum(hand[9:18]) <= 0:
            type_missing += 1
        if sum(hand[18:27]) <= 0:
            type_missing += 1
        if sum(hand[27:31]) <= 0:
            type_missing += 1
        if sum(hand[31:34]) <= 0:
            type_missing += 1
        result["all_types"] = type_missing

        result["mixed_triple_chows"] = min(
            _missing_for_req(
                hand,
                _merge_req(
                    _sequence_requirement(0, start),
                    _sequence_requirement(1, start),
                    _sequence_requirement(2, start),
                ),
            )
            for start in range(7)
        )

        starts = [0, 3, 6]
        mixed_straight_best = 9
        for suit_a in range(3):
            for suit_b in range(3):
                if suit_b == suit_a:
                    continue
                for suit_c in range(3):
                    if suit_c == suit_a or suit_c == suit_b:
                        continue
                    req = _merge_req(
                        _sequence_requirement(suit_a, starts[0]),
                        _sequence_requirement(suit_b, starts[1]),
                        _sequence_requirement(suit_c, starts[2]),
                    )
                    mixed_straight_best = min(mixed_straight_best, _missing_for_req(hand, req))
        result["mixed_straight"] = mixed_straight_best

        result["pure_straight"] = min(
            _missing_for_req(
                hand,
                _merge_req(
                    _sequence_requirement(suit, 0),
                    _sequence_requirement(suit, 3),
                    _sequence_requirement(suit, 6),
                ),
            )
            for suit in range(3)
        )

        mixed_shifted_best = 9
        for base_start in range(5):
            for suit_a in range(3):
                for suit_b in range(3):
                    if suit_b == suit_a:
                        continue
                    for suit_c in range(3):
                        if suit_c == suit_a or suit_c == suit_b:
                            continue
                        req = _merge_req(
                            _sequence_requirement(suit_a, base_start),
                            _sequence_requirement(suit_b, base_start + 1),
                            _sequence_requirement(suit_c, base_start + 2),
                        )
                        mixed_shifted_best = min(mixed_shifted_best, _missing_for_req(hand, req))
        result["mixed_shifted_chows"] = mixed_shifted_best

        result["pure_shifted_chows"] = min(
            _missing_for_req(
                hand,
                _merge_req(
                    _sequence_requirement(suit, base_start),
                    _sequence_requirement(suit, base_start + 1),
                    _sequence_requirement(suit, base_start + 2),
                ),
            )
            for suit in range(3)
            for base_start in range(5)
        )

        terminal_honor_ids = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
        result["outside_hand"] = int(max(0, hand_sum - sum(hand[i] for i in terminal_honor_ids)))

        return result


@lru_cache(maxsize=1)
def get_foresight_calculator() -> GbmjForesightCalculator:
    return GbmjForesightCalculator()


def compute_foresight_features(state) -> Tuple[List[float], List[int]]:
    """[V4 foresight feature] Return 7 route planes and 4 top discard tiles."""
    global _CPP_DISABLED
    concealed_hand = concealed_counts_from_state(state)
    fixed_melds = fixed_melds_from_state(state)
    hand = route_counts_from_state(state)
    has_open_meld = _open_meld_count(state) > 0
    cans = getattr(state, "last_cans", None)
    can_discard = cans is not None and getattr(cans, "can_discard", False)
    discard_candidates = state.discard_candidates() if can_discard else [False] * 34

    cpp_module = _get_cpp_module()
    if cpp_module is not None:
        try:
            route_values, top_discards = cpp_module.compute_foresight(
                concealed_hand,
                fixed_melds,
                discard_candidates,
                str(PROJECT_ROOT / "data"),
                CPP_CACHE_SIZE,
            )
            return list(route_values), list(top_discards)
        except Exception:
            # [V4 foresight C++] If the extension was built for the wrong
            # Python ABI or table input is malformed, fall back once instead of
            # killing long-running training jobs.
            _CPP_DISABLED = True

    calculator = get_foresight_calculator()
    route_values = calculator.route_closeness(hand, has_open_meld=has_open_meld)

    if not can_discard:
        return route_values, []

    return route_values, calculator.top_discards(
        hand,
        discard_candidates,
        has_open_meld=has_open_meld,
        k=FORESIGHT_TOP_DISCARD_PLANES,
    )
