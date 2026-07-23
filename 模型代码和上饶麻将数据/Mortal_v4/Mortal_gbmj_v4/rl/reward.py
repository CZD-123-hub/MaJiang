import math
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from mortal_part.consts import (
    ADDKONG_BASE,
    ADDKONG_COUNT,
    ANKANG_BASE,
    ANKANG_COUNT,
    CHOW_BASE,
    CHOW_COUNT,
    DISCARD_BASE,
    DISCARD_COUNT,
    MINGGANG_BASE,
    MINGGANG_COUNT,
    PASS_INDEX,
    PUNG_BASE,
    PUNG_COUNT,
    WIN_INDEX,
)
from mortal_part.mjai.event import Event
from mortal_part.state.foresight import (
    FORESIGHT_ROUTE_PLANES,
    compute_foresight_features,
    foresight_info_from_features,
    foresight_info_from_obs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRITIC_FEATURE_DIM = 8


@dataclass(frozen=True)
class FanRouteSpec:
    name: str
    fan: int
    table_file: str = ""


# Real GBMJ fan values are kept for table compatibility only.  Dense route
# shaping uses ROUTE_FREQUENCY_WEIGHTS instead of rule fan values.
FAN_ROUTE_SPECS = [
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
    FanRouteSpec("greater_than_five", 12, "gbmj_greater_than_five.bin"),
    FanRouteSpec("knitted_straight", 12, "gbmj_knitted_straight.bin"),
    FanRouteSpec("less_than_five", 12, "gbmj_less_than_five.bin"),
    FanRouteSpec("all_unrelated", 12, "gbmj_all_unrelated.bin"),
    FanRouteSpec("all_claimed", 6, ""),
]


ROUTE_FREQUENCY_WEIGHTS: Dict[str, float] = {
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
    "greater_than_five": 5.5,
    "knitted_straight": 5.5,
    "less_than_five": 5.5,
    "all_unrelated": 5.3,
    "all_claimed": 5.0,
}


@dataclass
class MJRMRewardConfig:
    """[V4 potential reward] Potential-shaping reward based on main-fan routes.

    The previous MJ_RM three-stage rewards are intentionally disabled by
    default because opening/meld/ordinary-shanten bonuses were too easy to
    exploit.  The new dense part is:

        r_shape = lambda * (gamma * Score(s_next) - Score(s))

    where Score(s) is summed over high-frequency GBMJ fan routes.
    Terminal win/self-draw/deal-in rewards stay as sparse anchors.
    """

    enabled: bool = True
    enable_potential_reward: bool = True
    potential_table_dir: str = "data"
    potential_weight: float = 0.005
    potential_gamma: float = 0.98
    potential_terminal_beta: float = 0.0
    potential_cache_size: int = 50000
    potential_cache_clear_every: int = 1
    potential_table_max_targets: int = 12000
    potential_score_mode: str = "frequency_weighted_topk"
    potential_delta_mode: str = "discounted"
    potential_source: str = "tables"
    potential_score_scale: float = 4.0
    potential_score_clip: float = 2.0
    potential_reward_clip: float = 0.25
    potential_distance_alpha: float = 1.6
    potential_top_k: int = 4
    potential_topk_decay: float = 0.5
    potential_max_route_distance: float = 6.0
    potential_all_claimed_min_melds: int = 2
    route_distance_delta_weight: float = 0.03
    ordinary_distance_delta_weight: float = 0.0
    distance_delta_clip: float = 2.0
    dense_ready_reward: float = 0.015
    dense_near_ready_reward: float = 0.006
    dense_late_far_penalty: float = -0.008
    dense_late_push_penalty: float = -0.012
    dense_late_turn: int = 10
    dense_near_distance: float = 2.0
    dense_far_distance: float = 3.0
    score_delta_reward_scale: float = 40.0
    score_delta_reward_clip: float = 2.0
    terminal_redistribute_decay: float = 0.92
    dangerous_push_penalty: float = -0.08
    dangerous_push_turn: int = 10
    dangerous_push_best_distance: float = 1.0
    self_draw_reward: float = 2.0
    win_reward: float = 1.5
    deal_in_penalty: float = -2.0
    # [V4 RL outcome-aligned reward] Opponent wins also hurt in GBMJ scoring.
    # These terms close the previous reward loophole where "other player wins,
    # but I did not deal in" was treated as neutral.
    other_self_draw_penalty: float = -0.8
    other_ron_penalty: float = -0.4
    tenpai_reward: float = 0.3
    noten_penalty: float = -0.2


def critic_features_from_info(info: Dict) -> np.ndarray:
    """[V4 RL closed-loop] Build rollout-only critic features.

    The critic now consumes summaries reconstructed from the same 11 foresight
    planes used by the actor.  This avoids the previous split-brain setup where
    actor features, reward shaping, and critic features were computed by three
    slightly different route evaluators.
    """
    score = float(info.get("potential_score", 0.0))
    best_distance = float(info.get("potential_best_distance", 14.0))
    route_values = list(info.get("foresight_route_values", ()))
    if len(route_values) < FORESIGHT_ROUTE_PLANES:
        route_values += [0.0] * (FORESIGHT_ROUTE_PLANES - len(route_values))
    route_values = [min(1.0, max(0.0, float(value))) for value in route_values[:FORESIGHT_ROUTE_PLANES]]
    best_route = max(route_values) if route_values else 0.0
    mean_route = sum(route_values) / max(1, len(route_values))
    good_routes = sum(1 for value in route_values if value >= 0.25)
    top_count = len(info.get("foresight_top_discards", ()))
    turn = float(info.get("turn", 0.0))
    hand_sum = float(sum(info.get("hand", ())))

    return np.asarray(
        [
            min(1.0, max(0.0, score / 20.0)),
            min(1.0, max(0.0, best_distance / 14.0)),
            min(1.0, max(0.0, best_route)),
            min(1.0, max(0.0, mean_route)),
            min(1.0, max(0.0, good_routes / float(FORESIGHT_ROUTE_PLANES))),
            min(1.0, max(0.0, turn / 30.0)),
            min(1.0, max(0.0, hand_sum / 20.0)),
            min(1.0, max(0.0, top_count / 4.0)),
        ],
        dtype=np.float32,
    )


def _action_group(action: int) -> str:
    if action == PASS_INDEX:
        return "pass"
    if action == WIN_INDEX:
        return "hu"
    if DISCARD_BASE <= action < DISCARD_BASE + DISCARD_COUNT:
        return "discard"
    if PUNG_BASE <= action < PUNG_BASE + PUNG_COUNT:
        return "pon"
    if CHOW_BASE <= action < CHOW_BASE + CHOW_COUNT:
        return "chi"
    if (
        MINGGANG_BASE <= action < MINGGANG_BASE + MINGGANG_COUNT
        or ANKANG_BASE <= action < ANKANG_BASE + ANKANG_COUNT
        or ADDKONG_BASE <= action < ADDKONG_BASE + ADDKONG_COUNT
    ):
        return "kan"
    return "other"


def _normalize_hand(hand) -> Tuple[int, ...]:
    values = list(hand)[:34]
    if len(values) < 34:
        values += [0] * (34 - len(values))
    return tuple(max(0, min(4, int(x))) for x in values)


def _tile_id(tile_or_id) -> int:
    if hasattr(tile_or_id, "id"):
        return int(tile_or_id.id)
    return int(tile_or_id)


def _route_counts_from_state(state) -> Tuple[int, ...]:
    """[V4 potential reward] Count hand + own fixed melds for route scoring.

    Using concealed hand only would punish useful Chi/Pon/Kong, because the
    consumed tiles leave handcards.  Potential should describe the whole route
    the player is building, so own melds are added back as fixed completed sets.
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


def _meld_counts_from_state(state) -> Tuple[int, int]:
    chi_count = len(getattr(state, "chis", ()))
    pon_count = len(getattr(state, "pons", ()))
    minkan_count = len(getattr(state, "minkans", ()))
    ankan_count = len(getattr(state, "ankans", ()))
    any_meld_count = chi_count + pon_count + minkan_count + ankan_count
    claimed_meld_count = chi_count + pon_count + minkan_count
    return int(any_meld_count), int(claimed_meld_count)


def _ordinary_distance_from_state(state) -> int:
    try:
        value = int(getattr(state, "shanten", 14))
    except Exception:
        value = 14
    return max(0, min(14, value))


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


class GbmjFanPotential:
    """[V4 potential reward] Main-fan potential calculator.

    Large exact target tables are expensive to scan inside PPO rollout, so this
    class uses fast route estimators for broad fans and only loads small exact
    tables when target_count <= potential_table_max_targets.  Increase that
    config if you want a slower but more table-driven experiment.
    """

    def __init__(
        self,
        table_dir: str,
        cache_size: int = 200000,
        table_max_targets: int = 12000,
        score_mode: str = "frequency_weighted_topk",
        distance_alpha: float = 1.6,
        top_k: int = 4,
        topk_decay: float = 0.5,
        max_route_distance: float = 6.0,
        all_claimed_min_melds: int = 2,
    ):
        self.table_dir = _safe_table_dir(table_dir)
        self.cache_size = max(0, int(cache_size))
        self.table_max_targets = max(0, int(table_max_targets))
        self.score_mode = str(score_mode)
        self.distance_alpha = max(0.1, float(distance_alpha))
        self.top_k = max(1, int(top_k))
        self.topk_decay = min(1.0, max(0.0, float(topk_decay)))
        self.max_route_distance = float(max_route_distance)
        self.all_claimed_min_melds = max(0, int(all_claimed_min_melds))
        self.cache: "OrderedDict[Tuple, Tuple[float, Dict[str, float]]]" = OrderedDict()
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
            targets = np.frombuffer(raw, dtype=np.uint8).reshape(target_count, 34).astype(np.int16)
            self.tables[spec.name] = (int(fan), targets)

    def score(
        self,
        hand,
        open_meld_count: int = 0,
        claimed_meld_count: int = 0,
    ) -> Tuple[float, Dict[str, float]]:
        hand_key = _normalize_hand(hand)
        open_meld_count = max(0, int(open_meld_count))
        claimed_meld_count = max(0, int(claimed_meld_count))
        # Do not use ordinary shanten as route potential.  GBMJ needs 8 fan,
        # so plain fast-tenpai progress must not be rewarded as a target route.
        key = (hand_key, open_meld_count, claimed_meld_count)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached

        distances = self._fast_distances(
            hand_key,
            open_meld_count=open_meld_count,
            claimed_meld_count=claimed_meld_count,
        )
        distances.update(self._table_distances(hand_key))

        if open_meld_count > 0:
            distances["seven_pairs"] = 14

        total = 0.0
        best_name = ""
        best_distance = 99.0
        best_value = -1.0
        candidates: List[Tuple[float, str, float]] = []
        weighted_topk = self.score_mode in ("frequency_weighted_topk", "route_weight_topk", "weighted_topk")
        for spec in FAN_ROUTE_SPECS:
            if spec.name not in distances:
                continue
            distance = max(0.0, float(distances[spec.name]))
            if weighted_topk and self.max_route_distance >= 0.0 and distance > self.max_route_distance:
                continue
            value = self._route_value(spec, distance)
            if value <= 0.0:
                continue
            candidates.append((value, spec.name, distance))
            if not weighted_topk:
                total += value
            if value > best_value:
                best_value = value
                best_name = spec.name
                best_distance = distance

        selected = candidates
        if weighted_topk:
            selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:self.top_k]
            total = 0.0
            for idx, (value, _, _) in enumerate(selected):
                total += (self.topk_decay ** idx) * value
            if selected:
                best_value, best_name, best_distance = selected[0]

        info = {
            "potential_score": float(total),
            "potential_best_distance": float(best_distance if best_name else 14.0),
            "potential_loaded_tables": float(len(self.tables)),
            "potential_route_count": float(len(distances)),
            "potential_selected_route_count": float(len(selected)),
            "potential_best_route_weight": float(ROUTE_FREQUENCY_WEIGHTS.get(best_name, 0.0)),
            "potential_best_route_value": float(best_value if best_name else 0.0),
        }
        if self.cache_size > 0:
            self.cache[key] = (float(total), info)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return float(total), info

    def _route_value(self, spec: FanRouteSpec, distance: float) -> float:
        denom = distance + 1.0
        if self.score_mode in ("frequency_weighted_topk", "route_weight_topk", "weighted_topk"):
            weight = float(ROUTE_FREQUENCY_WEIGHTS.get(spec.name, spec.fan))
            return weight / (denom ** self.distance_alpha)
        if self.score_mode == "fan_over_sqrt":
            return float(spec.fan) / math.sqrt(denom)
        return math.sqrt(float(spec.fan) / denom)

    def _table_distances(self, hand: Tuple[int, ...]) -> Dict[str, int]:
        if not self.tables:
            return {}
        hand_arr = np.asarray(hand, dtype=np.int16)
        result: Dict[str, int] = {}
        for name, (_, targets) in self.tables.items():
            # Exact target distance under the table convention:
            # complete target -> 0, one missing tile -> 1, etc.
            result[name] = int(np.maximum(targets - hand_arr, 0).sum(axis=1).min())
        return result

    def _fast_distances(
        self,
        hand: Tuple[int, ...],
        open_meld_count: int = 0,
        claimed_meld_count: int = 0,
    ) -> Dict[str, int]:
        hand_sum = sum(hand)
        suits = [range(0, 9), range(9, 18), range(18, 27)]
        suit_counts = [sum(hand[i] for i in suit) for suit in suits]
        honor_count = sum(hand[27:34])

        result: Dict[str, int] = {}

        # 七对：对子数和不同牌种数共同决定还差多少张。
        pair_count = sum(1 for c in hand if c >= 2)
        kind_count = sum(1 for c in hand if c > 0)
        result["seven_pairs"] = max(0, 7 - pair_count) + max(0, 7 - kind_count)

        if claimed_meld_count >= self.all_claimed_min_melds:
            pair_cost = min(max(0, 2 - c) for c in hand)
            missing_claims = max(0, 4 - int(claimed_meld_count))
            result["all_claimed"] = int(pair_cost + 2 * missing_claims)

        # 碰碰胡：选择一个对子和四个刻子，估算补齐所需张数。
        triplet_costs = [max(0, 3 - c) for c in hand]
        best_pungs = 14
        for pair_tid in range(34):
            pair_cost = max(0, 2 - hand[pair_tid])
            rest = sorted(triplet_costs[i] for i in range(34) if i != pair_tid)
            best_pungs = min(best_pungs, pair_cost + sum(rest[:4]))
        result["all_pungs"] = int(best_pungs)

        # 清一色 / 混一色 / 大于五 / 小于五属于牌域路线，直接按离开目标牌域的数量估算。
        result["full_flush"] = int(max(0, hand_sum - max(suit_counts)))
        result["half_flush"] = int(max(0, hand_sum - max(c + honor_count for c in suit_counts)))
        low_ids = set(list(range(0, 4)) + list(range(9, 13)) + list(range(18, 22)))
        high_ids = set(list(range(5, 9)) + list(range(14, 18)) + list(range(23, 27)))
        result["less_than_five"] = int(sum(hand[i] for i in range(34) if i not in low_ids))
        result["greater_than_five"] = int(sum(hand[i] for i in range(34) if i not in high_ids))

        # 五门齐：万/筒/条/风/箭五类至少各有一张。
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

        # 三色三同顺：三门同起点顺子。
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

        # 花龙：三门分别 123/456/789。
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

        # 清龙：同门 123/456/789。
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

        # 三色三步高：三门起点依次错一位，允许花色排列。
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

        # 一色三步高：同门起点依次错一位。
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

        # 全带幺粗估：越多幺九字越接近；精确版本可通过提高 table_max_targets 使用表。
        terminal_honor_ids = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
        result["outside_hand"] = int(max(0, hand_sum - sum(hand[i] for i in terminal_honor_ids)))

        return result


class MJRMRewardShaper:
    """[V4 potential reward] Rollout reward calculator used by PPO agents."""

    def __init__(self, cfg: MJRMRewardConfig):
        self.cfg = cfg
        self.potential = GbmjFanPotential(
            cfg.potential_table_dir,
            cache_size=cfg.potential_cache_size,
            table_max_targets=cfg.potential_table_max_targets,
            score_mode=cfg.potential_score_mode,
            distance_alpha=cfg.potential_distance_alpha,
            top_k=cfg.potential_top_k,
            topk_decay=cfg.potential_topk_decay,
            max_route_distance=cfg.potential_max_route_distance,
            all_claimed_min_melds=cfg.potential_all_claimed_min_melds,
        )

    @classmethod
    def from_config(cls, rl_cfg: Dict):
        cfg = MJRMRewardConfig(
            enabled=bool(rl_cfg.get("enable_mjrm_reward", True)),
            enable_potential_reward=bool(rl_cfg.get("enable_potential_reward", True)),
            potential_table_dir=str(rl_cfg.get("potential_table_dir", "data")),
            potential_weight=float(rl_cfg.get("potential_weight", 0.005)),
            potential_gamma=float(rl_cfg.get("potential_gamma", rl_cfg.get("gamma", 0.98))),
            potential_terminal_beta=float(rl_cfg.get("potential_terminal_beta", 0.0)),
            potential_cache_size=int(rl_cfg.get("potential_cache_size", 50000)),
            potential_table_max_targets=int(rl_cfg.get("potential_table_max_targets", 12000)),
            potential_score_mode=str(rl_cfg.get("potential_score_mode", "frequency_weighted_topk")),
            potential_delta_mode=str(rl_cfg.get("potential_delta_mode", "discounted")),
            potential_source=str(rl_cfg.get("potential_source", "tables")),
            potential_score_scale=float(rl_cfg.get("potential_score_scale", 4.0)),
            potential_score_clip=float(rl_cfg.get("potential_score_clip", 2.0)),
            potential_reward_clip=float(rl_cfg.get("potential_reward_clip", 0.25)),
            potential_distance_alpha=float(rl_cfg.get("potential_distance_alpha", 1.6)),
            potential_top_k=int(rl_cfg.get("potential_top_k", 4)),
            potential_topk_decay=float(rl_cfg.get("potential_topk_decay", 0.5)),
            potential_max_route_distance=float(rl_cfg.get("potential_max_route_distance", 6.0)),
            potential_all_claimed_min_melds=int(rl_cfg.get("potential_all_claimed_min_melds", 2)),
            route_distance_delta_weight=float(rl_cfg.get("route_distance_delta_weight", 0.03)),
            ordinary_distance_delta_weight=float(rl_cfg.get("ordinary_distance_delta_weight", 0.0)),
            distance_delta_clip=float(rl_cfg.get("distance_delta_clip", 2.0)),
            dense_ready_reward=float(rl_cfg.get("dense_ready_reward", 0.015)),
            dense_near_ready_reward=float(rl_cfg.get("dense_near_ready_reward", 0.006)),
            dense_late_far_penalty=float(rl_cfg.get("dense_late_far_penalty", -0.008)),
            dense_late_push_penalty=float(rl_cfg.get("dense_late_push_penalty", -0.012)),
            dense_late_turn=int(rl_cfg.get("dense_late_turn", 10)),
            dense_near_distance=float(rl_cfg.get("dense_near_distance", 2.0)),
            dense_far_distance=float(rl_cfg.get("dense_far_distance", 3.0)),
            score_delta_reward_scale=float(rl_cfg.get("score_delta_reward_scale", 40.0)),
            score_delta_reward_clip=float(rl_cfg.get("score_delta_reward_clip", 2.0)),
            terminal_redistribute_decay=float(rl_cfg.get("terminal_redistribute_decay", 0.92)),
            dangerous_push_penalty=float(rl_cfg.get("dangerous_push_penalty", -0.08)),
            dangerous_push_turn=int(rl_cfg.get("dangerous_push_turn", 10)),
            dangerous_push_best_distance=float(rl_cfg.get("dangerous_push_best_distance", 1.0)),
            self_draw_reward=float(rl_cfg.get("self_draw_reward", 2.0)),
            win_reward=float(rl_cfg.get("win_reward", 1.5)),
            deal_in_penalty=float(rl_cfg.get("deal_in_penalty", -2.0)),
            other_self_draw_penalty=float(rl_cfg.get("other_self_draw_penalty", -0.8)),
            other_ron_penalty=float(rl_cfg.get("other_ron_penalty", -0.4)),
            tenpai_reward=float(rl_cfg.get("tenpai_reward", 0.3)),
            noten_penalty=float(rl_cfg.get("noten_penalty", -0.2)),
        )
        return cls(cfg)

    def clear_cache(self) -> None:
        """[V4 potential reward] Drop rollout-local potential cache.

        The cache is already size-bounded, but clearing it between PPO
        iterations prevents Python object overhead from accumulating across
        long RL runs.
        """
        self.potential.cache.clear()

    def _foresight_potential_info(self, state, obs=None) -> Tuple[float, Dict]:
        """[V4 RL closed-loop] Prefer the already encoded actor foresight block."""
        if obs is not None:
            info = foresight_info_from_obs(obs)
            if info:
                return float(info.get("potential_score", 0.0)), info

        route_values, top_discards = compute_foresight_features(state)
        info = foresight_info_from_features(route_values, top_discards)
        return float(info.get("potential_score", 0.0)), info

    def describe_state_action(self, state, action: int, obs=None) -> Dict:
        hand = _route_counts_from_state(state)
        open_meld_count, claimed_meld_count = _meld_counts_from_state(state)
        ordinary_distance = _ordinary_distance_from_state(state)
        if not self.cfg.enable_potential_reward:
            score, info = 0.0, {}
        elif self.cfg.potential_source in ("obs_foresight", "foresight"):
            score, info = self._foresight_potential_info(state, obs=obs)
        else:
            score, info = self.potential.score(
                hand,
                open_meld_count=open_meld_count,
                claimed_meld_count=claimed_meld_count,
            )
        return {
            "action": int(action),
            "action_group": _action_group(int(action)),
            "turn": int(getattr(state, "at_turn", 0)),
            "hand": hand,
            "open_meld_count": float(open_meld_count),
            "claimed_meld_count": float(claimed_meld_count),
            "ordinary_distance": float(ordinary_distance),
            "potential_score": float(score),
            **info,
        }

    def opening_reward(self, state) -> Tuple[float, Dict[str, float]]:
        # [V4 potential reward] Opening structure reward removed.  The route
        # potential already rewards route-forming progress and avoids fixed
        # pair/sequence/triplet bonuses dominating PPO.
        return 0.0, {}

    def action_reward(self, action: int) -> Tuple[float, Dict[str, float]]:
        # [V4 potential reward] Meld reward removed.  Chi/Pon/Kong should be
        # useful only if it increases route potential or terminal outcome.
        return 0.0, {}

    def state_action_dense_reward(self, info: Dict) -> Tuple[float, Dict[str, float]]:
        """Small per-decision signal to avoid purely terminal credit."""
        if not self.cfg.enabled:
            return 0.0, {}
        reward = 0.0
        comps: Dict[str, float] = {}
        best_distance = float(info.get("potential_best_distance", 14.0))
        turn = int(info.get("turn", 0))
        action_group = str(info.get("action_group", ""))

        if best_distance <= 1.0:
            value = float(self.cfg.dense_ready_reward)
            reward += value
            comps["dense_ready_reward"] = value
        elif best_distance <= float(self.cfg.dense_near_distance):
            value = float(self.cfg.dense_near_ready_reward)
            reward += value
            comps["dense_near_ready_reward"] = value

        if turn >= int(self.cfg.dense_late_turn) and best_distance >= float(self.cfg.dense_far_distance):
            value = float(self.cfg.dense_late_far_penalty)
            if value != 0.0:
                reward += value
                comps["dense_late_far_penalty"] = value
            if action_group in {"chi", "pon", "kan"}:
                value = float(self.cfg.dense_late_push_penalty)
                if value != 0.0:
                    reward += value
                    comps["dense_late_push_penalty"] = value

        return reward, comps

    def shanten_transition_reward(self, prev_info: Dict, next_info: Dict) -> Tuple[float, Dict[str, float]]:
        """[V4 potential reward] Dense potential difference between decisions."""
        if not self.cfg.enabled or not self.cfg.enable_potential_reward:
            return 0.0, {}
        prev_raw_score = float(prev_info.get("potential_score", 0.0))
        next_raw_score = float(next_info.get("potential_score", 0.0))
        # [V4 clean RL reward] The raw summed route score is useful for
        # ranking hands, but too large for PPO rewards.  Normalize before
        # differencing so potential shaping cannot dominate terminal results.
        prev_score = self._normalize_potential_score(prev_raw_score)
        next_score = self._normalize_potential_score(next_raw_score)
        raw_delta = next_raw_score - prev_raw_score
        norm_raw_delta = next_score - prev_score
        # [V4 MJ_T reward] Default is the paper-style potential difference:
        # lambda * (gamma * Score(s_next) - Score(s)).  Other modes are kept
        # only for ablation, not for the main MJ_T reproduction path.
        if self.cfg.potential_delta_mode == "discounted":
            delta = self.cfg.potential_gamma * next_score - prev_score
        elif self.cfg.potential_delta_mode == "normalized_discounted":
            delta = (self.cfg.potential_gamma * next_score - prev_score) / max(1.0, abs(prev_score))
        else:
            delta = norm_raw_delta
        unclipped_reward = self.cfg.potential_weight * delta
        reward_clip = max(0.0, float(self.cfg.potential_reward_clip))
        if reward_clip > 0.0:
            reward = min(reward_clip, max(-reward_clip, unclipped_reward))
        else:
            reward = unclipped_reward
        distance_clip = max(0.0, float(self.cfg.distance_delta_clip))
        route_distance_delta = float(prev_info.get("potential_best_distance", 14.0)) - float(next_info.get("potential_best_distance", 14.0))
        ordinary_distance_delta = float(prev_info.get("ordinary_distance", 14.0)) - float(next_info.get("ordinary_distance", 14.0))
        if distance_clip > 0.0:
            route_distance_delta = min(distance_clip, max(-distance_clip, route_distance_delta))
            ordinary_distance_delta = min(distance_clip, max(-distance_clip, ordinary_distance_delta))
        route_distance_reward = float(self.cfg.route_distance_delta_weight) * route_distance_delta
        ordinary_distance_reward = float(self.cfg.ordinary_distance_delta_weight) * ordinary_distance_delta
        reward += route_distance_reward + ordinary_distance_reward
        return reward, {
            "potential_reward": reward,
            "potential_reward_unclipped": unclipped_reward,
            "route_distance_delta_reward": route_distance_reward,
            "ordinary_distance_delta_reward": ordinary_distance_reward,
            "route_distance_delta": route_distance_delta,
            "ordinary_distance_delta": ordinary_distance_delta,
            "potential_delta": delta,
            "potential_raw_delta": norm_raw_delta,
            "potential_score_raw_delta": raw_delta,
            "potential_prev": prev_score,
            "potential_next": next_score,
            "potential_raw_prev": prev_raw_score,
            "potential_raw_next": next_raw_score,
            "potential_best_distance": float(next_info.get("potential_best_distance", 0.0)),
            "potential_loaded_tables": float(next_info.get("potential_loaded_tables", 0.0)),
            "potential_route_count": float(next_info.get("potential_route_count", 0.0)),
            "potential_selected_route_count": float(next_info.get("potential_selected_route_count", 0.0)),
            "potential_best_route_weight": float(next_info.get("potential_best_route_weight", 0.0)),
            "potential_best_route_value": float(next_info.get("potential_best_route_value", 0.0)),
            # [V4 potential reward] Keep the old key neutral; train_ppo can
            # still read it safely if an old config accidentally enables Eq.4.
            "shanten_delta": 0.0,
        }

    def _normalize_potential_score(self, score: float) -> float:
        """[V4 clean RL reward] Normalize fan-potential to PPO reward scale."""
        scale = max(1.0e-6, float(self.cfg.potential_score_scale))
        return min(float(self.cfg.potential_score_clip), max(0.0, float(score) / scale))

    def _score_delta_reward(self, score_delta: float) -> float:
        scale = max(1.0e-6, float(self.cfg.score_delta_reward_scale))
        clip = max(0.0, float(self.cfg.score_delta_reward_clip))
        reward = float(score_delta) / scale
        return min(clip, max(-clip, reward))

    def terminal_potential_reward(self, last_info: Dict) -> Tuple[float, Dict[str, float]]:
        """[V4 MJ_T reward] Do not add an artificial terminal potential close.

        MJ_T's formula uses r_env + lambda * (gamma * Score(s_next) - Score(s))
        between real decision states.  The terminal outcome is handled by
        terminal_kyoku_reward(), so forcing Score(terminal)=0 would create an
        extra negative reward that is not stated in the paper and can punish a
        hand simply for having high route potential at the end.
        """
        return 0.0, {}

    def terminal_kyoku_reward(self, game_result, player_id: int, kyoku_index: int, segment: List[Dict]) -> Tuple[float, Dict[str, float]]:
        """[V4 potential reward] Sparse terminal anchor for one player/kyoku."""
        if not self.cfg.enabled:
            return 0.0, {}
        reward = 0.0
        comps = {
            "terminal_score_delta": 0.0,
            "terminal_score_delta_reward": 0.0,
            "terminal_self_draw": 0.0,
            "terminal_win": 0.0,
            "terminal_deal_in": 0.0,
            "terminal_other_self_draw": 0.0,
            "terminal_other_ron": 0.0,
            "terminal_tenpai": 0.0,
            "terminal_noten": 0.0,
        }

        has_hu = False
        score_delta = 0.0
        if kyoku_index < len(game_result.game_log):
            for ext in game_result.game_log[kyoku_index]:
                ev = ext.event.event
                if not isinstance(ev, Event.Hu):
                    continue
                has_hu = True
                if ev.deltas is not None and 0 <= player_id < len(ev.deltas):
                    score_delta += float(ev.deltas[player_id])
                winner = int(ev.player)
                target = int(ev.target)
                if winner == player_id:
                    if target == player_id:
                        comps["terminal_self_draw"] += self.cfg.self_draw_reward
                        reward += self.cfg.self_draw_reward
                    else:
                        comps["terminal_win"] += self.cfg.win_reward
                        reward += self.cfg.win_reward
                elif target == player_id:
                    comps["terminal_deal_in"] += self.cfg.deal_in_penalty
                    reward += self.cfg.deal_in_penalty
                elif target == winner:
                    comps["terminal_other_self_draw"] += self.cfg.other_self_draw_penalty
                    reward += self.cfg.other_self_draw_penalty
                else:
                    comps["terminal_other_ron"] += self.cfg.other_ron_penalty
                    reward += self.cfg.other_ron_penalty

        if has_hu:
            score_reward = self._score_delta_reward(score_delta)
            comps["terminal_score_delta"] += score_delta
            comps["terminal_score_delta_reward"] += score_reward
            reward += score_reward

        if not has_hu and segment:
            # [V4 potential reward] Tenpai reward is disabled by default to
            # avoid the old "只追听牌" interference.  This branch is kept only
            # for controlled ablation.
            last_info = segment[-1].get("reward_info", {})
            if float(last_info.get("potential_best_distance", 99.0)) <= 1.0:
                comps["terminal_tenpai"] += self.cfg.tenpai_reward
                reward += self.cfg.tenpai_reward
            else:
                comps["terminal_noten"] += self.cfg.noten_penalty
                reward += self.cfg.noten_penalty

        return reward, comps

    def redistribute_terminal_reward(self, segment: List[Dict], terminal_reward: float) -> None:
        """Distribute part of a hand's terminal reward across its decisions."""
        if not self.cfg.enabled:
            return
        beta = float(self.cfg.potential_terminal_beta)
        if beta == 0.0 or not segment or terminal_reward == 0.0:
            return
        decay = min(1.0, max(0.0, float(self.cfg.terminal_redistribute_decay)))
        weights = [decay ** float(len(segment) - 1 - idx) for idx in range(len(segment))]
        weight_sum = sum(weights)
        if weight_sum <= 1.0e-8:
            return

        for transition, weight in zip(segment, weights):
            weight = weight / weight_sum
            reward = beta * weight * float(terminal_reward)
            transition["reward"] += reward
            self.add_components(transition, {
                "terminal_redistribution": reward,
                "terminal_redistribution_weight": weight,
            })

    def apply_dangerous_push_penalty(self, segment: List[Dict], terminal_components: Dict[str, float]) -> None:
        if not self.cfg.enabled or not segment:
            return
        if float(terminal_components.get("terminal_deal_in", 0.0)) >= 0.0:
            return
        penalty = float(self.cfg.dangerous_push_penalty)
        if penalty == 0.0:
            return
        min_turn = int(self.cfg.dangerous_push_turn)
        max_ready_distance = float(self.cfg.dangerous_push_best_distance)
        push_groups = {"discard", "chi", "pon", "kan"}
        for transition in segment:
            info = transition.get("reward_info", {})
            if str(info.get("action_group", "")) not in push_groups:
                continue
            if int(info.get("turn", 0)) < min_turn:
                continue
            if float(info.get("potential_best_distance", 99.0)) <= max_ready_distance:
                continue
            transition["reward"] += penalty
            self.add_components(transition, {
                "dangerous_push_penalty": penalty,
                "dangerous_push_count": 1.0,
            })

    @staticmethod
    def add_components(transition: Dict, components: Dict[str, float]) -> None:
        values = transition.setdefault("reward_components", {})
        for key, value in components.items():
            values[key] = values.get(key, 0.0) + float(value)


def reward_component_stats(transitions: List[Dict]) -> Dict[str, float]:
    """[V4 potential reward] Aggregate reward components for TensorBoard."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for transition in transitions:
        for key, value in transition.get("reward_components", {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {
        key: totals[key] / max(1, counts.get(key, 0))
        for key in sorted(totals)
    }
