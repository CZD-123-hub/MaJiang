import gzip
import logging
import random
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from mortal_part.dataset.augmentation import build_augmentation_specs
from mortal_part.mjai.event import Event
from mortal_part.rankings import Rankings


GRP_FEATURE_DIM = 12
RANK_UTILITIES = (1.0, 0.33, -0.33, -1.0)


def _wind_to_int(value) -> int:
    if isinstance(value, str):
        return {"E": 0, "S": 1, "W": 2, "N": 3}.get(value.upper(), 0)
    try:
        return int(value) % 4
    except Exception:
        return 0


def _safe_scores(scores: Optional[Sequence[float]]) -> List[float]:
    if scores is None:
        return [500.0, 500.0, 500.0, 500.0]
    values = [float(x) for x in list(scores)[:4]]
    if len(values) < 4:
        values.extend([500.0] * (4 - len(values)))
    return values


def _rotate(values: Sequence[float], player_id: int) -> List[float]:
    player_id = int(player_id) % 4
    values = list(values)
    return values[player_id:] + values[:player_id]


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, value)))


def _rank_by_score(scores: Sequence[float], player_id: int) -> int:
    return int(Rankings(list(scores)).rank_by_player[int(player_id) % 4])


def _stage_feature_vector(
    scores_rel: Sequence[float],
    round_wind: int,
    kyoku: int,
    zhuang_rel: int,
    current_rank: int,
    total_rounds: int = 16,
) -> np.ndarray:
    """Compact global-game state used by the GRP.

    Scores are relative to the predicted player, so scores_rel[0] is self.
    This function is shared by offline GRP training and PPO rollout inference
    to avoid a train/serve feature mismatch.
    """
    scores = [float(x) for x in list(scores_rel)[:4]]
    if len(scores) < 4:
        scores.extend([500.0] * (4 - len(scores)))
    total_rounds = max(1, int(total_rounds))
    round_wind = int(round_wind) % 4
    kyoku = int(kyoku) % 4
    hand_index = max(0, min(total_rounds - 1, round_wind * 4 + kyoku))
    remaining = max(0, total_rounds - 1 - hand_index)

    self_score = scores[0]
    sorted_scores = sorted(scores, reverse=True)
    leader = sorted_scores[0]
    trailer = sorted_scores[-1]
    higher = [score for score in sorted_scores if score > self_score]
    lower = [score for score in sorted_scores if score < self_score]
    gap_to_next_higher = self_score - min(higher) if higher else 0.0
    gap_to_next_lower = self_score - max(lower) if lower else 0.0
    mean_score = sum(scores) / 4.0
    score_std = float(np.std(np.asarray(scores, dtype=np.float32)))

    return np.asarray(
        [
            hand_index / max(1.0, float(total_rounds - 1)),
            remaining / max(1.0, float(total_rounds - 1)),
            round_wind / 3.0,
            kyoku / 3.0,
            1.0 if int(zhuang_rel) % 4 == 0 else 0.0,
            (int(zhuang_rel) % 4) / 3.0,
            _clip((self_score - 500.0) / 500.0),
            _clip((self_score - leader) / 500.0),
            _clip(gap_to_next_higher / 500.0),
            _clip(gap_to_next_lower / 500.0),
            float(max(0, min(3, int(current_rank)))) / 3.0,
            _clip(score_std / 500.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def global_stage_features_from_state(state, total_rounds: int = 16) -> np.ndarray:
    """Build GRP stage features from a live PlayerState.

    PlayerState stores scores relative to the current player, matching the
    offline feature convention used by Grp.stage_features_for_player().
    """
    scores_rel = _safe_scores(getattr(state, "scores", None))
    current_rank = _rank_by_score(scores_rel, 0)
    return _stage_feature_vector(
        scores_rel=scores_rel,
        round_wind=int(getattr(state, "round_wind", 0)),
        kyoku=int(getattr(state, "kyoku", 0)),
        zhuang_rel=int(getattr(state, "zhuang_id", 0)),
        current_rank=current_rank,
        total_rounds=total_rounds,
    )


@dataclass
class KyokuStage:
    round_wind: int
    kyoku: int
    zhuang: int
    scores: Tuple[float, float, float, float]


class Grp:
    """Global Reward Predictor metadata for one multi-hand game log."""

    def __init__(
        self,
        stages: Sequence[KyokuStage],
        initial_scores: Sequence[float],
        final_scores: Sequence[float],
    ):
        self.stages = list(stages)
        self.initial_scores = _safe_scores(initial_scores)
        self.final_scores = _safe_scores(final_scores)
        self.rank_by_player = Rankings(self.final_scores).rank_by_player.astype(np.int64).tolist()

    @staticmethod
    def load_log(raw_log: str) -> "Grp":
        events = []
        for line in raw_log.strip().splitlines():
            line = line.strip()
            if line:
                events.append(Event.from_str(line))
        return Grp.load_events(events)

    @staticmethod
    def load_json_file(filename: str) -> "Grp":
        if str(filename).endswith(".gz"):
            with gzip.open(filename, "rt", encoding="utf-8") as f:
                return Grp.load_log(f.read())
        with open(filename, "r", encoding="utf-8") as f:
            return Grp.load_log(f.read())

    @staticmethod
    def load_json_files(filenames: Sequence[str]) -> List["Grp"]:
        return [Grp.load_json_file(filename) for filename in filenames]

    @staticmethod
    def load_events(events: Sequence[Event]) -> "Grp":
        stages: List[KyokuStage] = []
        initial_scores: Optional[List[float]] = None
        current_scores: Optional[List[float]] = None

        for wrapper in events:
            ev = wrapper.event
            if isinstance(ev, Event.StartKyoku):
                scores = _safe_scores(ev.scores)
                if initial_scores is None:
                    initial_scores = list(scores)
                current_scores = list(scores)
                stages.append(
                    KyokuStage(
                        round_wind=_wind_to_int(ev.wind),
                        kyoku=int(ev.kyoku) % 4,
                        zhuang=int(ev.zhuang) % 4,
                        scores=tuple(scores),
                    )
                )
            elif isinstance(ev, Event.Hu) and ev.deltas is not None:
                if current_scores is None:
                    current_scores = _safe_scores(initial_scores)
                for idx, delta in enumerate(list(ev.deltas)[:4]):
                    current_scores[idx] += float(delta)

        if initial_scores is None:
            initial_scores = [500.0, 500.0, 500.0, 500.0]
        final_scores = list(current_scores) if current_scores is not None else list(initial_scores)
        if not stages:
            stages.append(KyokuStage(0, 0, 0, tuple(initial_scores)))
        return Grp(stages, initial_scores, final_scores)

    def __len__(self) -> int:
        return len(self.stages)

    def is_empty(self) -> bool:
        return len(self.stages) == 0

    def stage_features_for_player(self, player_id: int, kyoku_idx: int, total_rounds: int = 16) -> np.ndarray:
        idx = max(0, min(int(kyoku_idx), len(self.stages) - 1))
        stage = self.stages[idx]
        player_id = int(player_id) % 4
        scores_rel = _rotate(stage.scores, player_id)
        zhuang_rel = (int(stage.zhuang) - player_id) % 4
        current_rank = _rank_by_score(stage.scores, player_id)
        return _stage_feature_vector(
            scores_rel=scores_rel,
            round_wind=stage.round_wind,
            kyoku=stage.kyoku,
            zhuang_rel=zhuang_rel,
            current_rank=current_rank,
            total_rounds=total_rounds,
        )

    def target_for_player(
        self,
        player_id: int,
        rank_weight: float = 1.0,
        score_weight: float = 0.25,
        score_scale: float = 200.0,
        target_clip: float = 1.5,
    ) -> float:
        player_id = int(player_id) % 4
        rank = max(0, min(3, int(self.rank_by_player[player_id])))
        rank_value = RANK_UTILITIES[rank]
        score_delta = float(self.final_scores[player_id] - self.initial_scores[player_id])
        score_value = _clip(score_delta / max(1.0e-6, float(score_scale)), -2.0, 2.0)
        target = float(rank_weight) * rank_value + float(score_weight) * score_value
        clip_value = float(target_clip)
        if clip_value > 0.0:
            target = _clip(target, -clip_value, clip_value)
        return float(target)

    def take_feature(self) -> np.ndarray:
        return np.asarray(
            [[stage.round_wind, stage.kyoku, stage.zhuang, *stage.scores] for stage in self.stages],
            dtype=np.float32,
        )

    def take_rank_by_player(self) -> List[int]:
        return list(self.rank_by_player)

    def take_final_scores(self) -> List[float]:
        return list(self.final_scores)


def _load_raw_log(filename: str) -> str:
    if str(filename).endswith(".gz"):
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            return f.read()
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def _iter_file_batches(files: Sequence[str], file_batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(files), int(file_batch_size)):
        batch = list(files[start:start + int(file_batch_size)])
        if batch:
            yield batch


class GlobalRewardDataset(IterableDataset):
    """Iterable dataset for V_global(s) regression.

    Each sample is one expert decision state with:
      obs: v3/v4 194-channel visible observation
      grp_features: compact whole-game stage features
      target: final whole-game utility for that player
    """

    def __init__(
        self,
        file_list: Sequence[str],
        file_batch_size: int = 8,
        num_epochs: int = 1,
        shuffle_files: bool = True,
        augmentation_factor: int = 1,
        shuffle_augmentation: bool = True,
        total_rounds: int = 16,
        rank_weight: float = 1.0,
        score_weight: float = 0.25,
        score_scale: float = 200.0,
        target_clip: float = 1.5,
        skip_bad_files: bool = True,
    ):
        super().__init__()
        self.file_list = list(file_list)
        self.file_batch_size = int(file_batch_size)
        self.num_epochs = int(num_epochs)
        self.shuffle_files = bool(shuffle_files)
        self.augmentation_factor = int(augmentation_factor)
        self.shuffle_augmentation = bool(shuffle_augmentation)
        self.total_rounds = int(total_rounds)
        self.rank_weight = float(rank_weight)
        self.score_weight = float(score_weight)
        self.score_scale = float(score_scale)
        self.target_clip = float(target_clip)
        self.skip_bad_files = bool(skip_bad_files)

    def __iter__(self):
        from mortal_part.dataset.gameplay import GameplayLoader

        for _ in range(self.num_epochs):
            files = list(self.file_list)
            worker = get_worker_info()
            if worker is not None:
                files = files[worker.id::worker.num_workers]
            if self.shuffle_files:
                random.shuffle(files)

            specs = build_augmentation_specs(self.augmentation_factor)
            if self.shuffle_augmentation and len(specs) > 1:
                random.shuffle(specs)

            for spec in specs:
                loader = GameplayLoader(augmentation_spec=spec)
                for batch_files in _iter_file_batches(files, self.file_batch_size):
                    for file_games in self._load_file_games(loader, batch_files):
                        for game in file_games:
                            yield from self._iter_game(game)

    def _load_file_games(self, loader, batch_files: Sequence[str]):
        if self.skip_bad_files:
            for filename in batch_files:
                try:
                    yield from loader.load_gz_log_files([filename])
                except Exception as exc:
                    logging.warning("skipped bad GRP log file: %s error=%r", filename, exc)
            return

        try:
            yield from loader.load_gz_log_files(list(batch_files))
            return
        except Exception as batch_exc:
            raise RuntimeError("failed to load GRP file batch: %s" % (list(batch_files),)) from batch_exc

    def _iter_game(self, game):
        grp = game.grp
        if grp is None or grp.is_empty():
            return
        target = grp.target_for_player(
            game.player_id,
            rank_weight=self.rank_weight,
            score_weight=self.score_weight,
            score_scale=self.score_scale,
            target_clip=self.target_clip,
        )
        for obs, kyoku_idx in zip(game.obs, game.at_kyoku):
            features = grp.stage_features_for_player(
                game.player_id,
                kyoku_idx,
                total_rounds=self.total_rounds,
            )
            yield (
                np.asarray(obs, dtype=np.float32),
                np.asarray(features, dtype=np.float32),
                np.float32(target),
            )


def collect_files(patterns: Sequence[str], project_root: Optional[Path] = None) -> List[str]:
    files = set()
    for pattern in patterns:
        path = Path(str(pattern))
        full_pattern = str(path if path.is_absolute() else Path(project_root or ".") / path)
        for filename in glob(full_pattern, recursive=True):
            files.add(str(Path(filename).resolve()))
    return sorted(files)
