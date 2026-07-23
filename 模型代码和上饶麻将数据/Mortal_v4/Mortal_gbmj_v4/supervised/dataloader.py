import random

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from mortal_part.dataset.augmentation import build_augmentation_specs
from mortal_part.consts import ACTION_SPACE
from mortal_part.dataset.gameplay import GameplayLoader


"""监督训练数据集。

这是一个 IterableDataset：它不把全部样本一次性读进内存，而是按文件批次
加载、回放并逐条 yield。这样适合数 GB 的对局日志，也允许通过数据增强
把同一局从不同座位/牌面变换中重复利用。
"""


class PolicyDataset(IterableDataset):
    # Single-model supervised dataset for the 235-way action policy.
    def __init__(
        self,
        file_list,
        file_batch_size=16,
        num_epochs=1,
        shuffle_files=True,
        augmentation_factor=1,
        shuffle_augmentation=True,
        source_specs=None,
        history_len=0,
        value_target_config=None,
    ):
        super().__init__()
        self.file_list = list(file_list)
        self.file_batch_size = int(file_batch_size)
        self.num_epochs = int(num_epochs)
        self.shuffle_files = bool(shuffle_files)
        self.augmentation_factor = int(augmentation_factor)
        self.shuffle_augmentation = bool(shuffle_augmentation)
        # [V4 rollback-history] Default is v3-style single-state training.
        # When history_len=0, no history observations/actions are stored.
        self.history_len = int(history_len)
        self.value_target_config = dict(value_target_config or {})
        # [V4 data-mix] Optional source-specific data exposure, e.g.
        # public data 12x augmentation and private data 2x augmentation.
        self.source_specs = list(source_specs or [])

    def __iter__(self):
        # 每次创建 DataLoader 迭代器时重新遍历 num_epochs；训练脚本通常把
        # num_epochs 设为 1，由外层 epoch 控制训练轮数。
        for _ in range(self.num_epochs):
            if self.source_specs:
                yield from self.iter_source_jobs()
            else:
                yield from self.iter_single_source()

    def iter_single_source(self):
        file_list = list(self.file_list)
        worker = get_worker_info()
        if worker is not None:
            # 多进程 worker 按文件切片，避免多个 worker 重复回放同一个文件。
            file_list = file_list[worker.id::worker.num_workers]
        if self.shuffle_files:
            random.shuffle(file_list)
        specs = build_augmentation_specs(self.augmentation_factor)
        if self.shuffle_augmentation and len(specs) > 1:
            random.shuffle(specs)
        for spec in specs:
            yield from self.load_files(file_list, spec)

    def iter_source_jobs(self):
        worker = get_worker_info()
        jobs = []
        for source in self.source_specs:
            files = list(source.get("files", []))
            if worker is not None:
                files = files[worker.id::worker.num_workers]
            if self.shuffle_files:
                random.shuffle(files)
            specs = build_augmentation_specs(int(source.get("augmentation_factor", 1)))
            if self.shuffle_augmentation and len(specs) > 1:
                random.shuffle(specs)
            for spec in specs:
                for start in range(0, len(files), self.file_batch_size):
                    batch_files = files[start:start + self.file_batch_size]
                    if batch_files:
                        jobs.append((source.get("name", "source"), batch_files, spec))

        # [V4 data-mix] Shuffle chunk-level jobs so public/private sources are
        # interleaved instead of appearing as long contiguous blocks.
        if self.shuffle_files:
            random.shuffle(jobs)
        for _, batch_files, spec in jobs:
            yield from self.load_file_batch(batch_files, spec)

    def load_files(self, file_list, augmentation_spec):
        for start in range(0, len(file_list), self.file_batch_size):
            batch_files = file_list[start:start + self.file_batch_size]
            yield from self.load_file_batch(batch_files, augmentation_spec)

    def load_file_batch(self, batch_files, augmentation_spec):
        # GameplayLoader 负责“事件流 -> 状态回放样本”；PolicyDataset 负责
        # 把样本整理成 DataLoader 能拼 batch 的 numpy 数组。
        loader = GameplayLoader(
            augmentation_spec=augmentation_spec,
            value_target_config=self.value_target_config,
        )
        for file_games in loader.load_gz_log_files(batch_files):
            for game in file_games:
                yield from self.iter_game(game)

    def iter_game(self, game):
        obs_list = game.take_obs()
        actions = game.take_actions()
        masks = game.take_masks()
        value_targets = game.take_value_targets()
        if len(value_targets) != len(actions):
            value_targets = [0.0] * len(actions)
        history_obs = []
        history_actions = []
        empty_hist_actions = np.zeros((0,), dtype=np.int64)
        empty_hist_obs = None
        for obs, action, mask, value_target in zip(obs_list, actions, masks, value_targets):
            action = int(action)
            # 数据清洗的两道硬检查：动作必须在 235 类范围内，并且必须是
            # 当时合法动作。错误样本跳过，避免交叉熵学习到非法行为。
            if not (0 <= action < ACTION_SPACE):
                continue
            if not mask[action]:
                continue
            if self.history_len > 0:
                pad_count = max(0, self.history_len - len(history_obs))
                if pad_count > 0:
                    zero_obs = np.zeros_like(obs, dtype=np.float32)
                    hist_obs = [zero_obs] * pad_count + history_obs[-self.history_len:]
                    hist_actions = [ACTION_SPACE] * pad_count + history_actions[-self.history_len:]
                else:
                    hist_obs = history_obs[-self.history_len:]
                    hist_actions = history_actions[-self.history_len:]
                hist_obs_arr = np.stack(hist_obs, axis=0).astype(np.float32, copy=False)
                hist_actions_arr = np.asarray(hist_actions, dtype=np.int64)
            else:
                if empty_hist_obs is None:
                    empty_hist_obs = np.zeros((0,) + tuple(obs.shape), dtype=np.float32)
                hist_obs_arr = empty_hist_obs
                hist_actions_arr = empty_hist_actions
            yield (
                # 当前 direct-235 模型实际使用 obs/mask/action；历史和值字段
                # 保留是为了兼容旧版训练、评估和 PPO 代码。
                obs.astype(np.float32, copy=False),
                hist_obs_arr,
                hist_actions_arr,
                np.asarray(mask, dtype=np.bool_),
                np.int64(action),
                np.float32(value_target),
            )
            if self.history_len > 0:
                history_obs.append(obs.astype(np.float32, copy=False))
                history_actions.append(action)
