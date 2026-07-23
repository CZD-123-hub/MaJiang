import gzip
from itertools import islice
from typing import List, Optional, Set

import numpy as np

from mortal_part.dataset.augmentation import apply_augmentation
from mortal_part.chi_type import ChiType
from mortal_part.consts import (
    ADDKONG_BASE,
    ANKANG_BASE,
    CHOW_BASE,
    DISCARD_BASE,
    MINGGANG_BASE,
    PASS_INDEX,
    PUNG_BASE,
    WIN_INDEX,
)
from mortal_part.dataset.grp import Grp
from mortal_part.mjai.event import Event
from mortal_part.state.mah_player_gb import PlayerState


"""把事件流转换为监督学习样本。

一局日志的基本处理链路是：文本事件 -> Event 对象 -> PlayerState 回放 ->
obs/mask/action 三元组。模型并不直接读取 JSON 字段，而是读取回放过程
中每个决策时刻的状态编码和专家下一步动作。
"""


class GameplayLoader:
    def __init__(
        self,
        oracle: bool = False,
        player_names: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        trust_seed: bool = False,
        always_include_kan_select: bool = False,
        augmentation_spec=None,
        value_target_config: Optional[dict] = None,
    ):
        self.oracle = oracle
        self.player_names = player_names if player_names else []
        self.excludes = excludes if excludes else []
        self.trust_seed = trust_seed
        self.always_include_kan_select = always_include_kan_select
        self.augmentation_spec = augmentation_spec
        self.value_target_config = dict(value_target_config or {})
        self.player_names_set: Set[str] = set(self.player_names)
        self.excludes_set: Set[str] = set(self.excludes)

    def load_log(self, raw_log: str) -> List['Gameplay']:
        # 每一行是一个 mjai 事件；事件顺序不能打乱，因为 PlayerState 依赖
        # 前序事件逐步更新手牌、牌河、剩余牌和合法动作。
        events = []
        for line in raw_log.strip().split('\n'):
            if not line.strip():
                continue
            events.append(Event.from_str(line))
        if self.augmentation_spec is not None:
            events = apply_augmentation(events, self.augmentation_spec)
        return self.load_events(events)

    def load_gz_log_files(self, gzip_filenames: List[str]) -> List[List['Gameplay']]:
        def load_json_file(filename: str) -> List['Gameplay']:
            if filename.endswith('.gz'):
                with gzip.open(filename, 'rt', encoding='utf-8') as f:
                    return self.load_log(f.read())
            with open(filename, 'r', encoding='utf-8') as f:
                return self.load_log(f.read())

        return [load_json_file(fn) for fn in gzip_filenames]

    def load_events(self, events: list[Event]) -> List['Gameplay']:
        if not events or not isinstance(events[0].event, Event.StartGame):
            raise ValueError('Empty or invalid game log')

        names = events[0].event.names
        # 一局日志会为每个指定玩家生成一条 Gameplay；默认四名玩家都会生成。
        player_ids = [
            idx
            for idx, name in enumerate(names)
            if (not self.player_names_set or name in self.player_names_set)
            and (not self.excludes_set or name not in self.excludes_set)
        ]
        return [Gameplay.load_events_by_player(self, events, player_id) for player_id in player_ids]


class Gameplay:
    def __init__(self, grp=None, player_id: int = 0):
        self.obs = []
        self.invisible_obs = []
        self.actions: List[int] = []
        self.masks = []
        self.value_targets: List[float] = []
        self.at_kyoku: List[int] = []
        self.dones: List[bool] = []
        self.apply_gamma: List[bool] = []
        self.at_turns: List[int] = []
        self.shantens: List[int] = []
        self.grp: Grp = grp
        self.player_id = player_id
        self.player_name = ''

    def take_obs(self) -> List[np.ndarray]:
        obs = self.obs
        self.obs = []
        return obs

    def take_invisible_obs(self) -> List[np.ndarray]:
        invisible_obs = self.invisible_obs
        self.invisible_obs = []
        return invisible_obs

    def take_actions(self) -> List[int]:
        actions = self.actions
        self.actions = []
        return actions

    def take_masks(self) -> List[np.ndarray]:
        masks = self.masks
        self.masks = []
        return masks

    def take_value_targets(self) -> List[float]:
        value_targets = self.value_targets
        self.value_targets = []
        return value_targets

    def take_at_kyoku(self) -> List[int]:
        at_kyoku = self.at_kyoku
        self.at_kyoku = []
        return at_kyoku

    def take_dones(self) -> List[bool]:
        dones = self.dones
        self.dones = []
        return dones

    def take_apply_gamma(self) -> List[bool]:
        apply_gamma = self.apply_gamma
        self.apply_gamma = []
        return apply_gamma

    def take_at_turns(self) -> List[int]:
        at_turns = self.at_turns
        self.at_turns = []
        return at_turns

    def take_shantens(self) -> List[int]:
        shantens = self.shantens
        self.shantens = []
        return shantens

    def take_grp(self) -> Grp:
        grp = self.grp
        self.grp = None
        return grp

    def take_player_id(self) -> int:
        return self.player_id

    @classmethod
    def load_events_by_player(cls, config: GameplayLoader, events: list[Event], player_id: int) -> 'Gameplay':
        grp = Grp.load_events(events)
        data = cls(grp=grp, player_id=player_id)
        ctx = LoaderContext(config=config, state=PlayerState(player_id), kyoku_idx=0)

        for i in range(len(events) - 3):
            # 用相邻事件窗口回放：cur 更新状态，next_event 提供专家动作标签。
            window = list(islice(events, i, i + 4))
            data.extend_from_event_window(ctx, window)

        data.dones = [k1 > k0 for k0, k1 in zip(data.at_kyoku, data.at_kyoku[1:])]
        data.dones.append(True)
        # Critic pretraining uses the same terminal reward scale as PPO.  The
        # policy still imitates the expert action; only the independent critic
        # consumes these value targets.
        kyoku_values, kyoku_has_hu = cls.build_terminal_value_targets(
            events,
            player_id,
            **ctx.config.value_target_config,
        )
        if ctx.config.value_target_config.get("enable_draw_tenpai_target", True):
            last_shanten_by_kyoku = {}
            for kyoku_idx, shanten in zip(data.at_kyoku, data.shantens):
                last_shanten_by_kyoku[int(kyoku_idx)] = int(shanten)
            tenpai_reward = float(ctx.config.value_target_config.get("tenpai_reward", 0.1))
            noten_penalty = float(ctx.config.value_target_config.get("noten_penalty", -0.1))
            for kyoku_idx, shanten in last_shanten_by_kyoku.items():
                if 0 <= kyoku_idx < len(kyoku_values) and not kyoku_has_hu[kyoku_idx]:
                    kyoku_values[kyoku_idx] += tenpai_reward if shanten <= 0 else noten_penalty
        data.value_targets = [
            float(kyoku_values[kyoku_idx]) if 0 <= kyoku_idx < len(kyoku_values) else 0.0
            for kyoku_idx in data.at_kyoku
        ]
        return data

    @staticmethod
    def build_terminal_value_targets(
        events: list[Event],
        player_id: int,
        self_draw_reward: float = 2.0,
        win_reward: float = 1.5,
        deal_in_penalty: float = -2.0,
        other_self_draw_penalty: float = -1.0,
        other_ron_penalty: float = -0.5,
        score_delta_reward_scale: float = 40.0,
        score_delta_reward_clip: float = 2.0,
        **_,
    ) -> tuple[List[float], List[bool]]:
        values: List[float] = []
        has_hu_values: List[bool] = []
        current_value = 0.0
        current_score_delta = 0.0
        current_has_hu = False
        saw_start_kyoku = False

        def score_delta_reward(score_delta: float) -> float:
            scale = max(1.0e-6, float(score_delta_reward_scale))
            clip = max(0.0, float(score_delta_reward_clip))
            reward = float(score_delta) / scale
            if clip > 0.0:
                reward = max(-clip, min(clip, reward))
            return reward

        def close_kyoku():
            total = current_value
            if current_has_hu:
                total += score_delta_reward(current_score_delta)
            values.append(float(total))
            has_hu_values.append(bool(current_has_hu))

        for ext in events:
            ev = ext.event
            if isinstance(ev, Event.StartKyoku):
                if saw_start_kyoku:
                    close_kyoku()
                saw_start_kyoku = True
                current_value = 0.0
                current_score_delta = 0.0
                current_has_hu = False
            elif isinstance(ev, Event.Hu):
                current_has_hu = True
                if ev.deltas is not None and 0 <= player_id < len(ev.deltas):
                    current_score_delta += float(ev.deltas[player_id])
                winner = int(ev.player)
                target = int(ev.target)
                if winner == player_id:
                    current_value += float(self_draw_reward) if target == player_id else float(win_reward)
                elif target == player_id:
                    current_value += float(deal_in_penalty)
                elif target == winner:
                    current_value += float(other_self_draw_penalty)
                else:
                    current_value += float(other_ron_penalty)
            elif isinstance(ev, Event.EndKyoku):
                if saw_start_kyoku:
                    close_kyoku()
                    saw_start_kyoku = False
                    current_value = 0.0
                    current_score_delta = 0.0
                    current_has_hu = False
        if saw_start_kyoku:
            close_kyoku()
        if not values:
            values.append(0.0)
            has_hu_values.append(False)
        return values, has_hu_values

    @staticmethod
    def encode_chow_label(tile_id: int, chi_type: ChiType) -> int:
        suit = tile_id // 9
        number = tile_id % 9
        if chi_type == ChiType.Low:
            seq_start = number
            variant = 0
        elif chi_type == ChiType.Mid:
            seq_start = number - 1
            variant = 1
        else:
            seq_start = number - 2
            variant = 2
        return CHOW_BASE + suit * 21 + seq_start * 3 + variant

    def extend_from_event_window(self, ctx: 'LoaderContext', window: list[Event]) -> None:
        cur = window[0]
        next_event = window[1]

        if isinstance(cur.event, Event.StartGame):
            self.player_name = cur.event.names[self.player_id]
        elif isinstance(cur.event, Event.EndKyoku):
            ctx.kyoku_idx += 1

        cans = ctx.state.update(cur)
        # 当前状态没有任何可执行动作时，不产生训练样本；只有真正需要模型
        # 决策的时刻才记录 obs、合法动作 mask 和专家 label。
        if not cans.can_act:
            return

        label = None
        if isinstance(next_event.event, Event.Dahai):
            label = DISCARD_BASE + next_event.event.tile.id
        elif isinstance(next_event.event, Event.Pon) and next_event.event.player == self.player_id:
            label = PUNG_BASE + next_event.event.tile.id
        elif isinstance(next_event.event, Event.MinGang) and next_event.event.player == self.player_id:
            label = MINGGANG_BASE + next_event.event.tile.id
        elif isinstance(next_event.event, Event.BuGang) and next_event.event.player == self.player_id:
            label = ADDKONG_BASE + next_event.event.tile.id
        elif isinstance(next_event.event, Event.AnGang) and next_event.event.player == self.player_id:
            label = ANKANG_BASE + next_event.event.consumed[0].id
        elif isinstance(next_event.event, Event.Chi) and next_event.event.player == self.player_id:
            label = self.encode_chow_label(
                next_event.event.tile.id,
                ChiType.from_tiles(next_event.event.consumed, next_event.event.tile),
            )
        else:
            has_any_ron = isinstance(next_event.event, Event.Hu)
            if has_any_ron and next_event.event.player == self.player_id:
                label = WIN_INDEX
            else:
                pass_case = (
                    (cans.can_chi and isinstance(next_event.event, Event.Tsumo))
                    or ((cans.can_daiminkan or cans.can_ron_hu or cans.can_pon) and not has_any_ron)
                )
                if pass_case:
                    label = PASS_INDEX

        if label is not None:
            self.add_entry(ctx, label)

    def add_entry(self, ctx: 'LoaderContext', label: int) -> None:
        # label 是 0~234 的单一分类目标；mask 用于判断预测是否违反规则。
        # 这一步完成了“状态 -> 特征工程 -> 监督样本”的最后连接。
        obs, mask = ctx.state.encode_obs(False)
        self.obs.append(obs)
        self.actions.append(label)
        self.masks.append(mask)
        self.at_kyoku.append(ctx.kyoku_idx)
        self.apply_gamma.append(DISCARD_BASE <= label < DISCARD_BASE + 34)
        self.at_turns.append(ctx.state.at_turn)
        self.shantens.append(ctx.state.shanten)


class LoaderContext:
    def __init__(self, config: GameplayLoader, state: PlayerState, kyoku_idx: int):
        self.config = config
        self.state = state
        self.kyoku_idx = kyoku_idx
