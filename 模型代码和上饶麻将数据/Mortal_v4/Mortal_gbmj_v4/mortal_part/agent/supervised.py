import copy
import threading
import time
from threading import Lock
from typing import List, Optional

import numpy as np

from mortal_part.agent.defs import BatchAgent, InvisibleState
from mortal_part.consts import (
    ACTION_SPACE,
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
from mortal_part.mjai.event import Event, EventExt, Metadata
from mortal_part.state.mah_player_gb import PlayerState
from mortal_part.tile import Tile


class WaitGroup:
    def __init__(self):
        self.count = 0
        self.cond = threading.Condition()

    def add(self, delta=1):
        with self.cond:
            self.count += delta

    def done(self):
        with self.cond:
            self.count -= 1
            if self.count <= 0:
                self.cond.notify_all()

    def wait(self):
        with self.cond:
            while self.count > 0:
                self.cond.wait()


class SyncFields:
    def __init__(self):
        self.states = []
        self.history_states = []
        self.history_actions = []
        self.masks = []
        self.action_idxs = {}


class SupervisedBatchAgent(BatchAgent):
    # [V4 local-play] BatchAgent adapter for v4's 235-way supervised
    # policy.  This mirrors MortalBatchAgent's arena contract but uses
    # the paper/v3 action layout instead of the old 41-action layout.
    def __init__(self, engine, player_ids: List[int]):
        if not all(0 <= player_id <= 3 for player_id in player_ids):
            raise ValueError("Player IDs must be within 0 to 3.")

        self.engine = engine
        # [V4 local-play v3-compat] Each local-play side can request its own
        # observation layout: v4/current=205 channels, v3/old=194 channels.
        self.obs_version = int(getattr(engine, "obs_version", 4))
        self.enable_quick_eval = engine.enable_quick_eval
        self.enable_rule_based_agari_guard = engine.enable_rule_based_agari_guard
        self.name_value = engine.name
        self.player_ids = player_ids

        self.actions = []
        self.q_values = []
        self.masks_recv = []
        self.is_greedy = []
        self.last_eval_elapsed = 0.0
        self.last_batch_size = 0

        self.evaluated = False
        self.quick_eval_reactions = [None] * len(player_ids) if self.enable_quick_eval else []
        self.sync_fields = SyncFields()
        self.lock = Lock()
        self.wg = WaitGroup()
        # [V4 history-hierarchical] Runtime keeps the last K own model
        # decisions so the Transformer sees the same kind of history as SL/RL.
        self.history_len = int(getattr(engine, "history_len", 0))
        self.history_states = [[] for _ in player_ids]
        self.history_action_ids = [[] for _ in player_ids]

    def _clear_batch_if_needed(self):
        if self.evaluated:
            self.sync_fields = SyncFields()
            self.actions = []
            self.q_values = []
            self.masks_recv = []
            self.is_greedy = []
            self.evaluated = False

    def name(self) -> str:
        return self.name_value

    def start_game(self, index: int) -> None:
        self.history_states[index] = []
        self.history_action_ids[index] = []

    def oracle_obs_version(self) -> Optional[int]:
        return None

    def evaluate(self):
        self.wg.wait()
        with self.lock:
            if not self.sync_fields.states:
                return
            start = time.time()
            self.last_batch_size = len(self.sync_fields.states)
            states = [np.asarray(state, dtype=np.float32) for state in self.sync_fields.states]
            masks = [np.asarray(mask, dtype=np.bool_) for mask in self.sync_fields.masks]
            history_states = [np.asarray(state, dtype=np.float32) for state in self.sync_fields.history_states]
            history_actions = [np.asarray(action, dtype=np.int64) for action in self.sync_fields.history_actions]
            self.actions, self.q_values, self.masks_recv, self.is_greedy = self.engine.react_batch(
                states,
                masks,
                None,
                history_states if history_states else None,
                history_actions if history_actions else None,
            )
            self.last_eval_elapsed = time.time() - start

    def gen_meta(self, state: PlayerState, action_idx: int):
        q_values = self.q_values[action_idx]
        masks = self.masks_recv[action_idx]
        is_greedy = self.is_greedy[action_idx]

        mask_bits = 0
        q_values_compact = []
        for idx, (q, m) in enumerate(zip(q_values, masks)):
            if m:
                mask_bits |= 1 << idx
                q_values_compact.append(float(q))

        return Metadata(
            q_values=q_values_compact,
            mask_bits=mask_bits,
            is_greedy=bool(is_greedy),
            batch_size=self.last_batch_size,
            shanten=state.shanten,
        )

    def set_scene(self, index: int, _: List[EventExt], state: PlayerState,
                  invisible_state: Optional[InvisibleState]) -> None:
        self._clear_batch_if_needed()
        cans = state.last_cans

        if self.enable_quick_eval and cans.can_discard and not cans.can_tsumo_hu and not cans.can_ankan and not cans.can_kakan:
            candidates = state.discard_candidates()
            only_candidate = None
            for tile_id, flag in enumerate(candidates):
                if flag:
                    if only_candidate is None:
                        only_candidate = tile_id
                    else:
                        only_candidate = None
                        break
            if only_candidate is not None:
                actor = self.player_ids[index]
                self.quick_eval_reactions[index] = Event("Dahai", player=actor, tile=Tile(only_candidate))
                return

        state_clone = copy.deepcopy(state)

        def process_features():
            feature, mask = state_clone.encode_obs(False, obs_version=self.obs_version)
            if self.history_len > 0:
                pad_count = max(0, self.history_len - len(self.history_states[index]))
                zero_feature = np.zeros_like(feature, dtype=np.float32)
                history_states = (
                    [zero_feature] * pad_count
                    + self.history_states[index][-self.history_len:]
                )
                history_actions = (
                    [ACTION_SPACE] * pad_count
                    + self.history_action_ids[index][-self.history_len:]
                )
                history_states = np.stack(history_states, axis=0).astype(np.float32, copy=False)
                history_actions = np.asarray(history_actions, dtype=np.int64)
            else:
                history_states = np.zeros((0,) + tuple(feature.shape), dtype=np.float32)
                history_actions = np.zeros((0,), dtype=np.int64)
            with self.lock:
                self.sync_fields.action_idxs[index] = len(self.sync_fields.states)
                self.sync_fields.states.append(feature)
                self.sync_fields.history_states.append(history_states)
                self.sync_fields.history_actions.append(history_actions)
                self.sync_fields.masks.append(mask)
            self.wg.done()

        self.wg.add(1)
        process_features()

    @staticmethod
    def _decode_chow_consumed(action: int, target_tile: Tile):
        variant = (action - CHOW_BASE) % 3
        if variant == 0:
            return [target_tile.next(), target_tile.next().next()]
        if variant == 1:
            return [target_tile.prev(), target_tile.next()]
        return [target_tile.prev().prev(), target_tile.prev()]

    def _fallback_without_hu(self, action_idx: int):
        q_values = list(self.q_values[action_idx])
        q_values[WIN_INDEX] = -1.0e9
        return max(range(len(q_values)), key=lambda idx: q_values[idx])

    def get_reaction(self, index: int, events: List[EventExt], state: PlayerState,
                     invisible_state: Optional[InvisibleState]) -> EventExt:
        if self.enable_quick_eval and self.quick_eval_reactions[index] is not None:
            ev = self.quick_eval_reactions[index]
            self.quick_eval_reactions[index] = None
            # [V4 history-hierarchical] Even forced single-discard shortcuts
            # must be visible to the history Transformer at later decisions.
            feature, _ = state.encode_obs(False, obs_version=self.obs_version)
            self.history_states[index].append(np.asarray(feature, dtype=np.float32).copy())
            if isinstance(ev.event, Event.Dahai):
                self.history_action_ids[index].append(int(DISCARD_BASE + ev.event.tile.id))
            else:
                self.history_action_ids[index].append(int(PASS_INDEX))
            return EventExt.no_meta(ev)

        if not self.evaluated:
            self.evaluate()
            self.evaluated = True

        start = time.time()
        with self.lock:
            action_idx = self.sync_fields.action_idxs[index]
            actor = self.player_ids[index]
            cans = state.last_cans
            action = int(self.actions[action_idx])

            if self.enable_rule_based_agari_guard and action == WIN_INDEX and not state.rule_based_agari():
                action = self._fallback_without_hu(action_idx)

            event = Event("NoneEvent")

            if action == PASS_INDEX:
                event = Event("NoneEvent")

            elif DISCARD_BASE <= action < DISCARD_BASE + DISCARD_COUNT:
                assert cans.can_discard, "failed discard check: %s" % state.brief_info()
                event = Event("Dahai", player=actor, tile=Tile(action - DISCARD_BASE))

            elif action == WIN_INDEX:
                assert cans.can_hu, "failed hu check: %s" % state.brief_info()
                tile = state.last_kawa_tile if actor != cans.target_actor else state.last_self_tsumo
                event = Event("Hu", player=actor, target=cans.target_actor, tile=tile, fan=0, deltas=None)

            elif MINGGANG_BASE <= action < MINGGANG_BASE + MINGGANG_COUNT:
                assert cans.can_daiminkan, "failed minggang check: %s" % state.brief_info()
                tile = state.last_kawa_tile or Tile(action - MINGGANG_BASE)
                event = Event("MinGang", player=actor, tile=tile, target=cans.target_actor, consumed=[tile, tile, tile])

            elif ANKANG_BASE <= action < ANKANG_BASE + ANKANG_COUNT:
                tile = Tile(action - ANKANG_BASE)
                assert cans.can_ankan and tile in state.ankan_candidates, "failed ankang check: %s" % state.brief_info()
                event = Event("AnGang", player=actor, consumed=[tile, tile, tile, tile])

            elif ADDKONG_BASE <= action < ADDKONG_BASE + ADDKONG_COUNT:
                tile = Tile(action - ADDKONG_BASE)
                assert cans.can_kakan and tile in state.kakan_candidates, "failed add-kong check: %s" % state.brief_info()
                event = Event("BuGang", player=actor, tile=tile, consumed=[tile])

            elif PUNG_BASE <= action < PUNG_BASE + PUNG_COUNT:
                assert cans.can_pon, "failed pung check: %s" % state.brief_info()
                tile = state.last_kawa_tile or Tile(action - PUNG_BASE)
                event = Event("Pon", player=actor, tile=tile, target=cans.target_actor, consumed=[tile, tile])

            elif CHOW_BASE <= action < CHOW_BASE + CHOW_COUNT:
                target_tile = state.last_kawa_tile
                assert target_tile is not None and cans.can_chi, "failed chow check: %s" % state.brief_info()
                consumed = self._decode_chow_consumed(action, target_tile)
                event = Event("Chi", player=actor, target=cans.target_actor, tile=target_tile, consumed=consumed)

            else:
                raise ValueError("unexpected 235-way action id: %s" % action)

            meta = self.gen_meta(state, action_idx)
            meta.eval_time_ns = int((time.time() - start + self.last_eval_elapsed) * 1.0e9)
            meta.batch_size = self.last_batch_size
            # [V4 history-hierarchical] Append the realized model decision
            # after it has passed rule guards/fallbacks.
            feature = np.asarray(self.sync_fields.states[action_idx], dtype=np.float32)
            self.history_states[index].append(feature.copy())
            self.history_action_ids[index].append(int(action))
            return EventExt(event=event, meta=meta)
