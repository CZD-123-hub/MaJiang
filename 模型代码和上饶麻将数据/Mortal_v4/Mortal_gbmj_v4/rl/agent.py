import copy
import time
from typing import List, Optional

import numpy as np

from mortal_part.agent.defs import BatchAgent, InvisibleState
from mortal_part.agent.supervised import SyncFields, WaitGroup
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
from mortal_part.dataset.grp import global_stage_features_from_state
from rl.reward import MJRMRewardShaper, critic_features_from_info


class PPOBatchAgent(BatchAgent):
    """[V4 potential reward] Batch arena agent that records PPO trajectories."""

    def __init__(self, engine, player_ids: List[int], rank_bonus=None, reward_shaper=None):
        self.engine = engine
        self.enable_quick_eval = engine.enable_quick_eval
        self.enable_rule_based_agari_guard = engine.enable_rule_based_agari_guard
        self.name_value = engine.name
        self.player_ids = player_ids
        self.rank_bonus = list(rank_bonus or [0.0, 0.0, 0.0, 0.0])
        # [V4 potential reward] Uses main-fan potential shaping plus terminal anchors.
        self.reward_shaper = reward_shaper or MJRMRewardShaper.from_config({})

        self.actions = []
        self.q_values = []
        self.masks_recv = []
        self.is_greedy = []
        self.log_probs = []
        self.values = []
        self.last_eval_elapsed = 0.0
        self.last_batch_size = 0

        self.evaluated = False
        self.quick_eval_reactions = [None] * len(player_ids) if self.enable_quick_eval else []
        self.sync_fields = SyncFields()
        self.lock = __import__("threading").Lock()
        self.wg = WaitGroup()

        self.current_kyoku = [[] for _ in player_ids]
        self.kyoku_segments = [[] for _ in player_ids]
        self.completed_transitions = []
        # [V4 history-hierarchical] PPO rollout uses the same K-step own
        # decision history as supervised training and local inference.
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
            self.log_probs = []
            self.values = []
            self.evaluated = False

    def name(self) -> str:
        return self.name_value

    def oracle_obs_version(self) -> Optional[int]:
        return None

    def start_game(self, index: int) -> None:
        self.current_kyoku[index] = []
        self.kyoku_segments[index] = []
        self.history_states[index] = []
        self.history_action_ids[index] = []

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
            out = self.engine.react_batch(
                states,
                masks,
                None,
                history_states if history_states else None,
                history_actions if history_actions else None,
            )
            self.actions = out["actions"]
            self.q_values = out["logits"]
            self.masks_recv = out["masks"]
            self.is_greedy = out["is_greedy"]
            self.log_probs = out["log_probs"]
            self.values = out["values"]
            self.last_eval_elapsed = time.time() - start

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
            feature, mask = state_clone.encode_obs(False)
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

    def gen_meta(self, state: PlayerState, action_idx: int):
        q_values = self.q_values[action_idx]
        masks = self.masks_recv[action_idx]
        mask_bits = 0
        q_values_compact = []
        for idx, (q, m) in enumerate(zip(q_values, masks)):
            if m:
                mask_bits |= 1 << idx
                q_values_compact.append(float(q))
        return Metadata(
            q_values=q_values_compact,
            mask_bits=mask_bits,
            is_greedy=False,
            batch_size=self.last_batch_size,
            shanten=state.shanten,
        )

    def _record_transition(self, index, action_idx, action, state: PlayerState, recordable=True):
        if not recordable:
            return

        obs = np.asarray(self.sync_fields.states[action_idx], dtype=np.float32)
        reward_info = self.reward_shaper.describe_state_action(state, action, obs=obs)
        # [V4 asymmetric critic] Store dense route-potential summaries only for
        # PPO value learning.  They are not part of the actor observation and
        # therefore do not affect platform inference compatibility.
        critic_features = critic_features_from_info(reward_info)
        grp_features = global_stage_features_from_state(state)

        # [V4 potential reward] The previous own decision receives
        # lambda * (gamma * Score(next) - Score(prev)) once the next own state is known.
        if self.current_kyoku[index]:
            prev = self.current_kyoku[index][-1]
            reward, comps = self.reward_shaper.shanten_transition_reward(prev["reward_info"], reward_info)
            prev["reward"] += float(reward)
            self.reward_shaper.add_components(prev, comps)

        transition = {
            "obs": obs.copy(),
            "history_obs": np.asarray(self.sync_fields.history_states[action_idx], dtype=np.float32).copy(),
            "history_actions": np.asarray(self.sync_fields.history_actions[action_idx], dtype=np.int64).copy(),
            "mask": np.asarray(self.sync_fields.masks[action_idx], dtype=np.bool_).copy(),
            "action": int(action),
            "old_log_prob": float(self.log_probs[action_idx]),
            "value": float(self.values[action_idx]),
            "reward": 0.0,
            "done": False,
            "reward_info": reward_info,
            "critic_features": critic_features,
            "grp_features": grp_features,
            "reward_components": {},
        }

        # [V4 potential reward] Kept as a no-op hook for ablation/backward compatibility.
        if not self.current_kyoku[index]:
            reward, comps = self.reward_shaper.opening_reward(state)
            transition["reward"] += float(reward)
            self.reward_shaper.add_components(transition, comps)

        # [V4 potential reward] Kept as a no-op hook; melds are rewarded only by potential/outcome.
        reward, comps = self.reward_shaper.action_reward(action)
        transition["reward"] += float(reward)
        self.reward_shaper.add_components(transition, comps)

        reward, comps = self.reward_shaper.state_action_dense_reward(reward_info)
        transition["reward"] += float(reward)
        self.reward_shaper.add_components(transition, comps)

        self.current_kyoku[index].append(transition)

    def get_reaction(self, index: int, events: List[EventExt], state: PlayerState,
                     invisible_state: Optional[InvisibleState]) -> EventExt:
        if self.enable_quick_eval and self.quick_eval_reactions[index] is not None:
            ev = self.quick_eval_reactions[index]
            self.quick_eval_reactions[index] = None
            # [V4 history-hierarchical] Keep forced single-discard shortcuts in
            # the recurrent decision history, even though they are not PPO
            # training samples.
            feature, _ = state.encode_obs(False)
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
            sampled_action = int(self.actions[action_idx])
            action = sampled_action
            recordable = True

            if self.enable_rule_based_agari_guard and action == WIN_INDEX and not state.rule_based_agari():
                action = self._fallback_without_hu(action_idx)
                # [V4 MJ_RM strict DPPO] Skip overridden Hu samples because
                # their stored log-prob belongs to the rejected sampled action.
                recordable = False

            event = Event("NoneEvent")

            if action == PASS_INDEX:
                event = Event("NoneEvent")
            elif DISCARD_BASE <= action < DISCARD_BASE + DISCARD_COUNT:
                event = Event("Dahai", player=actor, tile=Tile(action - DISCARD_BASE))
            elif action == WIN_INDEX:
                tile = state.last_kawa_tile if actor != cans.target_actor else state.last_self_tsumo
                event = Event("Hu", player=actor, target=cans.target_actor, tile=tile, fan=0, deltas=None)
            elif MINGGANG_BASE <= action < MINGGANG_BASE + MINGGANG_COUNT:
                tile = state.last_kawa_tile or Tile(action - MINGGANG_BASE)
                event = Event("MinGang", player=actor, tile=tile, target=cans.target_actor, consumed=[tile, tile, tile])
            elif ANKANG_BASE <= action < ANKANG_BASE + ANKANG_COUNT:
                tile = Tile(action - ANKANG_BASE)
                event = Event("AnGang", player=actor, consumed=[tile, tile, tile, tile])
            elif ADDKONG_BASE <= action < ADDKONG_BASE + ADDKONG_COUNT:
                tile = Tile(action - ADDKONG_BASE)
                event = Event("BuGang", player=actor, tile=tile, consumed=[tile])
            elif PUNG_BASE <= action < PUNG_BASE + PUNG_COUNT:
                tile = state.last_kawa_tile or Tile(action - PUNG_BASE)
                event = Event("Pon", player=actor, tile=tile, target=cans.target_actor, consumed=[tile, tile])
            elif CHOW_BASE <= action < CHOW_BASE + CHOW_COUNT:
                target_tile = state.last_kawa_tile
                consumed = self._decode_chow_consumed(action, target_tile)
                event = Event("Chi", player=actor, target=cans.target_actor, tile=target_tile, consumed=consumed)
            else:
                raise ValueError("unexpected 235-way action id: %s" % action)

            self._record_transition(index, action_idx, action, state, recordable=recordable)
            # [V4 history-hierarchical] History records the realized action even
            # when PPO skips the sample, e.g. after rule-based Hu fallback.
            feature = np.asarray(self.sync_fields.states[action_idx], dtype=np.float32)
            self.history_states[index].append(feature.copy())
            self.history_action_ids[index].append(int(action))

            meta = self.gen_meta(state, action_idx)
            meta.eval_time_ns = int((time.time() - start + self.last_eval_elapsed) * 1.0e9)
            meta.batch_size = self.last_batch_size
            return EventExt(event=event, meta=meta)

    def end_kyoku(self, index: int) -> None:
        self.kyoku_segments[index].append(self.current_kyoku[index])
        self.current_kyoku[index] = []

    def end_game(self, index: int, game_result) -> None:
        if self.current_kyoku[index]:
            self.kyoku_segments[index].append(self.current_kyoku[index])
            self.current_kyoku[index] = []

        player_id = self.player_ids[index]
        last_segment_idx = -1
        for idx in range(len(self.kyoku_segments[index]) - 1, -1, -1):
            if self.kyoku_segments[index][idx]:
                last_segment_idx = idx
                break
        game_rank = int(game_result.rankings().rank_by_player[player_id])
        game_rank_reward = 0.0
        if 0 <= game_rank < len(self.rank_bonus):
            game_rank_reward = float(self.rank_bonus[game_rank])

        for seg_idx, segment in enumerate(self.kyoku_segments[index]):
            if not segment:
                continue
            # [V4 RL framework audit] Close the potential-shaping trajectory
            # for the final own decision in this hand.  Intermediate decisions
            # are closed when the next own state is observed in _record_transition.
            reward, comps = self.reward_shaper.terminal_potential_reward(segment[-1].get("reward_info", {}))
            segment[-1]["reward"] += float(reward)
            self.reward_shaper.add_components(segment[-1], comps)

            reward, comps = self.reward_shaper.terminal_kyoku_reward(game_result, player_id, seg_idx, segment)
            segment[-1]["reward"] += float(reward)
            self.reward_shaper.add_components(segment[-1], comps)
            self.reward_shaper.apply_dangerous_push_penalty(segment, comps)
            # [V4 deal-in control] Spread part of the hand outcome backward
            # across this hand's own decisions, so win/deal-in signals do not
            # land only on the final recorded action.
            self.reward_shaper.redistribute_terminal_reward(segment, float(reward))
            if seg_idx == last_segment_idx and game_rank_reward != 0.0:
                segment[-1]["reward"] += game_rank_reward
                self.reward_shaper.add_components(segment[-1], {
                    "terminal_game_rank_bonus": game_rank_reward,
                    "terminal_game_rank_%d" % (game_rank + 1): 1.0,
                })
            segment[-1]["done"] = True
            self.completed_transitions.extend(segment)

        self.kyoku_segments[index] = []

    def pop_transitions(self):
        transitions = self.completed_transitions
        self.completed_transitions = []
        return transitions
