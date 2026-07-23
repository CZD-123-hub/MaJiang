import numpy as np

from mortal_part.consts import ORACLE_EXTRA_CHANNELS, oracle_obs_shape
from mortal_part.mjai.event import Event


class OracleTracker:
    """[V4] Training-only privileged information tracker.

    This tracker reconstructs hidden hands from complete logs.  Its output is
    used only by the teacher branch, so the deployed student still receives
    visible information plus legal runtime features only.
    """

    def __init__(self):
        self.hands = [[0] * 34 for _ in range(4)]
        self.started = False

    def _clear(self):
        self.hands = [[0] * 34 for _ in range(4)]
        self.started = False

    @staticmethod
    def _dec(counts, tile_id, amount=1):
        for _ in range(int(amount)):
            if 0 <= tile_id < 34 and counts[tile_id] > 0:
                counts[tile_id] -= 1

    @staticmethod
    def _inc(counts, tile_id, amount=1):
        if 0 <= tile_id < 34:
            counts[tile_id] += int(amount)

    def update(self, wrapper):
        event = wrapper.event
        if isinstance(event, Event.StartKyoku):
            self.hands = [[0] * 34 for _ in range(4)]
            for player, tiles in enumerate(event.tehais):
                for tile in tiles:
                    self._inc(self.hands[player], tile.id)
            self.started = True
            return
        if isinstance(event, Event.EndKyoku):
            self._clear()
            return
        if not self.started:
            return

        if isinstance(event, Event.Tsumo):
            self._inc(self.hands[event.player], event.tile.id)
        elif isinstance(event, Event.Dahai):
            self._dec(self.hands[event.player], event.tile.id)
        elif isinstance(event, (Event.Chi, Event.Pon, Event.MinGang)):
            for tile in event.consumed:
                self._dec(self.hands[event.player], tile.id)
        elif isinstance(event, Event.BuGang):
            for tile in event.consumed:
                self._dec(self.hands[event.player], tile.id)
        elif isinstance(event, Event.AnGang):
            for tile in event.consumed:
                self._dec(self.hands[event.player], tile.id)

    @staticmethod
    def _set_tile(arr, plane, tile_id, value):
        arr[plane, tile_id // 9, tile_id % 9] = np.float32(value)

    @staticmethod
    def _encode_count_planes(arr, start, counts, plane_count):
        for tile_id, count in enumerate(counts):
            capped = min(max(int(count), 0), plane_count)
            for level in range(capped):
                OracleTracker._set_tile(arr, start + level, tile_id, 1.0)

    def build_obs(self, visible_obs, state, player_id):
        extra = np.zeros((ORACLE_EXTRA_CHANNELS, 4, 9), dtype=np.float32)

        # [V4] 3 planes: hidden hands of next, opposite, previous player.
        for rel in range(1, 4):
            abs_player = (int(player_id) + rel) % 4
            for tile_id, count in enumerate(self.hands[abs_player]):
                if count > 0:
                    self._set_tile(extra, rel - 1, tile_id, min(float(count) / 4.0, 1.0))

        # [V4] 4 planes: remaining wall estimate after subtracting public tiles
        # and all reconstructed hands.  This is privileged during training.
        hidden_sum = [0] * 34
        for player_counts in self.hands:
            for tile_id, count in enumerate(player_counts):
                hidden_sum[tile_id] += int(count)
        public = getattr(state, "tiles_public", [0] * 34)
        wall = [max(0, min(4, 4 - int(public[i]) - int(hidden_sum[i]))) for i in range(34)]
        self._encode_count_planes(extra, 3, wall, 4)

        oracle_obs = np.concatenate([visible_obs.astype(np.float32, copy=False), extra], axis=0)
        if oracle_obs.shape != oracle_obs_shape:
            raise ValueError(f"oracle obs shape mismatch: {oracle_obs.shape} != {oracle_obs_shape}")
        return oracle_obs
