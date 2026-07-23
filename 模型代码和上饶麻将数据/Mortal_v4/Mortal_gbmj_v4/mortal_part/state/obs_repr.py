import math

import numpy as np

from mortal_part.chi_type import ChiType
from mortal_part.consts import (
    ACTION_SPACE,
    ADDKONG_BASE,
    ANKANG_BASE,
    CHOW_BASE,
    DISCARD_BASE,
    MINGGANG_BASE,
    PASS_INDEX,
    PUNG_BASE,
    WIN_INDEX,
    obs_shape,
)
from mortal_part.state.mah_player_gb import PlayerState


"""监督学习的特征工程入口。

麻将原始状态不能直接送入 CNN，因此这里把 PlayerState 转成固定大小的
浮点特征图 ``[C, 4, 9]``。4x9 的网格对应 34 张牌（最后几格用于字牌），
每个通道表示一种可学习的牌面事实，例如牌的数量、可见性、弃牌历史、
听牌状态和当前可执行动作。与此同时，``mask`` 保存当前状态下合法的
235 个动作，供训练评估和推理阶段约束输出。
"""
DISCARD_HISTORY_LEN = 28
SHANTEN_CAP = 6
BASE_OBS_CHANNELS = obs_shape[0]
FORESIGHT_OBS_CHANNELS = BASE_OBS_CHANNELS + 11


def obs_shape_for_version(version):
    # [V4 direct-235 no-foresight] Version 4 is the default 194-channel
    # visible encoder.  Version 5 is reserved only for loading old 205-channel
    # foresight checkpoints in local comparisons.
    if int(version) >= 5:
        return (FORESIGHT_OBS_CHANNELS, obs_shape[1], obs_shape[2])
    return obs_shape


class ObsEncoderContext:
    """[V4 direct-235 no-foresight] Visible core by default; v5 keeps 205-channel compatibility."""

    def __init__(self, state: PlayerState, version: int, at_kan_select: bool):
        self.state = state
        self.idx = 0
        self.version = int(version)
        self.at_kan_select = at_kan_select
        self.arr = np.zeros(obs_shape_for_version(self.version), dtype=np.float32)
        self.mask = np.zeros(ACTION_SPACE, dtype=bool)

    def encode(self):
        # 编码顺序就是模型输入通道的顺序；改变顺序会导致旧 checkpoint 失效。
        # 先编码牌面和历史，再编码规则/动作上下文，最后构造合法动作 mask。
        # Paper-compatible visible core, with a longer discard history.
        self.encode_wind(self.state.seat_wind)
        self.encode_wind(self.state.round_wind)
        self.encode_count_planes(self.state.tiles_public, 4)
        self.encode_count_planes(self.state.handcards, 4)
        self.encode_player_meld_features()
        self.encode_discard_history()

        # Practical v3 additions from already-maintained PlayerState fields.
        self.encode_visibility_features()
        self.encode_hand_progress_features()
        self.encode_king_feature()
        if self.version >= 5:
            self.encode_foresight_features()
        self.encode_action_context()
        self.encode_discard_recency()
        self.encode_table_scalar_features()
        # [V4 foresight feature] Do not append SPCalculator expectation-value planes.
        # The new lookahead is the compact 7-route + 4-top-discard feature block.
        self.build_action_mask()

        assert self.idx == self.arr.shape[0], f"obs encoder row mismatch: {self.idx} != {self.arr.shape[0]}"
        return self.arr, self.mask

    def clip01(self, value):
        return np.float32(min(max(float(value), 0.0), 1.0))

    def tile_pos(self, tile_id):
        return tile_id // 9, tile_id % 9

    def set_tile(self, plane_idx, tile_id, value=1.0):
        row, col = self.tile_pos(tile_id)
        self.arr[plane_idx, row, col] = np.float32(value)

    def set_tile_max(self, plane_idx, tile_id, value=1.0):
        row, col = self.tile_pos(tile_id)
        self.arr[plane_idx, row, col] = max(self.arr[plane_idx, row, col], np.float32(value))

    def fill_plane(self, value):
        self.arr[self.idx, :, :] = np.float32(value)
        self.idx += 1

    def encode_wind(self, wind_idx):
        self.set_tile(self.idx, 27 + (int(wind_idx) % 4), 1.0)
        self.idx += 1

    def encode_count_planes(self, counts, plane_count):
        # 用多个二值平面表达“数量”：例如 count=3 时点亮前 3 个平面，
        # 比把数量直接压成一个标量更容易让卷积学习牌的组合关系。
        for tile_id, count in enumerate(counts):
            capped = min(max(int(count), 0), plane_count)
            for level in range(capped):
                self.set_tile(self.idx + level, tile_id, 1.0)
        self.idx += plane_count

    def encode_scaled_counts(self, counts, scale):
        for tile_id, count in enumerate(counts):
            self.set_tile(self.idx, tile_id, self.clip01(float(count) / float(scale)))
        self.idx += 1

    def encode_bool_tiles(self, flags):
        for tile_id, flag in enumerate(flags):
            if flag:
                self.set_tile(self.idx, tile_id, 1.0)
        self.idx += 1

    def classify_fuuro(self, fulu):
        ids = sorted(tile.id for tile in fulu)
        if len(ids) == 3 and ids[0] == ids[1] == ids[2]:
            return "pong", ids[0]
        if len(ids) == 4 and ids[0] == ids[1] == ids[2] == ids[3]:
            return "kong", ids[0]
        if len(ids) == 3 and ids[0] + 1 == ids[1] and ids[1] + 1 == ids[2] and ids[0] // 9 == ids[2] // 9 and ids[0] < 27:
            return "chi", ids
        return "other", ids

    def encode_player_meld_features(self):
        for player_fuuro in self.state.fulu_overview:
            chi_counts = [0] * 34
            pong_tiles = set()
            kong_tiles = set()

            for fulu in player_fuuro:
                kind, payload = self.classify_fuuro(fulu)
                if kind == "chi":
                    for tile_id in payload:
                        chi_counts[tile_id] += 1
                elif kind == "pong":
                    pong_tiles.add(payload)
                elif kind == "kong":
                    kong_tiles.add(payload)

            self.encode_count_planes(chi_counts, 4)
            for tile_id in sorted(pong_tiles):
                self.set_tile(self.idx, tile_id, 1.0)
            self.idx += 1
            for tile_id in sorted(kong_tiles):
                self.set_tile(self.idx, tile_id, 1.0)
            self.idx += 1

    def encode_discard_history(self):
        for player_kawa in self.state.kawa_overview:
            for i in range(DISCARD_HISTORY_LEN):
                if i < len(player_kawa):
                    self.set_tile(self.idx + i, player_kawa[i].id, 1.0)
            self.idx += DISCARD_HISTORY_LEN

    def encode_visibility_features(self):
        seen = [min(max(int(count), 0), 4) for count in self.state.tiles_seen]
        remaining = [4 - count for count in seen]
        dead = [count >= 4 for count in seen]
        self.encode_scaled_counts(seen, 4)
        self.encode_scaled_counts(remaining, 4)
        self.encode_bool_tiles(dead)

    def encode_hand_progress_features(self):
        self.encode_bool_tiles(self.state.waits)

        shanten = min(max(int(self.state.shanten), 0), SHANTEN_CAP)
        for value in range(SHANTEN_CAP + 1):
            self.fill_plane(1.0 if shanten == value else 0.0)

        cans = self.state.last_cans
        if cans.can_discard:
            discard_candidates = self.state.discard_candidates()
            keep_discards = self.state.keep_shanten_discards
            next_discards = self.state.next_shanten_discards
            has_next = self.state.has_next_shanten_discard
        else:
            discard_candidates = [False] * 34
            keep_discards = [False] * 34
            next_discards = [False] * 34
            has_next = False

        self.encode_bool_tiles(discard_candidates)
        self.encode_bool_tiles(keep_discards)
        self.encode_bool_tiles(next_discards)
        self.fill_plane(1.0 if has_next else 0.0)

    def encode_king_feature(self):
        """追加上饶精牌位置；国标旧日志没有精牌时保持全零。"""
        king_tile_id = getattr(self.state, "king_tile_id", None)
        if king_tile_id is not None:
            self.set_tile(self.idx, int(king_tile_id), 1.0)
        self.idx += 1

    def encode_foresight_features(self):
        """[V4 foresight feature] Paper-style shallow lookahead features.

        Planes:
        - 7 scalar planes: main-route family closeness, in [0, 1]
        - 4 tile planes: top-4 local discard recommendations under D=1 route score
        """
        from mortal_part.state.foresight import FORESIGHT_TOP_DISCARD_PLANES, compute_foresight_features

        route_values, top_discards = compute_foresight_features(self.state)
        for value in route_values:
            self.fill_plane(self.clip01(value))

        top_discards = list(top_discards[:FORESIGHT_TOP_DISCARD_PLANES])
        for tile_id in top_discards:
            self.set_tile(self.idx, int(tile_id), 1.0)
            self.idx += 1
        for _ in range(FORESIGHT_TOP_DISCARD_PLANES - len(top_discards)):
            self.idx += 1

    def encode_action_context(self):
        # 动作上下文告诉模型“现在轮到什么决策”：上家打出的牌、可吃碰杠胡
        # 标志、杠牌候选和相对座位等，属于强规则约束特征。
        cans = self.state.last_cans
        target_tile = self.state.last_kawa_tile
        if target_tile is not None:
            self.set_tile(self.idx, target_tile.id, 1.0)
        self.idx += 1

        flags = (
            cans.can_pass,
            cans.can_discard,
            cans.can_tsumo_hu or cans.can_ron_hu,
            cans.can_tsumo_hu,
            cans.can_ron_hu,
            cans.can_pon,
            cans.can_chi_low,
            cans.can_chi_mid,
            cans.can_chi_high,
            cans.can_daiminkan,
            cans.can_ankan,
            cans.can_kakan,
        )
        for flag in flags:
            self.fill_plane(1.0 if flag else 0.0)

        for tile in self.state.ankan_candidates:
            self.set_tile(self.idx, tile.id, 1.0)
        self.idx += 1
        for tile in self.state.kakan_candidates:
            self.set_tile(self.idx, tile.id, 1.0)
        self.idx += 1

        target_actor = cans.target_actor
        if target_actor is not None:
            target_rel = self.state.rel(target_actor)
            if 0 <= target_rel < 4:
                self.fill_plane(1.0 if target_rel == 0 else 0.0)
                self.fill_plane(1.0 if target_rel == 1 else 0.0)
                self.fill_plane(1.0 if target_rel == 2 else 0.0)
                self.fill_plane(1.0 if target_rel == 3 else 0.0)
                return
        for _ in range(4):
            self.fill_plane(0.0)

    def encode_discard_recency(self):
        for player_kawa in self.state.kawa_overview:
            last_idx = len(player_kawa) - 1
            for turn, tile in enumerate(player_kawa):
                decay = math.exp(-0.2 * float(last_idx - turn))
                self.set_tile_max(self.idx, tile.id, decay)
            self.idx += 1

    def encode_table_scalar_features(self):
        self.fill_plane(self.clip01(float(self.state.tiles_left) / 84.0))
        self.fill_plane(self.clip01(float(self.state.at_turn) / float(DISCARD_HISTORY_LEN)))

        total_score = float(sum(score for score in self.state.scores if score > 0))
        score_denom = total_score if total_score > 0 else 2000.0
        for score in self.state.scores:
            self.fill_plane(self.clip01(float(score) / score_denom))

        self.fill_plane(self.clip01(float(self.state.flower_count) / 8.0))
        self.fill_plane(self.clip01(float(self.state.tehai_len_div3) / 4.0))
        self.fill_plane(1.0 if self.at_kan_select else 0.0)
        self.fill_plane(self.clip01(float(len(self.state.fulu_overview[0])) / 4.0))

    def encode_chow_index(self, tile_id, chi_type):
        suit = tile_id // 9
        number = tile_id % 9
        if suit >= 3:
            raise ValueError(f"honor tile cannot be chow target: {tile_id}")
        if chi_type == ChiType.Low:
            seq_start = number
            variant = 0
        elif chi_type == ChiType.Mid:
            seq_start = number - 1
            variant = 1
        else:
            seq_start = number - 2
            variant = 2
        if not (0 <= seq_start <= 6):
            raise ValueError(f"invalid chow sequence start: tile={tile_id} type={chi_type}")
        return CHOW_BASE + suit * 21 + seq_start * 3 + variant

    def build_action_mask(self):
        # mask 与监督标签共同构成训练样本：label 是专家动作，mask 是当时允许
        # 的动作集合。这里必须和上饶规则一致，否则会把正确动作误判成非法。
        cans = self.state.last_cans
        if cans.can_pass:
            self.mask[PASS_INDEX] = True

        if cans.can_discard:
            for tid, flag in enumerate(self.state.discard_candidates()):
                if flag:
                    self.mask[DISCARD_BASE + tid] = True

        if cans.can_tsumo_hu or cans.can_ron_hu:
            self.mask[WIN_INDEX] = True

        target_tile = self.state.last_kawa_tile
        if cans.can_daiminkan and target_tile is not None:
            self.mask[MINGGANG_BASE + target_tile.id] = True
        if cans.can_ankan:
            for tile in self.state.ankan_candidates:
                self.mask[ANKANG_BASE + tile.id] = True
        if cans.can_kakan:
            for tile in self.state.kakan_candidates:
                self.mask[ADDKONG_BASE + tile.id] = True
        if cans.can_pon and target_tile is not None:
            self.mask[PUNG_BASE + target_tile.id] = True
        if target_tile is not None:
            if cans.can_chi_low:
                self.mask[self.encode_chow_index(target_tile.id, ChiType.Low)] = True
            if cans.can_chi_mid:
                self.mask[self.encode_chow_index(target_tile.id, ChiType.Mid)] = True
            if cans.can_chi_high:
                self.mask[self.encode_chow_index(target_tile.id, ChiType.High)] = True

        assert int(self.mask.sum()) <= ACTION_SPACE
