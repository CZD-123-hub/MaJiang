from typing import List

from mortal_part.state.item import KawaItem, MoveType, Sutehai, ChiPon
from mortal_part.state.sp_tables import SinglePlayerTables
from mortal_part.tile import Tile
from mortal_part.state.action import ActionCandidate
from mortal_part.mjai.event import Event
from mortal_part.rankings import Rankings
from mortal_part.algo.point import Point
from mortal_part.shangrao_rules import ShangraoRuleAdapter
import copy
try:
    # 国标规则扩展是 Linux .so；上饶规则单测在 Windows 上不依赖它。
    from mortal_part import mortal_cpp as mc
except ImportError:
    mc = None

import logging
try:
    from colorlog import ColoredFormatter
except ImportError:
    ColoredFormatter = None

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
# 定义颜色格式
if ColoredFormatter is not None:
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
else:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
# 设置格式化器
console_handler.setFormatter(formatter)
# 配置日志
if False:
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler]
    )


class PlayerState(object):
    """
    玩家信息，包括推荐出牌、推荐动作、换三张推荐、定缺推荐、执行碰杠动作等
    以及
        action.rs 中的 impl PlayerState
        getter.rs 中的 impl PlayerState
        agent_helper.rs 中的 impl PlayerState
        update.rs 中的 impl PlayerState
    """

    def __init__(self, seat_id):
        """
        构造器
        @param name: 玩家名称        @param seat_id: 玩家座位号
        """
        self.seat_id = seat_id  # 座位号
        self.handcards = [0] * 34  # 玩家手牌 [int, 34], 对应1~9万/筒/索
        self.waits = [False] * 34  # 期望进张 [bool, 34]
        self.tiles_public = [0] * 34  # 所有公开牌（副露+弃牌） [int, 34] * 34
        self.tiles_seen = [0] * 34  # 自己可见的所有牌 [int, 34]

        # for SPCalculator
        self.keep_shanten_discards = [False] * 34  # 使向听数不变的牌 [bool, 34]
        self.next_shanten_discards = [False] * 34  # 使向听数减1的牌 [bool, 34]
        self.forbidden_tiles = [False] * 34  # 禁止操作的牌（？感觉用不上） [bool, 34]

        self.round_wind = 0  # 圈风
        self.seat_wind = 0  # 场风
        self.kyoku = 0
        self.scores = [0] * 4  # 玩家分数 [int, 4], Rotated to be relative, so `scores[0]` is the score of the player.
        self.rank = 0  # 玩家排名
        self.zhuang_id = 0

        self.kawa = [[] for _ in range(4)]  # 四家弃牌，[[Optional[KawaItem], 28], 4],
        self.kawa_overview = [[] for _ in range(4)]  # [[Tile, 28], 4] 四个玩家的弃牌
        self.fulu_overview = [[] for _ in range(4)]  # [[[Tile, 4], 4], 4] 四个玩家的副露
        self.ankan_overview_count = [0, 0, 0, 0]  # [[Tile, 4], 4] 四个玩家的暗杠

        self.at_turn = 0  # 当前回合数
        self.tiles_left = 84  # 剩余牌数
        self.intermediate_kan = []  # [Tile, 4], 作为kawa的属性用于编码
        self.intermediate_chi_pon = None  # 作为kawa的属性用于编码

        self.shanten = 13  # 初始向听数，待计算，之后写个方法计算初始向听数
        self.last_self_tsumo = None  # 最后自己摸的牌(包括当前情况，下同
        self.last_kawa_tile = None  # 场面上最后弃的牌
        self.last_cans = ActionCandidate()  # 最后可以进行的操作，ActionCandidate类型（还没写，是否仿照rust中的形式有待讨论

        self.ankan_candidates = []  # [Tile, 3]
        self.kakan_candidates = []  # [Tile, 3]
        self.chankan_chance = False  # 抢杠判断

        self.at_rinshan = 2  # 是否处于岭上状态（进行了杠操作, 改为int: 0表示刚刚进行了杠牌操作, 每进行一次摸牌操作+1

        self.chis = []  # 吃副露的中间牌
        self.pons = []  # 碰的目标牌
        self.minkans = []  # 明杠的目标牌, 加杠的结果也是明杠
        self.ankans = []  # 暗杠的目标牌
        self.flower_count = 0
        # 精牌 id（上饶日志可提供；国标日志保持 None）。规则适配完成前不参与国标计算。
        self.king_tile_id = None
        self.shangrao_rules = ShangraoRuleAdapter()
        # 手牌中碰杠的容量
        self.tehai_len_div3 = 4

        self.has_next_shanten_discard = False

    # ======================= action =============================
    # ============================================================

    def validate_reaction(self, action: Event) -> None:
        """
        Check if an action is a valid reaction to the current state.
        Raises ValueError if the action is invalid.
        @param action: 当前事件
        @return:
        """
        cans = self.last_cans
        # 处理流局或空事件
        if isinstance(action.event, Event.Ryukyoku):
            # 上饶日志把流局作为显式终局事件；精牌指示牌是否占一张牌墙要由
            # 回放核验确定，因此此阶段不强加国标的 tiles_left == 0 约束。
            if not self.uses_shangrao_rules() and self.tiles_left != 0:
                raise ValueError("cannot ryukyoku")
            return
        elif isinstance(action.event, Event.NoneEvent):
            return

        # 验证动作操作者
        actor = action.actor()
        if actor is None:
            raise ValueError("Action does not have actor and is not ryukyoku(流局)")
        if actor != self.seat_id:
            raise ValueError(f"Actor is {actor}, not self ({self.seat_id})")

        # 处理不同的事件类型
        # 打牌
        if isinstance(action.event, Event.Dahai):
            if not cans.can_discard:
                raise ValueError("Cannot discard")
            self.ensure_tiles_in_hand([action.event.tile])

        # 吃
        elif isinstance(action.event, Event.Chi):
            if action.event.target == action.event.player:
                raise ValueError("Chi from itself")
            if self.last_kawa_tile != action.event.tile:
                raise ValueError("Chi target is not the last kawa tile")
            if not cans.can_chi:
                raise ValueError("Cannot chi")
            self.ensure_tiles_in_hand(action.event.consumed)

        # 碰
        elif isinstance(action.event, Event.Pon):
            if action.event.target == action.event.player:
                raise ValueError("Pon from itself")
            if self.last_kawa_tile != action.event.tile:
                raise ValueError("Pon target is not the last kawa tile")
            if not cans.can_pon:
                raise ValueError("Cannot pon")
            self.ensure_tiles_in_hand(action.event.consumed)

        # 明杠
        elif isinstance(action.event, Event.MinGang):
            if action.event.target == action.event.player:
                raise ValueError("Daiminkan from itself")
            if self.last_kawa_tile != action.event.tile:
                raise ValueError("Daiminkan target is not the last kawa tile")
            if not cans.can_daiminkan:
                raise ValueError("Cannot daiminkan")
            self.ensure_tiles_in_hand(action.event.consumed)

        # 补杠
        elif isinstance(action.event, Event.BuGang):
            if not cans.can_kakan:
                raise ValueError("Cannot kakan")
            deaka_pai = action.event.tile
            if deaka_pai not in self.kakan_candidates:
                raise ValueError(f"Cannot kakan {action.event.tile}")
            self.ensure_tiles_in_hand([action.event.tile])

        # 暗杠
        elif isinstance(action.event, Event.AnGang):
            if not cans.can_ankan:
                raise ValueError("Cannot ankan")
            tile = action.event.consumed[0]
            if tile not in self.ankan_candidates:
                raise ValueError(f"Cannot ankan {tile}")
            self.ensure_tiles_in_hand(action.event.consumed)

        # 和牌
        elif isinstance(action.event, Event.Hu):
            if action.event.player != self.seat_id and not cans.can_hu:
                logger.warning(
                    "Validating hu from player %s to self %s and cans.can_hu = %s",
                    action.event.player, self.seat_id, cans.can_hu,
                )
                raise ValueError("Not self agari")
            elif action.event.player == self.seat_id and not cans.can_hu:
                raise ValueError("Cannot agari successfully")

            if cans.can_tsumo_hu:
                self.ensure_tiles_in_hand([action.event.tile])

        else:
            raise ValueError(f"Unexpected action {action}")

    def ensure_tiles_in_hand(self, tiles: List[Tile]) -> None:
        """
        确保操作的牌都在手牌中
        @param tiles:
        @return:
        """
        for tile in tiles:
            if self.handcards[tile.id] <= 0:
                raise ValueError(f"{tile} is not in hand")

    # =============================== getter ====================================
    # ============================================================================

    def is_zhuang(self):
        """
        对应 getter.rs 中的 impl PlayerState；
        @return:
        """
        return self.zhuang_id == 0

    # =============================== obs_repr ===================================
    # ============================================================================

    def encode_obs(self, at_kan_select: bool, obs_version: int = 4):
        """
        对应 obs_repr.rs 中的 impl PlayerState；
        对当前玩家状态进行编码，调用ObsEncoderContext中的encode_obs接口
        @param at_kan_select:
        @return:
        """
        from mortal_part.state.obs_repr import ObsEncoderContext
        # [V4 local-play v3-compat] v4 remains the default, but local-play can
        # request version=3 to feed old 194-channel v3 checkpoints.
        return ObsEncoderContext(self, version=obs_version, at_kan_select=at_kan_select).encode()

    # =============================== player_state ===============================
    # ============================================================================

    def update_json(self, mjai_json: str) -> ActionCandidate:
        """
        从JSON字符串更新玩家状态
        @param mjai_json: JSON格式的事件字符串
        @return: ActionCandidate: 可用动作候选
        """
        event = Event.from_str(mjai_json)
        return self.update(event)

    def validate_reaction_json(self, mjai_json):
        """
        验证反应动作是否有效
        @param mjai_json: JSON格式的动作字符串
        @return:
        """
        action = Event.from_str(mjai_json)
        return self.validate_reaction(action)

    def brief_info(self) -> str:
        """
        仅用于调试
        Return a human readable description of the current state.

        Returns:
            str: A string description of the state
        """
        # Convert waits to tile strings
        waits = [Tile(i) for i, wait in enumerate(self.waits) if wait]

        # Format kawa (discards)
        zipped_kawa = []
        max_kawa_len = max(len(kawa) for kawa in self.kawa) if any(self.kawa) else 0
        for i in range(max_kawa_len):
            row = []
            for player in range(4):
                if i < len(self.kawa[player]) and self.kawa[player][i] is not None:
                    row.append(str(self.kawa[player][i]))
                else:
                    row.append("-")
            zipped_kawa.append(f"{i:2}. {row[0]}\t{row[1]}\t{row[2]}\t{row[3]}")

        return f"""player (abs): {self.seat_id}
                oya (rel): {self.zhuang_id}
                turn: {self.at_turn}
                score (rel): {self.scores}
                tehai: {tiles_to_string(self.handcards)}
                fuuro: {self.fulu_overview[0]}
                ankan: {self.ankan_overview_count[0]}
                tehai len: {self.tehai_len_div3}
                shanten: {self.shanten} (actual: {self.real_time_shanten()})
                waits: {waits}
                action candidates: {self.last_cans}
                last self tsumo: {self.last_self_tsumo}
                last kawa tile: {self.last_kawa_tile}
                tiles left: {self.tiles_left}
                kawa:
                {chr(10).join(zipped_kawa)}"""

    # ============================= agent_helper =============================
    # ========================================================================

    def kans_count(self):
        """
        对应agent_helper.rs 中的 kans_count
        @return: 玩家杠牌数量
        """
        return len(self.minkans) + len(self.ankans)

    def uses_shangrao_rules(self) -> bool:
        """有精牌字段的事件流走上饶纯 Python 规则，不再调用国标 .so。"""
        return self.king_tile_id is not None

    def open_meld_count(self) -> int:
        """当前玩家已完成的副露组数（吃、碰、明杠、暗杠）。"""
        return len(self.chis) + len(self.pons) + len(self.minkans) + len(self.ankans)

    def own_open_melds_are_yaojiu(self) -> bool:
        """九幺胡需要所有已副露的牌也属于幺九或字牌。"""
        return all(
            tile.is_yaokyuu()
            for meld in self.fulu_overview[0]
            for tile in meld
        )

    def shangrao_can_hu(self, winning_tile: Tile, is_ron: bool) -> bool:
        return self.shangrao_rules.can_hu(
            self.handcards,
            self.open_meld_count(),
            self.king_tile_id,
            winning_tile.id if winning_tile is not None else None,
            is_ron=is_ron,
            open_melds_are_yaojiu=self.own_open_melds_are_yaojiu(),
        )

    def discard_candidates(self) -> List[bool]:
        """
        弃牌候选
        Used by `Agent` impls, must be called at 3n+2.
        @return:
        """
        assert self.last_cans.can_discard, "tehai is not 3n+2"
        ret = [False] * 34
        for i, count in enumerate(self.handcards):
            if count == 0:
                continue
            ret[i] = not self.forbidden_tiles[i]    # TODO 后续可能不需要这个forbidden_tiles
        return ret

    # def discard_candidates_with_unconditional_tenpai(self) -> List[bool]:
    #     """
    #     功能：计算在3n+2时，打出某张牌后再摸到一张牌就能胡牌的这样一张弃牌候选
    #     @return:
    #     """
    #     assert self.last_cans.can_discard, "must be 3n+2"
    #
    #     ret = [False] * 34
    #
    #     # 快速剪枝
    #     if self.tiles_left == 0 or self.shanten > 1 or (self.shanten == 1 and not any(self.next_shanten_discards)):
    #         return ret
    #
    #     # 已经自摸，或者副露后手牌还是自摸的情况，就直接返回了，没有必要计算
    #     if self.last_self_tsumo is not None:
    #         if self.waits[self.last_self_tsumo.id]:
    #             # 任何丢弃都会保持 agari，但我们要的是“听牌”不是“和了”，所以空表
    #             return ret
    #     # elif mc.ShantenCalculator.calc_all(self.handcards, self.tehai_len_div3) == -1:
    #     #     return ret
    #     else:
    #         if mc.ShantenCalculator.calc_all(self.handcards, self.tehai_len_div3) == -1:
    #             return ret
    #
    #     # 决定用哪组弃牌候选
    #     tenpai_discards = self.next_shanten_discards if self.shanten == 1 else self.keep_shanten_discards
    #
    #     for discard in range(34):
    #         if not tenpai_discards[discard] or self.handcards[discard] == 0:
    #             continue
    #
    #         # 模拟打掉一张
    #         hand_3n1 = self.handcards.copy()
    #         hand_3n1[discard] -= 1
    #         # 弃牌后不能听牌则继续
    #         if mc.ShantenCalculator.calc_all(hand_3n1, self.tehai_len_div3) > -1:
    #             continue
    #
    #         # 遍历剩余 34 种牌，当作“下一张摸牌”
    #         for tsumo in range(34):
    #             if tsumo == discard or hand_3n1[tsumo] == 4:
    #                 continue
    #
    #             # 可以和牌则认为有效
    #             if mc.AgariCalculator(hand_3n1, self.chis, self.pons, self.minkans, self.ankans, tsumo, False, self.round_wind, self.seat_wind) \
    #                 .agari(self.chankan_chance, self.at_rinshan, self.tiles_left <= 3 or self.at_turn >= 20, self.tiles_public[tsumo] == 3).fan >= 8:
    #                 ret[discard] = True
    #                 break  # 只要有一张有效摸牌即可，不必再验
    #     return ret

    def yaojiu_kind_count(self):
        """
        0  1  2  3  4  5  6  7  8
        9  10 11 12 13 14 15 16 17
        18 19 20 21 22 23 24 25 26
        27 28 29 30 31 32 33
        对应agent_helper.rs 中的 yaokyuu_kind_count
        @return:玩家手牌中的幺九牌数量
        """
        return self.handcards[0] + self.handcards[8] + self.handcards[9] + self.handcards[17] + self.handcards[18] + \
            self.handcards[26]

    def rule_based_agari(self) -> bool:
        if not self.last_cans.can_hu:
            return False
        else :  # 国标麻将特性以及比赛中是单局比赛，直接无脑和牌
            return True
        # return self.rule_based_agari_slow(
        #     self.last_cans.can_ron_hu,
        #     self.rel(self.last_cans.target_actor),
        # )

    def rule_based_agari_slow(self, is_ron: bool, target_rel: int) -> bool:
        # 1. 无脑放行区，不是最后一局，也不是最后一名
        if self.kyoku % 4 < 3 or self.rank < 3:
            return True

        # 2. 半放行区
        if self.round_wind == 2 and self.kyoku % 4 < 3:  # 未到 W4
            return True

        # 3. 计算此次和牌的得分
        max_win_point = self.agari_points(is_ron)

        if max_win_point is None:  # 无法和牌（无役）
            return False

        # 4. 模拟和了后分数
        exp_scores = self.scores.copy()
        if is_ron:
            exp_scores[0] += max_win_point.ron
            exp_scores[target_rel] -= (max_win_point.ron - 8 * 2)
            for idx in range(1, 4):
                if idx != target_rel:
                    exp_scores[idx] -= 8
        else:
            total = max_win_point.tsumo
            exp_scores[0] += total
            single = total / 3
            for idx in range(1, 4):
                exp_scores[idx] -= single

        # 不再垫底就和
        return self.get_rank(exp_scores) < 3

    def agari_points(self, is_ron: bool) -> Point:
        """
        和牌分数
        @param is_ron:是不是点炮胡
        @return:
        """
        if not (is_ron and self.last_cans.can_ron_hu or self.last_cans.can_tsumo_hu):
            # logging.debug("cannot hu")
            return None

        # 上饶计分不等于国标 fan；监督阶段只需要合法动作，不应伪造 Point。
        if self.uses_shangrao_rules():
            return None

        # 更新手牌
        tehai = copy.deepcopy(self.handcards)

        if is_ron:
            winning_tile = self.last_kawa_tile
        else:
            winning_tile = self.last_self_tsumo
            tehai[winning_tile.id] -= 1

        if winning_tile is None:
            logger.info("cannot find the winning tile")
            return None

        agari_calc = mc.AgariCalculator(
            tehai=tehai,
            chis=self.chis,
            pons=self.pons,
            minkans=self.minkans,
            ankans=self.ankans,
            winning_tile=winning_tile.id,
            is_ron=is_ron,
            round_wind=self.round_wind,
            seat_wind=self.seat_wind,
            flower_count=self.flower_count
        )
        agari = agari_calc.agari(self.chankan_chance, self.at_rinshan, self.tiles_left <= 3, self.tiles_public[winning_tile.id] == 3)
        if agari is None:
            # logging.info("not a hora hand")
            return None

        return Point(agari.fan)

    def real_time_shanten(self):
        """
        计算此时的实际向听数。与`self.shanten`不同，此函数能够正确计算在3n+2时的向听数
        @return:在3n+2时的向听数
        """
        # 1.不能弃牌，手牌数3n+1，当前的向听数是准确的
        if not self.last_cans.can_discard:
            # 3n+1, 直接返回当前向听数
            return self.shanten
        # 2.可以弃牌，手牌数3n+2，意味可能是自然摸牌后的状态，也可能是吃碰后的状态
        # 2.1如果不是听牌状态，则当前向听数也是准确的
        if self.shanten > 0:
            # 3n+2, 非听牌状态
            # 向听数 = self.shanten-1 如果有任何牌可以减小向听数
            if self.has_next_shanten_discard:
                return self.shanten - 1
            else:
                return self.shanten
        # 2.2.1听牌状态并且上一个动作是自摸，也可以直接返回
        if self.last_self_tsumo is not None:
            tile = self.last_self_tsumo.id
            # 3n+2, 听牌状态(shanten = 1), 如果听牌后自摸
            if self.waits[tile]:
                return -1
            return 0
        # 2.2.2听牌状态，上一个动作不是自摸，而是吃碰等副露动作，此时手牌数为3n+2，要么是听牌，要么是已经胡牌，只能是0或-1
        # 但是吃碰后又不能和牌，即便达到胡牌状态，向听数为-1也没意义，不让胡，所以只能是0
        return 0

    def single_player_tables(self) -> SinglePlayerTables:
        """[V4 expectation contrast] Build the original SPCalculator EV table.

        This restores the expectation-value/search feature used by the old
        Mortal-style encoder, so this experiment can compare v3-style visible
        features against visible features plus EV guidance.
        """
        # SPCalculator 是国标 EV 搜索，不能作为上饶观测特征。
        if self.uses_shangrao_rules() or self.tiles_left < 4:
            return SinglePlayerTables(None)

        cur_shanten = self.real_time_shanten()
        if cur_shanten < 0:
            return SinglePlayerTables(None)

        can_discard = self.last_cans.can_discard
        if can_discard:
            tsumos_left = self.tiles_left // 4
            calc_haitei = self.tiles_left % 4 == 0
        else:
            target_actor = self.last_cans.target_actor
            if target_actor is None:
                return SinglePlayerTables(None)
            target = self.rel(target_actor)
            # Chankan is intentionally ignored here, matching the original EV feature.
            tiles_left_at_next_tsumo = max(0, self.tiles_left - (4 - target))
            tsumos_left = tiles_left_at_next_tsumo // 4
            calc_haitei = tiles_left_at_next_tsumo % 4 == 0

        if tsumos_left < 1:
            return SinglePlayerTables(None)

        init_state = mc.InitState(
            tehai=copy.deepcopy(self.handcards),
            tiles_seen=self.tiles_seen,
        )

        sp_calc = mc.SPCalculator(
            tehai_len_div3=self.tehai_len_div3,
            chis=self.chis,
            pons=self.pons,
            minkans=self.minkans,
            ankans=self.ankans,
            sort_result=True,
            is_zhuang=self.zhuang_id == 0,
            maximize_win_prob=False,
            calc_tegawari=False,
            calc_shanten_down=False,
            calc_haitei=calc_haitei,
            chankan_chance=self.chankan_chance,
            at_rinshan=self.at_rinshan,
            round_wind=self.round_wind,
            seat_wind=self.seat_wind,
            flower_count=self.flower_count,
        )
        return SinglePlayerTables(max_ev_table=sp_calc.calc(init_state, can_discard, tsumos_left, cur_shanten))

    # =============================== update ==============================
    # =====================================================================

    def update(self, event: Event) -> ActionCandidate:
        """更新玩家状态"""
        return self.update_with_keep_cans(event, False)

    def update_with_keep_cans(self, event: Event, keep_cans_on_announce: bool) -> ActionCandidate:
        """
        如果 keep_cans_on_announce 为 true，那么在处理 Hu 事件时，
        会保持 self.last_cans、self.ankan_candidates 和 self.kakan_candidates 的值不变，这些值是从上一次更新时继承过来的。
        目前，将 keep_cans_on_announce 设置为 true 主要用于 validate_logs 函数中。
        @param event:
        @param keep_cans_on_announce:
        @return:
        """
        try:
            return self.update_inner(event, keep_cans_on_announce)
        except Exception as e:
            # 添加上下文信息
            context = f"on event {event}"
            raise RuntimeError(context) from e

    def update_inner(self, event: Event, keep_cans_on_announce: bool) -> ActionCandidate:
        if not keep_cans_on_announce or not event.is_in_game_announce():  # TODO 非和牌事件为什么要这么做
            self.last_cans = ActionCandidate(target_actor=event.actor())
            self.ankan_candidates.clear()
            self.kakan_candidates.clear()
        self.chankan_chance = False
        event_type = event.event
        # 根据事件类型处理
        if isinstance(event_type, Event.StartGame):
            self._handle_start_game(event_type)
        elif isinstance(event_type, Event.BotStartGame):
            self._handle_start_game(event_type)
        elif isinstance(event_type, Event.BotStartKyoku):
            self._handle_bot_start_kyoku(event_type)
        elif isinstance(event_type, Event.StartKyoku):
            self._handle_start_kyoku(event_type)
        elif isinstance(event_type, Event.Tsumo):
            self._handle_tsumo(event_type)
        elif isinstance(event_type, Event.Dahai):
            self._handle_dahai(event_type)
        elif isinstance(event_type, Event.Chi):
            self._handle_chi(event_type)
        elif isinstance(event_type, Event.Pon):
            self._handle_pon(event_type)
        elif isinstance(event_type, Event.MinGang):
            self._handle_daiminkan(event_type)
        elif isinstance(event_type, Event.BuGang):
            self._handle_kakan(event_type)
        elif isinstance(event_type, Event.AnGang):
            self._handle_ankan(event_type)
        elif isinstance(event_type, Event.Ryukyoku):
            self._handle_ryukyoku(event_type)
        elif isinstance(event_type, Event.BuHua):
            self._handle_buhua(event_type)
        # elif isinstance(event_type, Event.Hu):  # 需要处理分数变化
        #     self.update_scores(event_type)

        return self.last_cans

    def _handle_buhua(self, event_type):
        if event_type.player == self.seat_id:
            self.flower_count += 1
        if isinstance(event_type.replacement, Tile):
            tsumo = Event("Tsumo", player=event_type.player, tile=event_type.replacement)
            self._handle_tsumo(tsumo.event)

    def _handle_ryukyoku(self, event_type):
        pass

    def _handle_start_game(self, event: Event.StartGame) -> None:
        """
        处理游戏开始事件
        @param event:
        @return:
        """
        pass

    def _handle_start_kyoku(self, event: Event.StartKyoku) -> None:
        """
        处理游戏开局事件,需要根据数据格式调整编码
        @param event:
        @return:
        """
        self.handcards = [0] * 34  # 玩家手牌 [int, 34], 对应1~9万/筒/索
        self.waits = [False] * 34  # 期望进张 [bool, 34]
        self.tiles_public = [0] * 34  # 公共牌信息 [int, 34]
        self.tiles_seen = [0] * 34  # 自己可见的所有牌 [int, 34]

        # for SPCalculator
        self.keep_shanten_discards = [False] * 34  # 使向听数不变的牌 [bool, 34]
        self.next_shanten_discards = [False] * 34  # 使向听数减1的牌 [bool, 34]
        self.forbidden_tiles = [False] * 34  # 禁止操作的牌 [bool, 34]

        # 数据中没有庄家和场风信息，所以需要根据局数来推导，圈风信息已有
        self.round_wind = {"E": 0, "S": 1, "W": 2, "N": 3}[event.wind]  # 当前圈风
        self.kyoku = event.kyoku - 1  # 当前是第几局
        zhuang_id = event.zhuang  # 绝对的庄家编号
        self.zhuang_id = self.rel(zhuang_id)  # 相对的庄家座位号
        self.seat_wind = (4 - self.zhuang_id) % 4  # 当前玩家的场风
        self.scores = copy.copy(event.scores)  # 玩家分数 [int, 4]
        self.rotate_left(self.scores, self.seat_id)  # Rotated to be relative, `scores[0]` is the score of the player.
        self.rank = 0  # 玩家排名

        self.kawa = [[] for _ in range(4)]  # 四家弃牌，[[Optional[KawaItem], 28], 4]
        self.kawa_overview = [[] for _ in range(4)]  # [[Tile, 28], 4] 四个玩家的弃牌
        self.fulu_overview = [[] for _ in range(4)]  # [[[Tile, 4], 4], 4] 四个玩家的副露
        self.ankan_overview_count = [0, 0, 0, 0]  # [[Tile, 4], 4] 四个玩家的暗杠

        self.at_turn = 0  # 当前回合数
        self.tiles_left = 84  # 剩余牌数
        self.intermediate_kan = []  # [Tile, 4], 作为kawa的属性用于编码
        self.intermediate_chi_pon = None  # 作为kawa的属性用于编码

        self.shanten = 13  # 初始向听数，待计算，之后写个方法计算初始向听数
        self.last_self_tsumo = None  # 最后自己摸的牌(包括当前情况，下同
        self.last_kawa_tile = None  # 最后自己弃的牌
        self.last_cans = ActionCandidate()  # 最后可以进行的操作，ActionCandidate类型

        self.ankan_candidates = []  # [Tile, 3]
        self.kakan_candidates = []  # [Tile, 3]
        self.chankan_chance = False  # 抢杠判断

        self.at_rinshan = 2  # 是否处于岭上状态（进行了杠操作, 改为int: 0表示刚刚进行了杠牌操作, 每进行一次摸牌操作+1

        self.chis = []  # 吃的牌，记录中间张
        self.pons = []  # 碰的目标牌
        self.minkans = []  # 明杠的目标牌, 加杠的结果也是明杠
        self.ankans = []  # 暗杠的目标牌
        self.flower_count = event.flower_count[self.seat_id] if hasattr(event, "flower_count") else 0  # 花牌数量
        self.king_tile_id = event.king_card.id if getattr(event, "king_card", None) is not None else None
        # 手牌中碰杠的容量
        self.tehai_len_div3 = 4

        self.has_next_shanten_discard = False

        self.update_rank()
        # 设置初始手牌
        for tile in event.tehais[self.seat_id]:
            self.move_tile(tile, MoveType.TSUMO)

        self.update_shanten()   # 3n + 1
        self.update_waits()
        self.pad_kawa_at_start()

    def _handle_bot_start_kyoku(self, event: Event.BotStartKyoku) -> None:
        self.handcards = [0] * 34  # 玩家手牌 [int, 34], 对应1~9万/筒/索
        self.waits = [False] * 34  # 期望进张 [bool, 34]
        self.tiles_public = [0] * 34  # 公共牌信息 [int, 34]
        self.tiles_seen = [0] * 34  # 自己可见的所有牌 [int, 34]

        # for SPCalculator
        self.keep_shanten_discards = [False] * 34  # 使向听数不变的牌 [bool, 34]
        self.next_shanten_discards = [False] * 34  # 使向听数减1的牌 [bool, 34]
        self.forbidden_tiles = [False] * 34  # 禁止操作的牌 [bool, 34]

        # 数据中没有庄家和场风信息，所以需要根据局数来推导，圈风信息已有
        # self.round_wind = {"E": 0, "S": 1, "W": 2, "N": 3}[event.wind]  # 当前圈风
        self.kyoku = self.seat_id  # 当前是第几局
        zhuang_id = 0  # 绝对的庄家编号
        self.zhuang_id = self.rel(zhuang_id)  # 相对的庄家座位号
        self.seat_wind = self.seat_id  # 当前玩家的场风
        self.scores = [500, 500, 500, 500]  # 玩家分数 [int, 4]
        self.rotate_left(self.scores, self.seat_id)  # Rotated to be relative, `scores[0]` is the score of the player.
        self.rank = 0  # 玩家排名

        self.kawa = [[] for _ in range(4)]  # 四家弃牌，[[Optional[KawaItem], 28], 4]
        self.kawa_overview = [[] for _ in range(4)]  # [[Tile, 28], 4] 四个玩家的弃牌
        self.fulu_overview = [[] for _ in range(4)]  # [[[Tile, 4], 4], 4] 四个玩家的副露
        self.ankan_overview_count = [0, 0, 0, 0]  # [[Tile, 4], 4] 四个玩家的暗杠

        self.at_turn = 0  # 当前回合数
        self.tiles_left = 84  # 剩余牌数
        self.intermediate_kan = []  # [Tile, 4], 作为kawa的属性用于编码
        self.intermediate_chi_pon = None  # 作为kawa的属性用于编码

        self.shanten = 13  # 初始向听数，待计算，之后写个方法计算初始向听数
        self.last_self_tsumo = None  # 最后自己摸的牌(包括当前情况，下同
        self.last_kawa_tile = None  # 最后自己弃的牌
        self.last_cans = ActionCandidate()  # 最后可以进行的操作，ActionCandidate类型

        self.ankan_candidates = []  # [Tile, 3]
        self.kakan_candidates = []  # [Tile, 3]
        self.chankan_chance = False  # 抢杠判断

        self.at_rinshan = 2  # 是否处于岭上状态（进行了杠操作, 改为int: 0表示刚刚进行了杠牌操作, 每进行一次摸牌操作+1

        self.chis = []  # 吃的牌，记录中间张
        self.pons = []  # 碰的目标牌
        self.minkans = []  # 明杠的目标牌, 加杠的结果也是明杠
        self.ankans = []  # 暗杠的目标牌

        # 手牌中碰杠的容量
        self.tehai_len_div3 = 4

        self.has_next_shanten_discard = False

        self.update_rank()
        # 设置初始手牌
        for tile in event.tehais:
            self.move_tile(tile, MoveType.TSUMO)

        self.update_shanten()  # 3n + 1
        self.update_waits()
        self.pad_kawa_at_start()


    def _handle_tsumo(self, event: Event.Tsumo) -> None:
        """
        处理摸牌事件
        @param event:
        @return:
        """
        if self.tiles_left == 0:
            logger.error("event: %s", event)
            raise ValueError("no more tile to catch!!!")
        self.tiles_left -= 1
        if event.player != self.seat_id:
            return
        tile = event.tile

        # 处理自己摸牌的情况
        self.at_turn += 1
        self.last_cans.can_discard = True
        self.last_self_tsumo = tile
        self.move_tile(tile, MoveType.TSUMO)
        self.update_shanten_discards()  # 3n + 2

        # 精牌局的胡牌条件由上饶适配器决定，不能沿用国标八番门槛。
        if self.uses_shangrao_rules():
            self.last_cans.can_tsumo_hu = self.shangrao_can_hu(tile, is_ron=False)
        # 国标旧日志仍保留原来的 AgariCalculator 路径。
        elif self.waits[tile.id]:
            self.handcards[tile.id] -= 1
            agari = mc.AgariCalculator(self.handcards, self.chis, self.pons, self.minkans, self.ankans, tile.id, False,
                                       self.round_wind, self.seat_wind, self.flower_count) \
                .agari(False, self.at_rinshan, self.tiles_left <= 3, self.tiles_public[tile.id] == 3)
            self.handcards[tile.id] += 1
            if agari is not None and agari.fan >= 8:
                self.last_cans.can_tsumo_hu = True

        if self.tiles_left == 0:
            return

        # 判断是否可以暗杠、加杠
        for tid, count in enumerate(self.handcards):
            if count > 0:
                if count == 4:
                    self.last_cans.can_ankan = True
                    self.ankan_candidates.append(Tile(tile_id=tid))
                elif tid in self.pons:
                    self.last_cans.can_kakan = True
                    self.kakan_candidates.append(Tile(tile_id=tid))

    def _handle_dahai(self, event: Event.Dahai) -> None:
        """处理打牌事件"""
        actor_rel = self.rel(event.player)
        tile = event.tile
        if actor_rel == 0:
            self.move_tile(tile, MoveType.DISCARD)
        else:
            self.tiles_seen[tile.id] += 1
            self.tiles_public[tile.id] += 1

        sutehai = Sutehai(tile, False)
        kawa_item = KawaItem(chi_pon=self.intermediate_chi_pon,
                             kan=self.intermediate_kan,
                             sutehai=sutehai)
        self.intermediate_chi_pon = None
        self.kawa[actor_rel].append(kawa_item)
        self.kawa_overview[actor_rel].append(tile)
        self.last_kawa_tile = tile

        # 自己打牌的情况
        if actor_rel == 0:
            self.last_cans.can_discard = False
            self.forbidden_tiles = [False] * 34
            self.at_rinshan += 1
            if self.next_shanten_discards[tile.id]:
                self.shanten -= 1
            if not self.keep_shanten_discards[tile.id]:
                self.update_shanten()   # 3n + 1
            self.update_waits()
            return

        if self.uses_shangrao_rules():
            self.last_cans.can_ron_hu = self.shangrao_can_hu(tile, is_ron=True)
        elif self.waits[tile.id]:
            agari = mc.AgariCalculator(self.handcards, self.chis, self.pons, self.minkans, self.ankans, tile.id, True, self.round_wind, self.seat_wind, self.flower_count) \
                .agari(False, self.at_rinshan, self.tiles_left == 0, self.tiles_public[tile.id] == 4)
            if agari is not None and agari.fan >= 8:
                self.last_cans.can_ron_hu = True

        if self.tiles_left == 0:
            return

        # 判断是否可以吃碰杠
        if actor_rel == 3 and not tile.is_jihai() and self.tehai_len_div3 > 0:
            self.set_can_chi_from_tile(tile)
        if self.handcards[tile.id] >= 2:
            self.last_cans.can_pon = True
        if self.handcards[tile.id] == 3:
            self.last_cans.can_daiminkan = True

    def _handle_chi(self, event: Event.Chi) -> None:
        actor_rel = self.rel(event.player)
        consumed = event.consumed
        tile = event.tile
        # 1. 把三张牌放进副露区
        full_set = [consumed[0], consumed[1], tile]
        full_set.sort(key=lambda t: t.id)
        self.fulu_overview[actor_rel].append(full_set)
        self.intermediate_chi_pon = ChiPon(consumed, tile)

        # 2. 非自家：只目击
        if actor_rel != 0:
            for t in consumed:
                self.tiles_seen[t.id] += 1
                self.tiles_public[t.id] += 1
            return

        # 3. 自家：真正修改手牌与状态
        self.last_cans.can_discard = True
        self.tehai_len_div3 -= 1
        self.last_self_tsumo = None  # 强制让下一打不是摸切

        # 3.2 从手牌拿走 2 张搭子
        for t in consumed:
            self.move_tile(t, MoveType.FULU_CONSUME)
        # 3.3 记录顺子的中间序号（用于以后 UI 或再计算）
        ids = sorted([consumed[0].id, consumed[1].id, tile.id])
        self.chis.append(ids[1])  # 记录中间牌

        # 3.4 重新计算听牌及听牌弃牌候选
        self.update_shanten()   # 3n + 2
        self.update_shanten_discards()  # 3n + 2

    def _handle_pon(self, event: Event.Pon) -> None:
        """处理碰牌事件"""
        actor_rel = self.rel(event.player)
        tile = event.tile
        consumed = event.consumed
        fulu_set = [consumed[0], consumed[1], tile]

        # 更新副露信息
        self.fulu_overview[actor_rel].append(fulu_set)
        self.intermediate_chi_pon = ChiPon(consumed=consumed, target_tile=tile)
        self.pad_kawa_for_pon_or_daiminkan(event.player, event.target)

        if actor_rel != 0:
            # 其他人碰牌，更新见到的牌
            self.tiles_seen[consumed[0].id] += 2
            self.tiles_public[consumed[0].id] += 2
            return

        self.last_cans.can_discard = True
        self.tehai_len_div3 -= 1
        self.move_tile(consumed[0], MoveType.FULU_CONSUME)
        self.move_tile(consumed[1], MoveType.FULU_CONSUME)
        self.pons.append(tile.id)

        self.update_shanten()   # 3n + 2
        self.update_shanten_discards()  # 3n + 2

        # 碰完后不可以加杠或暗杠

    def _handle_daiminkan(self, event: Event.MinGang) -> None:
        """处理明杠事件"""
        actor_rel = self.rel(event.player)
        tile = event.tile
        consumed = event.consumed
        fulu_set = [consumed[0], consumed[1], consumed[2], tile]

        # 更新副露信息
        self.fulu_overview[actor_rel].append(fulu_set)
        self.intermediate_kan.append(tile)
        self.pad_kawa_for_pon_or_daiminkan(event.player, event.target)

        # 如果不是自己杠牌
        if actor_rel != 0:
            # 其他人杠牌，更新见到的牌
            self.tiles_seen[tile.id] = 4
            self.tiles_public[tile.id] = 4
            return

        self.at_rinshan = 0
        self.tehai_len_div3 -= 1
        self.move_tile(consumed[0], MoveType.FULU_CONSUME)
        self.move_tile(consumed[1], MoveType.FULU_CONSUME)
        self.move_tile(consumed[2], MoveType.FULU_CONSUME)
        self.minkans.append(tile.id)  # 杠牌副露加入

        self.update_shanten()  # 3n + 1
        self.update_waits()    # 3n + 1

    def _handle_kakan(self, event: Event.BuGang) -> None:
        """处理加杠事件"""
        actor_rel = self.rel(event.player)
        tile = event.tile

        # 更新副露信息
        for fulu in self.fulu_overview[actor_rel]:
            if fulu[0] == tile:
                fulu.append(tile)
                break

        self.intermediate_kan.append(tile)

        if actor_rel != 0:
            # 其他人加杠，更新见到的牌
            self.tiles_seen[tile.id] += 1
            self.tiles_public[tile.id] += 1
            self.last_kawa_tile = tile
            # 判断是否可以抢杠胡
            if self.waits[tile.id]:
                self.chankan_chance = True
                self.last_cans.can_ron_hu = True
            return

        self.at_rinshan = 0
        self.move_tile(tile, MoveType.FULU_CONSUME)
        self.pons.remove(tile.id)  # 碰牌副露移除
        self.minkans.append(tile.id)

        if self.next_shanten_discards[tile.id]:
            self.shanten -= 1
        elif not self.keep_shanten_discards[tile.id]:
            self.update_shanten()   # 3n + 1

        self.update_waits()  # 3n + 1

    def _handle_ankan(self, event: Event.AnGang) -> None:
        """处理暗杠事件"""
        actor_rel = self.rel(event.player)
        # 更新副露信息
        self.ankan_overview_count[actor_rel] += 1

        if actor_rel != 0:
            return

        tile = event.consumed[0]
        consumed = event.consumed

        self.intermediate_kan.append(tile)

        self.at_rinshan = 0
        self.tehai_len_div3 -= 1
        for c_tile in consumed:
            self.move_tile(c_tile, MoveType.FULU_CONSUME)
        self.ankans.append(tile.id)

        self.update_shanten()   # 3n + 1
        self.update_waits()

    def rel(self, actor):
        """
        返回一个与指定actor相对于当前对象（self）的玩家ID（player_id）的相对位置索引
        @param actor:
        @return:
        """
        return (actor + 4 - self.seat_id) % 4

    def move_tile(self, tile: Tile, move_type: MoveType) -> None:
        """移动牌"""
        if move_type == MoveType.TSUMO:
            self.tiles_seen[tile.id] += 1
            self.handcards[tile.id] += 1
        elif move_type == MoveType.DISCARD:
            if self.handcards[tile.id] <= 0:
                raise ValueError(f"Cannot discard tile {tile} that is not in hand")
            self.tiles_public[tile.id] += 1
            self.handcards[tile.id] -= 1
        elif move_type == MoveType.FULU_CONSUME:
            if self.handcards[tile.id] <= 0:
                raise ValueError(f"Cannot use tile {tile} for fuuro that is not in hand")
            self.tiles_public[tile.id] += 1
            self.handcards[tile.id] -= 1

    def pad_kawa_for_pon_or_daiminkan(self, abs_actor, abs_target):
        """
        为碰和大明杠动作填充弃牌序列
        @param abs_actor:
        @param abs_target:
        @return:
        """
        i = (abs_target + 1) % 4
        while i != abs_actor:
            rel = self.rel(i)
            self.kawa[rel].append(None)
            i = (i + 1) % 4

    def pad_kawa_at_start(self):
        """
        在游戏开始时，根据庄家位置在相应玩家的河牌中添加空位
        @return:
        """
        for i in range(self.zhuang_id):
            self.kawa[i].append(None)

    def set_can_chi_from_tile(self, tile: Tile):
        self.last_cans.can_chi_low = False
        self.last_cans.can_chi_mid = False
        self.last_cans.can_chi_high = False

        tile_id = tile.id
        literal_num = tile_id % 9 + 1

        # ---- low  chi ---------------------------------
        if literal_num <= 7 and self.handcards[tile_id + 1] > 0 and self.handcards[tile_id + 2] > 0:
            self.last_cans.can_chi_low = True

        # ---- mid  chi ------------------------
        if 2 <= literal_num <= 8 and self.handcards[tile_id - 1] > 0 and self.handcards[tile_id + 1] > 0:
            self.last_cans.can_chi_mid = True

        # ---- high chi ---------------------------------
        if literal_num >= 3 and self.handcards[tile_id - 2] > 0 and self.handcards[tile_id - 1] > 0:
            self.last_cans.can_chi_high = True

    def update_shanten(self) -> None:
        """
        计算向听数, 在3n+1时调用。
        @return:
        """
        if self.uses_shangrao_rules():
            xts = self.shangrao_rules.shanten(
                self.handcards, self.open_meld_count(), self.king_tile_id
            )
        else:
            if mc is None:
                raise RuntimeError("国标 mortal_cpp.so 不可用；请使用含 king_card 的上饶事件流或在 Linux 运行")
            xts = mc.ShantenCalculator.calc_all(self.handcards, self.tehai_len_div3)
        if xts > 6:
            logging.warning(f"向听数{xts}，不在[0,6]中")
        self.shanten = max(xts, 0)

    def update_shanten_discards(self):
        """
        更新向听数和弃牌信息
        计算每张牌作为弃牌时的向听数变化，并标记进张和维持向听数的牌
        只能在手牌为 3n+2 的情况下调用
        """
        assert self.last_cans.can_discard, "tehai is not 3n+2"

        # 重置标记数组
        self.next_shanten_discards = [False] * 34
        self.keep_shanten_discards = [False] * 34
        self.has_next_shanten_discard = False

        # 临时手牌用于计算
        tehai = self.handcards
        # 遍历所有种类的牌
        for tid in range(34):
            # `self.forbidden_tiles[tid]` is not checked here, but it is acceptable
            # because forbidden tiles are always keep - shanten discards,
            # so it won't affect the result of `has_next_shanten_discard`.
            # We will take forbidden_tiles into account when generating discard candidates.
            count = self.handcards[tid]
            # 跳过没有的牌
            if count == 0:
                continue
            tehai[tid] -= 1  # 模拟打出一张牌
            if self.uses_shangrao_rules():
                shanten_after = self.shangrao_rules.shanten(
                    tehai, self.open_meld_count(), self.king_tile_id
                )
            else:
                if mc is None:
                    raise RuntimeError("国标 mortal_cpp.so 不可用；无法计算国标向听数")
                shanten_after = mc.ShantenCalculator.calc_all(tehai, self.tehai_len_div3)
            tehai[tid] += 1
            if shanten_after < self.shanten:
                self.next_shanten_discards[tid] = True
                self.has_next_shanten_discard = True
            elif shanten_after == self.shanten:
                self.keep_shanten_discards[tid] = True

    def update_waits(self):
        """
        更新听牌状态，检查当前手牌的待牌情况，记录所有可能的和牌；
        必须确保当前的手牌是3n+1，并且`self.shanten`是最新的且正确的。
        """
        # 确保手牌数量为3n+1
        assert not self.last_cans.can_discard, "tehai is not 3n+1"

        # 重置待牌数组
        self.waits = [False] * 34

        if self.uses_shangrao_rules():
            self.waits = self.shangrao_rules.waits(
                self.handcards,
                self.open_meld_count(),
                self.king_tile_id,
                self.tiles_seen,
                open_melds_are_yaojiu=self.own_open_melds_are_yaojiu(),
            )
            return

        # 如果还没听牌则直接返回
        if self.shanten > 0:
            return

        # 检查每种牌是否是待牌
        for tid in range(34):
            # 已经有4张相同的牌则跳过
            if self.handcards[tid] == 4 or self.tiles_seen[tid] == 4:
                continue

            # 模拟摸到这张牌
            self.handcards[tid] += 1
            # 计算加入这张牌后的向听数
            if mc.ShantenCalculator.calc_all(self.handcards, self.tehai_len_div3) == -1:
                # 只有未被使用完的牌才能等待
                self.waits[tid] = (self.tiles_seen[tid] < 4)
            self.handcards[tid] -= 1

    def update_rank(self):
        """更新当前玩家的排名"""
        self.rank = self.get_rank(self.scores)

    def get_rank(self, scores_rel):
        """
        计算玩家排名
        @param scores_rel: 从视角玩家开始的相对分数列表
        @return: 当前玩家的排名(1-4)
        """
        # 转换为绝对位置的分数
        scores_rel_copy = scores_rel.copy()
        # 右移玩家ID位，使得分数位置对应绝对座位  用np.roll()也行
        scores_abs = scores_rel_copy[-self.seat_id:] + scores_rel_copy[:-self.seat_id]

        # 使用Rankings类计算排名
        rankings = Rankings(scores_abs)
        return rankings.rank_by_player[self.seat_id]

    def rotate_left(self, arr, mid):
        """原地旋转数组"""
        if not arr or mid == 0:
            return
        n = len(arr)
        mid = mid % n
        if mid == 0:
            return

        def reverse(start, end):
            while start < end:
                arr[start], arr[end] = arr[end], arr[start]
                start += 1
                end -= 1

        reverse(0, mid - 1)
        reverse(mid, n - 1)
        reverse(0, n - 1)


def tiles_to_string(tiles: List[int]) -> str:
    """
    将牌的数组表示转换为字符串表示。
    @param tiles: 27种牌的数量数组
    @return: 牌的字符串表示
    """
    suhai_parts = []  # 数牌集合，川麻只处理数牌就好

    # 处理数牌 (萬子、筒子、索子)
    for kind in range(3):  # 0=萬, 1=筒, 2=索
        chunk = tiles[kind * 9:(kind + 1) * 9]
        partial = ""
        not_empty = False

        for num, count in enumerate(chunk):
            if count > 0:
                literal_num = num + 1
                partial += str(literal_num) * count
                not_empty = True

        if not_empty:
            c = 'm' if kind == 0 else 'p' if kind == 1 else 's'
            partial += c
            suhai_parts.append(partial)

    suhai = " ".join(suhai_parts)
    return suhai


if __name__ == '__main__':
    print("")
