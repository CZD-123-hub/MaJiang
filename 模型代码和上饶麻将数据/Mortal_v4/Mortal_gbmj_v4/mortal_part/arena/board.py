from typing import List, Tuple
from enum import Enum
import hashlib
import random

from mortal_part.mjai.event import Event, EventExt
from mortal_part.state.mah_player_gb import PlayerState
from mortal_part.vec_ops import vec_add_assign
from mortal_part.tile import Tile
from mortal_part.arena.result import KyokuResult
import logging

from colorlog import ColoredFormatter

console_handler = logging.StreamHandler()
# 定义颜色格式
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
# 设置格式化器
console_handler.setFormatter(formatter)
# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler]
)

class AgentContext:
    """代理上下文类"""

    def __init__(self, player_states: List[PlayerState], log: List[EventExt]):
        self.player_states = player_states
        self.log = log

class Poll(Enum):
    IN_GAME = "InGame"
    END = "End"


# 棋盘类实现，所有字段都是公开的，以便调用者可以直接设置牌山（yama）和分数（scores）。
class Board:
    def __init__(self, kyoku=0, scores=None):
        self.kyoku: int = kyoku  # 从0开始, 阶段数
        self.scores: List[int] = scores if scores is not None else [0] * 4
        self.haipai: List[List[Tile]] = [[], [], [], []]  # 玩家的配牌情况 [[Tile; 13]; 4]
        self.yama: List[Tile] = []

    def init_from_seed(self, game_seed: Tuple[int, int]):
        """
        根据游戏种子初始化牌局
        @param game_seed:
        @return:
        """
        nonce, key = game_seed
        kyoku_seed = hashlib.sha3_256()  # 使用 SHA3-256 算法对这些字节进行哈希计算
        kyoku_seed.update(nonce.to_bytes(8, 'little'))  # 将 nonce 转换为 8 字节的小端字节序
        kyoku_seed.update(key.to_bytes(8, 'little'))  # 将 key 转换为 8 字节的小端字节序
        kyoku_seed.update(bytes(self.kyoku))  # 将当前的局数转换为字节, 川麻中就一局不需要再次哈希
        seed_bytes = kyoku_seed.digest()

        random.seed(int.from_bytes(seed_bytes, 'big'))

        # 使用UNSHUFFLED常量，复制一份以避免修改原始数据
        seq = UNSHUFFLED.copy()
        random.shuffle(seq)

        # 分发手牌
        for i in range(4):
            self.haipai[i] = seq[i * 13:(i + 1) * 13]

        idx = 13 * 4

        self.yama = seq[idx:idx + 84]
        idx += 84

        assert idx == len(seq), f"Index {idx} should equal sequence length {len(seq)}"

    def into_state(self) -> 'BoardState':
        """
        转换为游戏状态
        @return:
        """
        zhuang = self.kyoku % 4
        round_wind = self.kyoku // 4
        player_states = [PlayerState(i) for i in range(4)]
        for i in range(4):
            # 玩家 i 的场风：相对于庄家 dealer 的偏移
            seat_wind = (i - zhuang + 4) % 4
            player_states[i].round_wind = round_wind
            player_states[i].seat_wind = seat_wind
        return BoardState(self, zhuang=zhuang, player_states=player_states)


class BoardState:
    """游戏状态类"""

    def __init__(self, board, zhuang, player_states):
        self.board: Board = board
        self.zhuang: int = zhuang
        self.player_states: List[PlayerState] = player_states  # [PlayerState; 4]

        self.has_hu = False
        self.kyoku_deltas = [0] * 4

        self.tiles_left = 84
        self.tsumo_actor = 0

        self.log = []  # [EventExt]

    def poll(self, reactions: List[EventExt]) -> Poll:
        """
        轮询游戏状态，判断是否还在游戏中
        @param reactions: [EventExt; 4] 四个玩家的动作列表
        @return:
        """
        while True:
            poll = self.step(reactions)
            if poll == Poll.IN_GAME:
                # 检查是否有玩家可以行动
                if any(player.last_cans.can_act for player in self.player_states):
                    return poll
            elif poll == Poll.END:
                # 结束局并添加日志
                self.add_log_no_meta(Event("EndKyoku"))
                # 更新这个游戏的最终牌桌分数
                vec_add_assign(self.board.scores, self.kyoku_deltas)
                return poll
            # 重置反应列表
            reactions = [EventExt.no_meta(Event('NoneEvent')) for _ in range(4)]

    def agent_context(self) -> AgentContext:
        """
        返回代理上下文
        @return:
        """
        return AgentContext(self.player_states, self.log)

    def end(self) -> KyokuResult:
        """
        返回整局游戏（4盘）的结果
        @return:
        """
        return KyokuResult(
            kyoku=self.board.kyoku,
            has_hu=self.has_hu,
            scores=self.board.scores
        )

    def take_log(self) -> List[EventExt]:
        """
        获取并清空日志
        @return:
        """
        log = self.log
        self.log = []
        return log

    def add_log(self, ev: EventExt):
        """
        添加带元数据的日志
        @param ev:
        @return:
        """
        self.log.append(ev)

    def add_log_no_meta(self, ev: Event):
        """
        添加不带元数据的日志
        @param ev:
        @return:
        """
        self.log.append(EventExt.no_meta(ev))

    def broadcast(self, ev: Event):
        """
        广播事件给所有玩家
        @param ev:
        @return:
        """
        for state in self.player_states:
            state.update(ev)

    def haipai(self):
        """
        开局时进行配牌，牌山剩余13张牌
        @return:
        """
        # 创建开局事件
        round_wind = self.board.kyoku // 4 # 圈风
        wind_str = ["E", "S", "W", "N"]
        start_kyoku = Event("StartKyoku",
                            wind=wind_str[round_wind],
                            kyoku=self.zhuang + 1,
                            zhuang=self.zhuang,
                            scores=self.board.scores,
                            tehais=self.board.haipai,
                            flower_count=[0, 0, 0, 0]
                            )
        self.broadcast(start_kyoku)
        self.add_log_no_meta(start_kyoku)

        # 处理第一张摸牌
        if not self.board.yama:
            raise ValueError("invalid yama: empty at init")

        tile = self.board.yama.pop()
        self.tiles_left -= 1

        first_tsumo = Event("Tsumo", self.zhuang, tile)
        self.broadcast(first_tsumo)
        self.add_log_no_meta(first_tsumo)

    def exhaustive_ryukyoku(self):
        """
        处理手牌情况耗尽的流局
        @return:
        """
        pass
    def handle_hora(self, single_actor: int, single_target: int, tile: Tile, reactions: List[EventExt]) -> None:
        """
        处理和牌情况
        @param tile: 胡的牌
        @param single_actor: 和牌者
        @param single_target: 放铳者/自摸者
        @param reactions: 玩家反应列表
        """
        self.has_hu = True
        is_ron = single_actor != single_target

        # 一炮多响，只计算按点炮者逆时针方向的第一个玩家的和牌结果
        for i in range(4):
            seat = (single_target + i) % 4
            if isinstance(reactions[seat].event.event, Event.Hu):
                point = self.player_states[seat].agari_points(is_ron)
                if point is None:
                    print(self.player_states[seat].handcards)
                    if is_ron:
                        print(self.player_states[seat].last_kawa_tile)
                    else:
                        print(self.player_states[seat].last_self_tsumo)
                    logging.error("和牌玩家无法计算和牌点数")
                    return
                deltas = [0] * 4
                if is_ron:
                    deltas[single_target] -= point.fan
                    for j in range(4):
                        if j != seat:
                            deltas[j] -= 8
                else:
                    for j in range(4):
                       if j != seat:
                           deltas[j] -= point.fan
                           deltas[j] -= 8
                deltas[seat] += point.ron if is_ron else point.tsumo
                vec_add_assign(self.kyoku_deltas, deltas)

                # 记录和牌事件
                hora = Event("Hu", player=seat, target=single_target, tile=tile, fan=point.fan, deltas=deltas)
                self.add_log_no_meta(hora)
                return

    def step(self, reactions: List[EventExt]) -> Poll:
        """
        处理游戏的每一步
        @param reactions: [EventExt; 4] 四个玩家的动作列表
        @return: 游戏状态（进行中/结束）
        """
        # 初始配牌
        if self.tiles_left == 84:
            self.haipai()
            return Poll.IN_GAME

        # 验证所有玩家的反应
        for actor, ev in enumerate(reactions):
            self.player_states[actor].validate_reaction(ev.event)

        # 选择优先级最高的事件
        # 优先级：和牌 > 杠碰 > 其他 > 无动作
        ev = min(reactions, key=lambda x: {
            'Hu': 0,
            'MinGang': 1,
            'Pon': 1,
            'NoneEvent': 3
        }.get(type(x.event.event).__name__, 2))
        # 根据事件类型处理
        # 处理None事件
        if isinstance(ev.event.event, Event.NoneEvent):
            # 检查是否流局
            if self.tiles_left == 0:
                # self.exhaustive_ryukyoku()
                return Poll.END
            # 处理摸牌事件
            if not self.board.yama:  # 如果牌山不为空
                raise ValueError(f"tiles left > 0 ({self.tiles_left}) but yama is empty")

            tile = self.board.yama.pop()

            self.tiles_left -= 1
            tsumo = Event("Tsumo", self.tsumo_actor, tile)

            self.broadcast(tsumo)
            self.add_log_no_meta(tsumo)

        # 处理打牌事件
        elif isinstance(ev.event.event, Event.Dahai):
            self.broadcast(ev.event)
            self.add_log(ev)
            self.tsumo_actor = (ev.event.event.player + 1) % 4

        # 处理吃、碰事件
        elif isinstance(ev.event.event, (Event.Pon, Event.Chi)):
            self.broadcast(ev.event)
            self.add_log(ev)

        # 处理暗杠事件
        elif isinstance(ev.event.event, Event.AnGang):
            self.broadcast(ev.event)
            self.add_log(ev)
            self.tsumo_actor = ev.event.event.player

        # 处理明杠和加杠事件
        elif isinstance(ev.event.event, (Event.MinGang, Event.BuGang)):
            self.broadcast(ev.event)
            self.add_log(ev)
            self.tsumo_actor = ev.event.event.player

        # 处理和牌事件
        elif isinstance(ev.event.event, Event.Hu):
            self.handle_hora(ev.event.event.player, ev.event.event.target, ev.event.event.tile, reactions)
            return Poll.END

        elif isinstance(ev.event.event, Event.Ryukyoku):
            return Poll.END

        else:
            raise ValueError(f"unexpected event: {ev.event}")

        return Poll.IN_GAME


# 定义未清洗前的所有牌
UNSHUFFLED = [
    Tile(0), Tile(0), Tile(0), Tile(0),  # 1m
    Tile(1), Tile(1), Tile(1), Tile(1),  # 2m
    Tile(2), Tile(2), Tile(2), Tile(2),  # 3m
    Tile(3), Tile(3), Tile(3), Tile(3),  # 4m
    Tile(4), Tile(4), Tile(4), Tile(4),  # 5m
    Tile(5), Tile(5), Tile(5), Tile(5),  # 6m
    Tile(6), Tile(6), Tile(6), Tile(6),  # 7m
    Tile(7), Tile(7), Tile(7), Tile(7),  # 8m
    Tile(8), Tile(8), Tile(8), Tile(8),  # 9m

    Tile(9), Tile(9), Tile(9), Tile(9),  # 1p
    Tile(10), Tile(10), Tile(10), Tile(10),  # 2p
    Tile(11), Tile(11), Tile(11), Tile(11),  # 3p
    Tile(12), Tile(12), Tile(12), Tile(12),  # 4p
    Tile(13), Tile(13), Tile(13), Tile(13),  # 5p
    Tile(14), Tile(14), Tile(14), Tile(14),  # 6p
    Tile(15), Tile(15), Tile(15), Tile(15),  # 7p
    Tile(16), Tile(16), Tile(16), Tile(16),  # 8p
    Tile(17), Tile(17), Tile(17), Tile(17),  # 9p

    Tile(18), Tile(18), Tile(18), Tile(18),  # 1s
    Tile(19), Tile(19), Tile(19), Tile(19),  # 2s
    Tile(20), Tile(20), Tile(20), Tile(20),  # 3s
    Tile(21), Tile(21), Tile(21), Tile(21),  # 4s
    Tile(22), Tile(22), Tile(22), Tile(22),  # 5s
    Tile(23), Tile(23), Tile(23), Tile(23),  # 6s
    Tile(24), Tile(24), Tile(24), Tile(24),  # 7s
    Tile(25), Tile(25), Tile(25), Tile(25),  # 8s
    Tile(26), Tile(26), Tile(26), Tile(26),  # 9s

    Tile(27), Tile(27), Tile(27), Tile(27),  # East
    Tile(28), Tile(28), Tile(28), Tile(28),  # South
    Tile(29), Tile(29), Tile(29), Tile(29),  # West
    Tile(30), Tile(30), Tile(30), Tile(30),  # North
    Tile(31), Tile(31), Tile(31), Tile(31),  # Zhong
    Tile(32), Tile(32), Tile(32), Tile(32),  # Fa
    Tile(33), Tile(33), Tile(33), Tile(33),  # Bai
]

if __name__ == '__main__':
    print(UNSHUFFLED)
    seed = (12345678, 87654321)
    seed = tuple(map(int, seed))
    board = Board(kyoku=0, scores=[500, 500, 500, 500])
    board.init_from_seed(seed)
    print(board.haipai)
    print(board.yama)
