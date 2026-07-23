import copy
from typing import Optional
from time import time
from tqdm import tqdm

from mortal_part.agent.defs import BatchAgent
from mortal_part.arena.board import *
from mortal_part.mjai.event import Event, EventExt
from mortal_part.state.mah_player_gb import PlayerState
from mortal_part.arena.result import GameResult


class Index:
    """
    用于Game查找特定Agent（game -> agent）和Agent查找特定玩家ID（agent -> game）的索引类
    """

    def __init__(self, agent_idx: int, player_id_idx: int):
        # For `Game` to find a specific `Agent` (game -> agent).
        self.agent_idx = agent_idx
        # For `Agent` to find a specific player ID (agent -> game).
        self.player_id_idx = player_id_idx

    def __repr__(self):
        return f"Index(agent_idx={self.agent_idx}, player_id_idx={self.player_id_idx})"

class Game:

    """
    游戏状态跟踪和管理类
    """
    def __init__(self, length: int, seed: Tuple[int, int], indexes: List[Index], scores: List[int]):
        self.length: int = length  # 16表示全庄战、12表示西风战、8表示半庄战，4表示东风战
        self.seed: Tuple[int, int] = seed
        self.indexes: List[Index] = indexes  # [Index; 4]

        self.last_reactions = [EventExt.no_meta(Event('NoneEvent'))] * 4  # [EventExt; 4] 用于poll阶段的缓存

        self.board: BoardState = BoardState(board=None, zhuang=0, player_states=[PlayerState(i) for i in range(4)])  # BoardState()
        self.kyoku: int = 0  # 当前对局数
        self.scores: List[int] = scores  # 各家分数

        self.game_log: List[List[EventExt]] = []  # 游戏记录

        self.kyoku_started: bool = False  # 当前局是否已开始
        self.ended: bool = False  # 游戏是否结束

    """
    查询游戏状态，检查是否有玩家可以行动或游戏是否结束
    """
    def poll(self, agents: List[BatchAgent]) -> None:
        if self.ended:
            return

        if not self.kyoku_started:
            # TODO 在打满指定盘数时（4盘游戏）结束游戏，要打多少局需要改这个self.length
            if self.kyoku >= self.length:
                self.ended = True
                return

            # 初始化新的牌局
            next_board = Board(kyoku=self.kyoku, scores=self.scores.copy())
            next_board.init_from_seed(self.seed)
            self.board = next_board.into_state()
            self.kyoku_started = True

        reactions = copy.deepcopy(self.last_reactions)
        self.last_reactions = [EventExt.no_meta(Event('NoneEvent'))] * 4
        poll = self.board.poll(reactions)

        if poll == Poll.IN_GAME:
            # 游戏继续进行，处理玩家行动
            ctx = self.board.agent_context()
            for player_id, state in enumerate(ctx.player_states):
                if not state.last_cans.can_act:
                    continue

                idx = self.indexes[player_id]
                agents[idx.agent_idx].set_scene(
                    idx.player_id_idx,
                    ctx.log,
                    state,
                    None  # 改成直接传None省去对隐藏信息的编码
                )

        elif poll == Poll.END:
            # 当前局结束，处理结算
            self.kyoku_started = False

            # 通知所有代理当前局结束
            for idx in self.indexes:
                agents[idx.agent_idx].end_kyoku(idx.player_id_idx)

            # 获取并处理结果
            kyoku_result = self.board.end()
            self.scores = kyoku_result.scores

            # 保存日志
            logs = self.board.take_log()
            self.game_log.append(logs)

            # 处理正常结束
            self.kyoku += 1
            return self.poll(agents)

    def commit(self, agents: List[BatchAgent]) -> Optional[GameResult]:
        """
        提交更改并在游戏结束时返回结果
        """
        if self.ended:
            # 创建游戏结果
            # [V4 local-play fix] Agent implementations are inconsistent:
            # old Mortal shadows .name with a string, while the v4 supervised
            # agent keeps name() as a method.  Normalize here so StartGame
            # logs contain plain strings instead of bound method objects.
            names = []
            for i in range(4):
                agent = agents[self.indexes[i].agent_idx]
                agent_name = getattr(agent, "name", "")
                if callable(agent_name):
                    agent_name = agent_name()
                names.append(str(agent_name))
            game_result = GameResult(
                names=names,
                scores=self.scores,
                zhuang=(self.kyoku - 1) // 4,
                seed=self.seed,
                game_log=self.game_log
            )

            # 通知所有代理游戏结束
            for idx in self.indexes:
                agents[idx.agent_idx].end_game(idx.player_id_idx, game_result)
            return game_result

        # 处理玩家行动
        ctx = self.board.agent_context()
        # 给可以行动的玩家生成推荐动作
        for player_id, state in enumerate(ctx.player_states):
            if not state.last_cans.can_act:
                continue

            idx = self.indexes[player_id]
            self.last_reactions[player_id] = agents[idx.agent_idx].get_reaction(
                idx.player_id_idx,
                ctx.log,
                state,
                None  # 改成直接传None省去对隐藏信息的编码
            )

        return None


class BatchGame:
    """
    批量游戏控制器
    """
    def __init__(self, length: int = 4, init_scores: List[int] = None,
                 disable_progress_bar: bool = False):
        self.length = length
        self.init_scores = init_scores or [500] * 4
        self.disable_progress_bar = disable_progress_bar

    """
    创建一个东风战游戏
    """
    @classmethod
    def east_game(cls, disable_progress_bar: bool = False) -> 'BatchGame':
        return cls(length=4, init_scores=[500] * 4,
                   disable_progress_bar=disable_progress_bar)

    """
    创建一个半庄游戏
    """
    @classmethod
    def south_game(cls, disable_progress_bar: bool = False) -> 'BatchGame':
        return cls(length=8, init_scores=[500] * 4,
                   disable_progress_bar=disable_progress_bar)

    """
    创建一个西风战游戏
    """
    @classmethod
    def west_game(cls, disable_progress_bar: bool = False) -> 'BatchGame':
        return cls(length=12, init_scores=[500] * 4,
                   disable_progress_bar=disable_progress_bar)

    """
    创建一个全庄游戏
    """
    @classmethod
    def north_game(cls, disable_progress_bar: bool = False) -> 'BatchGame':
        return cls(length=16, init_scores=[500] * 4,
                   disable_progress_bar=disable_progress_bar)

    def run(self, agents: List[BatchAgent], indexes: List[List[Index]],
            seeds: List[Tuple[int, int]]) -> List[GameResult]:
        """
        并行运行多个游戏
        """
        if not agents:
            raise ValueError("agents列表不能为空")
        if not indexes:
            raise ValueError("indexes列表不能为空")
        if len(indexes) != len(seeds):
            raise ValueError(f"indexes长度({len(indexes)})必须等于seeds长度({len(seeds)})")

        # 初始化游戏
        games = []
        for game_idx, (idxs, seed) in enumerate(zip(indexes, seeds)):
            for i, idx in enumerate(idxs):
                agents[idx.agent_idx].start_game(idx.player_id_idx)
            game = Game(length=self.length, seed=seed, indexes=idxs, scores=self.init_scores.copy())
            games.append((game_idx, game))

        game_results = [GameResult() for _ in range(len(games))]
        to_remove = []
        cycles = 0
        actions = 0

        # 设置进度条
        pbar = None if self.disable_progress_bar else tqdm(
            total=len(games),
            desc="正在处理游戏",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        )

        start_time = time()

        # 主游戏循环
        while games:
            # 查询阶段
            for _, game in games:
                game.poll(agents)
            # 单局测试用
            # game_idx, game = games[0]
            # game.poll(agents)

            # 提交阶段
            for idx_for_rm, (game_idx, game) in enumerate(games):
                if game_result := game.commit(agents):
                    game_results[game_idx] = game_result
                    to_remove.append(idx_for_rm)

            # 单局测试用
            # if game_result := game.commit(agents):
            #     game_results[game_idx] = game_result
            #     to_remove.append(0)

            # 移除已完成的游戏
            for idx_for_rm in reversed(to_remove):
                games.pop(idx_for_rm)
                if pbar:
                    pbar.update(1)

            to_remove.clear()
            cycles += 1
            actions += len(games)
            # 更新进度条信息
            if pbar:
                elapsed = time() - start_time
                pbar.set_postfix({
                    'cycles': f"{cycles} ({cycles / elapsed:.3f} 轮/秒)",
                    'actions': f"{actions} ({actions / elapsed:.3f} 动作/秒)"
                })

        if pbar:
            pbar.close()

        return game_results