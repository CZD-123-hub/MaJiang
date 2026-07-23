import logging
import os
import gzip
from typing import Optional, Callable, List, Tuple

from mortal_part.agent.defs import BatchAgent
from mortal_part.agent.py_agent import new_py_agent
from mortal_part.arena.game import Index, BatchGame
from mortal_part.arena.result import GameResult
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


class OneVsThree:
    def __init__(self, disable_progress_bar: bool = False, log_dir: Optional[str] = None):
        self.disable_progress_bar = disable_progress_bar
        self.log_dir = log_dir

    def py_vs_py(self, challenger, champion, seed_start: Tuple[int, int], seed_count: int) -> List[int]:
        results = self.run_batch(
            lambda player_ids: new_py_agent(challenger, player_ids),
            lambda player_ids: new_py_agent(champion, player_ids),
            seed_start,
            seed_count
        )
        rankings = [0] * 4
        for i, result in enumerate(results):
            rank = result.rankings().rank_by_player[i % 4]
            rankings[rank] += 1
        return rankings

    def run_batch(self,
                  new_challenger_agent: Callable[[List[int]], 'BatchAgent'],
                  new_champion_agent: Callable[[List[int]], 'BatchAgent'],
                  seed_start: Tuple[int, int],
                  seed_count: int) -> List[GameResult]:
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # 日志记录
        logging.info(
            f"seed: [{seed_start[0]}, {seed_start[0] + seed_count}) "
            f"w/ {seed_start[1]:#x}, start {seed_count} groups, {seed_count * 4} hanchans")

        seeds = [(seed, seed_start[1]) for seed in range(seed_start[0], seed_start[0] + seed_count) for _ in range(4)]

        challenger_player_ids = list(range(4)) * seed_count
        # 3个擂主的4种划分方法
        champion_player_ids_per_seed = [
            1, 2, 3,  # split A
            0, 2, 3,  # split B
            0, 1, 3,  # split C
            0, 1, 2,  # split D
        ]
        champion_player_ids = champion_player_ids_per_seed * seed_count

        agents = [
            new_challenger_agent(challenger_player_ids),
            new_champion_agent(champion_player_ids)
        ]
        batch_game = BatchGame.east_game(self.disable_progress_bar)

        challenger_idx = 0
        champion_idx = 0
        agent_idxs_per_seed = [
            [0, 1, 1, 1],  # split A
            [1, 0, 1, 1],  # split B
            [1, 1, 0, 1],  # split C
            [1, 1, 1, 0],  # split D
        ]
        indexes = []
        for _ in range(seed_count):
            # 中层循环：4种split配置
            for agent_idxs_per_split in agent_idxs_per_seed:
                game_indexes = []  # 创建当前游戏的索引列表
                # 内层循环：4个玩家
                for agent_idx in agent_idxs_per_split:
                    if agent_idx == 0:
                        player_id_idx = challenger_idx
                        challenger_idx += 1
                    else:
                        player_id_idx = champion_idx
                        champion_idx += 1
                    game_indexes.append(Index(
                        agent_idx=agent_idx,
                        player_id_idx=player_id_idx
                    ))
                indexes.append(game_indexes)

        results = batch_game.run(agents, indexes, seeds)

        if self.log_dir:
            print("dumping game logs....................")
            for i, game_result in enumerate(results):
                split_name = ["a", "b", "c", "d"][i % 4]
                seed, key = game_result.seed
                filename = os.path.join(self.log_dir, f"{seed}_{key}_{split_name}.json.gz")

                log = game_result.dump_json_log()
                with gzip.open(filename, 'wt') as f:
                    f.write(log)
        return results
