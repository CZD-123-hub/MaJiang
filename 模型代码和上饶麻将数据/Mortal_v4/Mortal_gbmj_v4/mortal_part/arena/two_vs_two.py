import logging
import os
import gzip
import threading
from typing import Optional, Callable, List, Tuple

from mortal_part.agent.defs import BatchAgent
from mortal_part.agent.py_agent import new_py_agent
from mortal_part.arena.game import BatchGame, Index
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


class TwoVsTwo:
    def __init__(self, disable_progress_bar: bool = False, log_dir: Optional[str] = None):
        self.disable_progress_bar = disable_progress_bar
        self.log_dir = log_dir

    def py_vs_py(self, challenger, champion, seed_start: Tuple[int, int], seed_count: int) -> None:
        def run_in_thread():
            self.run_batch(
                lambda player_ids: new_py_agent(challenger, player_ids),
                lambda player_ids: new_py_agent(champion, player_ids),
                seed_start,
                seed_count
            )

        # 使用线程运行任务
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()  # 等待线程完成

    def run_batch(self, new_challenger_agent: Callable[[List[int]], 'BatchAgent'],
                  new_champion_agent: Callable[[List[int]], 'BatchAgent'],
                  seed_start: Tuple[int, int], seed_count: int) -> List[GameResult]:
        # 创建日志目录
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # 日志记录
        logging.info(
            f"seed: [{seed_start[0]}, {seed_start[0] + seed_count}) "
            f"w/ {seed_start[1]:#x}, start {seed_count} groups, {seed_count * 2} hanchans")

        # 生成种子
        seeds = [(seed, seed_start[1]) for seed in range(seed_start[0], seed_start[0] + seed_count) for _ in range(2)]

        # 玩家ID分配
        challenger_player_ids_per_seed = [0, 2, 1, 3]
        challenger_player_ids = challenger_player_ids_per_seed * seed_count

        champion_player_ids_per_seed = [1, 3, 0, 2]
        champion_player_ids = champion_player_ids_per_seed * seed_count

        # 创建代理
        agents = [
            new_challenger_agent(challenger_player_ids),
            new_champion_agent(champion_player_ids)
        ]

        # 创建批量游戏
        batch_game = BatchGame.south_game(self.disable_progress_bar)

        # 生成索引
        challenger_idx = 0
        champion_idx = 0
        agent_idxs_per_seed = [
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ]
        indexes = []
        for agent_idxs_per_split in agent_idxs_per_seed * seed_count:
            for agent_idx in agent_idxs_per_split:
                player_id_idx = challenger_idx if agent_idx == 0 else champion_idx
                index = Index(agent_idx=agent_idx, player_id_idx=player_id_idx)
                if agent_idx == 0:
                    challenger_idx += 1
                else:
                    champion_idx += 1
                indexes.append(index)

        # 运行批量游戏
        results = batch_game.run(agents, indexes, seeds)

        # 保存日志
        if self.log_dir:  # TODO 更具体的日志保存问题，后面再实现
            logging.info("dumping game logs")

            for i, game_result in enumerate(results):
                split_name = ["a", "b"][i % 2]
                seed, key = game_result.seed
                filename = os.path.join(self.log_dir, f"{seed}_{key}_{split_name}.json.gz")

                log = game_result.dump_json_log()
                with gzip.open(filename, 'wt') as f:
                    f.write(log)

        return results

    def run_one(
            self,
            new_challenger_agent: Callable[[List[int]], 'BatchAgent'],
            new_champion_agent: Callable[[List[int]], 'BatchAgent'],
            seed: Tuple[int, int], split: int) -> GameResult:
        # 创建日志目录
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        logging.info(f"seed: {seed[0]} w/ {seed[1]:#x}, split: {split}, start 1 hanchan")

        # 根据 split 选择玩家 ID
        challenger_player_ids = [0, 2] if split == 0 else [1, 3]
        champion_player_ids = [1, 3] if split == 0 else [0, 2]

        # 创建代理
        agents = [
            new_challenger_agent(challenger_player_ids),
            new_champion_agent(champion_player_ids),
        ]
        batch_game = BatchGame.south_game(self.disable_progress_bar)

        # 生成索引
        indexes = [[
            Index(agent_idx=0, player_id_idx=0),
            Index(agent_idx=1, player_id_idx=0),
            Index(agent_idx=0, player_id_idx=1),
            Index(agent_idx=1, player_id_idx=1),
        ]] if split == 0 else [[
            Index(agent_idx=1, player_id_idx=0),
            Index(agent_idx=0, player_id_idx=0),
            Index(agent_idx=1, player_id_idx=1),
            Index(agent_idx=0, player_id_idx=1),
        ]]

        # 运行单个游戏
        results = batch_game.run(agents, indexes, [seed])

        # 保存日志
        if self.log_dir:  # TODO 更具体的日志保存问题，后面再实现
            logging.info("dumping game logs")
            split_name = ["a", "b"][split]
            seed, key = seed
            filename = os.path.join(self.log_dir, f"{seed}_{key}_{split_name}.json.gz")

            log = results[0].dump_json_log()
            with gzip.open(filename, 'wt') as f:
                f.write(log)

        return results[0]
