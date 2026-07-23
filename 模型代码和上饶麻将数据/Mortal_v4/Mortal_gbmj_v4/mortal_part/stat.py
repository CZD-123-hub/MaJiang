import glob
import gzip
import json
import os

from tqdm import tqdm

from mortal_part.mjai.event import *
from mortal_part.rankings import *


class Stat:
    def __init__(self):
        self.game = 0
        self.round = 0
        self.zhuang_id = 0
        self.point = 0
        self.rank_1 = 0
        self.rank_2 = 0
        self.rank_3 = 0
        self.rank_4 = 0
        self.tobi = 0  # 击飞 川麻是否需要待定
        self.fulu = 0
        self.fulu_num = 0
        self.fulu_point = 0
        self.fulu_hu = 0
        self.fulu_hu_jun = 0
        self.fulu_hu_point = 0
        self.fulu_houjuu = 0  # houjuu 指放铳
        self.hu = 0
        self.hu_as_zhuang = 0
        self.hu_jun = 0
        self.hu_point_zhuang = 0
        self.hu_point_ko = 0
        self.houjuu = 0
        self.houjuu_jun = 0
        self.houjuu_to_zhuang = 0
        self.houjuu_point_to_zhuang = 0
        self.houjuu_point_to_ko = 0
        self.ryukyoku = 0
        self.ryukyoku_point = 0

    def __iadd__(self, other):
        """使用反射自动累加所有字段"""
        if not isinstance(other, Stat):
            return NotImplemented

        for attr in vars(self):
            current_value = getattr(self, attr)
            other_value = getattr(other, attr)
            setattr(self, attr, current_value + other_value)

        return self

    @classmethod
    def from_game(cls, events: [Event], player_id):
        stat = cls()
        stat.game = 1

        cur_scores = [0] * 4
        cur_zhuang = 0
        jun = 0
        fulu_num = 0

        for event in events:
            if isinstance(event, Event):
                event = event.event

            if isinstance(event, Event.StartKyoku):
                stat.round += 1
                cur_scores = event.scores
                cur_zhuang = event.zhuang
                if cur_zhuang == player_id:
                    stat.zhuang_id += 1
                jun = 0
                fulu_num = 0

            elif isinstance(event, Event.Dahai) and event.player == player_id:
                jun += 1

            elif isinstance(event, (Event.Pon, Event.MinGang)) and event.player == player_id:
                fulu_num += 1

            elif isinstance(event, Event.Hu):
                deltas = event.deltas
                for i in range(4):
                    cur_scores[i] += deltas[i]

                if event.player == player_id:
                    point = deltas[player_id]
                    stat.hu += 1
                    stat.hu_jun += jun

                    if cur_zhuang == player_id:
                        stat.hu_as_zhuang += 1
                        stat.hu_point_zhuang += point
                    else:
                        stat.hu_point_ko += point
                    if fulu_num > 0:
                        stat.fulu_hu += 1
                        stat.fulu_hu_jun += jun
                        stat.fulu_hu_point += point
                        stat.fulu_point += point
                elif event.target == player_id:
                    # 玩家点炮（放铳）
                    point = deltas[player_id]
                    stat.houjuu += 1
                    stat.houjuu_jun += jun

                    if cur_zhuang == event.player:  # 点炮给庄家
                        stat.houjuu_to_zhuang += 1
                        stat.houjuu_point_to_zhuang += point
                    else:  # 点炮给闲家
                        stat.houjuu_point_to_ko += point
                    if fulu_num > 0:
                        stat.fulu_houjuu += 1
                        stat.fulu_point += point

            elif isinstance(event, Event.Ryukyoku):
                stat.ryukyoku += 1

            elif isinstance(event, Event.EndKyoku):
                if fulu_num > 0:
                    stat.fulu += 1
                    stat.fulu_num += fulu_num

        rk = Rankings(cur_scores)

        final_score = cur_scores[player_id]
        stat.point = final_score - 500
        if final_score < 0:
            stat.tobi = 1

        rank = rk.rank_by_player[player_id]
        setattr(stat, f'rank_{rank + 1}', 1)

        return stat

    def _update_fulu_hu(self, point, jun):
        self.fulu_hu += 1
        self.fulu_hu_jun += jun
        self.fulu_hu_point += point
        self.fulu_point += point

    @classmethod
    def from_dir(cls, directory: str, player_name: str, disable_progress_bar: bool = False) -> 'Stat':
        """
        从目录中读取并分析所有对局记录
        @param directory:
        @param player_name:
        @param disable_progress_bar:
        @return:
        """
        json_files = glob.glob(os.path.join(directory, "**/*.json"), recursive=True)
        json_gz_files = glob.glob(os.path.join(directory, "**/*.json.gz"), recursive=True)
        all_files = json_files + json_gz_files

        stats = cls()
        for path in tqdm(all_files, disable=disable_progress_bar):
            try:
                if path.endswith('.gz'):
                    with gzip.open(path, 'rt', encoding='utf-8') as f:
                        raw_log = f.read()
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        raw_log = f.read()

                events = [Event.from_str(line) for line in raw_log.splitlines()]
                if not events:
                    continue

                first_event = events[0]
                if not first_event.event.names or not isinstance(first_event.event, Event.StartGame):
                    continue

                names = first_event.event.names
                for i, name in enumerate(names):
                    if name == player_name:
                        game_stat = cls.from_game(events, i)
                        stats += game_stat

            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        return stats

    @classmethod
    def from_log(cls, log: str, player_id: int) -> 'Stat':
        """
        从单个对局记录分析统计数据
        @param log: 对局记录数据
        @param player_id:
        @return:
        """
        events = [Event.from_str(line) for line in log.splitlines()]
        return cls.from_game(events, player_id)

    def total_pt(self, pts: List[int]) -> int:
        """计算总点数"""
        return (self.rank_1 * pts[0] + self.rank_2 * pts[1] +
                self.rank_3 * pts[2] + self.rank_4 * pts[3])

    def avg_pt(self, pts: List[int]) -> float:
        """计算平均点数"""
        return self.total_pt(pts) / self.game if self.game else 0

    @property
    def avg_rank(self) -> float:
        """平均顺位"""
        return self.avg_pt([1, 2, 3, 4])

    @property
    def rank_1_rate(self) -> float:
        """一位率"""
        return self.rank_1 / self.game if self.game else 0

    @property
    def rank_2_rate(self) -> float:
        """二位率"""
        return self.rank_2 / self.game if self.game else 0

    @property
    def rank_3_rate(self) -> float:
        """三位率"""
        return self.rank_3 / self.game if self.game else 0

    @property
    def rank_4_rate(self) -> float:
        """四位率"""
        return self.rank_4 / self.game if self.game else 0

    @property
    def tobi_rate(self) -> float:
        """飞人率"""
        return self.tobi / self.game if self.game else 0

    @property
    def avg_point_per_game(self) -> float:
        """场均得点"""
        return self.point / self.game if self.game else 0

    @property
    def avg_point_per_round(self) -> float:
        """每巡得点"""
        return self.point / self.round if self.round else 0

    @property
    def avg_point_per_hu(self) -> float:
        """平均和了点数"""
        return (self.hu_point_ko + self.hu_point_zhuang) / self.hu if self.hu else 0

    @property
    def avg_point_per_zhuang_hu(self) -> float:
        """庄家和了平均点数"""
        return self.hu_point_zhuang / self.hu_as_zhuang if self.hu_as_zhuang else 0

    @property
    def avg_point_per_ko_hu(self) -> float:
        """子家和了平均点数"""
        ko_hu = self.hu - self.hu_as_zhuang
        return self.hu_point_ko / ko_hu if ko_hu else 0

    @property
    def avg_point_per_fulu_hu(self) -> float:
        """副露和了平均点数"""
        return self.fulu_hu_point / self.fulu_hu if self.fulu_hu else 0

    @property
    def avg_point_per_ryukyoku(self) -> float:
        """流局平均得点"""
        return self.ryukyoku_point / self.ryukyoku if self.ryukyoku else 0

    @property
    def avg_hu_jun(self) -> float:
        """和了巡目"""
        return self.hu_jun / self.hu if self.hu else 0

    @property
    def avg_fulu_hu_jun(self) -> float:
        """副露和了巡目"""
        return self.fulu_hu_jun / self.fulu_hu if self.fulu_hu else 0

    @property
    def avg_point_per_houjuu(self) -> float:
        """放铳平均点数"""
        return (self.houjuu_point_to_ko + self.houjuu_point_to_zhuang) / self.houjuu if self.houjuu else 0

    @property
    def avg_point_per_houjuu_to_zhuang(self) -> float:
        """放铳给庄家平均点数"""
        return self.houjuu_point_to_zhuang / self.houjuu_to_zhuang if self.houjuu_to_zhuang else 0

    @property
    def avg_point_per_houjuu_to_ko(self) -> float:
        """放铳给子家平均点数"""
        ko_houjuu = self.houjuu - self.houjuu_to_zhuang
        return self.houjuu_point_to_ko / ko_houjuu if ko_houjuu else 0

    @property
    def avg_houjuu_jun(self) -> float:
        """放铳巡目"""
        return self.houjuu_jun / self.houjuu if self.houjuu else 0

    @property
    def hu_rate(self) -> float:
        """和了率"""
        return self.hu / self.round if self.round else 0

    @property
    def houjuu_rate(self) -> float:
        """放铳率"""
        return self.houjuu / self.round if self.round else 0

    @property
    def fulu_rate(self) -> float:
        """副露率"""
        return self.fulu / self.round if self.round else 0

    @property
    def ryukyoku_rate(self) -> float:
        """流局率"""
        return self.ryukyoku / self.round if self.round else 0

    @property
    def hu_rate_as_zhuang(self) -> float:
        """亲家和了率"""
        return self.hu_as_zhuang / self.zhuang_id if self.zhuang_id else 0

    @property
    def hu_as_zhuang_rate(self) -> float:
        """和了中亲家和了比例"""
        return self.hu_as_zhuang / self.hu if self.hu else 0

    @property
    def houjuu_to_zhuang_rate(self) -> float:
        """放铳中放铳给亲家比例"""
        return self.houjuu_to_zhuang / self.houjuu if self.houjuu else 0

    @property
    def avg_fulu_num(self) -> float:
        """平均副露数"""
        return self.fulu_num / self.fulu if self.fulu else 0

    @property
    def hu_rate_after_fulu(self) -> float:
        """副露后和了率"""
        return self.fulu_hu / self.fulu if self.fulu else 0

    @property
    def houjuu_rate_after_fulu(self) -> float:
        """副露后放铳率"""
        return self.fulu_houjuu / self.fulu if self.fulu else 0

    @property
    def avg_fulu_point(self) -> float:
        """副露收支"""
        return self.fulu_point / self.fulu if self.fulu else 0

    def __str__(self) -> str:
        """字符串表示"""
        return f"Stat(games={self.game}, avg_rank={self.avg_rank:.2f})"

    def __repr__(self) -> str:
        """详细表示"""
        return self.__str__()
