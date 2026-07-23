import json
from typing import Tuple
from mortal_part.mjai.event import Event, EventExt, Metadata  # 引入相关类
from mortal_part.rankings import Rankings
from mortal_part.tile import Tile


class KyokuResult:
    """
    表示局结果的类
    """

    def __init__(self, kyoku, has_hu, scores):
        self.kyoku = kyoku
        self.has_hu = has_hu
        self.scores = scores  # 传入一个长度为 4 的列表

    def __repr__(self):
        return f"KyokuResult(kyoku={self.kyoku}, has_hu={self.has_hu}, scores={self.scores}"


class GameResult:
    """
    表示整个游戏结果的类
    """

    def __init__(self, names=None, scores=None, zhuang=0, seed: Tuple[int, int] = None,
                 game_log=None):
        self.names = [""] * 4 if names is None else names  # 长度为 4 的玩家名字列表
        self.scores = [0] * 4 if scores is None else scores  # 长度为 4 的玩家分数列表
        self.zhuang = zhuang  # 场风，0表示东风场，1表示南风场等
        self.seed = (0, 0) if seed is None else seed  # 元组，包含种子信息
        self.game_log = [] if game_log is None else game_log  # 每个事件是 EventExt 的实例

    def rankings(self):
        """
        生成玩家排名
        """
        return Rankings(self.scores)

    def dump_json_log(self):
        """
        导出日志为 JSON 格式
        """
        log = []

        # 添加开始游戏事件
        # [V4 local-play fix] StartGame signature is
        # (names, zhuang, wind, seed).  Pass seed by keyword instead of
        # accidentally putting it into the wind slot.
        start_game_event = Event('StartGame', self.names, self.zhuang, self.zhuang, seed=self.seed)
        log.append(json.dumps(start_game_event.to_dict(), ensure_ascii=False, default=lambda x: x.to_dict()))

        # 添加游戏日志中的所有事件
        for ev in filter(None, self.game_log):
            for e in ev:
                log.append(json.dumps(e.to_dict(), ensure_ascii=False, default=lambda x: x.to_dict()))

        # 添加结束游戏事件
        log.append(json.dumps(Event('EndGame').to_dict(), ensure_ascii=False, default=lambda x: x.to_dict()))

        # 转换为 JSON 字符串
        return '\n'.join(log) + '\n'

    def __repr__(self):
        return (f"GameResult(names={self.names}, scores={self.scores}, "
                f"seed={self.seed}, game_log={self.game_log})")


# 示例使用
if __name__ == "__main__":
    # 示例数据
    names = ["玩家1", "玩家2", "玩家3", "玩家4"]
    scores = [25000, 30000, 20000, 15000]
    seed = (12345, 67890)
    ee = EventExt(Event('Tsumo', 0, Tile(0)), Metadata(None, None))
    ee2 = EventExt(Event('Dahai', 1, Tile(1)), Metadata(1.0, None))
    ee3 = EventExt(Event('Pon', 2, Tile(2), 1, [Tile(2), Tile(2)]), Metadata(None, None))
    game_log = [[ee, ee2], [ee3]]
    game_result = GameResult(names, scores, seed, game_log)
    print(game_result.rankings())
    print(game_result.dump_json_log())
