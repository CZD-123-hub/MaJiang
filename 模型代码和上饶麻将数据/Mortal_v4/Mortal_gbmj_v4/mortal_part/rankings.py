import numpy as np


class Rankings:
    def __init__(self, scores):
        """
        初始化 Rankings 对象，计算 player_by_rank 和 rank_by_player。

        :param scores: List[int]，四个玩家的分数。
        """
        self.player_by_rank = np.argsort([-score for score in scores]).astype(np.uint8)
        self.rank_by_player = np.empty(4, dtype=np.uint8)
        for rank, player_id in enumerate(self.player_by_rank):
            self.rank_by_player[player_id] = rank

    def __repr__(self):
        return f"Rankings(player_by_rank={self.player_by_rank.tolist()}, rank_by_player={self.rank_by_player.tolist()})"
