from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from mortal_part.arena.result import GameResult
from mortal_part.mjai.event import EventExt
from mortal_part.state.mah_player_gb import PlayerState


InvisibleState = np.ndarray


class BatchAgent(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def oracle_obs_version(self) -> Optional[int]:
        return None

    @abstractmethod
    def set_scene(self, index: int, log: List[EventExt], state: PlayerState,
                  invisible_state: Optional[InvisibleState]) -> None:
        pass

    @abstractmethod
    def get_reaction(self, index: int, log: List[EventExt], state: PlayerState,
                     invisible_state: Optional[InvisibleState]) -> EventExt:
        pass

    def start_game(self, index: int) -> None:
        pass

    def end_kyoku(self, index: int) -> None:
        pass

    def end_game(self, index: int, game_result: GameResult) -> None:
        pass
