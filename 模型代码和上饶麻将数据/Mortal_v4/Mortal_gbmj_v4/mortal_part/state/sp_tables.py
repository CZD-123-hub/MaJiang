from typing import Any, List

try:
    # 仅用于类型标注；Windows 上做上饶规则回放时没有国标 Linux .so。
    from mortal_part.mortal_cpp import Candidate
except ImportError:
    Candidate = Any

class SinglePlayerTables:
    def __init__(self, max_ev_table: List[Candidate] | None):
        self.max_ev_table = max_ev_table
