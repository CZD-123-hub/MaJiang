"""运行九江红中四人同 AI 自博弈。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jiujiang_ai.self_play import run_self_play


def main() -> None:
    parser = argparse.ArgumentParser(description="九江红中四人同 AI 自博弈")
    parser.add_argument("--rounds", type=int, default=10, help="对局数，默认 10")
    parser.add_argument("--seed", type=int, default=None, help="随机种子；指定后可复现")
    parser.add_argument("--verbose", action="store_true", help="实时打印摸牌和动作过程")
    args = parser.parse_args()
    callback = _print_event if args.verbose else None
    result = run_self_play(rounds=args.rounds, seed=args.seed, event_callback=callback)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _print_event(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
