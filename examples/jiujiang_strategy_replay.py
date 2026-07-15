from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jiujiang_ai.strategy_replay import SUPPORTED_STRATEGIES, compare_strategy_snapshots


def load_snapshots(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("snapshot file must be a JSON array of game-state objects")
    return payload


def build_sample_snapshots() -> list[dict[str, Any]]:
    hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
    return [
        {
            "snapshot_id": "sample-1",
            "action_cards": {"7": [[0x01], [0x08]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
            "action_seq": [],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="九江红中麻将策略离线回放对比")
    parser.add_argument("--input", help="局面快照 JSON 数组文件")
    parser.add_argument("--sample", action="store_true", help="使用内置局面样例")
    parser.add_argument(
        "--strategies",
        default="heuristic,multi_route,multi_route_tree",
        help=f"逗号分隔的策略，支持：{','.join(SUPPORTED_STRATEGIES)}",
    )
    args = parser.parse_args()
    if not args.input and not args.sample:
        raise SystemExit("请提供 --input 或 --sample")
    strategies = tuple(item.strip() for item in args.strategies.split(",") if item.strip())
    snapshots = build_sample_snapshots() if args.sample else load_snapshots(args.input)
    report = compare_strategy_snapshots(snapshots, strategies=strategies)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
