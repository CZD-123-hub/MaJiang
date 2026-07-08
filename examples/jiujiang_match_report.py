from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 允许脚本在 D:\MaJiang 之外运行时，仍然能导入 jiujiang_ai 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jiujiang_ai.stats import summarize_match_report


def build_sample_rounds() -> list[dict[str, Any]]:
    """构造一组本地演示用对局结果，方便先验证汇总脚本是否正常工作。"""
    return [
        {"winner": 0, "scores": [3, -1, -1, -1], "win_type": "zimo"},
        {"winner": 2, "score_delta": {"0": 0, "1": -2, "2": 2, "3": 0}, "dianpao_player": 1},
        {"winner": 3, "scores": [-2, 0, 0, 2], "dianpao_player": 0},
    ]


def load_rounds(path: str | Path) -> list[dict[str, Any]]:
    """从 JSON 文件读取一批 round_end 结果，要求文件内容是局结果数组。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("round result file must contain a JSON array")
    return payload


def generate_match_report(
    rounds: list[dict[str, Any]],
    our_players: list[int | str] | None = None,
) -> dict[str, Any]:
    """生成对打汇总报告，默认返回整体统计，可选附带我方座位视角摘要。"""
    return summarize_match_report(rounds, our_players=our_players)


def _parse_players(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="九江红中麻将对打结果汇总脚本")
    parser.add_argument("--input", help="包含 round_end 结果数组的 JSON 文件路径")
    parser.add_argument("--our-players", help="我方座位列表，例如 0,2")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="不读取文件，直接使用脚本内置演示数据生成汇总结果",
    )
    args = parser.parse_args()

    if args.sample:
        rounds = build_sample_rounds()
    elif args.input:
        rounds = load_rounds(args.input)
    else:
        raise SystemExit("请提供 --input 或 --sample")

    report = generate_match_report(rounds, our_players=_parse_players(args.our_players))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
