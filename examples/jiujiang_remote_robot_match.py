"""创建测试服机器人房间，并持续收集九江麻将 AI 对局结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jiujiang_ai.remote_match import (
    DEFAULT_ROBOT_ROOM_URL,
    start_continuous_remote_robot_matches,
    start_sequential_remote_robot_matches,
    write_report,
)
from jiujiang_ai.stats import DEFAULT_ROUND_LOG_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="测试服机器人九江麻将对局")
    parser.add_argument("--place-id", type=int, required=True, help="测试服 place_id")
    parser.add_argument(
        "--rooms",
        "--rounds",
        dest="rooms",
        type=int,
        help="指定局数；不填则持续开房，按 Ctrl+C 停止",
    )
    parser.add_argument("--url", default=DEFAULT_ROBOT_ROOM_URL, help="机器人开房接口")
    parser.add_argument("--timeout", type=int, default=1800, help="单个房间等待结算的秒数，默认 1800")
    parser.add_argument("--poll", type=float, default=2.0, help="读取结算日志的间隔秒数，默认 2")
    parser.add_argument("--log-path", default=str(DEFAULT_ROUND_LOG_PATH), help="本地 round_end JSONL 路径")
    parser.add_argument("--output", default="logs/jiujiang_remote_robot_report.json", help="汇总报告输出路径")
    args = parser.parse_args()

    if args.rooms is not None:
        print(f"将顺序进行 {args.rooms} 局：每局仅创建 1 个机器人房间。", flush=True)
        report = start_sequential_remote_robot_matches(
            place_id=args.place_id,
            rooms=args.rooms,
            url=args.url,
            log_path=args.log_path,
            timeout_seconds=args.timeout,
            poll_seconds=args.poll,
            progress_callback=_print_progress,
        )
        output = write_report(report, args.output)
    else:
        print("持续模式：每局仅创建 1 个房间；按 Ctrl+C 停止并写入报告。", flush=True)

        def save_progress(report: dict) -> None:
            # 每局结束就刷新一次报告，意外关闭时也尽量保留已完成结果。
            write_report(report, args.output)

        report = start_continuous_remote_robot_matches(
            place_id=args.place_id,
            url=args.url,
            log_path=args.log_path,
            timeout_seconds=args.timeout,
            poll_seconds=args.poll,
            progress_callback=_print_progress,
            report_callback=save_progress,
        )
        output = write_report(report, args.output)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入：{output}")


def _print_progress(event: dict) -> None:
    if event["event"] == "opening_round":
        total = event["total"]
        label = f"第 {event['round']} 局" if total is None else f"第 {event['round']}/{total} 局"
        print(f"{label}：正在请求开房…", flush=True)
    elif event["event"] == "room_created":
        print(
            f"第 {event['round']} 局开房返回：成功 {event['created_count']}，"
            f"失败 {event['failed_count']}，房间号 {event['room_ids']}",
            flush=True,
        )
    elif event["event"] == "round_progress":
        print(f"结算进度：{event['completed']}/{event['expected']} 局", flush=True)
    elif event["event"] == "room_error":
        print(f"第 {event['round']} 局开房异常：{event['error']}；稍后重试。", flush=True)


if __name__ == "__main__":
    main()
