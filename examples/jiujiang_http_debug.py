from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

# 允许脚本从 D:\MaJiang 之外运行时，仍然能导入 jiujiang_ai 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jiujiang_ai.api import get_action


def build_sample_data() -> dict[str, Any]:
    """构造一份九江红中本地调试用局面。"""
    hand = [
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x11,
        0x12,
        0x13,
        0x21,
        0x22,
        0x23,
        0x09,
        0x18,
    ]
    return {
        "room_id": 1,
        "game_area_id": 10021,
        "acting_player_position": 0,
        "acting_do_player_position": 0,
        "played_cards": [[], [], [], []],
        "player_hand_cards": [hand, [], [], []],
        "action_seq": [],
        "last_action": [],
        "player_chi_cards": [[], [], [], []],
        "player_peng_cards": [[], [], [], []],
        "player_gang_cards": [[], [], [], []],
        "player_bugang_cards": [[], [], [], []],
        "player_angang_cards": [[], [], [], []],
        "player_bu_cards": [[], [], [], []],
        # 候选弃牌只给两个，便于观察 AI 是否从候选列表里选择。
        "action_cards": {"7": [[0x09], [0x18]]},
        "remain_card_stack": [],
    }


def post_get_action(url: str, data: dict[str, Any]) -> list[Any]:
    """向 /get_action 发送一份局面数据，并返回 [action_type, action_card]。"""
    request = urllib.request.Request(
        url,
        data=json.dumps({"data": data}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="九江红中麻将 AI HTTP 调试脚本")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/get_action",
        help="本地或远程 /get_action 地址",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="不走 HTTP，直接调用 Python get_action(data)",
    )
    args = parser.parse_args()

    data = build_sample_data()
    if args.direct:
        result = list(get_action(data))
    else:
        result = post_get_action(args.url, data)

    print(json.dumps({"request_action_cards": data["action_cards"], "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
