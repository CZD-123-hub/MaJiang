# -*- coding: utf-8 -*-
"""
将"被覆盖的决策"与"HTTP响应耗时"关联，判断是否超时导致。
- http.log 的 response 行里带有 action_card 和 elapsed_ms（但时间戳是本地时间）
- action.jsonl 带有 ISO 时间戳、决策牌、以及可交叉验证的 played_cards
思路：按秒级时间戳把两边对齐，比较"被覆盖组"和"被执行组"的耗时分布。
"""

import sys, io, json, re
from collections import defaultdict
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ACTION_DISCARD = 7


def load_actions(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _list_diff(after, before):
    bc = defaultdict(int)
    for t in before:
        bc[t] += 1
    diff = []
    for t in after:
        if bc[t] > 0:
            bc[t] -= 1
        else:
            diff.append(t)
    return diff


def build_override_map(entries):
    """返回 {second_key: [(decided, was_overridden), ...]}，用ISO秒做key"""
    grouped = defaultdict(list)
    for e in entries:
        p = e.get("player_position")
        if not isinstance(p, int) or not (0 <= p <= 3):
            continue
        grouped[(e.get("room_id"), p)].append(e)

    # 按局切分（played长度回退=新局）
    records = []  # (iso_second, decided_tile, overridden_bool)
    for (room, player), seq in grouped.items():
        games, cur, prev = [], [], -1
        for e in seq:
            pc = e.get("table_state", {}).get("played_cards") or []
            cl = len(pc[player] or []) if player < len(pc) else 0
            if cl < prev:
                if cur:
                    games.append(cur)
                cur = []
            cur.append(e)
            prev = cl
        if cur:
            games.append(cur)

        for game in games:
            for idx, e in enumerate(game):
                if e.get("action_type") != ACTION_DISCARD:
                    continue
                ac = e.get("action_card") or []
                if not ac:
                    continue
                decided = ac[0]
                pc = e.get("table_state", {}).get("played_cards") or []
                if player >= len(pc):
                    continue
                before = list(pc[player] or [])
                nxt = None
                for later in game[idx+1:]:
                    lpc = later.get("table_state", {}).get("played_cards") or []
                    if player < len(lpc):
                        nxt = list(lpc[player] or [])
                        break
                if nxt is None:
                    continue
                new_tiles = _list_diff(nxt, before)
                if not new_tiles:
                    continue
                overridden = decided not in new_tiles
                ts = e.get("timestamp", "")
                # ISO 秒级 key: 2026-07-15T17:36:31
                sec = ts[:19]
                records.append((sec, e.get("room_id"), player, decided, overridden))
    return records


def parse_http_responses(path):
    """返回 {second_key: [elapsed_ms,...]}，用本地时间秒做key"""
    by_sec = defaultdict(list)
    # 行例: 2026-07-15 17:34:48,730 INFO get_action response ... elapsed_ms=29.7
    pat = re.compile(r'^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+.*get_action response.*elapsed_ms=([\d.]+)')
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = pat.match(line)
            if m:
                date, tm, elapsed = m.group(1), m.group(2), float(m.group(3))
                key = f"{date}T{tm}"
                by_sec[key].append(elapsed)
    return by_sec


def main():
    print("=" * 70)
    print("超时关联分析：被覆盖的决策 vs 响应耗时")
    print("=" * 70)
    print()

    entries = load_actions("logs/jiujiang_action.jsonl")
    records = build_override_map(entries)
    http_by_sec = parse_http_responses("logs/jiujiang_http.log")

    # 按秒对齐：每条决策记录找同一秒的http响应耗时（取最大，保守）
    honored_ms, overridden_ms = [], []
    matched = 0
    for sec, room, player, decided, overridden in records:
        candidates = http_by_sec.get(sec)
        if not candidates:
            continue
        matched += 1
        elapsed = max(candidates)
        if overridden:
            overridden_ms.append(elapsed)
        else:
            honored_ms.append(elapsed)

    print(f"决策记录总数: {len(records)}")
    print(f"成功对齐到HTTP耗时的: {matched}")
    print()

    def stats(name, arr):
        if not arr:
            print(f"{name}: 无数据")
            return
        arr = sorted(arr)
        n = len(arr)
        print(f"{name} (n={n}):")
        print(f"  平均耗时: {sum(arr)/n:.1f} ms")
        print(f"  中位数:   {arr[n//2]:.1f} ms")
        print(f"  P90:      {arr[int(n*0.9)]:.1f} ms")
        print(f"  最大:     {arr[-1]:.1f} ms")

    print("【核心对比】")
    print("-" * 70)
    stats("被执行的决策(honored)", honored_ms)
    print()
    stats("被覆盖的决策(overridden)", overridden_ms)
    print()

    # 按耗时阈值看被覆盖率
    print("【不同耗时区间的被覆盖率】")
    print("-" * 70)
    buckets = [(0, 300), (300, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10**9)]
    combined = [(e, False) for e in honored_ms] + [(e, True) for e in overridden_ms]
    for lo, hi in buckets:
        in_bucket = [ov for e, ov in combined if lo <= e < hi]
        if in_bucket:
            rate = sum(in_bucket) / len(in_bucket) * 100
            label = f"{lo}-{hi}ms" if hi < 10**9 else f">{lo}ms"
            print(f"  {label:>14}: 共{len(in_bucket):4d}手, 被覆盖率 {rate:5.1f}%")
    print()

    print("=" * 70)
    print("判读")
    print("=" * 70)
    if overridden_ms and honored_ms:
        ov_med = sorted(overridden_ms)[len(overridden_ms)//2]
        hn_med = sorted(honored_ms)[len(honored_ms)//2]
        if ov_med > hn_med * 1.5:
            print(f"被覆盖决策的耗时中位数({ov_med:.0f}ms)明显高于被执行的({hn_med:.0f}ms)")
            print("=> 强烈支持：AI响应太慢，测试服超时后打了默认牌")
        else:
            print(f"被覆盖({ov_med:.0f}ms)和被执行({hn_med:.0f}ms)耗时接近")
            print("=> 超时不是唯一原因，可能还有协议/字段理解问题")


if __name__ == "__main__":
    main()
