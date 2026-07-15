# -*- coding: utf-8 -*-
"""
验证座位0和座位2的动作是否真的由用户AI返回，还是超时后被测试服默认打牌覆盖。

核心思路（交叉验证）：
1. jiujiang_action.jsonl 记录了每次AI决策返回的 action_card 和当时的 table_state
2. 对于同一房间同一玩家，第N次决策返回的弃牌，应在第N+1次请求的 played_cards 里出现
3. 如果实际打出的是另一张牌 => 说明AI返回被丢弃，测试服用了默认牌（疑似超时/异常）
4. 同时统计响应耗时分布，找出可能触发超时的慢响应
"""

import sys
import io
import json
from collections import defaultdict

# 修复Windows控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

USER_SEATS = [0, 2]
ACTION_DISCARD = 7


def load_action_log(path):
    """加载AI决策日志"""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def split_into_games(seq, player):
    """
    把同一(room,player)的决策序列按"局"切分。
    played_cards[player]长度回退 => 新的一局开始。
    """
    games = []
    current = []
    prev_len = -1
    for e in seq:
        pc = e.get("table_state", {}).get("played_cards") or []
        cur_len = len(pc[player] or []) if player < len(pc) else 0
        # 长度回退说明是新局（上一局打了很多牌，新局从0开始）
        if cur_len < prev_len:
            if current:
                games.append(current)
            current = []
        current.append(e)
        prev_len = cur_len
    if current:
        games.append(current)
    return games


def verify_discards(entries):
    """
    交叉验证弃牌决策是否被实际执行。
    先按 (room_id, player) 分组，再按局切分，只在同一局内比较 played_cards 增量。
    """
    # 按房间+玩家分组，保留时间顺序
    grouped = defaultdict(list)
    for e in entries:
        room = e.get("room_id")
        player = e.get("player_position")
        if not isinstance(player, int) or not (0 <= player <= 3):
            continue
        grouped[(room, player)].append(e)

    results = {
        "total_discards": 0,
        "verified_honored": 0,   # AI弃牌确实被打出
        "overridden": 0,          # AI弃牌被替换成别的牌（疑似超时默认）
        "unverifiable": 0,        # 无后续记录，无法验证（通常是本局最后一手）
        "override_details": [],
        "per_seat": {s: {"total": 0, "honored": 0, "overridden": 0, "unverifiable": 0} for s in range(4)},
    }

    for (room, player), seq in grouped.items():
        # 按局切分，只在同一局内做增量比较
        for game in split_into_games(seq, player):
            for idx, e in enumerate(game):
                if e.get("action_type") != ACTION_DISCARD:
                    continue
                action_card = e.get("action_card") or []
                if not action_card:
                    continue
                decided_tile = action_card[0]

                ts = e.get("table_state", {})
                played_before = ts.get("played_cards") or []
                if player >= len(played_before):
                    continue
                my_played_before = list(played_before[player] or [])

                results["total_discards"] += 1
                if player in results["per_seat"]:
                    results["per_seat"][player]["total"] += 1

                # 查找同一局内该玩家的下一条记录，看 played_cards 增量
                next_state = None
                for later in game[idx + 1:]:
                    lts = later.get("table_state", {})
                    lplayed = lts.get("played_cards") or []
                    if player < len(lplayed):
                        next_state = list(lplayed[player] or [])
                        break

                if next_state is None:
                    results["unverifiable"] += 1
                    results["per_seat"][player]["unverifiable"] += 1
                    continue

                # played_cards 的增量 = 这一手之后该玩家实际新打出的牌
                new_tiles = _list_diff(next_state, my_played_before)

                if not new_tiles:
                    # 没有增量（可能被碰/杠打断，或状态未推进），无法确认
                    results["unverifiable"] += 1
                    results["per_seat"][player]["unverifiable"] += 1
                    continue

                # AI决定打的牌是否在新增打出的牌里
                if decided_tile in new_tiles:
                    results["verified_honored"] += 1
                    results["per_seat"][player]["honored"] += 1
                else:
                    results["overridden"] += 1
                    results["per_seat"][player]["overridden"] += 1
                    results["override_details"].append({
                        "room": room,
                        "player": player,
                        "ai_decided": decided_tile,
                        "actually_played": new_tiles,
                        "timestamp": e.get("timestamp"),
                    })

    return results


def _list_diff(after, before):
    """返回 after 相对 before 多出来的元素（考虑重复）"""
    before_count = defaultdict(int)
    for t in before:
        before_count[t] += 1
    diff = []
    for t in after:
        if before_count[t] > 0:
            before_count[t] -= 1
        else:
            diff.append(t)
    return diff


def parse_http_log(path):
    """解析HTTP日志，统计响应耗时和缺失响应"""
    requests = {}  # (room, player, turn) -> request info
    responses = []
    slow_responses = []
    missing_responses = 0

    request_lines = 0
    response_lines = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if "get_action request" in line:
                request_lines += 1
            elif "get_action response" in line:
                response_lines += 1
                # 提取 elapsed_ms 和 player_id
                elapsed = _extract_float(line, "elapsed_ms=")
                player = _extract_int(line, "player_id=")
                if elapsed is not None:
                    responses.append({"elapsed_ms": elapsed, "player": player})
                    if elapsed > 500:  # 超过500ms视为慢响应
                        slow_responses.append({"elapsed_ms": elapsed, "player": player, "line": line.strip()})

    return {
        "request_count": request_lines,
        "response_count": response_lines,
        "missing_responses": request_lines - response_lines,
        "responses": responses,
        "slow_responses": slow_responses,
    }


def _extract_float(line, key):
    try:
        part = line.split(key)[1].split()[0]
        return float(part)
    except (IndexError, ValueError):
        return None


def _extract_int(line, key):
    try:
        part = line.split(key)[1].split()[0]
        return int(part)
    except (IndexError, ValueError):
        return None


def main():
    print("=" * 70)
    print("座位0/座位2 动作真实性验证报告")
    print("=" * 70)
    print()

    # 1. 交叉验证弃牌
    print("【第一部分：弃牌决策交叉验证】")
    print("-" * 70)
    print("方法：对比AI返回的弃牌 与 下一次请求时实际打出的牌")
    print()

    entries = load_action_log("logs/jiujiang_action.jsonl")
    print(f"AI决策日志总条数: {len(entries)}")

    results = verify_discards(entries)
    print(f"可验证的弃牌决策总数: {results['total_discards']}")
    print(f"  [确认执行] AI弃牌被实际打出: {results['verified_honored']}")
    print(f"  [被覆盖]   AI弃牌被替换:     {results['overridden']}")
    print(f"  [无法验证] 无后续记录:       {results['unverifiable']}")
    print()

    verifiable = results['verified_honored'] + results['overridden']
    if verifiable > 0:
        honor_rate = results['verified_honored'] / verifiable * 100
        print(f"在可确认的 {verifiable} 手中，AI决策执行率: {honor_rate:.2f}%")
        override_rate = results['overridden'] / verifiable * 100
        print(f"被测试服默认牌覆盖的比例: {override_rate:.2f}%")
    print()

    # 分座位
    print("分座位统计（用户AI = 座位0/2）：")
    for seat in USER_SEATS:
        s = results["per_seat"][seat]
        v = s["honored"] + s["overridden"]
        rate = (s["honored"] / v * 100) if v > 0 else 0
        print(f"  座位{seat}: 总{s['total']}手, 确认执行{s['honored']}, 被覆盖{s['overridden']}, 无法验证{s['unverifiable']} => 执行率{rate:.1f}%")
    print()

    # 覆盖详情
    if results["override_details"]:
        print(f"[!] 发现 {len(results['override_details'])} 处AI决策未被执行（前10条）：")
        for d in results["override_details"][:10]:
            print(f"    房间{d['room']} 座位{d['player']}: AI想打{d['ai_decided']}, 实际打了{d['actually_played']}")
    else:
        print("[OK] 未发现AI弃牌决策被覆盖的情况")
    print()

    # 2. 响应耗时分析
    print("【第二部分：HTTP响应耗时分析】")
    print("-" * 70)

    http = parse_http_log("logs/jiujiang_http.log")
    print(f"get_action 请求总数: {http['request_count']}")
    print(f"get_action 响应总数: {http['response_count']}")
    print(f"缺失响应数（请求无对应响应，疑似崩溃/无返回）: {http['missing_responses']}")
    print()

    all_elapsed = [r["elapsed_ms"] for r in http["responses"] if r["elapsed_ms"] is not None]
    if all_elapsed:
        all_elapsed.sort()
        n = len(all_elapsed)
        print(f"响应耗时统计（共{n}个响应）：")
        print(f"  最小: {all_elapsed[0]:.1f} ms")
        print(f"  最大: {all_elapsed[-1]:.1f} ms")
        print(f"  平均: {sum(all_elapsed)/n:.1f} ms")
        print(f"  中位数(P50): {all_elapsed[n//2]:.1f} ms")
        print(f"  P90: {all_elapsed[int(n*0.9)]:.1f} ms")
        print(f"  P99: {all_elapsed[int(n*0.99)]:.1f} ms")
        print()

        # 按不同超时阈值统计
        for threshold in [500, 1000, 2000, 3000]:
            over = sum(1 for e in all_elapsed if e > threshold)
            print(f"  超过 {threshold}ms 的响应: {over} 个 ({over/n*100:.2f}%)")
    print()

    # 慢响应详情
    if http["slow_responses"]:
        print(f"[!] 慢响应（>500ms）共 {len(http['slow_responses'])} 个（前5条）：")
        for s in http["slow_responses"][:5]:
            print(f"    座位{s['player']}: {s['elapsed_ms']:.1f} ms")
    else:
        print("[OK] 未发现超过500ms的慢响应，超时风险很低")
    print()

    print("=" * 70)
    print("结论")
    print("=" * 70)
    if results["overridden"] == 0 and http["missing_responses"] == 0:
        print("座位0和座位2的所有动作都是你的AI返回的，未发现超时默认牌。")
    elif results["overridden"] > 0:
        print(f"发现 {results['overridden']} 手AI决策未被执行，需要进一步排查是否超时。")
    if http["missing_responses"] > 0:
        print(f"发现 {http['missing_responses']} 个请求没有响应，可能是AI崩溃或超时无返回。")


if __name__ == "__main__":
    main()
