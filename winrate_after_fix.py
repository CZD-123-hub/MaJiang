# -*- coding: utf-8 -*-
"""
基于优化生效点（2026-07-15 16:40）之后的 round_end 记录，重新计算真实胜率。
处理要点：
- 只取 timestamp >= CUTOFF 的记录
- 去重（文件里存在完全相同的相邻重复行）
- 兼容不同格式：winner / winners(多赢) / win_player_position
"""
import sys, io, json
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CUTOFF = '2026-07-15T16:40:00'
USER_SEATS = [0, 2]


def load_rounds(path):
    rows = []
    seen = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get('timestamp', '') < CUTOFF:
                continue
            # 去重：用 timestamp+data 的字符串签名
            sig = e.get('timestamp', '') + json.dumps(e.get('data', {}), sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            rows.append(e)
    return rows


def main():
    rows = load_rounds('logs/jiujiang_round_end.jsonl')

    total = 0
    draws = 0
    wins_by_seat = defaultdict(int)
    zimo_by_seat = defaultdict(int)
    dianpao_win_by_seat = defaultdict(int)
    dianpao_loss_by_seat = defaultdict(int)
    score_by_seat = defaultdict(float)
    has_score = 0

    for e in rows:
        d = e.get('data', {})

        # 流局
        if d.get('huangzhuang') or d.get('is_draw'):
            draws += 1
            total += 1
            continue

        # 赢家：支持单赢/多赢/平台字段
        winners = []
        if d.get('winner') is not None:
            winners = [d['winner']]
        elif isinstance(d.get('winners'), list):
            winners = d['winners']
        elif d.get('win_player_position') is not None:
            wp = d['win_player_position']
            if isinstance(wp, int) and 0 <= wp <= 3:
                winners = [wp]

        if not winners:
            continue

        total += 1
        for w in winners:
            wins_by_seat[w] += 1

        win_type = d.get('win_type')
        dp = d.get('dianpao_player')
        if dp is None:
            dp = d.get('dp_player_position')
            # 平台自摸时 dp 等于赢家本人，需排除
            if dp in winners:
                dp = None

        if win_type == 'zimo':
            for w in winners:
                zimo_by_seat[w] += 1
        elif win_type == 'dianpao':
            for w in winners:
                dianpao_win_by_seat[w] += 1
            if isinstance(dp, int) and 0 <= dp <= 3:
                dianpao_loss_by_seat[dp] += 1

        # 分数
        scores = d.get('scores') or d.get('total_score')
        if isinstance(scores, list) and len(scores) == 4:
            has_score += 1
            for i in range(4):
                try:
                    score_by_seat[i] += float(scores[i])
                except (TypeError, ValueError):
                    pass

    print("=" * 66)
    print("优化后真实胜率（2026-07-15 16:40 之后）")
    print("=" * 66)
    print(f"有效局数: {total}  (其中流局: {draws})")
    print(f"含分数记录: {has_score} 局")
    print()

    decisive = total - draws
    print("分座位表现:")
    print("-" * 66)
    for s in range(4):
        team = "用户AI" if s in USER_SEATS else "学长AI"
        wr = wins_by_seat[s] / decisive * 100 if decisive else 0
        print(f"座位{s} ({team}): 胜{wins_by_seat[s]:3d}局 胜率{wr:5.1f}%  "
              f"自摸{zimo_by_seat[s]} 点炮胡{dianpao_win_by_seat[s]} 点炮{dianpao_loss_by_seat[s]}  "
              f"总分{score_by_seat[s]:+.0f}")
    print()

    user_wins = sum(wins_by_seat[s] for s in USER_SEATS)
    user_score = sum(score_by_seat[s] for s in USER_SEATS)
    senior_wins = sum(wins_by_seat[s] for s in [1, 3])
    senior_score = sum(score_by_seat[s] for s in [1, 3])

    print("团队对比:")
    print("-" * 66)
    print(f"用户AI(0+2): 胜{user_wins}局  胜率{user_wins/decisive*100:.1f}%  总分{user_score:+.0f}")
    print(f"学长AI(1+3): 胜{senior_wins}局  胜率{senior_wins/decisive*100:.1f}%  总分{senior_score:+.0f}")
    print()
    print(f"基准: 单座位期望胜率25%, 两座位期望胜率50%")
    print(f"用户AI实际胜率: {user_wins/decisive*100:.1f}%  ({'高于' if user_wins/decisive>0.5 else '低于'}50%基准)")


if __name__ == "__main__":
    main()
