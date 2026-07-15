"""
分析九江麻将对局日志
座位0和座位2：用户开发的AI
座位1和座位3：学长开发的AI
"""

import json
from collections import defaultdict
from typing import Dict, List, Any

def analyze_match_log(file_path: str, start_line: int = 704):
    """分析对局日志"""

    # 统计数据
    player_stats = {
        0: {"wins": 0, "losses": 0, "zimo": 0, "dianpao_win": 0, "dianpao_loss": 0, "total_score": 0, "rounds": 0},
        1: {"wins": 0, "losses": 0, "zimo": 0, "dianpao_win": 0, "dianpao_loss": 0, "total_score": 0, "rounds": 0},
        2: {"wins": 0, "losses": 0, "zimo": 0, "dianpao_win": 0, "dianpao_loss": 0, "total_score": 0, "rounds": 0},
        3: {"wins": 0, "losses": 0, "zimo": 0, "dianpao_win": 0, "dianpao_loss": 0, "total_score": 0, "rounds": 0},
    }

    # 对局详情列表
    rounds_detail = []

    # 读取日志文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 从指定行开始分析
    for i, line in enumerate(lines[start_line:], start=start_line):
        try:
            entry = json.loads(line.strip())
            data = entry.get('data', {})

            # 跳过没有winner字段的记录
            if 'winner' not in data:
                continue

            winner = data.get('winner')
            win_type = data.get('win_type')
            dianpao_player = data.get('dianpao_player')
            scores = data.get('scores', [])

            # 跳过无效数据
            if winner is None:
                continue

            # 记录每局详情
            round_info = {
                'line': i + 1,
                'winner': winner,
                'win_type': win_type,
                'dianpao_player': dianpao_player,
                'scores': scores
            }
            rounds_detail.append(round_info)

            # 更新统计
            for player_id in range(4):
                player_stats[player_id]['rounds'] += 1

                if len(scores) == 4:
                    player_stats[player_id]['total_score'] += scores[player_id]

                    if scores[player_id] > 0:
                        player_stats[player_id]['wins'] += 1
                    elif scores[player_id] < 0:
                        player_stats[player_id]['losses'] += 1

            # 更新胜者统计
            if win_type == 'zimo':
                player_stats[winner]['zimo'] += 1
            elif win_type == 'dianpao':
                player_stats[winner]['dianpao_win'] += 1
                if dianpao_player is not None:
                    player_stats[dianpao_player]['dianpao_loss'] += 1

        except (json.JSONDecodeError, KeyError) as e:
            continue

    return player_stats, rounds_detail


def print_analysis(stats: Dict, rounds: List[Any]):
    """打印分析结果"""

    print("=" * 80)
    print("九江麻将对局分析报告 (第704行以后)")
    print("=" * 80)
    print()

    # 队伍划分
    team_user = [0, 2]  # 用户开发的AI
    team_senior = [1, 3]  # 学长开发的AI

    print("【队伍划分】")
    print(f"  用户AI: 座位0, 座位2")
    print(f"  学长AI: 座位1, 座位3")
    print()

    # 总体统计
    print("【总体统计】")
    print("-" * 80)
    for player_id in range(4):
        s = stats[player_id]
        team = "用户AI" if player_id in team_user else "学长AI"
        win_rate = (s['wins'] / s['rounds'] * 100) if s['rounds'] > 0 else 0
        avg_score = s['total_score'] / s['rounds'] if s['rounds'] > 0 else 0

        print(f"座位{player_id} ({team}):")
        print(f"  对局数: {s['rounds']}")
        print(f"  胜局数: {s['wins']} ({win_rate:.1f}%)")
        print(f"  自摸: {s['zimo']} | 点炮胡: {s['dianpao_win']} | 点炮: {s['dianpao_loss']}")
        print(f"  总分: {s['total_score']:.1f} | 平均分: {avg_score:.2f}")
        print()

    # 队伍对比
    print("【队伍对比】")
    print("-" * 80)

    user_total_score = sum(stats[p]['total_score'] for p in team_user)
    senior_total_score = sum(stats[p]['total_score'] for p in team_senior)
    user_total_wins = sum(stats[p]['wins'] for p in team_user)
    senior_total_wins = sum(stats[p]['wins'] for p in team_senior)
    user_total_rounds = sum(stats[p]['rounds'] for p in team_user)
    senior_total_rounds = sum(stats[p]['rounds'] for p in team_senior)

    print(f"用户AI团队:")
    print(f"  总分: {user_total_score:.1f}")
    print(f"  总胜局: {user_total_wins} / {user_total_rounds} ({user_total_wins/user_total_rounds*100:.1f}%)")
    print(f"  平均每局得分: {user_total_score/user_total_rounds:.2f}")
    print()

    print(f"学长AI团队:")
    print(f"  总分: {senior_total_score:.1f}")
    print(f"  总胜局: {senior_total_wins} / {senior_total_rounds} ({senior_total_wins/senior_total_rounds*100:.1f}%)")
    print(f"  平均每局得分: {senior_total_score/senior_total_rounds:.2f}")
    print()

    # 关键问题分析
    print("【关键问题分析】")
    print("-" * 80)

    # 1. 点炮率分析
    print("1. 防守问题 - 点炮分析:")
    for player_id in range(4):
        s = stats[player_id]
        team = "用户AI" if player_id in team_user else "学长AI"
        dianpao_rate = (s['dianpao_loss'] / s['rounds'] * 100) if s['rounds'] > 0 else 0
        print(f"  座位{player_id} ({team}): 点炮 {s['dianpao_loss']} 次, 点炮率 {dianpao_rate:.1f}%")
    print()

    # 2. 自摸vs点炮胡分析
    print("2. 进攻效率 - 胡牌方式分析:")
    for player_id in range(4):
        s = stats[player_id]
        team = "用户AI" if player_id in team_user else "学长AI"
        total_hu = s['zimo'] + s['dianpao_win']
        zimo_ratio = (s['zimo'] / total_hu * 100) if total_hu > 0 else 0
        print(f"  座位{player_id} ({team}): 总胡牌 {total_hu} 次 (自摸 {s['zimo']}, 点炮胡 {s['dianpao_win']}), 自摸率 {zimo_ratio:.1f}%")
    print()

    # 3. 座位2表现异常分析
    print("3. 座位2 (用户AI) 表现突出:")
    s2 = stats[2]
    print(f"  座位2胜率: {s2['wins']/s2['rounds']*100:.1f}%")
    print(f"  座位0胜率: {stats[0]['wins']/stats[0]['rounds']*100:.1f}%")
    print(f"  座位2总分: {s2['total_score']:.1f}")
    print(f"  座位0总分: {stats[0]['total_score']:.1f}")
    print(f"  → 座位2显著优于座位0，可能的原因:")
    print(f"    - 座位位置优势?")
    print(f"    - 随机性?")
    print(f"    - 算法在不同座位表现不一致?")
    print()

    # 优化建议
    print("【优化建议】")
    print("-" * 80)

    # 计算用户AI的弱点
    user_dianpao_total = sum(stats[p]['dianpao_loss'] for p in team_user)
    user_dianpao_rate = user_dianpao_total / user_total_rounds * 100
    senior_dianpao_total = sum(stats[p]['dianpao_loss'] for p in team_senior)
    senior_dianpao_rate = senior_dianpao_total / senior_total_rounds * 100

    print("1. 防守优化 (危险牌判断):")
    print(f"   - 用户AI点炮率: {user_dianpao_rate:.1f}%")
    print(f"   - 学长AI点炮率: {senior_dianpao_rate:.1f}%")
    if user_dianpao_rate > senior_dianpao_rate:
        print(f"   [!] 用户AI点炮率较高，需要加强:")
        print(f"     - 增强危险牌识别算法")
        print(f"     - 优化弃牌安全性评估")
        print(f"     - 当对手听牌时，调整出牌策略，优先出安全牌")
    print()

    user_zimo_total = sum(stats[p]['zimo'] for p in team_user)
    senior_zimo_total = sum(stats[p]['zimo'] for p in team_senior)
    user_total_hu = sum(stats[p]['zimo'] + stats[p]['dianpao_win'] for p in team_user)
    senior_total_hu = sum(stats[p]['zimo'] + stats[p]['dianpao_win'] for p in team_senior)
    user_zimo_rate = (user_zimo_total / user_total_hu * 100) if user_total_hu > 0 else 0
    senior_zimo_rate = (senior_zimo_total / senior_total_hu * 100) if senior_total_hu > 0 else 0

    print("2. 进攻优化 (速度与效率):")
    print(f"   - 用户AI自摸率: {user_zimo_rate:.1f}% ({user_zimo_total}/{user_total_hu})")
    print(f"   - 学长AI自摸率: {senior_zimo_rate:.1f}% ({senior_zimo_total}/{senior_total_hu})")
    if user_zimo_rate > senior_zimo_rate:
        print(f"   [+] 用户AI自摸率较高，说明听牌速度较快")
    else:
        print(f"   [!] 用户AI自摸率较低，需要优化:")
        print(f"     - 优化手牌拆分策略，追求更快听牌")
        print(f"     - 提升有效进张数量的评估权重")
    print()

    print("3. 座位差异分析:")
    print(f"   - 座位0平均分: {stats[0]['total_score']/stats[0]['rounds']:.2f}")
    print(f"   - 座位2平均分: {stats[2]['total_score']/stats[2]['rounds']:.2f}")
    if abs(stats[0]['total_score']/stats[0]['rounds'] - stats[2]['total_score']/stats[2]['rounds']) > 0.5:
        print(f"   [!] 两个座位表现差异较大，建议:")
        print(f"     - 检查代码是否对座位位置有依赖")
        print(f"     - 增加更多对局测试，排除随机因素")
    print()

    print("4. 具体优化方向:")
    print(f"   a) 向听数评估:")
    print(f"      - 当前可能过于保守，可以更激进地降低向听数")
    print(f"      - 在1向听或听牌阶段，优化有效进张的计算")
    print()
    print(f"   b) 弃牌策略:")
    print(f"      - 实现基于对手出牌历史的危险牌判断")
    print(f"      - 根据对手的碰牌、杠牌信息推断危险牌")
    print(f"      - 在他人听牌时，优先弃安全牌（已出现过的牌）")
    print()
    print(f"   c) 博弈树评估权重调整:")
    print(f"      - 提高'有效进张数量'的权重")
    print(f"      - 降低'孤张数量'的惩罚权重")
    print(f"      - 考虑加入'牌型价值'评估（如对倒、两面听优于边张、嵌张）")
    print()

    # 最近20局趋势
    print("【最近20局趋势】")
    print("-" * 80)
    recent_rounds = rounds[-20:] if len(rounds) > 20 else rounds
    recent_stats = defaultdict(lambda: {"wins": 0, "score": 0})

    for r in recent_rounds:
        if len(r['scores']) == 4:
            for pid in range(4):
                if r['scores'][pid] > 0:
                    recent_stats[pid]['wins'] += 1
                recent_stats[pid]['score'] += r['scores'][pid]

    for pid in range(4):
        team = "用户AI" if pid in team_user else "学长AI"
        print(f"座位{pid} ({team}): 胜{recent_stats[pid]['wins']}局, 总分{recent_stats[pid]['score']:.1f}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    log_file = "logs/jiujiang_round_end.jsonl"
    stats, rounds = analyze_match_log(log_file, start_line=704)
    print_analysis(stats, rounds)
