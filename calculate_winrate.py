"""
计算用户AI的准确胜率
"""

import json

def calculate_winrate(file_path: str, start_line: int = 704):
    """计算从指定行开始的胜率"""

    # 用户AI: 座位0和座位2
    user_seats = [0, 2]

    # 统计数据
    total_rounds = 0
    user_wins = 0
    seat_stats = {0: {"rounds": 0, "wins": 0}, 2: {"rounds": 0, "wins": 0}}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines[start_line:], start=start_line):
        try:
            entry = json.loads(line.strip())
            data = entry.get('data', {})

            if 'winner' not in data:
                continue

            winner = data.get('winner')
            if winner is None:
                continue

            # 统计这一局
            total_rounds += 1

            # 检查用户AI是否获胜
            if winner in user_seats:
                user_wins += 1
                seat_stats[winner]['wins'] += 1

            # 统计各座位参与局数
            for seat in user_seats:
                seat_stats[seat]['rounds'] += 1

        except (json.JSONDecodeError, KeyError):
            continue

    # 计算胜率
    overall_winrate = (user_wins / total_rounds * 100) if total_rounds > 0 else 0

    print("=" * 60)
    print("用户AI胜率统计（第704行以后）")
    print("=" * 60)
    print()
    print(f"总对局数: {total_rounds}")
    print(f"用户AI获胜局数: {user_wins}")
    print(f"用户AI总胜率: {overall_winrate:.2f}%")
    print()
    print("分座位统计:")
    print(f"  座位0: {seat_stats[0]['wins']}/{seat_stats[0]['rounds']} = {seat_stats[0]['wins']/seat_stats[0]['rounds']*100:.2f}%")
    print(f"  座位2: {seat_stats[2]['wins']}/{seat_stats[2]['rounds']} = {seat_stats[2]['wins']/seat_stats[2]['rounds']*100:.2f}%")
    print()

    # 理论期望
    print("参考基准:")
    print(f"  四人麻将理论期望胜率: 25.00%")
    print(f"  两个座位理论期望胜率: 50.00%")
    print()

    if overall_winrate > 25:
        print(f"✓ 用户AI胜率高于单座位期望 ({overall_winrate:.2f}% > 25%)")
    else:
        print(f"✗ 用户AI胜率低于单座位期望 ({overall_winrate:.2f}% < 25%)")

    print("=" * 60)

    return overall_winrate, total_rounds, user_wins

if __name__ == "__main__":
    calculate_winrate("logs/jiujiang_round_end.jsonl", start_line=704)
