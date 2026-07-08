import sys
from pathlib import Path

# 让这个脚本无论从哪里启动，都能找到 D:\MaJiang\mahjong_ai 这个包。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_ai.hand_split import split_hand
from mahjong_ai.search_tree import expand_discard_tree
from mahjong_ai.tiles import format_tiles, parse_tiles, tile_name


def print_combinations(hand_text: str) -> None:
    """打印任务一：手牌组合拆分结果。"""
    hand = parse_tiles(hand_text)
    print("手牌:", format_tiles(hand))
    print()
    print("前 8 个较优组合:")
    for index, combo in enumerate(split_hand(hand, limit=8), start=1):
        print(f"组合{index}:")
        print("  刻子集合:", [format_tiles(group) for group in combo.triplets])
        print("  顺子集合:", [format_tiles(group) for group in combo.sequences])
        print("  搭子集合:", [format_tiles(group) for group in combo.pairs + combo.taatsu])
        print("  向听数:", combo.shanten)
        print("  剩余牌:", format_tiles(combo.leftovers))
        print()


def print_discard_search(hand_text: str) -> None:
    """打印任务二：候选弃牌评分和推荐出牌。"""
    hand = parse_tiles(hand_text)
    result = expand_discard_tree(hand)
    print("候选弃牌评分:")
    for discard, decision in sorted(result.discard_scores.items(), key=lambda item: item[1].score, reverse=True):
        best_draws = ", ".join(
            f"{tile_name(child.draw)}({child.remaining}张, 向听{child.shanten})"
            for child in decision.children[:5]
        )
        print(
            f"  打 {tile_name(discard):>5}: score={decision.score:.2f}, "
            f"弃后向听={decision.shanten_after_discard}, "
            f"有效进张数={decision.effective_count}, "
            f"较优摸牌=[{best_draws}]"
        )
    print()
    print("推荐出牌:", tile_name(result.best_discard))
    print("扩展节点数:", result.node_count)


if __name__ == "__main__":
    # 如果命令行后面传入了手牌，就使用命令行参数。
    # 如果没有传入，就在控制台提示用户输入；直接回车则使用默认样例。
    default_sample = "1W 2W 3W 2T 3T 4T 5B 6B 7B EAST EAST EAST RED NORTH"
    sample = " ".join(sys.argv[1:])
    if not sample:
        sample = input(
            "请输入14张手牌，牌之间用空格隔开；直接回车使用默认样例：\n"
            "例如：1W 2W 3W 2T 3T 4T 5B 6B 7B EAST EAST EAST RED NORTH\n> "
        ).strip()
    sample = sample or default_sample
    print_combinations(sample)
    print_discard_search(sample)
