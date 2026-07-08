import unittest

from mahjong_ai.search_tree import choose_discard, expand_discard_tree
from mahjong_ai.tiles import parse_tiles


class SearchTreeTests(unittest.TestCase):
    def test_expand_discard_tree_scores_every_unique_discard(self):
        # 每一种不同的手牌都应该生成一个候选弃牌节点。
        hand = parse_tiles("1W 2W 3W 2T 3T 4T 5B 6B 7B EAST EAST EAST RED RED")

        result = expand_discard_tree(hand)

        self.assertEqual(set(result.discard_scores), set(hand))
        self.assertGreater(result.node_count, len(set(hand)))
        self.assertEqual(result.best_discard, choose_discard(hand).discard)

    def test_choose_discard_prefers_isolated_tile_over_complete_sets(self):
        # 已经成型的顺子/刻子不应轻易拆掉，孤张通常更适合先打。
        hand = parse_tiles("1W 2W 3W 2T 3T 4T 5B 6B 7B EAST EAST EAST RED NORTH")

        decision = choose_discard(hand)

        self.assertIn(decision.discard, parse_tiles("RED NORTH"))
        self.assertGreaterEqual(decision.score, 0)
        self.assertTrue(decision.children)


if __name__ == "__main__":
    unittest.main()
