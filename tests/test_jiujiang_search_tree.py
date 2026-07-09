import unittest

from jiujiang_ai.search_tree import choose_discard, expand_discard_tree
from jiujiang_ai.tiles import HONGZHONG


class JiujiangSearchTreeTests(unittest.TestCase):
    def test_expand_discard_tree_scores_every_candidate_discard(self):
        # 第一阶段搜索树需要对每个候选弃牌都建立一个显式根分支。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x18]
        candidates = [[0x09], [0x18]]

        result = expand_discard_tree(hand, candidates)

        self.assertEqual(set(result.discard_scores), {0x09, 0x18})
        self.assertGreater(result.node_count, len(candidates))
        self.assertEqual(result.best_discard, choose_discard(hand, candidates).discard)

    def test_choose_discard_prefers_isolated_tile_over_complete_structure(self):
        # 已成型的顺子不应轻易拆掉，第一阶段树搜索至少要能优先打孤张。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        candidates = [[0x01], [0x08]]

        decision = choose_discard(hand, candidates)

        self.assertEqual(decision.discard, 0x08)
        self.assertTrue(decision.children)

    def test_expand_discard_tree_respects_remaining_counts(self):
        # 搜索树展开摸牌子节点时，要尊重外部传入的剩余牌张数，而不是默认所有牌都可摸。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x18]
        candidates = [[0x18]]
        remaining_counts = {tile: 0 for tile in range(0x01, 0x2A)}
        remaining_counts[0x09] = 2
        remaining_counts[0x35] = 1

        result = expand_discard_tree(hand, candidates, remaining_counts=remaining_counts)
        decision = result.discard_scores[0x18]

        self.assertEqual({child.draw for child in decision.children}, {0x09, 0x35})

    def test_ting_branch_reports_expected_path_metrics(self):
        # 第二阶段需要把“这张弃牌之后的路径期望”显式算出来，方便后续贴近 v5 的路径收益。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x18, 0x21, 0x22, 0x23, 0x09]

        result = expand_discard_tree(hand, [[0x18]])
        decision = result.discard_scores[0x18]

        self.assertEqual(decision.hu_child_count, 2)
        self.assertEqual(decision.hu_total_remaining, 7)
        self.assertGreater(decision.expected_path_value, 0)
        hu_children = {child.draw: child for child in decision.children if child.is_hu}
        self.assertEqual(set(hu_children), {0x09, HONGZHONG})
        self.assertTrue(all(child.path_value > 0 for child in hu_children.values()))
        self.assertGreater(decision.expected_hu_value, 0)
        self.assertGreater(decision.expected_ting_value, 0)

    def test_expected_path_value_distinguishes_candidate_quality(self):
        # 显式路径期望应该能区分两个候选弃牌，而不只是给一个最终推荐结果。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        candidates = [[0x01], [0x08]]

        result = expand_discard_tree(hand, candidates)

        self.assertGreater(
            result.discard_scores[0x08].expected_path_value,
            result.discard_scores[0x01].expected_path_value,
        )

    def test_children_expose_path_component_values(self):
        # 子节点不仅要有总路径分，还要能拆成胡牌/进听/向听改良三类收益。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x18, 0x21, 0x22, 0x23, 0x09]

        result = expand_discard_tree(hand, [[0x18]])
        decision = result.discard_scores[0x18]
        hu_child = next(child for child in decision.children if child.draw == 0x09)
        non_hu_child = next(child for child in decision.children if not child.is_hu)

        self.assertGreater(hu_child.hu_value, 0)
        self.assertEqual(hu_child.ting_value, 0)
        self.assertEqual(hu_child.improvement_value, 0)
        self.assertEqual(
            hu_child.path_value,
            hu_child.hu_value + hu_child.ting_value + hu_child.improvement_value + hu_child.follow_up_score_bonus,
        )

        self.assertEqual(non_hu_child.hu_value, 0)
        self.assertGreaterEqual(non_hu_child.ting_value, 0)
        self.assertGreaterEqual(non_hu_child.improvement_value, 0)
        self.assertEqual(
            non_hu_child.path_value,
            non_hu_child.hu_value
            + non_hu_child.ting_value
            + non_hu_child.improvement_value
            + non_hu_child.follow_up_score_bonus,
        )

    def test_improvement_path_value_is_positive_when_draw_allows_better_follow_up(self):
        # 有些摸牌不会立刻胡，也不会直接进听，但能让“摸后再打一张”的局面更接近成型。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]

        result = expand_discard_tree(hand, [[0x01]])
        decision = result.discard_scores[0x01]

        self.assertGreater(decision.expected_improvement_value, 0)

    def test_draw_node_exposes_best_follow_up_state(self):
        # 第二层雏形里，每个摸牌节点都应该给出“摸后最佳弃牌”的后继状态摘要。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]

        result = expand_discard_tree(hand, [[0x01]])
        decision = result.discard_scores[0x01]
        follow_up_child = next(child for child in decision.children if child.improvement_value > 0)

        self.assertIsNotNone(follow_up_child.best_follow_up_discard)
        self.assertGreaterEqual(follow_up_child.follow_up_shanten, 0)
        self.assertGreaterEqual(follow_up_child.follow_up_effective_count, 0)
        self.assertIsInstance(follow_up_child.follow_up_winning_tiles, dict)

    def test_decision_aggregates_follow_up_quality(self):
        # 候选弃牌结果需要汇总第二层后继质量，方便后续接更完整的树评分。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]

        result = expand_discard_tree(hand, [[0x01], [0x08]])

        self.assertGreaterEqual(result.discard_scores[0x01].expected_follow_up_effective_count, 0)
        self.assertGreaterEqual(result.discard_scores[0x08].expected_follow_up_effective_count, 0)
        self.assertLess(
            result.discard_scores[0x08].expected_follow_up_shanten,
            result.discard_scores[0x01].expected_follow_up_shanten,
        )

    def test_draw_node_exposes_controlled_follow_up_nodes(self):
        # 两层雏形不应只返回摘要，还应显式给出少量可比较的“摸后再弃”节点。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]

        result = expand_discard_tree(hand, [[0x01]])
        decision = result.discard_scores[0x01]
        follow_up_child = next(child for child in decision.children if child.improvement_value > 0)

        self.assertTrue(follow_up_child.follow_up_nodes)
        self.assertLessEqual(len(follow_up_child.follow_up_nodes), 3)
        self.assertEqual(follow_up_child.best_follow_up_discard, follow_up_child.follow_up_nodes[0].discard)

    def test_follow_up_nodes_are_sorted_by_quality(self):
        # 后继弃牌节点应该按质量排序，最优后继放在第一个，便于后续继续扩树。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]

        result = expand_discard_tree(hand, [[0x01]])
        decision = result.discard_scores[0x01]
        follow_up_child = next(child for child in decision.children if child.improvement_value > 0)

        scores = [node.score for node in follow_up_child.follow_up_nodes]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_draw_node_path_value_includes_follow_up_score_bonus(self):
        # 当后继弃牌节点质量更高时，当前摸牌路径价值也应该被抬高。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]

        result = expand_discard_tree(hand, [[0x01]])
        decision = result.discard_scores[0x01]
        follow_up_child = max(
            (child for child in decision.children if not child.is_hu),
            key=lambda child: child.follow_up_score_bonus,
        )
        plain_child = min(
            (child for child in decision.children if not child.is_hu),
            key=lambda child: child.follow_up_score_bonus,
        )

        self.assertGreater(follow_up_child.path_value, plain_child.path_value)

    def test_decision_score_reflects_follow_up_score_bonus(self):
        # 当前弃牌总分应该体现第二层后继节点的累计收益。
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]

        result = expand_discard_tree(hand, [[0x01], [0x08]])

        self.assertGreater(
            result.discard_scores[0x08].score,
            result.discard_scores[0x01].score,
        )


if __name__ == "__main__":
    unittest.main()
