import unittest
from unittest.mock import patch

import jiujiang_ai.evaluator as evaluator_module
from jiujiang_ai.evaluator import choose_discard, score_discards
from jiujiang_ai.tiles import HONGZHONG


class JiujiangEvaluatorTests(unittest.TestCase):
    def test_prefers_isolated_tile_over_breaking_meld(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        candidates = [[0x01], [0x08]]

        decision = choose_discard(hand, candidates)

        self.assertEqual(decision.discard, 0x08)

    def test_scores_only_provided_candidates(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        scores = score_discards(hand, [[0x01], [0x08]])

        self.assertEqual(set(scores), {0x01, 0x08})

    def test_duplicate_tile_candidates_are_evaluated_once(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]

        with patch.object(evaluator_module, "analyze_hand", wraps=evaluator_module.analyze_hand) as analyze_mock:
            score_discards(hand, [[0x08]])
            single_candidate_calls = analyze_mock.call_count
            analyze_mock.reset_mock()
            scores = score_discards(hand, [[0x08], [0x08]])

        self.assertEqual(set(scores), {0x08})
        self.assertEqual(analyze_mock.call_count, single_candidate_calls)

    def test_expired_deadline_uses_fast_legal_fallback(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        candidates = [[0x01], [0x08]]

        with patch.object(
            evaluator_module,
            "choose_fast_discard",
            wraps=evaluator_module.choose_fast_discard,
        ) as fast_mock:
            decision = choose_discard(hand, candidates, deadline=0.0)

        self.assertIn([decision.discard], candidates)
        fast_mock.assert_called_once()

    def test_prefers_better_shanten_when_obvious(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x18]
        decision = choose_discard(hand, [[0x09], [0x18]])

        self.assertIn(decision.discard, {0x09, 0x18})

    def test_effective_count_uses_real_winning_tiles_when_ting(self):
        # 打掉 8 条后真实听 9 万和红中，评估器应记录具体胡牌张数。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x18, 0x21, 0x22, 0x23, 0x09]

        scores = score_discards(hand, [[0x18]])
        decision = scores[0x18]

        self.assertEqual(decision.winning_tiles[0x09], 3)
        self.assertEqual(decision.winning_tiles[HONGZHONG], 4)
        self.assertEqual(decision.effective_count, 7)

    def test_choose_discard_prefers_real_ting_over_shanten_approximation(self):
        # 三张红中在手时，打普通牌可听第四张红中直接胡；不能被向听近似分诱导去打红中。
        hand = [HONGZHONG, HONGZHONG, 0x02, 0x03, 0x03, 0x13, 0x29, 0x06, 0x26, 0x28, 0x24, HONGZHONG, 0x11, 0x09]
        candidates = [[tile] for tile in sorted(set(hand))]

        decision = choose_discard(hand, candidates)

        self.assertNotEqual(decision.discard, HONGZHONG)
        self.assertTrue(decision.winning_tiles)


    def test_choose_discard_prefers_visible_safe_tile_when_offense_is_equal(self):
        # 两个候选弃牌进攻价值相同时，优先打场上已经出现过的相对安全牌。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x18]
        candidates = [[0x09], [0x18]]

        decision = choose_discard(hand, candidates, visible_discards={0x18: 2})

        self.assertEqual(decision.discard, 0x18)


if __name__ == "__main__":
    unittest.main()
