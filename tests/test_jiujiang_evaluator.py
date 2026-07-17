import unittest
from unittest.mock import patch

import jiujiang_ai.evaluator as evaluator_module
from jiujiang_ai.evaluator import TwoPlyDiscardDecision, choose_discard, choose_two_ply_discard, score_discards
from jiujiang_ai.hu import HuOptions
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

    def test_qidui_option_contributes_real_waits_to_discard_scoring(self):
        # 六对加单张的 13 张牌，开启七对后应识别出补同牌即胡的真实听口。
        hand = [
            0x01, 0x01, 0x04, 0x04, 0x07, 0x07,
            0x11, 0x11, 0x14, 0x14, 0x17, 0x17,
            0x21, 0x29,
        ]

        disabled = score_discards(hand, [[0x29]])[0x29]
        enabled = score_discards(hand, [[0x29]], options=HuOptions(allow_qidui=True))[0x29]

        self.assertEqual(disabled.winning_tiles, {})
        self.assertEqual(enabled.winning_tiles[0x21], 3)
        # 同时红中也能与单张补成第七对，因此总听牌张数为同牌 3 张 + 红中 4 张。
        self.assertEqual(enabled.effective_count, 7)

    def test_run_hongzhong_multiplier_increases_candidate_win_value(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x18, 0x21, 0x22, 0x23, 0x09]

        normal = score_discards(hand, [[0x18]])[0x18]
        doubled = score_discards(hand, [[0x18]], win_multiplier_by_discard={0x18: 2})[0x18]

        self.assertGreater(doubled.score, normal.score)

    def test_two_ply_returns_a_legal_discard_and_exposes_path_summary(self):
        hand = [0x01, 0x02, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x07, 0x08, 0x19]
        candidates = [[tile] for tile in sorted(set(hand))]

        decision = choose_two_ply_discard(hand, candidates)

        self.assertIn(decision.discard, hand)
        self.assertIsInstance(decision, TwoPlyDiscardDecision)
        self.assertGreater(decision.explored_draw_types, 0)
        self.assertGreater(decision.expected_path_value, 0)

    def test_two_ply_keeps_first_ply_result_when_expansion_hits_deadline(self):
        hand = [0x01, 0x02, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x07, 0x08, 0x19]
        candidates = [[tile] for tile in sorted(set(hand))]
        baseline = choose_discard(hand, candidates)

        with patch.object(
            evaluator_module,
            "_two_ply_draw_nodes",
            side_effect=evaluator_module._DecisionDeadlineExceeded,
        ):
            decision = choose_two_ply_discard(hand, candidates)

        self.assertEqual(decision, baseline)


if __name__ == "__main__":
    unittest.main()
