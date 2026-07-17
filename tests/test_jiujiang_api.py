import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import jiujiang_ai.api as api_module
from jiujiang_ai.api import _legal_discard_candidates, get_action, round_end
from jiujiang_ai.decision_log import load_decision_logs
from jiujiang_ai.rules import (
    ACTION_ANGANG,
    ACTION_BUGANG,
    ACTION_DISCARD,
    ACTION_GANG,
    ACTION_HU,
    ACTION_PASS,
    ACTION_PENG,
    ACTION_TING,
)
from jiujiang_ai.tiles import HONGZHONG


class JiujiangApiTests(unittest.TestCase):
    def test_filters_hongzhong_discard_when_run_hongzhong_disabled(self):
        hand = [HONGZHONG, 0x01]
        data = {"room_options": {"run_hongzhong_double": False}}

        candidates = _legal_discard_candidates([[HONGZHONG], [0x01]], hand, data)

        self.assertEqual(candidates, [[0x01]])

    def test_keeps_hongzhong_discard_when_run_hongzhong_enabled(self):
        hand = [HONGZHONG, 0x01]
        data = {"room_options": {"run_hongzhong_double": True}}

        candidates = _legal_discard_candidates([[HONGZHONG], [0x01]], hand, data)

        self.assertEqual(candidates, [[HONGZHONG], [0x01]])

    def test_only_hongzhong_discard_candidate_returns_pass_when_run_hongzhong_disabled(self):
        data = {
            "action_cards": {"7": [[HONGZHONG]]},
            "player_hand_cards": [[HONGZHONG, 0x01, 0x04, 0x07, 0x11, 0x14, 0x17, 0x21, 0x24, 0x27, 0x02, 0x05, 0x08, 0x29], [], [], []],
            "acting_do_player_position": 0,
            "room_options": {"run_hongzhong_double": False},
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_PASS)
        self.assertEqual(action_card, [])

    def test_hu_has_priority(self):
        data = {
            "action_cards": {"4": [], "7": [[0x01]]},
            "player_hand_cards": [[0x01] * 2, [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_HU)
        self.assertEqual(action_card, [])

    def test_discards_from_candidates(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {"action_cards": {"7": [[0x01], [0x08]]}, "player_hand_cards": [hand, [], [], []], "acting_do_player_position": 0}

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x08])

    def test_discard_only_request_skips_peng_and_gang_value_work(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {
            "action_cards": {"7": [[0x01], [0x08]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        with patch.object(api_module, "hand_value", side_effect=AssertionError("unexpected hand_value call")):
            action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x08])

    def test_fourteen_tile_discard_skips_waiting_hand_probe(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {
            "action_cards": {"7": [[0x01], [0x08]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        with patch.object(api_module, "winning_tile_counts") as winning_mock:
            action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x08])
        winning_mock.assert_not_called()

    def test_busy_discard_evaluation_uses_fast_fallback(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {
            "action_cards": {"7": [[0x01], [0x08]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }
        fake_decision = type("FastDecision", (), {"discard": 0x01})()

        api_module._DISCARD_EVALUATION_SLOTS.acquire()
        api_module._DISCARD_EVALUATION_SLOTS.acquire()
        try:
            with patch.object(api_module, "choose_fast_discard", return_value=fake_decision) as fast_mock, patch.object(
                api_module, "_choose_discard_decision"
            ) as full_mock:
                action_type, action_card = get_action(data)
        finally:
            api_module._DISCARD_EVALUATION_SLOTS.release()
            api_module._DISCARD_EVALUATION_SLOTS.release()

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x01])
        fast_mock.assert_called_once()
        full_mock.assert_not_called()

    def test_discards_visible_safe_tile_from_action_history_when_offense_is_equal(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x18]
        data = {
            "action_cards": {"7": [[0x09], [0x18]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
            "played_cards": [[], [], [], []],
            "action_seq": [[1, ACTION_DISCARD, 0x18], [2, ACTION_DISCARD, 0x18]],
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x18])

    def test_rejects_chi_and_hongzhong_peng_then_passes(self):
        data = {"action_cards": {"1": [[0x01, 0x02, 0x03]], "2": [[HONGZHONG, HONGZHONG, HONGZHONG]], "0": []}}

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_PASS)
        self.assertEqual(action_card, [])

    def test_selects_non_hongzhong_gang(self):
        data = {"action_cards": {"3": [[0x02, 0x02, 0x02, 0x02]]}}

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_GANG)
        self.assertEqual(action_card, [0x02, 0x02, 0x02, 0x02])

    def test_returns_angang_action_type(self):
        hand = [0x01, 0x01, 0x01, 0x01, 0x05, 0x06, 0x06, 0x14, 0x21, 0x21, 0x24, 0x25, 0x27, 0x28]
        data = {
            "action_cards": {"5": [[0x01, 0x01, 0x01, 0x01]], "0": []},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_ANGANG)
        self.assertEqual(action_card, [0x01, 0x01, 0x01, 0x01])

    def test_returns_bugang_action_type(self):
        hand = [0x07, 0x11, 0x12, 0x13, 0x17, 0x18, 0x19, 0x21, 0x21, 0x24, 0x25, 0x26, 0x27, 0x21]
        data = {
            "action_cards": {"6": [[0x21, 0x21, 0x21, 0x21]], "0": []},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
            "player_peng_cards": [[[0x21, 0x21, 0x21]], [], [], []],
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_BUGANG)
        self.assertEqual(action_card, [0x21, 0x21, 0x21, 0x21])

    def test_gang_immediate_score_matches_settlement_rule(self):
        self.assertEqual(api_module._gang_immediate_gain(ACTION_GANG), 3.0)
        self.assertEqual(api_module._gang_immediate_gain(ACTION_BUGANG), 3.0)
        self.assertEqual(api_module._gang_immediate_gain(ACTION_ANGANG), 6.0)

    def test_skips_angang_when_gang_would_make_hand_worse(self):
        # 这副牌把四张 6 筒直接暗杠后，手里有效结构明显减少，第一版收益判断应选择过牌。
        hand = [0x06, 0x07, 0x11, 0x14, 0x17, 0x25, 0x26, 0x26, 0x26, 0x26, 0x28, 0x29, HONGZHONG, HONGZHONG]
        data = {
            "action_cards": {"5": [[0x26, 0x26, 0x26, 0x26]], "0": []},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_PASS)
        self.assertEqual(action_card, [])

    def test_keeps_angang_when_gang_does_not_reduce_progress(self):
        # 这副牌暗杠 1 万后整体向听不变，当前第一版策略仍然允许杠牌。
        hand = [0x01, 0x01, 0x01, 0x01, 0x05, 0x06, 0x06, 0x14, 0x21, 0x21, 0x24, 0x25, 0x27, 0x28]
        data = {
            "action_cards": {"5": [[0x01, 0x01, 0x01, 0x01]], "0": []},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_ANGANG)
        self.assertEqual(action_card, [0x01, 0x01, 0x01, 0x01])

    def test_skips_gang_when_current_hand_is_already_ting(self):
        # 当前 13 张已经听 9 万/红中时，不为了杠牌破坏听牌结构。
        hand = [0x01, 0x02, 0x03, 0x02, 0x02, 0x02, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09]
        data = {
            "action_cards": {"3": [[0x02, 0x02, 0x02, 0x02]], "0": []},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_PASS)
        self.assertEqual(action_card, [])

    def test_skips_peng_when_current_hand_is_already_ting(self):
        # 已经听牌时，碰牌会改变手牌结构，第一版策略选择保守过牌。
        hand = [0x01, 0x02, 0x03, 0x02, 0x02, 0x04, 0x05, 0x06, 0x21, 0x22, 0x23, 0x09, 0x09]
        data = {
            "action_cards": {"2": [[0x02, 0x02, 0x02]], "0": []},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_PASS)
        self.assertEqual(action_card, [])

    def test_skips_peng_after_passing_same_tile_before_next_discard(self):
        # 过碰过圈：玩家 0 对玩家 1 打出的 6 万选择过，自己下次出牌前不能再碰 6 万。
        hand = [0x15, 0x12, 0x29, 0x06, 0x26, 0x02, 0x13, 0x12, 0x23, 0x17, 0x28, 0x06, 0x01]
        data = {
            "action_cards": {"2": [[0x06, 0x06, 0x06]], "0": []},
            "action_seq": [[1, ACTION_DISCARD, 0x06], [0, ACTION_PASS]],
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_PASS)
        self.assertEqual(action_card, [])

    def test_peng_pass_restriction_resets_after_own_discard(self):
        # 玩家 0 后续已经出过牌，之前对 6 万的过碰限制重置，可以重新判断是否碰。
        hand = [0x15, 0x12, 0x29, 0x06, 0x26, 0x02, 0x13, 0x12, 0x23, 0x17, 0x28, 0x06, 0x01]
        data = {
            "action_cards": {"2": [[0x06, 0x06, 0x06]], "0": []},
            "action_seq": [[1, ACTION_DISCARD, 0x06], [0, ACTION_PASS], [0, ACTION_DISCARD, 0x01]],
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_PENG)
        self.assertEqual(action_card, [0x06, 0x06, 0x06])

    def test_peng_route_rejects_best_forced_discard_that_worsens_shanten(self):
        hand = [0x15, 0x12, 0x29, 0x06, 0x26, 0x02, 0x13, 0x12, 0x23, 0x17, 0x28, 0x06, 0x01]
        data = {"acting_do_player_position": 0}
        before_shanten = api_module.analyze_hand(hand).shanten
        with patch.object(api_module, "hand_value", return_value=10.0), patch.object(
            api_module, "_best_peng_post_discard_shanten", return_value=before_shanten + 1
        ):
            result = api_module._best_peng({ACTION_PENG: [[0x06, 0x06, 0x06]]}, hand, data, api_module.HuOptions())

        self.assertIsNone(result)

    def test_peng_route_accepts_strictly_better_non_regressing_route(self):
        hand = [0x15, 0x12, 0x29, 0x06, 0x26, 0x02, 0x13, 0x12, 0x23, 0x17, 0x28, 0x06, 0x01]
        data = {"acting_do_player_position": 0}
        before_shanten = api_module.analyze_hand(hand).shanten
        with patch.object(api_module, "hand_value", side_effect=[10.0, 10.1]), patch.object(
            api_module, "_best_peng_post_discard_shanten", return_value=before_shanten
        ):
            result = api_module._best_peng({ACTION_PENG: [[0x06, 0x06, 0x06]]}, hand, data, api_module.HuOptions())

        self.assertEqual(result, [0x06, 0x06, 0x06])

    def test_ting_is_selected_when_no_higher_priority_action_exists(self):
        data = {"action_cards": {"0": [], "8": []}}

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_TING)
        self.assertEqual(action_card, [])

    def test_four_hongzhong_hand_can_hu_without_prompted_hu_action(self):
        data = {
            "action_cards": {"7": [[HONGZHONG], [0x01]]},
            "player_hand_cards": [[HONGZHONG, HONGZHONG, HONGZHONG, HONGZHONG, 0x01], [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_HU)
        self.assertEqual(action_card, [])

    def test_round_end_acknowledges_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "jiujiang_ai.stats.DEFAULT_ROUND_LOG_PATH",
            Path(temp_dir) / "round_end.jsonl",
        ):
            result = round_end(
                {
                    "winner": 0,
                    "win_type": "zimo",
                    "player_hand_cards": [[HONGZHONG, 0x11, 0x12], [], [], []],
                    "room_options": {"zama_count": 1},
                    "zama_cards": [0x01],
                }
            )

        self.assertEqual(result["status"], "ok")
        self.assertIn("settlement", result)
        self.assertEqual(result["settlement"]["score_by_player"], [9, -3, -3, -3])

    def test_discard_defaults_to_heuristic_evaluator_when_tree_search_disabled(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {
            "action_cards": {"7": [[0x01], [0x08]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
            "room_options": {"search_tree_enabled": False, "two_ply_search_enabled": False},
        }

        fake_tree_decision = type("TreeDecision", (), {"discard": 0x01})()

        with patch.object(api_module, "choose_discard", wraps=api_module.choose_discard) as heuristic_mock, patch.object(
            api_module, "choose_tree_discard", return_value=fake_tree_decision
        ) as tree_mock:
            action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x08])
        heuristic_mock.assert_called_once()
        tree_mock.assert_not_called()

    def test_discard_uses_tree_search_when_enabled_and_falls_back_on_failure(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {
            "action_cards": {"7": [[0x01], [0x08]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
            "room_options": {"search_tree_enabled": True},
        }

        fake_tree_decision = type("TreeDecision", (), {"discard": 0x01})()

        with patch.object(api_module, "choose_tree_discard", return_value=fake_tree_decision) as tree_mock, patch.object(
            api_module, "choose_discard", wraps=api_module.choose_discard
        ) as heuristic_mock:
            action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x01])
        tree_mock.assert_called_once()
        heuristic_mock.assert_not_called()

        with patch.object(api_module, "choose_tree_discard", side_effect=ValueError("tree failed")) as tree_mock, patch.object(
            api_module, "choose_discard", wraps=api_module.choose_discard
        ) as heuristic_mock:
            action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x08])
        tree_mock.assert_called_once()
        heuristic_mock.assert_called_once()

    def test_discard_uses_multi_route_engine_when_enabled(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {
            "action_cards": {"7": [[0x01], [0x08]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
            "room_options": {"multi_route_enabled": True},
        }
        fake_decision = type("MultiRouteDecision", (), {"discard": 0x01})()

        with patch.object(api_module, "choose_multi_route_discard", return_value=fake_decision) as multi_route_mock, patch.object(
            api_module, "choose_discard", wraps=api_module.choose_discard
        ) as heuristic_mock, patch.object(api_module, "choose_tree_discard") as tree_mock:
            action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x01])
        multi_route_mock.assert_called_once()
        heuristic_mock.assert_not_called()
        tree_mock.assert_not_called()

    def test_discard_uses_bounded_two_ply_search_when_enabled(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {
            "action_cards": {"7": [[0x01], [0x08]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
            "room_options": {"two_ply_search_enabled": True},
        }
        fake_decision = type("TwoPlyDecision", (), {"discard": 0x01})()

        with patch.object(api_module, "choose_two_ply_discard", return_value=fake_decision) as two_ply_mock, patch.object(
            api_module, "choose_discard"
        ) as heuristic_mock:
            action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x01])
        two_ply_mock.assert_called_once()
        heuristic_mock.assert_not_called()

    def test_discard_uses_multi_route_tree_when_enabled(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {
            "action_cards": {"7": [[0x01], [0x08]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
            "room_options": {"multi_route_tree_enabled": True},
        }
        fake_decision = type("TreeDecision", (), {"discard": 0x08})()

        with patch.object(api_module, "choose_tree_discard", return_value=fake_decision) as tree_mock, patch.object(
            api_module, "choose_multi_route_discard"
        ) as multi_route_mock:
            action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x08])
        tree_mock.assert_called_once()
        self.assertTrue(tree_mock.call_args.kwargs["use_multi_route"])
        self.assertIs(tree_mock.call_args.kwargs["decision_data"], data)
        multi_route_mock.assert_not_called()

    def test_discard_logs_summary_only_when_enabled(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "decisions.jsonl"
            data = {
                "room_id": 7,
                "action_cards": {"7": [[0x01], [0x08]]},
                "player_hand_cards": [hand, [], [], []],
                "acting_do_player_position": 0,
                "room_options": {"decision_log_enabled": True, "decision_log_path": str(log_path)},
            }

            action_type, action_card = get_action(data)
            records = load_decision_logs(log_path)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["action_card"], action_card)
        self.assertEqual(records[0]["context"]["room_id"], 7)


if __name__ == "__main__":
    unittest.main()
