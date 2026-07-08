import unittest

from jiujiang_ai.api import _legal_discard_candidates, get_action, round_end
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
        result = round_end({"winner": 0})

        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
