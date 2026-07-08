import unittest

from jiujiang_ai.api import get_action, round_end
from jiujiang_ai.rules import ACTION_DISCARD, ACTION_GANG, ACTION_HU, ACTION_PASS, ACTION_PENG, ACTION_TING
from jiujiang_ai.tiles import HONGZHONG


class JiujiangApiTests(unittest.TestCase):
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
