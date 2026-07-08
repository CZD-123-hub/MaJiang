import unittest

from jiujiang_ai.api import get_action
from jiujiang_ai.hu import HuOptions, can_hu
from jiujiang_ai.rules import ACTION_HU
from jiujiang_ai.tiles import HONGZHONG


class JiujiangHuTests(unittest.TestCase):
    def test_standard_pinghu_hand_can_hu(self):
        # 标准平胡：四组面子 + 一对将。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x09]

        self.assertTrue(can_hu(hand))

    def test_thirteen_tiles_cannot_hu_without_four_hongzhong(self):
        # 普通胡牌必须是 3n+2 张，避免把听牌状态误判为已胡。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09]

        self.assertFalse(can_hu(hand))

    def test_hongzhong_can_complete_pair(self):
        # 红中可作为万能牌补成将牌。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, HONGZHONG]

        self.assertTrue(can_hu(hand))

    def test_hongzhong_can_complete_sequence(self):
        # 红中可作为万能牌补顺子里的缺张。
        hand = [0x01, 0x02, HONGZHONG, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x09]

        self.assertTrue(can_hu(hand))

    def test_four_hongzhong_can_hu_directly(self):
        # 九江红中规则：四红中可直接胡。
        self.assertTrue(can_hu([HONGZHONG, HONGZHONG, HONGZHONG, HONGZHONG]))

    def test_qidui_requires_room_option(self):
        # 七对是房间选项，默认不开放。
        hand = [0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x11, 0x11, 0x12, 0x12, 0x21, 0x21]

        self.assertFalse(can_hu(hand))
        self.assertTrue(can_hu(hand, HuOptions(allow_qidui=True)))

    def test_qidui_can_use_hongzhong_to_complete_pair(self):
        # 开启七对后，红中可以补一张孤张组成对子。
        hand = [0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x11, 0x11, 0x12, 0x12, 0x21, HONGZHONG]

        self.assertTrue(can_hu(hand, HuOptions(allow_qidui=True)))

    def test_non_winning_hand_returns_false(self):
        # 结构不够四面子一将，也无法七对时不能胡。
        hand = [0x01, 0x01, 0x02, 0x04, 0x06, 0x08, 0x11, 0x13, 0x15, 0x17, 0x21, 0x24, 0x27, 0x29]

        self.assertFalse(can_hu(hand, HuOptions(allow_qidui=True)))

    def test_get_action_uses_local_can_hu_fallback(self):
        # 如果外部 action_cards 没给胡，但本地完整胡牌判断成立，则仍然选择胡。
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x09]
        data = {
            "action_cards": {"7": [[0x09]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_HU)
        self.assertEqual(action_card, [])

    def test_get_action_reads_qidui_room_option(self):
        # 房间配置开启七对后，API 兜底胡牌判断也要认可七对。
        hand = [0x01, 0x01, 0x02, 0x02, 0x03, 0x03, 0x04, 0x04, 0x11, 0x11, 0x12, 0x12, 0x21, 0x21]
        data = {
            "action_cards": {"7": [[0x21]]},
            "player_hand_cards": [hand, [], [], []],
            "acting_do_player_position": 0,
            "room_options": {"allow_qidui": True},
        }

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_HU)
        self.assertEqual(action_card, [])


if __name__ == "__main__":
    unittest.main()
