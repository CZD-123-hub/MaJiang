import unittest

from jiujiang_ai.rules import (
    ACTION_CHI,
    ACTION_DISCARD,
    ACTION_GANG,
    ACTION_HU,
    ACTION_PASS,
    ACTION_PENG,
    is_legal_operation,
    normalize_action_cards,
)
from jiujiang_ai.tiles import HONGZHONG


class JiujiangRuleTests(unittest.TestCase):
    def test_action_constants_match_harness_values(self):
        self.assertEqual(ACTION_PASS, 0)
        self.assertEqual(ACTION_CHI, 1)
        self.assertEqual(ACTION_PENG, 2)
        self.assertEqual(ACTION_GANG, 3)
        self.assertEqual(ACTION_HU, 4)
        self.assertEqual(ACTION_DISCARD, 7)

    def test_chi_is_never_legal(self):
        self.assertFalse(is_legal_operation(ACTION_CHI, [0x01, 0x02, 0x03]))

    def test_hongzhong_cannot_peng_or_gang(self):
        self.assertFalse(is_legal_operation(ACTION_PENG, [HONGZHONG, HONGZHONG, HONGZHONG]))
        self.assertFalse(is_legal_operation(ACTION_GANG, [HONGZHONG, HONGZHONG, HONGZHONG, HONGZHONG]))

    def test_non_hongzhong_peng_and_gang_are_legal(self):
        self.assertTrue(is_legal_operation(ACTION_PENG, [0x02, 0x02, 0x02]))
        self.assertTrue(is_legal_operation(ACTION_GANG, [0x02, 0x02, 0x02, 0x02]))

    def test_normalize_action_cards_accepts_string_keys(self):
        normalized = normalize_action_cards({"7": [[0x01]], "4": []})

        self.assertIn(ACTION_DISCARD, normalized)
        self.assertIn(ACTION_HU, normalized)
        self.assertEqual(normalized[ACTION_DISCARD], [[0x01]])


if __name__ == "__main__":
    unittest.main()
