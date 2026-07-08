import unittest

from jiujiang_ai.win_context import detect_win_context


class JiujiangWinContextTests(unittest.TestCase):
    def test_detects_zimo_from_win_type(self):
        context = detect_win_context({"winner": 0, "win_type": "zimo"})

        self.assertEqual(context.win_type, "zimo")
        self.assertEqual(context.winners, [0])
        self.assertIsNone(context.dianpao_player)
        self.assertFalse(context.is_multi_win)

    def test_detects_dianpao_from_discard_player(self):
        context = detect_win_context({"winner": 2, "discard_player": 1})

        self.assertEqual(context.win_type, "dianpao")
        self.assertEqual(context.winners, [2])
        self.assertEqual(context.dianpao_player, 1)
        self.assertFalse(context.is_multi_win)

    def test_detects_qianggang_from_explicit_win_type(self):
        context = detect_win_context({"winner": 3, "win_type": "qianggang", "pao_player": 1})

        self.assertEqual(context.win_type, "qianggang")
        self.assertEqual(context.winners, [3])
        self.assertEqual(context.dianpao_player, 1)

    def test_detects_gangkai_from_alias(self):
        context = detect_win_context({"winner": 1, "win_type": "杠上开花"})

        self.assertEqual(context.win_type, "gangkai")
        self.assertEqual(context.winners, [1])
        self.assertIsNone(context.dianpao_player)

    def test_returns_unknown_when_fields_are_missing(self):
        context = detect_win_context({})

        self.assertEqual(context.win_type, "unknown")
        self.assertEqual(context.winners, [])
        self.assertIsNone(context.dianpao_player)
        self.assertFalse(context.is_multi_win)

    def test_marks_multi_win_when_multiple_winners_exist(self):
        context = detect_win_context({"winners": [0, 2], "dianpao_player": 1})

        self.assertEqual(context.win_type, "dianpao")
        self.assertEqual(context.winners, [0, 2])
        self.assertEqual(context.dianpao_player, 1)
        self.assertTrue(context.is_multi_win)


if __name__ == "__main__":
    unittest.main()
