import unittest

from jiujiang_ai.hand_split import (
    _analyze_counts,
    _counts_tuple,
    _split_counts,
    analyze_hand,
    analyze_normalized_counts,
    is_four_hongzhong,
)
from jiujiang_ai.tiles import HONGZHONG


class JiujiangHandSplitTests(unittest.TestCase):
    def test_complete_pinghu_without_hongzhong(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x09]

        result = analyze_hand(hand)

        self.assertEqual(result.shanten, 0)
        self.assertEqual(result.hongzhong_used, 0)

    def test_one_hongzhong_completes_pair(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, HONGZHONG]

        result = analyze_hand(hand)

        self.assertEqual(result.shanten, 0)
        self.assertGreaterEqual(result.hongzhong_used, 1)

    def test_hongzhong_improves_incomplete_hand(self):
        no_red = analyze_hand([0x01, 0x02, 0x03, 0x04, 0x05, 0x11, 0x12, 0x21, 0x22, 0x09, 0x09, 0x18, 0x19])
        with_red = analyze_hand([0x01, 0x02, 0x03, 0x04, 0x05, 0x11, 0x12, 0x21, 0x22, 0x09, 0x09, 0x18, HONGZHONG])

        self.assertLessEqual(with_red.shanten, no_red.shanten)

    def test_four_hongzhong_detection(self):
        self.assertTrue(is_four_hongzhong([HONGZHONG, HONGZHONG, HONGZHONG, HONGZHONG]))
        self.assertFalse(is_four_hongzhong([HONGZHONG, HONGZHONG, HONGZHONG, 0x01]))

    def test_split_counts_deduplicates_equivalent_partial_shapes(self):
        hand = (0x01, 0x02, 0x03, 0x11, 0x12, 0x13)

        partials = _split_counts(_counts_tuple(hand))
        summaries = {(item.melds, item.pairs, item.taatsu, item.leftovers) for item in partials}

        self.assertEqual(len(partials), len(set(partials)))
        self.assertEqual(
            summaries,
            {
                (2, 0, 0, 0),
                (1, 0, 1, 1),
                (1, 0, 0, 3),
                (0, 0, 2, 2),
                (0, 0, 1, 4),
                (0, 0, 0, 6),
            },
        )

    def test_analyze_hand_cache_uses_canonical_tile_counts(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x08, 0x09]
        _analyze_counts.cache_clear()

        analyze_hand(hand)
        first = _analyze_counts.cache_info()
        analyze_hand(list(reversed(hand)))
        second = _analyze_counts.cache_info()

        self.assertEqual(second.misses, first.misses)
        self.assertEqual(second.hits, first.hits + 1)

    def test_normalized_counts_analysis_matches_hand_analysis(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x08, HONGZHONG]
        ordinary = tuple(sorted(tile for tile in hand if tile != HONGZHONG))

        direct = analyze_hand(hand)
        normalized = analyze_normalized_counts(_counts_tuple(ordinary), hand.count(HONGZHONG))

        self.assertEqual(normalized, direct)


if __name__ == "__main__":
    unittest.main()
