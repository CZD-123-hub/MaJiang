import unittest

from jiujiang_ai.tiles import (
    HONGZHONG,
    JIUJIANG_TILE_CODES,
    format_tiles,
    remaining_tile_counts,
    validate_hand,
)


class JiujiangTileTests(unittest.TestCase):
    def test_accepts_suited_tiles_and_hongzhong(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, HONGZHONG]

        validate_hand(hand)

        self.assertIn(HONGZHONG, JIUJIANG_TILE_CODES)
        self.assertEqual(format_tiles([0x01, 0x19, 0x29, HONGZHONG]), ["1W", "9T", "9B", "RED"])

    def test_rejects_non_jiujiang_honor_tiles(self):
        with self.assertRaises(ValueError):
            validate_hand([0x31])

        with self.assertRaises(ValueError):
            validate_hand([0x37])

    def test_rejects_more_than_four_of_same_tile(self):
        with self.assertRaises(ValueError):
            validate_hand([0x01, 0x01, 0x01, 0x01, 0x01])

    def test_remaining_tile_counts_uses_four_copies(self):
        counts = remaining_tile_counts([0x01, 0x01, HONGZHONG])

        self.assertEqual(counts[0x01], 2)
        self.assertEqual(counts[HONGZHONG], 3)


if __name__ == "__main__":
    unittest.main()
