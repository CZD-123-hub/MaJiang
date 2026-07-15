import unittest

from jiujiang_ai.hand_routes import HandRoute, TaatsuKind, enumerate_hand_routes
from jiujiang_ai.route_metrics import RouteMetrics, measure_routes, retain_routes
from jiujiang_ai.tiles import HONGZHONG


class JiujiangHandRouteTests(unittest.TestCase):
    def test_keeps_multiple_routes_for_an_ambiguous_hand(self):
        # 12345 既可以优先拆出 123，也可以保留 12/34 等后续组合；
        # 路线生成阶段不能因为顺子优先而只留下一个结果。
        hand = [
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x11,
            0x12,
            0x13,
            0x21,
            0x22,
            0x23,
            0x06,
            0x08,
            0x09,
        ]

        routes = enumerate_hand_routes(hand)

        self.assertGreater(len(routes), 1)
        self.assertGreater(len({route.signature for route in routes}), 1)

    def test_deduplicates_equivalent_routes_after_generation(self):
        hand = [
            0x01,
            0x01,
            0x02,
            0x02,
            0x03,
            0x03,
            0x11,
            0x12,
            0x13,
            0x21,
            0x22,
            0x23,
            0x08,
            0x09,
        ]

        routes = enumerate_hand_routes(hand)

        self.assertEqual(len(routes), len({route.signature for route in routes}))

    def test_exposes_taatsu_shape_instead_of_a_single_count(self):
        hand = [
            0x01,
            0x02,
            0x04,
            0x06,
            0x11,
            0x12,
            0x13,
            0x21,
            0x22,
            0x23,
            0x05,
            0x05,
            0x08,
            0x09,
        ]

        routes = enumerate_hand_routes(hand)
        shapes = {taatsu.kind for route in routes for taatsu in route.taatsu}

        self.assertIn(TaatsuKind.PENCHAN, shapes)
        self.assertIn(TaatsuKind.KANCHAN, shapes)

    def test_records_hongzhong_pair_completion_as_a_route_plan(self):
        hand = [
            HONGZHONG,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x11,
            0x12,
            0x13,
            0x21,
            0x22,
            0x23,
            0x09,
        ]

        routes = enumerate_hand_routes(hand)

        self.assertTrue(
            any("pair_with_single" in assignment for route in routes for assignment in route.hongzhong_assignments)
        )

    def test_hongzhong_plan_does_not_reuse_a_single_tile_twice(self):
        hand = [
            HONGZHONG,
            HONGZHONG,
            HONGZHONG,
            0x01,
            0x02,
            0x03,
            0x11,
            0x12,
            0x13,
            0x21,
            0x22,
            0x23,
            0x04,
            0x06,
        ]

        routes = enumerate_hand_routes(hand)

        for route in routes:
            pair_tiles = {item.rsplit(":", 1)[1] for item in route.hongzhong_assignments if item.startswith("pair_with_single")}
            meld_tiles = {item.rsplit(":", 1)[1] for item in route.hongzhong_assignments if item.startswith("single_to_meld")}
            self.assertFalse(pair_tiles & meld_tiles)

    def test_route_metrics_distinguish_taatsu_quality(self):
        hand = [
            0x01,
            0x02,
            0x04,
            0x06,
            0x11,
            0x12,
            0x13,
            0x21,
            0x22,
            0x23,
            0x05,
            0x05,
            0x08,
            0x09,
        ]
        routes = enumerate_hand_routes(hand)
        metrics = measure_routes(hand, routes)

        penchan = next(route for route in routes if any(item.kind == TaatsuKind.PENCHAN for item in route.taatsu))
        kanchan = next(route for route in routes if any(item.kind == TaatsuKind.KANCHAN for item in route.taatsu))

        self.assertGreater(metrics[penchan.signature].effective_tile_kinds, 0)
        self.assertGreater(metrics[kanchan.signature].effective_tile_kinds, 0)

    def test_retention_keeps_one_more_shanten_route_when_its_draws_are_broader(self):
        lower_shanten = _route(shanten=1)
        broader_route = _route(shanten=2)
        metrics = {
            lower_shanten.signature: _metric(lower_shanten, effective_count=4, flexibility=3.0),
            broader_route.signature: _metric(broader_route, effective_count=12, flexibility=8.0),
        }

        kept = retain_routes((lower_shanten, broader_route), metrics, shanten_window=1)

        self.assertEqual({route.shanten for route in kept}, {1, 2})


def _route(*, shanten: int) -> HandRoute:
    return HandRoute(
        melds=(),
        pairs=(),
        taatsu=(),
        isolated=(0x01 if shanten == 1 else 0x02,),
        fixed_melds=0,
        hongzhong_count=0,
        hongzhong_used=0,
        hongzhong_assignments=(),
        red_completed_taatsu_indexes=(),
        red_completed_melds=0,
        shanten=shanten,
    )


def _metric(route: HandRoute, *, effective_count: int, flexibility: float) -> RouteMetrics:
    return RouteMetrics(
        route=route,
        effective_tiles={},
        effective_count=effective_count,
        effective_tile_kinds=effective_count,
        flexibility=flexibility,
        ryanmen_count=0,
        replacement_count=0,
    )


if __name__ == "__main__":
    unittest.main()
