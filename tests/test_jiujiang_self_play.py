import unittest

from jiujiang_ai.self_play import play_round, run_self_play


class JiujiangSelfPlayTests(unittest.TestCase):
    def test_same_seed_produces_same_round_summary(self):
        first = play_round(seed=20260714, decision_fn=self._first_legal_action)
        second = play_round(seed=20260714, decision_fn=self._first_legal_action)

        self.assertEqual(first["summary"], second["summary"])

    def test_ai_only_receives_its_own_hand(self):
        observed_payloads = []

        def observe_and_discard(data):
            observed_payloads.append(data)
            position = data["acting_do_player_position"]
            candidates = data["action_cards"]
            if "4" in candidates:
                return 4, []
            if "7" in candidates:
                return 7, candidates["7"][0]
            return 0, []

        play_round(seed=7, decision_fn=observe_and_discard)

        self.assertTrue(observed_payloads)
        for payload in observed_payloads:
            position = payload["acting_do_player_position"]
            hands = payload["player_hand_cards"]
            self.assertTrue(hands[position])
            self.assertTrue(all(not hand for index, hand in enumerate(hands) if index != position))

    def test_batch_play_finishes_requested_number_of_rounds(self):
        result = run_self_play(rounds=3, seed=42, decision_fn=self._first_legal_action)

        self.assertEqual(result["rounds"], 3)
        self.assertEqual(len(result["round_results"]), 3)
        self.assertEqual(result["wins"] + result["draws"], 3)
        self.assertEqual(sum(result["wins_by_player"].values()), result["wins"])
        self.assertEqual(len(result["total_score_by_player"]), 4)

    def test_round_emits_live_draw_and_action_events(self):
        events = []

        play_round(seed=9, decision_fn=self._first_legal_action, event_callback=events.append)

        self.assertTrue(any(event["event"] == "draw" for event in events))
        self.assertTrue(any(event["event"] == "action" for event in events))
        self.assertEqual(events[-1]["event"], "round_end")

    @staticmethod
    def _first_legal_action(data):
        candidates = data["action_cards"]
        for action_type in ("4", "3", "2", "5", "6", "7"):
            if action_type in candidates:
                cards = candidates[action_type]
                return int(action_type), cards[0] if cards else []
        return 0, []


if __name__ == "__main__":
    unittest.main()
