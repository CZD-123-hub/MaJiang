from typing import Dict, List, Tuple

import numpy as np

from mortal_part.agent.supervised import SupervisedBatchAgent
from mortal_part.arena.game import BatchGame, Index
from mortal_part.mjai.event import Event
from rl.agent import PPOBatchAgent


def _indexes_for_seed_count(seed_count):
    challenger_player_ids = list(range(4)) * seed_count
    champion_player_ids_per_seed = [
        1, 2, 3,
        0, 2, 3,
        0, 1, 3,
        0, 1, 2,
    ]
    champion_player_ids = champion_player_ids_per_seed * seed_count
    agent_idxs_per_seed = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]

    indexes = []
    challenger_idx = 0
    champion_idx = 0
    for _ in range(seed_count):
        for agent_idxs in agent_idxs_per_seed:
            game_indexes = []
            for agent_idx in agent_idxs:
                if agent_idx == 0:
                    player_id_idx = challenger_idx
                    challenger_idx += 1
                else:
                    player_id_idx = champion_idx
                    champion_idx += 1
                game_indexes.append(Index(agent_idx=agent_idx, player_id_idx=player_id_idx))
            indexes.append(game_indexes)
    return challenger_player_ids, champion_player_ids, indexes


def _make_batch_game(game_length, disable_progress_bar):
    # [V4 MJ_RM strict DPPO] Full four-wind games expose all round-wind contexts.
    game_length = int(game_length)
    if game_length == 4:
        return BatchGame.east_game(disable_progress_bar=disable_progress_bar)
    if game_length == 8:
        return BatchGame.south_game(disable_progress_bar=disable_progress_bar)
    if game_length == 12:
        return BatchGame.west_game(disable_progress_bar=disable_progress_bar)
    if game_length == 16:
        return BatchGame.north_game(disable_progress_bar=disable_progress_bar)
    return BatchGame(length=game_length, init_scores=[500] * 4, disable_progress_bar=disable_progress_bar)


def _round_outcome(kyoku_log, player_id):
    """[V4 compact metrics] Extract per-hand challenger outcome.

    Rewards still come exclusively from rl/reward.py.  These values are only
    rollout diagnostics.
    """
    deltas = [0.0, 0.0, 0.0, 0.0]
    has_hu = False
    self_hu = False
    self_zimo = False
    self_ron = False
    self_houjuu = False
    other_zimo = False
    other_ron = False

    for ext in kyoku_log:
        ev = ext.event.event
        if not isinstance(ev, Event.Hu):
            continue
        has_hu = True
        if ev.deltas is not None:
            for idx, delta in enumerate(ev.deltas):
                deltas[idx] += float(delta)

        winner = int(ev.player)
        target = int(ev.target)
        if winner == player_id:
            self_hu = True
            if target == player_id:
                self_zimo = True
            else:
                self_ron = True
        elif target == player_id:
            self_houjuu = True
        elif target == winner:
            other_zimo = True
        else:
            other_ron = True

    return {
        "deltas": deltas,
        "score_delta": float(deltas[player_id]),
        "has_hu": has_hu,
        "self_hu": self_hu,
        "self_zimo": self_zimo,
        "self_ron": self_ron,
        "self_houjuu": self_houjuu,
        "other_zimo": other_zimo,
        "other_ron": other_ron,
    }


def _round_rank_from_delta(deltas, player_id):
    # [V4 compact metrics] Do not merge tied 2nd/3rd/4th places.  Zimo and
    # non-dealer ron often create equal loser deltas, so we use player_id as a
    # stable tie-breaker to keep round_rank_2/3/4 separated in TensorBoard.
    order = sorted(range(4), key=lambda pid: (-float(deltas[pid]), pid))
    return order.index(player_id) + 1


def collect_one_vs_three(
    actor_engine,
    baseline_engine,
    seed_start: Tuple[int, int],
    seed_count: int,
    rank_bonus: List[float],
    game_length: int = 16,
    rank_edge_fourth_penalty: float = 1.25,
    disable_progress_bar=False,
    reward_shaper=None,
) -> Tuple[List[Dict], Dict]:
    """[V4 MJ_RM strict DPPO] Collect challenger trajectories vs frozen v3."""
    challenger_ids, champion_ids, indexes = _indexes_for_seed_count(int(seed_count))
    challenger = PPOBatchAgent(
        actor_engine,
        challenger_ids,
        rank_bonus=rank_bonus,
        reward_shaper=reward_shaper,
    )
    champion = SupervisedBatchAgent(baseline_engine, champion_ids)
    seeds = [(seed, seed_start[1]) for seed in range(seed_start[0], seed_start[0] + seed_count) for _ in range(4)]

    batch_game = _make_batch_game(game_length, disable_progress_bar=disable_progress_bar)
    results = batch_game.run([challenger, champion], indexes, seeds)
    transitions = challenger.pop_transitions()

    rounds = 0
    ranked_rounds = 0
    round_rankings = [0, 0, 0, 0]
    game_rankings = [0, 0, 0, 0]
    score_deltas = []
    hu_count = 0
    zimo_count = 0
    ron_win_count = 0
    houjuu_count = 0
    other_zimo_count = 0
    other_ron_count = 0
    draw_count = 0

    for i, result in enumerate(results):
        player_id = i % 4
        game_rank = int(result.rankings().rank_by_player[player_id])
        if 0 <= game_rank < 4:
            game_rankings[game_rank] += 1
        for kyoku_log in result.game_log:
            rounds += 1
            outcome = _round_outcome(kyoku_log, player_id)
            score_deltas.append(outcome["score_delta"])
            if outcome["has_hu"]:
                ranked_rounds += 1
                round_rank = _round_rank_from_delta(outcome["deltas"], player_id)
                round_rankings[round_rank - 1] += 1
            else:
                draw_count += 1
            hu_count += int(outcome["self_hu"])
            zimo_count += int(outcome["self_zimo"])
            ron_win_count += int(outcome["self_ron"])
            houjuu_count += int(outcome["self_houjuu"])
            other_zimo_count += int(outcome["other_zimo"])
            other_ron_count += int(outcome["other_ron"])

    round_total = max(1, rounds)
    ranked_total = max(1, ranked_rounds)
    game_total = max(1, len(results))
    avg_round_rank = float(sum((idx + 1) * value for idx, value in enumerate(round_rankings)) / ranked_total)
    avg_game_rank = float(sum((idx + 1) * value for idx, value in enumerate(game_rankings)) / game_total)
    zimo_rate = zimo_count / round_total
    ron_win_rate = ron_win_count / round_total
    deal_in_rate = houjuu_count / round_total
    other_zimo_rate = other_zimo_count / round_total
    other_ron_rate = other_ron_count / round_total
    # [V4 RL outcome-aligned reward] Keep the checkpoint metric in the same
    # unit system as terminal rewards, instead of selecting by raw reward noise.
    reward_cfg = getattr(reward_shaper, "cfg", None)
    self_draw_reward = float(getattr(reward_cfg, "self_draw_reward", 2.0))
    win_reward = float(getattr(reward_cfg, "win_reward", 1.0))
    deal_in_penalty = float(getattr(reward_cfg, "deal_in_penalty", -2.0))
    other_self_draw_penalty = float(getattr(reward_cfg, "other_self_draw_penalty", -0.8))
    other_ron_penalty = float(getattr(reward_cfg, "other_ron_penalty", -0.35))
    outcome_utility = (
        self_draw_reward * zimo_rate
        + win_reward * ron_win_rate
        + deal_in_penalty * deal_in_rate
        + other_self_draw_penalty * other_zimo_rate
        + other_ron_penalty * other_ron_rate
    )
    game_rank_1 = game_rankings[0] / game_total
    game_rank_2 = game_rankings[1] / game_total
    game_rank_3 = game_rankings[2] / game_total
    game_rank_4 = game_rankings[3] / game_total
    game_rank_pt = (
        90.0 * game_rank_1
        + 45.0 * game_rank_2
        - 135.0 * game_rank_4
    )
    round_rank_1 = round_rankings[0] / ranked_total
    round_rank_2 = round_rankings[1] / ranked_total
    round_rank_3 = round_rankings[2] / ranked_total
    round_rank_4 = round_rankings[3] / ranked_total
    round_rank_pt = (
        90.0 * round_rank_1
        + 45.0 * round_rank_2
        - 135.0 * round_rank_4
    )
    round_rank_edge = round_rank_1 - float(rank_edge_fourth_penalty) * round_rank_4
    game_rank_edge = game_rank_1 - float(rank_edge_fourth_penalty) * game_rank_4
    return transitions, {
        # [V4 rank-aligned PPO] Keep both hand-level diagnostics and full-game
        # rank metrics.  The latter are the checkpoint-selection signal when
        # optimizing per-game first/fourth rates.
        "games": int(len(results)),
        "rounds": int(rounds),
        "round_ranked": int(ranked_rounds),
        "transitions": int(len(transitions)),
        "avg_score_delta": float(np.mean(score_deltas)) if score_deltas else 0.0,
        "win_rate": hu_count / round_total,
        "zimo_rate": zimo_rate,
        "ron_win_rate": ron_win_rate,
        # [V4 RL framework audit] Use the same key consumed by train_ppo.py.
        # The previous name was houjuu_rate, so deal-in looked permanently 0
        # in progress/TensorBoard even when terminal_deal_in rewards existed.
        "deal_in_rate": deal_in_rate,
        "other_zimo_rate": other_zimo_rate,
        "other_ron_rate": other_ron_rate,
        "draw_rate": draw_count / round_total,
        "outcome_utility": outcome_utility,
        "round_rank_1": round_rank_1,
        "round_rank_2": round_rank_2,
        "round_rank_3": round_rank_3,
        "round_rank_4": round_rank_4,
        "avg_round_rank": avg_round_rank,
        "round_rank_pt": round_rank_pt,
        "round_rank_edge": round_rank_edge,
        "game_rank_1": game_rank_1,
        "game_rank_2": game_rank_2,
        "game_rank_3": game_rank_3,
        "game_rank_4": game_rank_4,
        "avg_game_rank": avg_game_rank,
        "game_rank_pt": game_rank_pt,
        "game_rank_edge": game_rank_edge,
    }
