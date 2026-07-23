#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supervised GBMJ bot for Botzone-style stdin/stdout interaction.

This bot intentionally does not depend on the original RL engine. It reuses the
same PlayerState/Event logic as dataset generation and calls the supervised
policy directly on encoded observations.
"""

import argparse
import copy
import sys
import traceback
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mortal_part.mjai.event import Event
from mortal_part.state.mah_player_gb import PlayerState
from mortal_part.tile import Tile
from supervised.policy import SupervisedPolicy, resolve_checkpoint_path


class BotzoneBot:
    """Botzone adapter for the supervised GBMJ policy."""

    # Botzone tile -> internal tile string.
    TILE_MAP = {
        "W1": "1m", "W2": "2m", "W3": "3m", "W4": "4m", "W5": "5m",
        "W6": "6m", "W7": "7m", "W8": "8m", "W9": "9m",
        "B1": "1p", "B2": "2p", "B3": "3p", "B4": "4p", "B5": "5p",
        "B6": "6p", "B7": "7p", "B8": "8p", "B9": "9p",
        "T1": "1s", "T2": "2s", "T3": "3s", "T4": "4s", "T5": "5s",
        "T6": "6s", "T7": "7s", "T8": "8s", "T9": "9s",
        "F1": "E", "F2": "S", "F3": "W", "F4": "N",
        "J1": "C", "J2": "F", "J3": "P",
    }
    REVERSE_TILE_MAP = {v: k for k, v in TILE_MAP.items()}

    def __init__(self, policy: SupervisedPolicy, player_id: int, deterministic: bool = True, temperature: float = 1.0):
        self.policy = policy
        self.state = PlayerState(player_id)
        self.player_id = player_id
        self.deterministic = deterministic
        self.temperature = temperature

        self.game_started = False
        self.seat_id = player_id
        self.last_player = 0
        self.last_discard_tile: Optional[Tile] = None
        self.last_action: Optional[str] = None
        self.last_tsumo_tile: Optional[Tile] = None

        # [Added for supervised bot] When the model plans a chi/pon and also
        # chooses the follow-up discard, we must roll back before processing the
        # actual platform confirmation line. This mirrors the original bot logic.
        self.state_for_transaction: Optional[PlayerState] = None

        # [Added for supervised bot] Botzone may echo self ankan as "GANG"
        # without the tile argument, so keep the chosen tile for replay.
        self.pending_ankan_tile: Optional[Tile] = None

    def to_mortal_str(self, botzone_tile: str) -> str:
        return self.TILE_MAP.get(botzone_tile, botzone_tile)

    def to_botzone_str(self, mortal_tile: str) -> str:
        return self.REVERSE_TILE_MAP.get(mortal_tile, mortal_tile)

    def parse_tile(self, botzone_tile: str) -> Tile:
        return Tile.from_str(self.to_mortal_str(botzone_tile))

    def tile_to_str(self, tile: Tile) -> str:
        return self.to_botzone_str(repr(tile))

    def clear_kyoku(self) -> None:
        self.game_started = False
        self.last_player = 0
        self.last_discard_tile = None
        self.last_action = None
        self.last_tsumo_tile = None
        self.state_for_transaction = None
        self.pending_ankan_tile = None

    def _select_action(self, at_kan_select: bool = False) -> int:
        obs, mask = self.state.encode_obs(at_kan_select)
        return self.policy.select_action(
            obs,
            mask,
            deterministic=self.deterministic,
            temperature=self.temperature,
        )

    def _process_event(self, event: Event, can_act: bool = True) -> Optional[str]:
        cans = self.state.update(event)
        if not can_act or not cans.can_act:
            return None
        return self._choose_response()

    def _choose_followup_discard(self, call_event: Event, response_prefix: str) -> str:
        # Save current state, then simulate the call and let the policy choose
        # the mandatory follow-up discard from the post-call state.
        self.state_for_transaction = copy.deepcopy(self.state)
        response = self._process_event(call_event, can_act=True)
        if response is None or not response.startswith("PLAY "):
            raise RuntimeError(f"expected PLAY after {response_prefix}, got {response!r}")
        discard_tile = response.split()[1]
        return f"{response_prefix} {discard_tile}"

    def _build_chi_consumed(self, action: int) -> List[Tile]:
        if self.last_discard_tile is None:
            raise RuntimeError("chi requested without any last discard tile")
        tid = self.last_discard_tile.id
        if action == 37:  # chi_low
            return [Tile(tid + 1), Tile(tid + 2)]
        if action == 38:  # chi_mid
            return [Tile(tid - 1), Tile(tid + 1)]
        if action == 39:  # chi_high
            return [Tile(tid - 2), Tile(tid - 1)]
        raise ValueError(f"not a chi action: {action}")

    def _resolve_self_kan(self) -> str:
        tile_by_action = {}
        for tile in self.state.kakan_candidates:
            tile_by_action[tile.id] = ("BUGANG", tile)
        for tile in self.state.ankan_candidates:
            tile_by_action[tile.id] = ("GANG", tile)

        if not tile_by_action:
            raise RuntimeError("kan selected but there is no self kan candidate")

        if len(tile_by_action) == 1:
            command, tile = next(iter(tile_by_action.values()))
        else:
            tile_action = self._select_action(at_kan_select=True)
            if tile_action not in tile_by_action:
                raise RuntimeError(f"illegal kan-select action: {tile_action}")
            command, tile = tile_by_action[tile_action]

        if command == "GANG":
            self.pending_ankan_tile = tile
        return f"{command} {self.tile_to_str(tile)}"

    def _choose_response(self) -> str:
        action = self._select_action(at_kan_select=False)
        cans = self.state.last_cans

        if 0 <= action <= 33:
            return f"PLAY {self.tile_to_str(Tile(action))}"
        if action == 36:
            return "HU"
        if action == 40:
            return "PASS"
        if action == 35:
            if cans.can_daiminkan and not cans.can_discard and not cans.can_kakan and not cans.can_ankan:
                return "GANG"
            return self._resolve_self_kan()
        if action == 34:
            if self.last_discard_tile is None:
                raise RuntimeError("pon selected without any last discard tile")
            call_event = Event(
                "Pon",
                self.seat_id,
                self.last_discard_tile,
                self.last_player,
                [self.last_discard_tile, self.last_discard_tile],
            )
            return self._choose_followup_discard(call_event, "PENG")
        if action in (37, 38, 39):
            consumed = self._build_chi_consumed(action)
            call_event = Event(
                "Chi",
                self.seat_id,
                self.last_player,
                self.last_discard_tile,
                consumed,
            )
            consumed_all = sorted([*consumed, self.last_discard_tile], key=lambda tile: tile.id)
            return self._choose_followup_discard(call_event, f"CHI {self.tile_to_str(consumed_all[1])}")

        raise RuntimeError(f"unexpected action id: {action}")

    def react_dahai(self, tile: Tile, player: int, action_type: str) -> str:
        self.last_discard_tile = tile
        self.last_action = action_type
        self.last_player = player
        response = self._process_event(Event("Dahai", player, tile), can_act=True)
        return response or "PASS"

    def react_interface(self, line: str) -> Optional[str]:
        parts = line.strip().split()
        if not parts:
            return None
        cmd = int(parts[0])

        if cmd == 0:
            if not self.game_started:
                self.game_started = True
                seat_id = int(parts[1])
                round_wind = int(parts[2])
                self._process_event(Event("BotStartGame", seat_id, round_wind), can_act=False)
                # BotStartGame does not initialize these inside PlayerState.
                self.state.seat_id = seat_id
                self.state.round_wind = round_wind
                self.state.zhuang_id = 0
                self.seat_id = seat_id
            return "PASS"

        if cmd == 1:
            my_tiles = [self.parse_tile(tile) for tile in parts[5:18]]
            self._process_event(Event("BotStartKyoku", my_tiles), can_act=False)
            self.pending_ankan_tile = None
            return "PASS"

        if cmd == 2:
            tile = self.parse_tile(parts[1])
            self.last_tsumo_tile = tile
            self.last_action = "DRAW"
            self.last_player = self.seat_id
            response = self._process_event(Event("Tsumo", self.seat_id, tile), can_act=True)
            return response or "PASS"

        if cmd != 3:
            return "PASS"

        player = int(parts[1])
        action_type = parts[2]

        if self.state_for_transaction is not None:
            self.state = self.state_for_transaction
            self.state_for_transaction = None

        if action_type == "DRAW":
            self.last_action = action_type
            self.last_player = player
            return "PASS"

        if action_type == "PLAY":
            tile = self.parse_tile(parts[3])
            return self.react_dahai(tile, player, action_type)

        if action_type == "PENG":
            tile = self.parse_tile(parts[3])
            self._process_event(
                Event("Pon", player, self.last_discard_tile, self.last_player, [self.last_discard_tile, self.last_discard_tile]),
                can_act=False,
            )
            return self.react_dahai(tile, player, action_type)

        if action_type == "CHI":
            chi_mid_tile_id = self.parse_tile(parts[3]).id
            discard_tile = self.parse_tile(parts[4])
            if self.last_discard_tile.id < chi_mid_tile_id:
                consumed = [Tile(self.last_discard_tile.id + 1), Tile(self.last_discard_tile.id + 2)]
            elif self.last_discard_tile.id == chi_mid_tile_id:
                consumed = [Tile(self.last_discard_tile.id - 1), Tile(self.last_discard_tile.id + 1)]
            else:
                consumed = [Tile(self.last_discard_tile.id - 2), Tile(self.last_discard_tile.id - 1)]
            self._process_event(Event("Chi", player, self.last_player, self.last_discard_tile, consumed), can_act=False)
            return self.react_dahai(discard_tile, player, action_type)

        if action_type == "GANG":
            if self.last_action == "DRAW":
                tile = self.pending_ankan_tile
                if len(parts) > 3:
                    tile = self.parse_tile(parts[3])
                if player == self.seat_id and tile is None and len(self.state.ankan_candidates) == 1:
                    tile = self.state.ankan_candidates[0]

                consumed = None if tile is None else [tile, tile, tile, tile]
                self._process_event(Event("AnGang", player, consumed), can_act=True)
                self.pending_ankan_tile = None
                self.last_action = action_type
                self.last_player = player
                return "PASS"

            response = self._process_event(
                Event("MinGang", player, self.last_discard_tile, self.last_player, [self.last_discard_tile, self.last_discard_tile, self.last_discard_tile]),
                can_act=True,
            )
            self.last_action = action_type
            self.last_player = player
            return response or "PASS"

        if action_type == "BUGANG":
            tile = self.parse_tile(parts[3])
            response = self._process_event(Event("BuGang", player, tile, [tile]), can_act=True)
            self.last_action = action_type
            self.last_player = player
            return response or "PASS"

        if action_type == "HU":
            self.clear_kyoku()
            return "PASS"

        return "PASS"


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Botzone bot for the supervised GBMJ project")
    parser.add_argument("player_id", nargs="?", type=int, default=0, help="seat id in [0, 3]")
    parser.add_argument("--checkpoint", default="best", help="'best', 'latest', or a checkpoint path")
    parser.add_argument("--device", default="cpu", help="torch device, default: cpu")
    parser.add_argument("--stochastic", action="store_true", help="sample from policy instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature when --stochastic is set")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.player_id not in range(4):
        parser.error("player_id must be in [0, 3]")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    policy = SupervisedPolicy.from_path(checkpoint_path, device=args.device)
    bot = BotzoneBot(
        policy=policy,
        player_id=args.player_id,
        deterministic=not args.stochastic,
        temperature=args.temperature,
    )

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            response = bot.react_interface(line)
            if response is not None:
                print(response, flush=True)

        except KeyboardInterrupt:
            break
        except Exception:
            print("PASS", flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
