import argparse
import secrets
import shutil
from pathlib import Path

import numpy as np

from supervised.config import PROJECT_ROOT


def parse_int(value):
    if isinstance(value, int):
        return value
    return int(str(value), 0)


def parse_obs_version(value):
    # "auto" lets each checkpoint choose its own obs encoder from the stored
    # first-conv channel count when comparing v4 direct-235 and v3 policies.
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "auto", "none"):
        return None
    return int(text, 0)


def format_obs_version(value):
    return "auto" if value is None else "v%d" % int(value)


def resolve_device(device):
    # [V4 local-play] Lazy torch import keeps --help usable even on a
    # machine that only edits configs and does not have the train env.
    import torch
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def infer_parallel_gpu_ids(args):
    # [V4 parallel-play] If --gpu-ids is omitted, keep --device semantics:
    # cuda:1 means only GPU 1, cpu means CPU, auto means all visible GPUs.
    if args.gpu_ids is not None:
        return args.gpu_ids
    device = str(args.device).lower()
    if device == "auto":
        return "auto"
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    return "cpu"


def build_engine(name, checkpoint, device, args, obs_version):
    from supervised.policy import SupervisedEngine
    return SupervisedEngine.from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        name=name,
        obs_version=obs_version,
        enable_amp=not args.no_amp,
        enable_quick_eval=not args.no_quick_eval,
        enable_rule_based_agari_guard=not args.no_agari_guard,
        deterministic=not args.stochastic,
        boltzmann_epsilon=args.boltzmann_epsilon,
        boltzmann_temp=args.boltzmann_temp,
        top_p=args.top_p,
    )


def print_stat(stat, rankings):
    rankings = np.asarray(rankings, dtype=np.float64)
    total = rankings.sum()
    avg_rank = float(rankings @ np.arange(1, 5) / total) if total else 0.0
    rank_pt = float(rankings @ np.array([90, 45, 0, -135], dtype=np.float64) / total) if total else 0.0
    rank_rates = rankings / total if total else np.zeros(4, dtype=np.float64)

    print("=== ranking summary ===")
    print("games: %d" % int(total))
    print("challenger_rankings: %s" % rankings.astype(int).tolist())
    print("avg_rank: %.4f, rank_pt: %.4f" % (avg_rank, rank_pt))
    print("rank_rates: 1st=%.4f 2nd=%.4f 3rd=%.4f 4th=%.4f" %
          (rank_rates[0], rank_rates[1], rank_rates[2], rank_rates[3]))

    print("=== round summary ===")
    print("stat_games: %d, rounds: %d" % (stat.game, stat.round))
    print("rank_1_rate: %.4f" % stat.rank_1_rate)
    print("rank_2_rate: %.4f" % stat.rank_2_rate)
    print("rank_3_rate: %.4f" % stat.rank_3_rate)
    print("rank_4_rate: %.4f" % stat.rank_4_rate)
    print("avg_point_per_game: %.4f" % stat.avg_point_per_game)
    print("avg_point_per_round: %.4f" % stat.avg_point_per_round)
    print("hu_rate: %.4f" % stat.hu_rate)
    print("houjuu_rate: %.4f" % stat.houjuu_rate)
    print("fulu_rate: %.4f" % stat.fulu_rate)
    print("ryukyoku_rate: %.4f" % stat.ryukyoku_rate)
    print("avg_point_per_hu: %.4f" % stat.avg_point_per_hu)
    print("avg_point_per_houjuu: %.4f" % stat.avg_point_per_houjuu)


def main():
    parser = argparse.ArgumentParser(description="Local 1-vs-3 play test for v4 supervised GBMJ policies.")
    parser.add_argument("--challenger", default="/data/mortal_gbmj/Mortal_gbmj_v3/checkpoints/v3_msres2former_public_aug12_finetune_2%val/gbmj_policy_best.pth", help="'best', 'latest', or checkpoint path for challenger.")
    parser.add_argument("--champion", default="/data/mortal_gbmj/Mortal_gbmj_v3/checkpoints/v3_msres2former_public_aug12_finetune/gbmj_policy_best.pth", help="'best', 'latest', or checkpoint path for champion/baseline.")
    parser.add_argument("--challenger-obs-version", default="3", help="auto, 3, 4, or 5.")
    parser.add_argument("--champion-obs-version", default="3", help="auto, 3, 4, or 5.")
    parser.add_argument("--device", default="cuda:1", help="auto/cpu/cuda:0/...; auto chooses cuda:0 when available.")
    parser.add_argument("--champion-device", default=None)
    parser.add_argument("--games", type=int, default=10000, help="Approximate hanchan count. Actual count is rounded down to a multiple of 4.")
    parser.add_argument("--seed-start", type=int, default=11200)
    parser.add_argument("--seed-key", default=None, help="Integer seed key, e.g. 0x2000. Random if omitted.")
    parser.add_argument("--log-dir", default="logs/local_play")
    parser.add_argument("--keep-logs", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="Use multiprocessing one-vs-three runner.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker processes when --parallel is enabled.")
    parser.add_argument("--gpu-ids", default=None, help="Parallel only: auto, cpu, 0, or 0,1. Defaults to --device semantics.")
    parser.add_argument("--torch-threads-per-worker", type=int, default=2)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--boltzmann-epsilon", type=float, default=0.0)
    parser.add_argument("--boltzmann-temp", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-quick-eval", action="store_true")
    parser.add_argument("--no-agari-guard", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    args = parser.parse_args()
    challenger_obs_version = parse_obs_version(args.challenger_obs_version)
    champion_obs_version = parse_obs_version(args.champion_obs_version)

    # [V4 local-play] Import arena/stat only after argparse has handled
    # --help.  Actual play still needs the native mortal_cpp extension,
    # exactly like the original Mortal local arena.
    from mortal_part.stat import Stat
    from supervised.policy import resolve_checkpoint_path

    seed_count = max(1, args.games // 4)
    actual_games = seed_count * 4
    seed_key = secrets.randbits(64) if args.seed_key is None else parse_int(args.seed_key)

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    if log_dir.exists() and not args.keep_logs:
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    challenger_path = resolve_checkpoint_path(args.challenger)
    champion_path = resolve_checkpoint_path(args.champion)

    print("challenger:", challenger_path)
    print("champion:  ", champion_path)
    print("obs_versions: challenger=%s champion=%s" % (
        format_obs_version(challenger_obs_version),
        format_obs_version(champion_obs_version),
    ))
    print("games:", actual_games, "seed_start:", args.seed_start, "seed_key:", hex(seed_key))
    print("log_dir:", log_dir)

    if args.parallel:
        # [V4 parallel-play] Fast path for many local games.  Each worker
        # loads both checkpoints once and writes logs for Stat aggregation.
        from mortal_part.arena.one_vs_three_parallel import OneVsThreeParallel

        gpu_ids = infer_parallel_gpu_ids(args)
        print("mode: parallel, workers:", args.workers, "gpu_ids:", gpu_ids)
        env = OneVsThreeParallel(disable_progress_bar=args.disable_progress_bar, log_dir=str(log_dir))
        # env = OneVsThreeParallel(disable_progress_bar=args.disable_progress_bar, log_dir=None)
        rankings = env.py_vs_py(
            checkpoint_paths={"mortal": str(challenger_path), "baseline": str(champion_path)},
            seed_start=(args.seed_start, seed_key),
            seed_count=seed_count,
            num_workers=args.workers,
            gpu_ids=gpu_ids,
            engine_options={
                "enable_amp": not args.no_amp,
                "enable_quick_eval": not args.no_quick_eval,
                "enable_rule_based_agari_guard": not args.no_agari_guard,
                "deterministic": not args.stochastic,
                "boltzmann_epsilon": args.boltzmann_epsilon,
                "boltzmann_temp": args.boltzmann_temp,
                "top_p": args.top_p,
                "torch_threads": args.torch_threads_per_worker,
                "obs_versions": {
                    "mortal": challenger_obs_version,
                    "baseline": champion_obs_version,
                },
            },
        )
    else:
        from mortal_part.arena.one_vs_three import OneVsThree

        device = resolve_device(args.device)
        champion_device = resolve_device(args.champion_device or args.device)
        print("mode: serial")
        print("device:", device, "champion_device:", champion_device)

        challenger = build_engine("mortal", args.challenger, device, args, challenger_obs_version)
        champion = build_engine("baseline", args.champion, champion_device, args, champion_obs_version)
        print("resolved_obs_versions: challenger=v%d champion=v%d" % (challenger.obs_version, champion.obs_version))

        env = OneVsThree(disable_progress_bar=args.disable_progress_bar, log_dir=str(log_dir))
        rankings = env.py_vs_py(
            challenger=challenger,
            champion=champion,
            seed_start=(args.seed_start, seed_key),
            seed_count=seed_count,
        )

    stat = Stat.from_dir(str(log_dir), "mortal", disable_progress_bar=args.disable_progress_bar)
    print_stat(stat, rankings)


if __name__ == "__main__":
    main()
