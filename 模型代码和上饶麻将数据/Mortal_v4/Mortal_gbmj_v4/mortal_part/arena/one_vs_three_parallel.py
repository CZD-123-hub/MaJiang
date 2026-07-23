# mortal_part/arena/one_vs_three_parallel.py

import gzip
import math
import os
import queue
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
import torch.multiprocessing as mp

from mortal_part.agent.py_agent import new_py_agent
from mortal_part.arena.game import BatchGame, Index


_engines = None
_worker_info = None


def _normalize_gpu_ids(gpu_ids):
    if gpu_ids == "auto":
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return [-1]
    if gpu_ids in (None, "", "cpu"):
        return [-1]
    if isinstance(gpu_ids, int):
        return [gpu_ids]
    if isinstance(gpu_ids, str):
        values = []
        for item in gpu_ids.split(","):
            item = item.strip().lower()
            if not item:
                continue
            if item == "cpu":
                values.append(-1)
            else:
                values.append(int(item))
        return values or [-1]
    values = [int(item) for item in gpu_ids]
    return values or [-1]


def _device_for_worker(gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device("cuda:%d" % gpu_id)
    return torch.device("cpu")


def _build_engine(name, checkpoint, device, engine_options):
    # [V4 parallel-play] Each worker loads the 235-way supervised policy
    # directly.  This replaces the copied Mortal Brain+DQN checkpoint path.
    from supervised.policy import SupervisedEngine
    obs_versions = dict(engine_options.get("obs_versions", {}))
    obs_version = obs_versions.get(name)
    if obs_version in (None, "", "auto"):
        obs_version = None
    else:
        obs_version = int(obs_version)

    return SupervisedEngine.from_checkpoint(
        checkpoint=checkpoint,
        device=str(device),
        name=name,
        obs_version=obs_version,
        enable_amp=bool(engine_options.get("enable_amp", True)),
        enable_quick_eval=bool(engine_options.get("enable_quick_eval", True)),
        enable_rule_based_agari_guard=bool(engine_options.get("enable_rule_based_agari_guard", True)),
        deterministic=bool(engine_options.get("deterministic", True)),
        boltzmann_epsilon=float(engine_options.get("boltzmann_epsilon", 0.0)),
        boltzmann_temp=float(engine_options.get("boltzmann_temp", 1.0)),
        top_p=float(engine_options.get("top_p", 1.0)),
    )


def _init_worker(gpu_id: int, checkpoint_paths: Dict[str, str], engine_options: Dict):
    global _engines, _worker_info

    torch.set_num_threads(max(1, int(engine_options.get("torch_threads", 2))))
    device = _device_for_worker(int(gpu_id))

    _worker_info = {"gpu_id": int(gpu_id), "device": str(device)}
    print("[V4 parallel-play] worker gpu=%s device=%s loading checkpoints..." % (gpu_id, device), flush=True)

    _engines = {
        "mortal": _build_engine("mortal", checkpoint_paths["mortal"], device, engine_options),
        "baseline": _build_engine("baseline", checkpoint_paths["baseline"], device, engine_options),
    }

    print("[V4 parallel-play] worker gpu=%s ready." % gpu_id, flush=True)


def _indexes_for_seed_group():
    challenger_player_ids = list(range(4))
    champion_player_ids = [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]

    agent_idxs_per_seed = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]

    indexes_list = []
    challenger_idx = 0
    champion_idx = 0
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
        indexes_list.append(game_indexes)

    return challenger_player_ids, champion_player_ids, indexes_list


def _worker_run(task: Tuple[int, int, Optional[str]]) -> Tuple[int, List[int]]:
    global _engines, _worker_info
    if _engines is None:
        raise RuntimeError("parallel worker has not been initialized")

    seed_val, key, log_dir = task
    gpu_id = int(_worker_info["gpu_id"])

    challenger_player_ids, champion_player_ids, indexes_list = _indexes_for_seed_group()
    challenger_agent = new_py_agent(_engines["mortal"], challenger_player_ids)
    champion_agent = new_py_agent(_engines["baseline"], champion_player_ids)

    seeds = [(seed_val, key)] * 4
    batch_game = BatchGame.east_game(disable_progress_bar=True)
    results = batch_game.run([challenger_agent, champion_agent], indexes_list, seeds)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        for i, game_result in enumerate(results):
            split_name = ["a", "b", "c", "d"][i]
            filename = os.path.join(log_dir, "%s_%s_%s.json.gz" % (seed_val, key, split_name))
            with gzip.open(filename, "wt") as f:
                f.write(game_result.dump_json_log())

    rankings = [0] * 4
    for i, result in enumerate(results):
        rank = result.rankings().rank_by_player[i % 4]
        rankings[rank] += 1

    return gpu_id, rankings


def _run_gpu_pool(gpu_id, tasks, workers, checkpoint_paths, engine_options, result_queue):
    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(gpu_id, checkpoint_paths, engine_options),
    ) as pool:
        for result in pool.imap_unordered(_worker_run, tasks):
            result_queue.put(result)


class OneVsThreeParallel:
    # [V4 parallel-play] Parallel local 1-vs-3 runner for supervised
    # checkpoints.  The public API mirrors the copied Mortal runner so
    # local_play.py can switch between serial and parallel modes.
    def __init__(self, disable_progress_bar: bool = False, log_dir: Optional[str] = None):
        self.disable_progress_bar = disable_progress_bar
        self.log_dir = log_dir
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    def py_vs_py(
        self,
        checkpoint_paths: Dict[str, str],
        seed_start: Tuple[int, int],
        seed_count: int,
        num_workers: int = 4,
        gpu_ids="auto",
        engine_options: Optional[Dict] = None,
    ) -> List[int]:
        engine_options = dict(engine_options or {})
        gpu_ids = _normalize_gpu_ids(gpu_ids)
        num_workers = max(1, int(num_workers))
        seed_count = max(0, int(seed_count))

        print("[V4 parallel-play] workers=%d gpu_ids=%s games=%d" % (num_workers, gpu_ids, seed_count * 4))
        print("[V4 parallel-play] challenger=%s" % checkpoint_paths["mortal"])
        print("[V4 parallel-play] champion=%s" % checkpoint_paths["baseline"])

        if seed_count == 0:
            return [0, 0, 0, 0]

        all_tasks = [
            (seed_val, seed_start[1], self.log_dir)
            for seed_val in range(seed_start[0], seed_start[0] + seed_count)
        ]
        all_rankings = [0] * 4
        pbar = None if self.disable_progress_bar else tqdm(total=seed_count, desc="LOCAL-PARALLEL")

        if len(gpu_ids) == 1:
            with mp.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(gpu_ids[0], checkpoint_paths, engine_options),
            ) as pool:
                for _, rankings in pool.imap_unordered(_worker_run, all_tasks):
                    for i in range(4):
                        all_rankings[i] += rankings[i]
                    if pbar:
                        pbar.update(1)
        else:
            workers_per_gpu = max(1, num_workers // len(gpu_ids))
            tasks_per_gpu = int(math.ceil(len(all_tasks) / float(len(gpu_ids))))

            with mp.Manager() as manager:
                result_queue = manager.Queue()
                gpu_processes = []

                for idx, gpu_id in enumerate(gpu_ids):
                    start = idx * tasks_per_gpu
                    end = min((idx + 1) * tasks_per_gpu, len(all_tasks))
                    gpu_tasks = all_tasks[start:end]
                    if not gpu_tasks:
                        continue
                    process = mp.Process(
                        target=_run_gpu_pool,
                        args=(gpu_id, gpu_tasks, workers_per_gpu, checkpoint_paths, engine_options, result_queue),
                    )
                    process.start()
                    gpu_processes.append(process)

                completed = 0
                while completed < seed_count:
                    try:
                        _, rankings = result_queue.get(timeout=2.0)
                    except queue.Empty:
                        failed = [process for process in gpu_processes if process.exitcode not in (None, 0)]
                        if failed:
                            for process in gpu_processes:
                                if process.is_alive():
                                    process.terminate()
                            codes = [process.exitcode for process in failed]
                            raise RuntimeError("parallel gpu process failed before finishing: %s" % codes)
                        continue

                    for i in range(4):
                        all_rankings[i] += rankings[i]
                    completed += 1
                    if pbar:
                        pbar.update(1)

                for process in gpu_processes:
                    process.join()
                    if process.exitcode not in (0, None):
                        raise RuntimeError("parallel gpu process exited with code %s" % process.exitcode)

        if pbar:
            pbar.close()

        total = sum(all_rankings)
        avg_rank = sum((i + 1) * c for i, c in enumerate(all_rankings)) / total if total else 0.0
        print("[V4 parallel-play] rankings=%s avg_rank=%.4f" % (all_rankings, avg_rank))
        return all_rankings
