import sys
import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mortal_part.consts import ACTION_SPACE
from supervised.config import config
from supervised.data_module import build_loader
from supervised.splits import build_data_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()

    dataset_cfg = config["dataset"]
    control_cfg = config["control"]
    splits = build_data_splits(dataset_cfg)
    files = splits[args.split]
    loader = build_loader(files, dataset_cfg, control_cfg, training=False)

    counts = [0 for _ in range(ACTION_SPACE)]
    samples = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        _, _, target = batch
        for idx in target.tolist():
            if 0 <= idx < ACTION_SPACE:
                counts[idx] += 1
                samples += 1

    print(json.dumps({
        "split": args.split,
        "files": len(files),
        "samples": samples,
        "action_counts": {str(idx): count for idx, count in enumerate(counts)},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
