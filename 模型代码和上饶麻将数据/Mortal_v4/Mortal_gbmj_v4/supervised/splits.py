import hashlib
import json
import random
from glob import glob
from pathlib import Path

import torch

from supervised.config import PROJECT_ROOT


"""训练/验证/测试文件划分。

划分单位是“对局日志文件”而不是单个状态样本。先按文件划分再回放，能
避免同一局的相邻状态同时出现在训练集和验证集，减少评估泄漏。
"""


def _as_project_path(pattern):
    path = Path(pattern)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _collect_files(patterns):
    files = set()
    for pattern in patterns:
        for file in glob(_as_project_path(pattern), recursive=True):
            files.add(str(Path(file).resolve()))
    return sorted(files)


def _train_globs(dataset_cfg):
    # [V4 data-mix] Keep backward compatibility with train_globs, but also let
    # public/private train globs define the training universe directly.
    patterns = []
    patterns.extend(dataset_cfg.get("train_globs", []))
    patterns.extend(dataset_cfg.get("public_train_globs", []))
    patterns.extend(dataset_cfg.get("private_train_globs", []))
    return patterns


def _split_signature(dataset_cfg, train_files, val_files, test_files):
    payload = {
        "train_globs": dataset_cfg.get("train_globs", []),
        "public_train_globs": dataset_cfg.get("public_train_globs", []),
        "private_train_globs": dataset_cfg.get("private_train_globs", []),
        "val_globs": dataset_cfg.get("val_globs", []),
        "test_globs": dataset_cfg.get("test_globs", []),
        "split_seed": dataset_cfg.get("split_seed", 0),
        "val_ratio": dataset_cfg.get("val_ratio", 0.0),
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _check_no_overlap(splits):
    # 数据集划分有交集时直接报错；监督模型的验证指标必须建立在独立文件上。
    seen = {}
    for split_name, files in splits.items():
        for file in files:
            old = seen.setdefault(file, split_name)
            if old != split_name:
                raise ValueError(f"file appears in both {old} and {split_name}: {file}")


def build_data_splits(dataset_cfg):
    split_index = dataset_cfg.get("split_index")
    split_index = PROJECT_ROOT / split_index if split_index else None

    # train_globs/public/private_train_globs 共同定义训练候选文件集合。
    train_files = _collect_files(_train_globs(dataset_cfg))
    explicit_val_files = _collect_files(dataset_cfg.get("val_globs", []))
    test_files = _collect_files(dataset_cfg.get("test_globs", []))

    overlap = set(explicit_val_files) & set(test_files)
    if overlap:
        sample = sorted(overlap)[0]
        raise ValueError(f"file appears in both val and test: {sample}")

    holdout = set(explicit_val_files) | set(test_files)
    if holdout:
        train_files = sorted(file for file in train_files if file not in holdout)

    signature = _split_signature(dataset_cfg, train_files, explicit_val_files, test_files)

    if split_index and split_index.exists():
        # split_index 固化随机划分，保证重新启动训练时训练/验证集合不漂移。
        cached = torch.load(split_index, map_location="cpu", weights_only=False)
        if cached.get("signature") == signature:
            splits = cached["splits"]
            _check_no_overlap(splits)
            return splits

    if explicit_val_files:
        val_files = explicit_val_files
    else:
        rng = random.Random(int(dataset_cfg.get("split_seed", 0)))
        shuffled = list(train_files)
        rng.shuffle(shuffled)
        val_ratio = float(dataset_cfg.get("val_ratio", 0.0))
        val_count = int(round(len(shuffled) * val_ratio))
        if val_ratio > 0 and len(shuffled) > 1:
            val_count = max(1, min(val_count, len(shuffled) - 1))
        val_files = sorted(shuffled[:val_count])
        train_files = sorted(shuffled[val_count:])

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    _check_no_overlap(splits)

    if split_index:
        split_index.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"signature": signature, "splits": splits}, split_index)

    return splits
