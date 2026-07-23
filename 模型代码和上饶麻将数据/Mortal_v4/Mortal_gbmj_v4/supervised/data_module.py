from glob import glob
from pathlib import Path

from torch.utils.data import DataLoader

from supervised.config import PROJECT_ROOT
from supervised.dataloader import PolicyDataset


"""根据 config.toml 构建训练/验证 DataLoader。

这里是“配置文件”和“数据集实现”的连接层：负责解析 glob、限制数据
来源、设置 batch/worker/pin_memory，并把最终文件列表交给 PolicyDataset。
"""


def _as_project_path(pattern):
    path = Path(pattern)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _collect_files(patterns):
    # 所有相对路径都以项目根目录为基准；训练前应检查 glob 是否真的匹配到文件。
    files = set()
    for pattern in patterns:
        for file in glob(_as_project_path(pattern), recursive=True):
            files.add(str(Path(file).resolve()))
    return sorted(files)


def _build_source_specs(files, dataset_cfg):
    # [V4 data-mix] Source-specific augmentation.  The train split still
    # controls which files are allowed; source globs only assign files to a
    # public/private augmentation schedule.
    # 先把 source 文件限制在 train split 内，避免 public/private 配置把验证集
    # 意外混入训练集，造成数据泄漏。
    train_files = set(str(Path(file).resolve()) for file in files)
    source_specs = []
    source_defs = (
        ("public", dataset_cfg.get("public_train_globs", []), dataset_cfg.get("public_augmentation_factor", None)),
        ("private", dataset_cfg.get("private_train_globs", []), dataset_cfg.get("private_augmentation_factor", None)),
    )
    for name, patterns, aug_factor in source_defs:
        if not patterns:
            continue
        source_files = [file for file in _collect_files(patterns) if file in train_files]
        if not source_files:
            continue
        source_specs.append({
            "name": name,
            "files": source_files,
            "augmentation_factor": int(aug_factor if aug_factor is not None else dataset_cfg.get("augmentation_factor", 1)),
        })
    return source_specs


def build_loader(files, dataset_cfg, control_cfg, training):
    # training=True 时使用增强和 shuffle；验证集固定不增强，保证指标可比较。
    num_workers = int(dataset_cfg["num_workers"] if training else dataset_cfg.get("eval_num_workers", 0))
    batch_size = int(control_cfg["batch_size"] if training else control_cfg.get("eval_batch_size", control_cfg["batch_size"]))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    persistent_workers = bool(dataset_cfg.get("persistent_workers", False)) and num_workers > 0
    multiprocessing_context = dataset_cfg.get("multiprocessing_context", None)
    augmentation_factor = int(dataset_cfg.get("augmentation_factor", 1)) if training else 1
    source_specs = _build_source_specs(files, dataset_cfg) if training else None
    dataset = PolicyDataset(
        file_list=files,
        file_batch_size=int(dataset_cfg["file_batch_size"]),
        num_epochs=1,
        shuffle_files=bool(training and dataset_cfg.get("shuffle_files", True)),
        augmentation_factor=augmentation_factor,
        shuffle_augmentation=bool(dataset_cfg.get("shuffle_augmentation", True)),
        source_specs=source_specs,
        # [V4 rollback-history] Default to v3-style single-state samples.
        # Set history_len explicitly only when re-enabling sequence modeling.
        history_len=int(dataset_cfg.get("history_len", 0)),
        value_target_config=dataset_cfg.get("value_target_config", None),
    )

    kwargs = {
        "batch_size": batch_size,
        "drop_last": bool(training),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(dataset_cfg.get("prefetch_factor", 2))
    if multiprocessing_context:
        kwargs["multiprocessing_context"] = str(multiprocessing_context)
    return DataLoader(dataset, **kwargs)
