import argparse
import copy
import hashlib
import json
import logging
import random
import sys
import time
from glob import glob
from pathlib import Path

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from mortal_part.dataset.grp import GRP_FEATURE_DIM, GlobalRewardDataset
from rl.grp_model import GlobalRewardPredictor, load_encoder_from_policy_checkpoint
from supervised.config import PROJECT_ROOT, config


def resolve_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _as_glob(pattern):
    path = Path(str(pattern))
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _collect_files(patterns):
    files = set()
    for pattern in patterns:
        for filename in glob(_as_glob(pattern), recursive=True):
            files.add(str(Path(filename).resolve()))
    return sorted(files)


def _split_signature(grp_cfg, train_files, val_files):
    payload = {
        "train_globs": grp_cfg.get("train_globs", []),
        "val_globs": grp_cfg.get("val_globs", []),
        "split_seed": grp_cfg.get("split_seed", 0),
        "val_ratio": grp_cfg.get("val_ratio", 0.0),
        "train_files": train_files,
        "val_files": val_files,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_splits(grp_cfg):
    train_files = _collect_files(grp_cfg.get("train_globs", []))
    explicit_val_files = _collect_files(grp_cfg.get("val_globs", []))
    if explicit_val_files:
        holdout = set(explicit_val_files)
        train_files = [file for file in train_files if file not in holdout]

    split_index = grp_cfg.get("split_index", "")
    split_path = resolve_path(split_index) if split_index else None
    signature = _split_signature(grp_cfg, train_files, explicit_val_files)
    if split_path and split_path.exists():
        cached = torch.load(str(split_path), map_location="cpu", weights_only=False)
        if cached.get("signature") == signature:
            return cached["splits"]

    if explicit_val_files:
        val_files = explicit_val_files
    else:
        shuffled = list(train_files)
        random.Random(int(grp_cfg.get("split_seed", 0))).shuffle(shuffled)
        val_ratio = float(grp_cfg.get("val_ratio", 0.02))
        val_count = int(round(len(shuffled) * val_ratio))
        if val_ratio > 0.0 and len(shuffled) > 1:
            val_count = max(1, min(val_count, len(shuffled) - 1))
        val_files = sorted(shuffled[:val_count])
        train_files = sorted(shuffled[val_count:])

    splits = {"train": sorted(train_files), "val": sorted(val_files)}
    if split_path:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"signature": signature, "splits": splits}, str(split_path))
    return splits


def build_arg_parser():
    grp_cfg = config.get("grp", {})
    control_cfg = config.get("control", {})
    parser = argparse.ArgumentParser(description="Train the Global Reward Predictor (GRP).")
    parser.add_argument("--device", default=str(grp_cfg.get("device", control_cfg.get("device", "cuda"))))
    parser.add_argument("--checkpoint-dir", default=str(grp_cfg.get("checkpoint_dir", "checkpoints/grp_v3_global_reward")))
    parser.add_argument("--tensorboard-dir", default=str(grp_cfg.get("tensorboard_dir", "runs/grp_v3_global_reward")))
    parser.add_argument("--init-policy-checkpoint", default=str(grp_cfg.get("init_policy_checkpoint", "")))
    parser.add_argument("--epochs", type=int, default=int(grp_cfg.get("epochs", 3)))
    parser.add_argument("--batch-size", type=int, default=int(grp_cfg.get("batch_size", 512)))
    parser.add_argument("--eval-batch-size", type=int, default=int(grp_cfg.get("eval_batch_size", grp_cfg.get("batch_size", 512))))
    parser.add_argument("--lr", type=float, default=float(grp_cfg.get("lr", 2.0e-4)))
    parser.add_argument("--weight-decay", type=float, default=float(grp_cfg.get("weight_decay", 1.0e-4)))
    parser.add_argument("--eps", type=float, default=float(grp_cfg.get("eps", 1.0e-8)))
    parser.add_argument("--betas", type=float, nargs=2, default=[float(x) for x in grp_cfg.get("betas", [0.9, 0.999])])
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default=str(grp_cfg.get("optimizer", "adamw")).lower())
    parser.add_argument("--loss", choices=("huber", "mse"), default=str(grp_cfg.get("loss", "huber")).lower())
    parser.add_argument("--huber-delta", type=float, default=float(grp_cfg.get("huber_delta", 0.5)))
    parser.add_argument("--max-grad-norm", type=float, default=float(grp_cfg.get("max_grad_norm", 0.5)))
    parser.add_argument("--augmentation-factor", type=int, default=int(grp_cfg.get("augmentation_factor", 1)))
    parser.add_argument("--file-batch-size", type=int, default=int(grp_cfg.get("file_batch_size", 8)))
    parser.add_argument("--num-workers", type=int, default=int(grp_cfg.get("num_workers", 8)))
    parser.add_argument("--eval-num-workers", type=int, default=int(grp_cfg.get("eval_num_workers", 4)))
    parser.add_argument("--prefetch-factor", type=int, default=int(grp_cfg.get("prefetch_factor", 2)))
    parser.add_argument("--total-rounds", type=int, default=int(grp_cfg.get("total_rounds", 16)))
    parser.add_argument("--rank-weight", type=float, default=float(grp_cfg.get("rank_weight", 1.0)))
    parser.add_argument("--score-weight", type=float, default=float(grp_cfg.get("score_weight", 0.25)))
    parser.add_argument("--score-scale", type=float, default=float(grp_cfg.get("score_scale", 200.0)))
    parser.add_argument("--target-clip", type=float, default=float(grp_cfg.get("target_clip", 1.5)))
    parser.add_argument("--feature-hidden", type=int, default=int(grp_cfg.get("feature_hidden", 64)))
    parser.add_argument("--value-hidden", type=int, default=int(grp_cfg.get("value_hidden", config.get("model", {}).get("head_hidden", 512))))
    parser.add_argument("--log-every", type=int, default=int(grp_cfg.get("log_every", 100)))
    parser.add_argument("--save-every", type=int, default=int(grp_cfg.get("save_every", 2000)))
    parser.add_argument("--eval-every", type=int, default=int(grp_cfg.get("eval_every", 5000)))
    parser.add_argument("--eval-max-batches", type=int, default=int(grp_cfg.get("eval_max_batches", 0)))
    parser.add_argument("--max-train-batches", type=int, default=int(grp_cfg.get("max_train_batches", 0)))
    parser.set_defaults(
        resume=bool(grp_cfg.get("resume", True)),
        enable_amp=bool(grp_cfg.get("enable_amp", control_cfg.get("enable_amp", False))),
        freeze_encoder=bool(grp_cfg.get("freeze_encoder", False)),
        shuffle_files=bool(grp_cfg.get("shuffle_files", True)),
        shuffle_augmentation=bool(grp_cfg.get("shuffle_augmentation", True)),
        skip_bad_files=bool(grp_cfg.get("skip_bad_files", True)),
    )
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--enable-amp", dest="enable_amp", action="store_true")
    parser.add_argument("--disable-amp", dest="enable_amp", action="store_false")
    parser.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true")
    parser.add_argument("--train-encoder", dest="freeze_encoder", action="store_false")
    parser.add_argument("--shuffle-files", dest="shuffle_files", action="store_true")
    parser.add_argument("--no-shuffle-files", dest="shuffle_files", action="store_false")
    parser.add_argument("--skip-bad-files", dest="skip_bad_files", action="store_true")
    parser.add_argument("--no-skip-bad-files", dest="skip_bad_files", action="store_false")
    return parser


class Running:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.pred_sum = 0.0
        self.pred_sq_sum = 0.0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.count = 0
        self.batches = 0

    def update(self, metrics):
        for key in ("loss_sum", "abs_sum", "sq_sum", "pred_sum", "pred_sq_sum", "target_sum", "target_sq_sum"):
            setattr(self, key, getattr(self, key) + float(metrics[key]))
        self.count += int(metrics["count"])
        self.batches += int(metrics.get("batches", 1))

    def compute(self):
        count = max(1, self.count)
        pred_mean = self.pred_sum / count
        target_mean = self.target_sum / count
        pred_var = max(0.0, self.pred_sq_sum / count - pred_mean * pred_mean)
        target_var = max(0.0, self.target_sq_sum / count - target_mean * target_mean)
        mse = self.sq_sum / count
        ev = 1.0 - mse / target_var if target_var > 1.0e-8 else 0.0
        return {
            "loss": self.loss_sum / count,
            "mae": self.abs_sum / count,
            "rmse": math.sqrt(max(0.0, mse)),
            "pred_mean": pred_mean,
            "pred_std": math.sqrt(pred_var),
            "target_mean": target_mean,
            "target_std": math.sqrt(target_var),
            "explained_variance": ev,
            "count": self.count,
            "batches": self.batches,
        }

    def pop(self):
        metrics = self.compute()
        self.reset()
        return metrics


def build_loader(files, args, training: bool):
    dataset = GlobalRewardDataset(
        file_list=files,
        file_batch_size=int(args.file_batch_size),
        num_epochs=1,
        shuffle_files=bool(training and args.shuffle_files),
        augmentation_factor=int(args.augmentation_factor) if training else 1,
        shuffle_augmentation=bool(args.shuffle_augmentation),
        total_rounds=int(args.total_rounds),
        rank_weight=float(args.rank_weight),
        score_weight=float(args.score_weight),
        score_scale=float(args.score_scale),
        target_clip=float(args.target_clip),
        skip_bad_files=bool(args.skip_bad_files),
    )
    num_workers = int(args.num_workers if training else args.eval_num_workers)
    kwargs = {
        "batch_size": int(args.batch_size if training else args.eval_batch_size),
        "drop_last": bool(training),
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def compute_loss(model, batch, device, args):
    obs, features, target = batch
    obs = torch.nan_to_num(obs.to(device, non_blocking=True), nan=0.0, posinf=1.0, neginf=0.0)
    features = features.to(device=device, non_blocking=True).float()
    target = target.to(device=device, non_blocking=True).float()
    with torch.autocast(device_type=device.type, enabled=bool(args.enable_amp and device.type == "cuda")):
        pred = model(obs, features)
        if args.loss == "mse":
            loss = F.mse_loss(pred.float(), target.float())
        else:
            loss = F.smooth_l1_loss(pred.float(), target.float(), beta=float(args.huber_delta))
    with torch.no_grad():
        pred_f = pred.detach().float()
        target_f = target.detach().float()
        err = pred_f - target_f
        count = int(target_f.numel())
        metrics = {
            "loss_sum": float(loss.detach().cpu()) * count,
            "abs_sum": float(err.abs().sum().cpu()),
            "sq_sum": float(err.square().sum().cpu()),
            "pred_sum": float(pred_f.sum().cpu()),
            "pred_sq_sum": float(pred_f.square().sum().cpu()),
            "target_sum": float(target_f.sum().cpu()),
            "target_sq_sum": float(target_f.square().sum().cpu()),
            "count": count,
            "batches": 1,
        }
    return loss, metrics


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    running = Running()
    max_batches = int(args.eval_max_batches)
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        _, metrics = compute_loss(model, batch, device, args)
        running.update(metrics)
    return running.compute()


def write_metrics(writer, prefix, metrics, step):
    for key, value in metrics.items():
        if key in ("count", "batches"):
            continue
        writer.add_scalar(f"{prefix}/{key}", float(value), step)


def save_checkpoint(path, model, optimizer, steps, epoch, best_metrics, args, model_cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "steps": int(steps),
            "epoch": int(epoch),
            "best_metrics": dict(best_metrics or {}),
            "config": copy.deepcopy(config),
            "model_cfg": dict(model_cfg),
            "grp": {
                "feature_dim": GRP_FEATURE_DIM,
                "feature_hidden": int(args.feature_hidden),
                "value_hidden": int(args.value_hidden),
                "rank_weight": float(args.rank_weight),
                "score_weight": float(args.score_weight),
                "score_scale": float(args.score_scale),
                "target_clip": float(args.target_clip),
                "total_rounds": int(args.total_rounds),
            },
            "source": "v4_global_reward_predictor",
        },
        str(path),
    )


def maybe_resume(latest_file, model, optimizer, device, enabled):
    if not enabled or not latest_file.exists():
        return 0, 0, None
    state = torch.load(str(latest_file), map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    logging.info("resumed GRP from %s at step=%s epoch=%s", latest_file, state.get("steps", 0), state.get("epoch", 0))
    return int(state.get("steps", 0)), int(state.get("epoch", 0)), state.get("best_metrics")


def run_training(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(str(args.device))
    torch.backends.cudnn.benchmark = bool(config.get("control", {}).get("enable_cudnn_benchmark", False))

    grp_cfg = config.get("grp", {})
    splits = build_splits(grp_cfg)
    logging.info("train files: %s", f"{len(splits['train']):,}")
    logging.info("val files: %s", f"{len(splits['val']):,}")
    if not splits["train"]:
        raise RuntimeError("GRP train split is empty; set [grp].train_globs to your 16-hand full-game logs")
    if not splits["val"]:
        raise RuntimeError("GRP val split is empty; set [grp].val_ratio > 0 or provide [grp].val_globs")

    train_loader = build_loader(splits["train"], args, training=True)
    val_loader = build_loader(splits["val"], args, training=False)

    model_cfg = dict(config.get("model", {}))
    model = GlobalRewardPredictor(
        feature_dim=GRP_FEATURE_DIM,
        feature_hidden=int(args.feature_hidden),
        value_hidden=int(args.value_hidden),
        **model_cfg,
    ).to(device)
    init_policy_checkpoint = str(args.init_policy_checkpoint)
    if init_policy_checkpoint:
        copied = load_encoder_from_policy_checkpoint(model, resolve_path(init_policy_checkpoint), device)
        logging.info("initialized GRP encoder from policy checkpoint: copied=%s path=%s", copied, resolve_path(init_policy_checkpoint))
    if bool(args.freeze_encoder):
        for name, param in model.named_parameters():
            if name.startswith("spatial_encoder."):
                param.requires_grad_(False)
        logging.info("frozen GRP spatial encoder")

    optim_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optim_cls(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        betas=tuple(float(x) for x in args.betas),
        eps=float(args.eps),
        weight_decay=float(args.weight_decay),
    )
    scaler = GradScaler(enabled=bool(args.enable_amp and device.type == "cuda"))

    ckpt_dir = resolve_path(args.checkpoint_dir)
    log_dir = resolve_path(args.tensorboard_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    latest_file = ckpt_dir / "grp_latest.pth"
    best_file = ckpt_dir / "grp_best.pth"
    steps, start_epoch, best_metrics = maybe_resume(latest_file, model, optimizer, device, bool(args.resume))
    best_val_loss = float(best_metrics.get("loss", "inf")) if isinstance(best_metrics, dict) else float("inf")

    for epoch in range(start_epoch, int(args.epochs)):
        epoch_start = time.time()
        model.train()
        running = Running()
        epoch_running = Running()
        train_iter = tqdm(train_loader, desc=f"GRP epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(train_iter):
            if int(args.max_train_batches) > 0 and batch_idx >= int(args.max_train_batches):
                break
            loss, metrics = compute_loss(model, batch, device, args)
            if not torch.isfinite(loss.detach()):
                raise RuntimeError("non-finite GRP loss at step %s: %s" % (steps + 1, float(loss.detach().cpu())))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(args.max_grad_norm) > 0.0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            running.update(metrics)
            epoch_running.update(metrics)
            if int(args.log_every) > 0 and steps % int(args.log_every) == 0:
                train_metrics = running.pop()
                write_metrics(writer, "grp_train", train_metrics, steps)
                writer.flush()
                train_iter.set_postfix(loss=f"{train_metrics['loss']:.4f}", ev=f"{train_metrics['explained_variance']:.3f}")
            if int(args.save_every) > 0 and steps % int(args.save_every) == 0:
                save_checkpoint(latest_file, model, optimizer, steps, epoch, best_metrics, args, model_cfg)
            if int(args.eval_every) > 0 and steps % int(args.eval_every) == 0:
                val_metrics = evaluate(model, val_loader, device, args)
                write_metrics(writer, "grp_val", val_metrics, steps)
                writer.flush()

        train_metrics = epoch_running.compute()
        write_metrics(writer, "grp_train_epoch", train_metrics, steps)
        val_metrics = evaluate(model, val_loader, device, args)
        write_metrics(writer, "grp_val", val_metrics, steps)
        writer.flush()
        elapsed = time.time() - epoch_start
        logging.info(
            "epoch=%s step=%s train_loss=%.6f val_loss=%.6f val_mae=%.6f ev=%.4f elapsed=%.1fs",
            epoch + 1,
            steps,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["mae"],
            val_metrics["explained_variance"],
            elapsed,
        )
        save_checkpoint(latest_file, model, optimizer, steps, epoch + 1, val_metrics, args, model_cfg)
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_metrics = dict(val_metrics)
            save_checkpoint(best_file, model, optimizer, steps, epoch + 1, best_metrics, args, model_cfg)
            logging.info("saved best GRP checkpoint: %s", best_file)

    writer.close()
    logging.info("GRP training done. latest=%s best=%s", latest_file, best_file)


def main():
    args = build_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
