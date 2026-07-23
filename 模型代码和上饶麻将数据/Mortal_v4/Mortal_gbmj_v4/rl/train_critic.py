import argparse
import copy
import logging
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from rl.model import (
    actor_state_dict,
    assert_independent_modules,
    load_actor_and_critic_from_checkpoint,
)
from supervised.config import PROJECT_ROOT, config
from supervised.data_module import build_loader
from supervised.splits import build_data_splits


def resolve_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _cpu_state_dict(module):
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def parameter_count(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


class CriticRunningAverages:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.abs_error_sum = 0.0
        self.sq_error_sum = 0.0
        self.pred_sum = 0.0
        self.pred_sq_sum = 0.0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.nonzero_targets = 0
        self.count = 0
        self.batches = 0

    def update(self, metrics):
        self.loss_sum += float(metrics["loss_sum"])
        self.abs_error_sum += float(metrics["abs_error_sum"])
        self.sq_error_sum += float(metrics["sq_error_sum"])
        self.pred_sum += float(metrics["pred_sum"])
        self.pred_sq_sum += float(metrics["pred_sq_sum"])
        self.target_sum += float(metrics["target_sum"])
        self.target_sq_sum += float(metrics["target_sq_sum"])
        self.nonzero_targets += int(metrics["nonzero_targets"])
        self.count += int(metrics["count"])
        self.batches += int(metrics.get("batches", 1))

    def compute(self):
        count = max(1, self.count)
        pred_mean = self.pred_sum / count
        target_mean = self.target_sum / count
        pred_var = max(0.0, self.pred_sq_sum / count - pred_mean * pred_mean)
        target_var = max(0.0, self.target_sq_sum / count - target_mean * target_mean)
        mse = self.sq_error_sum / count
        if target_var > 1.0e-8:
            explained_variance = 1.0 - mse / target_var
        else:
            explained_variance = 0.0
        return {
            "loss": self.loss_sum / count,
            "mae": self.abs_error_sum / count,
            "rmse": math.sqrt(max(0.0, mse)),
            "pred_mean": pred_mean,
            "pred_std": math.sqrt(pred_var),
            "target_mean": target_mean,
            "target_std": math.sqrt(target_var),
            "nonzero_target_rate": self.nonzero_targets / count,
            "explained_variance": explained_variance,
            "count": self.count,
            "batches": self.batches,
        }

    def pop(self):
        metrics = self.compute()
        self.reset()
        return metrics


def build_arg_parser():
    critic_cfg = config.get("critic_pretrain", {})
    control_cfg = config.get("control", {})
    dataset_cfg = config.get("dataset", {})
    optim_cfg = config.get("optim", {})
    rl_cfg = config.get("rl", {})

    default_init_checkpoint = critic_cfg.get(
        "init_checkpoint",
        rl_cfg.get("init_checkpoint", ""),
    )
    default_batch_size = int(critic_cfg.get("batch_size", control_cfg.get("batch_size", 512)))
    default_eval_batch_size = int(
        critic_cfg.get("eval_batch_size", control_cfg.get("eval_batch_size", default_batch_size))
    )

    parser = argparse.ArgumentParser(
        description="Pretrain the independent PPO critic from supervised game logs.",
    )
    parser.add_argument("--init-checkpoint", default=str(default_init_checkpoint))
    parser.add_argument(
        "--checkpoint-dir",
        default=str(critic_cfg.get("checkpoint_dir", "checkpoints/critic_pretrain")),
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=str(critic_cfg.get("tensorboard_dir", "runs/critic_pretrain")),
    )
    parser.add_argument("--device", default=str(critic_cfg.get("device", rl_cfg.get("device", control_cfg.get("device", "cuda")))))
    parser.add_argument("--epochs", type=int, default=int(critic_cfg.get("epochs", 3)))
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=default_eval_batch_size)
    parser.add_argument("--lr", type=float, default=float(critic_cfg.get("lr", rl_cfg.get("value_lr", optim_cfg.get("lr", 2.0e-4)))))
    parser.add_argument("--weight-decay", type=float, default=float(critic_cfg.get("weight_decay", rl_cfg.get("weight_decay", 0.0))))
    parser.add_argument("--eps", type=float, default=float(critic_cfg.get("eps", rl_cfg.get("eps", 1.0e-8))))
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=[float(x) for x in critic_cfg.get("betas", rl_cfg.get("betas", [0.9, 0.999]))],
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default=str(critic_cfg.get("optimizer", rl_cfg.get("optimizer", "adam"))).lower(),
    )
    parser.add_argument("--max-grad-norm", type=float, default=float(critic_cfg.get("max_grad_norm", rl_cfg.get("max_grad_norm", 0.5))))
    parser.add_argument(
        "--loss",
        choices=("huber", "mse"),
        default=str(critic_cfg.get("loss", "huber")).lower(),
    )
    parser.add_argument("--huber-delta", type=float, default=float(critic_cfg.get("huber_delta", 1.0)))
    parser.add_argument("--target-scale", type=float, default=float(critic_cfg.get("target_scale", 1.0)))
    parser.add_argument("--target-clip", type=float, default=float(critic_cfg.get("target_clip", 3.0)))
    parser.add_argument(
        "--self-draw-reward",
        type=float,
        default=float(critic_cfg.get("self_draw_reward", rl_cfg.get("self_draw_reward", 2.0))),
    )
    parser.add_argument(
        "--win-reward",
        type=float,
        default=float(critic_cfg.get("win_reward", rl_cfg.get("win_reward", 1.5))),
    )
    parser.add_argument(
        "--deal-in-penalty",
        type=float,
        default=float(critic_cfg.get("deal_in_penalty", rl_cfg.get("deal_in_penalty", -2.0))),
    )
    parser.add_argument(
        "--other-self-draw-penalty",
        type=float,
        default=float(critic_cfg.get("other_self_draw_penalty", rl_cfg.get("other_self_draw_penalty", -1.0))),
    )
    parser.add_argument(
        "--other-ron-penalty",
        type=float,
        default=float(critic_cfg.get("other_ron_penalty", rl_cfg.get("other_ron_penalty", -0.5))),
    )
    parser.add_argument(
        "--score-delta-reward-scale",
        type=float,
        default=float(critic_cfg.get("score_delta_reward_scale", rl_cfg.get("score_delta_reward_scale", 40.0))),
    )
    parser.add_argument(
        "--score-delta-reward-clip",
        type=float,
        default=float(critic_cfg.get("score_delta_reward_clip", rl_cfg.get("score_delta_reward_clip", 2.0))),
    )
    parser.add_argument(
        "--tenpai-reward",
        type=float,
        default=float(critic_cfg.get("tenpai_reward", rl_cfg.get("tenpai_reward", 0.1))),
    )
    parser.add_argument(
        "--noten-penalty",
        type=float,
        default=float(critic_cfg.get("noten_penalty", rl_cfg.get("noten_penalty", -0.1))),
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=int(critic_cfg.get("augmentation_factor", dataset_cfg.get("augmentation_factor", 1))),
        help="Set to 12 for all suit permutations plus number mirror.",
    )
    parser.set_defaults(
        override_source_augmentation=bool(critic_cfg.get("override_source_augmentation", True)),
        enable_amp=bool(critic_cfg.get("enable_amp", control_cfg.get("enable_amp", False))),
        resume=bool(critic_cfg.get("resume", False)),
        zero_critic_feature_proj=bool(critic_cfg.get("zero_critic_feature_proj", True)),
        enable_draw_tenpai_target=bool(critic_cfg.get("enable_draw_tenpai_target", True)),
    )
    parser.add_argument(
        "--override-source-augmentation",
        dest="override_source_augmentation",
        action="store_true",
        help="Apply --augmentation-factor to public/private source schedules too.",
    )
    parser.add_argument(
        "--preserve-source-augmentation",
        dest="override_source_augmentation",
        action="store_false",
        help="Keep public_augmentation_factor/private_augmentation_factor from config.toml.",
    )
    parser.add_argument("--enable-amp", dest="enable_amp", action="store_true")
    parser.add_argument("--disable-amp", dest="enable_amp", action="store_false")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--zero-critic-feature-proj", dest="zero_critic_feature_proj", action="store_true")
    parser.add_argument("--keep-critic-feature-proj", dest="zero_critic_feature_proj", action="store_false")
    parser.add_argument("--enable-draw-tenpai-target", dest="enable_draw_tenpai_target", action="store_true")
    parser.add_argument("--disable-draw-tenpai-target", dest="enable_draw_tenpai_target", action="store_false")
    parser.add_argument("--log-every", type=int, default=int(critic_cfg.get("log_every", control_cfg.get("log_every", 100))))
    parser.add_argument("--save-every", type=int, default=int(critic_cfg.get("save_every", 0)))
    parser.add_argument("--eval-every", type=int, default=int(critic_cfg.get("eval_every", 0)))
    parser.add_argument("--eval-max-batches", type=int, default=int(critic_cfg.get("eval_max_batches", control_cfg.get("eval_max_batches", 0) or 0)))
    parser.add_argument("--max-train-batches", type=int, default=int(critic_cfg.get("max_train_batches", 0)))
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--critic-feature-dim", type=int, default=int(critic_cfg.get("critic_feature_dim", rl_cfg.get("critic_feature_dim", 0))))
    parser.add_argument("--critic-feature-hidden", type=int, default=int(critic_cfg.get("critic_feature_hidden", rl_cfg.get("critic_feature_hidden", 64))))
    parser.add_argument("--strict-actor", action="store_true", default=bool(critic_cfg.get("strict_actor", False)))
    return parser


def build_training_configs(args):
    dataset_cfg = copy.deepcopy(config["dataset"])
    control_cfg = copy.deepcopy(config["control"])
    model_cfg = copy.deepcopy(config["model"])

    dataset_cfg["augmentation_factor"] = int(args.augmentation_factor)
    if bool(args.override_source_augmentation):
        dataset_cfg["public_augmentation_factor"] = int(args.augmentation_factor)
        dataset_cfg["private_augmentation_factor"] = int(args.augmentation_factor)
    if args.num_workers is not None:
        dataset_cfg["num_workers"] = int(args.num_workers)
    if args.eval_num_workers is not None:
        dataset_cfg["eval_num_workers"] = int(args.eval_num_workers)
    dataset_cfg["value_target_config"] = {
        "self_draw_reward": float(args.self_draw_reward),
        "win_reward": float(args.win_reward),
        "deal_in_penalty": float(args.deal_in_penalty),
        "other_self_draw_penalty": float(args.other_self_draw_penalty),
        "other_ron_penalty": float(args.other_ron_penalty),
        "score_delta_reward_scale": float(args.score_delta_reward_scale),
        "score_delta_reward_clip": float(args.score_delta_reward_clip),
        "tenpai_reward": float(args.tenpai_reward),
        "noten_penalty": float(args.noten_penalty),
        "enable_draw_tenpai_target": bool(args.enable_draw_tenpai_target),
    }

    control_cfg["batch_size"] = int(args.batch_size)
    control_cfg["eval_batch_size"] = int(args.eval_batch_size)
    return dataset_cfg, control_cfg, model_cfg


def move_critic_batch(batch, device, target_scale=1.0, target_clip=0.0):
    if len(batch) < 6:
        raise RuntimeError(
            "critic pretraining requires value_target from PolicyDataset; got a legacy batch"
        )
    obs, history_obs, history_actions, _, _, target = batch[:6]
    obs = torch.nan_to_num(obs.to(device, non_blocking=True), nan=0.0, posinf=1.0, neginf=0.0)
    history_obs = history_obs.to(device=device, non_blocking=True)
    history_actions = history_actions.to(device=device, non_blocking=True).long()
    target = target.to(device=device, non_blocking=True).float() * float(target_scale)
    if target_clip and target_clip > 0.0:
        target = target.clamp(min=-float(target_clip), max=float(target_clip))
    return obs, history_obs, history_actions, target


def zero_critic_feature_projection(critic):
    proj = getattr(critic, "critic_feature_proj", None)
    if proj is None:
        return False
    changed = False
    for module in proj.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            changed = True
    return changed


def compute_critic_loss(critic, batch, device, enable_amp, args):
    obs, history_obs, history_actions, target = move_critic_batch(
        batch,
        device,
        target_scale=float(args.target_scale),
        target_clip=float(args.target_clip),
    )
    amp_enabled = bool(enable_amp and device.type == "cuda")
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        pred = critic(
            obs,
            history_obs=history_obs,
            history_actions=history_actions,
            critic_features=None,
        )
        if args.loss == "mse":
            loss = F.mse_loss(pred.float(), target.float())
        else:
            loss = F.smooth_l1_loss(pred.float(), target.float(), beta=float(args.huber_delta))

    with torch.no_grad():
        pred_detached = pred.detach().float()
        target_detached = target.detach().float()
        error = pred_detached - target_detached
        count = int(target_detached.numel())
        metrics = {
            "loss_sum": float(loss.detach()) * count,
            "abs_error_sum": float(error.abs().sum().cpu()),
            "sq_error_sum": float(error.square().sum().cpu()),
            "pred_sum": float(pred_detached.sum().cpu()),
            "pred_sq_sum": float(pred_detached.square().sum().cpu()),
            "target_sum": float(target_detached.sum().cpu()),
            "target_sq_sum": float(target_detached.square().sum().cpu()),
            "nonzero_targets": int((target_detached.abs() > 1.0e-6).sum().cpu()),
            "count": count,
            "batches": 1,
        }
    return loss, metrics


@torch.no_grad()
def evaluate_critic(critic, data_loader, device, args, max_batches=0):
    critic.eval()
    running = CriticRunningAverages()
    for batch_idx, batch in enumerate(data_loader):
        if max_batches and batch_idx >= int(max_batches):
            break
        _, metrics = compute_critic_loss(
            critic,
            batch,
            device,
            enable_amp=bool(args.enable_amp),
            args=args,
        )
        running.update(metrics)
    return running.compute()


def write_metrics(writer, prefix, metrics, step):
    for key, value in metrics.items():
        if key in ("count", "batches"):
            continue
        writer.add_scalar(f"{prefix}/{key}", float(value), step)


def save_critic_checkpoint(path, actor, critic, optimizer, steps, epoch, best_metrics, args, model_cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor_state_dict(actor),
            "critic": _cpu_state_dict(critic),
            "critic_optimizer": optimizer.state_dict(),
            "steps": int(steps),
            "epoch": int(epoch),
            "best_metrics": dict(best_metrics or {}),
            "config": {
                **copy.deepcopy(config),
                "model": dict(model_cfg),
                "critic_pretrain": vars(args),
            },
            "critic_pretrain": True,
            "source": "v4_independent_critic_supervised_value_pretrain",
        },
        str(path),
    )


def maybe_resume(latest_file, critic, optimizer, device, enabled):
    if not enabled or not latest_file.exists():
        return 0, 0, None
    state = torch.load(str(latest_file), map_location=device, weights_only=False)
    if "critic" not in state:
        raise RuntimeError(f"checkpoint has no critic weights: {latest_file}")
    critic.load_state_dict(state["critic"], strict=False)
    if "critic_optimizer" in state:
        optimizer.load_state_dict(state["critic_optimizer"])
    steps = int(state.get("steps", 0))
    epoch = int(state.get("epoch", 0))
    best_metrics = state.get("best_metrics")
    logging.info("resumed critic pretrain from %s at step=%s epoch=%s", latest_file, steps, epoch)
    return steps, epoch, best_metrics


def run_training(args):
    if not args.init_checkpoint:
        raise RuntimeError("missing --init-checkpoint and [critic_pretrain].init_checkpoint")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device(str(args.device))
    torch.backends.cudnn.benchmark = bool(config.get("control", {}).get("enable_cudnn_benchmark", False))

    dataset_cfg, control_cfg, model_cfg = build_training_configs(args)
    splits = build_data_splits(dataset_cfg)
    for split_name, files in splits.items():
        logging.info("%s files: %s", split_name, f"{len(files):,}")
    if not splits["train"]:
        raise RuntimeError("train split is empty")
    if not splits["val"]:
        raise RuntimeError("val split is empty; set val_ratio > 0 or provide val_globs")

    logging.info(
        "augmentation_factor=%s override_source_augmentation=%s public=%s private=%s",
        dataset_cfg.get("augmentation_factor"),
        bool(args.override_source_augmentation),
        dataset_cfg.get("public_augmentation_factor"),
        dataset_cfg.get("private_augmentation_factor"),
    )
    train_loader = build_loader(splits["train"], dataset_cfg, control_cfg, training=True)
    val_loader = build_loader(splits["val"], dataset_cfg, control_cfg, training=False)

    init_checkpoint = resolve_path(args.init_checkpoint)
    actor, critic, clean_model_cfg = load_actor_and_critic_from_checkpoint(
        init_checkpoint,
        device=device,
        model_cfg=model_cfg,
        critic_feature_dim=int(args.critic_feature_dim),
        critic_feature_hidden=int(args.critic_feature_hidden),
        strict_actor=bool(args.strict_actor),
    )
    actor.eval()
    for param in actor.parameters():
        param.requires_grad_(False)
    assert_independent_modules(actor, critic)
    if bool(args.zero_critic_feature_proj) and int(args.critic_feature_dim) > 0:
        if zero_critic_feature_projection(critic):
            logging.info("zeroed critic feature projection; supervised batches provide no extra critic features")

    logging.info("init_checkpoint=%s", init_checkpoint)
    logging.info("critic params=%s", f"{parameter_count(critic):,}")
    logging.info(
        "optimizer=%s lr=%s weight_decay=%s loss=%s target_scale=%s target_clip=%s",
        args.optimizer,
        args.lr,
        args.weight_decay,
        args.loss,
        args.target_scale,
        args.target_clip,
    )
    logging.info("critic value target config=%s", dataset_cfg.get("value_target_config", {}))

    optim_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optim_cls(
        critic.parameters(),
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

    latest_file = ckpt_dir / "gbmj_critic_latest.pth"
    best_file = ckpt_dir / "gbmj_critic_best.pth"
    steps, start_epoch, best_metrics = maybe_resume(
        latest_file,
        critic,
        optimizer,
        device,
        enabled=bool(args.resume),
    )

    eval_max_batches = max(0, int(args.eval_max_batches))
    max_train_batches = max(0, int(args.max_train_batches))
    best_val_loss = float(best_metrics.get("loss", "inf")) if isinstance(best_metrics, dict) else float("inf")

    for epoch in range(start_epoch, int(args.epochs)):
        epoch_start = time.time()
        critic.train()
        log_running = CriticRunningAverages()
        epoch_running = CriticRunningAverages()
        train_iter = tqdm(
            train_loader,
            desc=f"CRITIC epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(train_iter):
            if max_train_batches and batch_idx >= max_train_batches:
                break
            loss, metrics = compute_critic_loss(
                critic,
                batch,
                device,
                enable_amp=bool(args.enable_amp),
                args=args,
            )
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"non-finite critic loss at step {steps + 1}: {float(loss.detach().cpu())}")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(args.max_grad_norm) > 0.0:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(critic.parameters(), float(args.max_grad_norm))
                if not torch.isfinite(grad_norm):
                    logging.warning(
                        "skip non-finite critic gradient at step %s: grad_norm=%s",
                        steps + 1,
                        float(grad_norm.detach().cpu()),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            log_running.update(metrics)
            epoch_running.update(metrics)
            if int(args.log_every) > 0 and steps % int(args.log_every) == 0:
                train_metrics = log_running.pop()
                write_metrics(writer, "critic_train", train_metrics, steps)
                writer.flush()
                train_iter.set_postfix(
                    loss=f"{train_metrics['loss']:.4f}",
                    mae=f"{train_metrics['mae']:.4f}",
                    nz=f"{train_metrics['nonzero_target_rate']:.3f}",
                )

            if int(args.save_every) > 0 and steps % int(args.save_every) == 0:
                save_critic_checkpoint(
                    latest_file,
                    actor,
                    critic,
                    optimizer,
                    steps,
                    epoch,
                    best_metrics,
                    args,
                    clean_model_cfg,
                )

            if int(args.eval_every) > 0 and steps % int(args.eval_every) == 0:
                val_metrics = evaluate_critic(
                    critic,
                    val_loader,
                    device,
                    args,
                    max_batches=eval_max_batches,
                )
                write_metrics(writer, "critic_val", val_metrics, steps)
                writer.flush()

        train_metrics = epoch_running.compute()
        write_metrics(writer, "critic_train_epoch", train_metrics, steps)
        val_metrics = evaluate_critic(
            critic,
            val_loader,
            device,
            args,
            max_batches=eval_max_batches,
        )
        write_metrics(writer, "critic_val", val_metrics, steps)
        writer.flush()

        elapsed = time.time() - epoch_start
        logging.info(
            "epoch=%s step=%s train_loss=%.6f train_mae=%.6f val_loss=%.6f val_mae=%.6f ev=%.4f elapsed=%.1fs",
            epoch + 1,
            steps,
            train_metrics["loss"],
            train_metrics["mae"],
            val_metrics["loss"],
            val_metrics["mae"],
            val_metrics["explained_variance"],
            elapsed,
        )

        latest_metrics = dict(val_metrics)
        save_critic_checkpoint(
            latest_file,
            actor,
            critic,
            optimizer,
            steps,
            epoch + 1,
            latest_metrics,
            args,
            clean_model_cfg,
        )
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_metrics = dict(val_metrics)
            save_critic_checkpoint(
                best_file,
                actor,
                critic,
                optimizer,
                steps,
                epoch + 1,
                best_metrics,
                args,
                clean_model_cfg,
            )
            logging.info("saved best critic checkpoint: %s", best_file)

    writer.close()
    logging.info("critic pretraining done. latest=%s best=%s", latest_file, best_file)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
