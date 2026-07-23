import argparse
import logging
import sys
from pathlib import Path

# [V4 MJ_RM strict DPPO] Allow both launch styles:
#   python -m rl.train_ppo
#   python rl/train_ppo.py
PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from supervised.config import PROJECT_ROOT, config
from supervised.policy import SupervisedEngine
from rl.model import (
    load_actor_and_critic_from_checkpoint,
    load_actor_from_checkpoint,
    save_actor_checkpoint,
    save_ppo_checkpoint,
)
from rl.grp_model import apply_grp_potential_rewards, load_grp_checkpoint
from rl.policy import PPOEngine, categorical_from_logits
from rl.reward import CRITIC_FEATURE_DIM, MJRMRewardShaper, reward_component_stats
from rl.rollout import collect_one_vs_three


def resolve_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def first_nonfinite_parameter(model):
    """[V4 RL NaN guard] Return the first non-finite parameter name, if any."""
    for name, param in model.named_parameters():
        if not torch.isfinite(param.detach()).all():
            return name
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and not torch.isfinite(buf.detach()).all():
            return name
    return None


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _batch_slice(batch, key, idx, device):
    value = batch[key][idx]
    if value.device != device:
        value = value.to(device, non_blocking=True)
    return value


def _transition_critic_features(transitions, critic_feature_dim):
    """[V4 asymmetric critic] Stack rollout-only critic feature vectors."""
    if critic_feature_dim <= 0:
        return np.zeros((len(transitions), 0), dtype=np.float32)
    values = []
    zero = np.zeros((critic_feature_dim,), dtype=np.float32)
    for transition in transitions:
        feature = transition.get("critic_features")
        if feature is None:
            values.append(zero)
            continue
        arr = np.asarray(feature, dtype=np.float32).reshape(-1)
        if arr.shape[0] != critic_feature_dim:
            fixed = zero.copy()
            copy_len = min(critic_feature_dim, arr.shape[0])
            fixed[:copy_len] = arr[:copy_len]
            arr = fixed
        values.append(arr)
    return np.stack(values, axis=0).astype(np.float32, copy=False)


def refresh_transition_values(critic_model, transitions, device, critic_feature_dim, batch_size=4096, enable_amp=True):
    """[V4 asymmetric critic] Recompute V(s) with critic-only features.

    Rollout action sampling does not need critic extras, but GAE/TD targets do.
    Refreshing values here keeps PPO advantages aligned with the asymmetric
    critic while preserving the actor's deployment-time observation contract.
    """
    if not transitions:
        return
    critic_model.eval()
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(transitions), batch_size):
            chunk = transitions[start:start + batch_size]
            obs = torch.as_tensor(
                np.stack([t["obs"] for t in chunk], axis=0),
                dtype=torch.float32,
                device=device,
            )
            # [V4 history-hierarchical] Recompute V(s) from the same K-step
            # history used during rollout, otherwise GAE is evaluating a
            # different state representation than the actor sampled from.
            history_obs = torch.as_tensor(
                np.stack([t["history_obs"] for t in chunk], axis=0),
                dtype=torch.float32,
                device=device,
            )
            history_actions = torch.as_tensor(
                np.stack([t["history_actions"] for t in chunk], axis=0),
                dtype=torch.long,
                device=device,
            )
            critic_features = torch.as_tensor(
                _transition_critic_features(chunk, critic_feature_dim),
                dtype=torch.float32,
                device=device,
            )
            if device.type == "cuda":
                with torch.autocast(device.type, enabled=bool(enable_amp)):
                    values = critic_model(
                        obs,
                        history_obs=history_obs,
                        history_actions=history_actions,
                        critic_features=critic_features,
                    )
            else:
                values = critic_model(
                    obs,
                    history_obs=history_obs,
                    history_actions=history_actions,
                    critic_features=critic_features,
                )
            for transition, value in zip(chunk, values.detach().float().cpu().tolist()):
                transition["value"] = float(value)


def freeze_actor_backbone(actor_model):
    """Freeze the v3 policy representation and train only the policy head."""
    trainable = 0
    frozen = 0
    for name, param in actor_model.named_parameters():
        if name.startswith("head."):
            param.requires_grad_(True)
            trainable += int(param.numel())
        else:
            param.requires_grad_(False)
            frozen += int(param.numel())
    return trainable, frozen


def tensor_batch(transitions, device, gamma, gae_lambda, critic_feature_dim=0, advantage_mode="returns"):
    rewards = np.asarray([t["reward"] for t in transitions], dtype=np.float32)
    dones = np.asarray([t["done"] for t in transitions], dtype=np.float32)
    values = np.asarray([t["value"] for t in transitions], dtype=np.float32)
    critic_features = _transition_critic_features(transitions, int(critic_feature_dim))

    # [V4 stable PPO] Reward-to-go is used only for critic warmup.  The actor
    # is initialized from supervised learning, but the value head is newly
    # created, so it needs a short calibration stage before PPO trusts V(s).
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        returns[t] = running_return

    # [V4 deal-in-control PPO] Keep TD targets for ablations, but the default
    # advantage/value target below uses full hand returns.  A freshly attached
    # critic bootstrapping from its own untrained V(s_next) was too weak for
    # sparse win/deal-in signals.
    td_targets = np.zeros_like(rewards, dtype=np.float32)
    gae_advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_nonterminal = 1.0 - dones[t]
        next_value = values[t + 1] if (t + 1) < len(values) and next_nonterminal > 0.0 else 0.0
        td_targets[t] = rewards[t] + gamma * next_value * next_nonterminal
        delta = td_targets[t] - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        gae_advantages[t] = last_gae

    if str(advantage_mode).lower() in ("return", "returns", "mc", "monte_carlo"):
        advantages = returns - values
    else:
        advantages = gae_advantages

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

    return {
        "obs": torch.as_tensor(np.stack([t["obs"] for t in transitions], axis=0), dtype=torch.float32, device=device),
        # [V4 history-hierarchical] PPO update keeps the same state definition
        # as supervised training: current observation plus last K own decisions.
        "history_obs": torch.as_tensor(
            np.stack([t["history_obs"] for t in transitions], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "history_actions": torch.as_tensor(
            np.stack([t["history_actions"] for t in transitions], axis=0),
            dtype=torch.long,
            device=device,
        ),
        "masks": torch.as_tensor(np.stack([t["mask"] for t in transitions], axis=0), dtype=torch.bool, device=device),
        "actions": torch.as_tensor([t["action"] for t in transitions], dtype=torch.long, device=device),
        "old_log_probs": torch.as_tensor([t["old_log_prob"] for t in transitions], dtype=torch.float32, device=device),
        "critic_features": torch.as_tensor(critic_features, dtype=torch.float32, device=device),
        "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
        "td_targets": torch.as_tensor(td_targets, dtype=torch.float32, device=device),
        "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=device),
        # [V4 potential reward] Kept for backward compatibility.  The current
        # config disables ordinary-shanten adaptive loss, so this remains zero.
        "shanten_delta": torch.as_tensor(
            [float(t.get("reward_components", {}).get("shanten_delta", 0.0)) for t in transitions],
            dtype=torch.float32,
            device=device,
        ),
    }


def critic_warmup_update(critic_model, critic_optimizer, batch, cfg):
    """[V4 independent critic] Fit critic before actor PPO updates.

    The actor starts from supervised learning; the critic is separate and has
    no supervised value label.  Warmup trains only the critic network so early
    advantages do not come from an uncalibrated value function.
    """
    critic_model.train()
    device = _model_device(critic_model)

    n = int(batch["actions"].shape[0])
    minibatch = min(int(cfg.get("critic_warmup_minibatch_size", cfg["minibatch_size"])), n)
    epochs = int(cfg.get("critic_warmup_epochs", 3))
    value_coef = float(cfg["value_coef"])
    max_grad_norm = float(cfg["max_grad_norm"])
    target_key = "returns" if str(cfg.get("value_target", "returns")).lower() in ("return", "returns", "mc", "monte_carlo") else "td_targets"

    totals = {"loss": 0.0, "value_loss": 0.0, "updates": 0}
    for _ in range(epochs):
        perm = torch.randperm(n, device=batch["actions"].device)
        for start in range(0, n, minibatch):
            idx = perm[start:start + minibatch]
            values = critic_model(
                _batch_slice(batch, "obs", idx, device),
                history_obs=_batch_slice(batch, "history_obs", idx, device),
                history_actions=_batch_slice(batch, "history_actions", idx, device),
                critic_features=_batch_slice(batch, "critic_features", idx, device),
            )
            # [V4 deal-in-control PPO] Fit the critic directly to full hand
            # returns during warmup, so early advantages are not dominated by
            # a bootstrapped value head with no supervised labels.
            value_loss = F.mse_loss(values, _batch_slice(batch, target_key, idx, device))
            loss = value_coef * value_loss

            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_grad_norm)
            critic_optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["value_loss"] += float(value_loss.detach().cpu())
            totals["updates"] += 1

    denom = max(1, totals.pop("updates"))
    return {
        "loss": totals["loss"] / denom,
        "policy_loss": 0.0,
        "value_loss": totals["value_loss"] / denom,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "signed_kl": 0.0,
        "clipfrac": 0.0,
        "early_stop": 0.0,
        "shanten_loss": 0.0,
        "critic_warmup": 1.0,
    }


def build_optimizers(actor_model, critic_model, cfg):
    """[V4 independent critic] Separate actor and critic optimizers."""
    policy_lr = float(cfg.get("policy_lr", cfg.get("lr", 1.0e-5)))
    value_lr = float(cfg.get("value_lr", policy_lr))
    optim_name = str(cfg.get("optimizer", "adam")).lower()
    optim_cls = torch.optim.AdamW if optim_name == "adamw" else torch.optim.Adam
    common = dict(
        betas=tuple(float(x) for x in cfg.get("betas", [0.9, 0.999])),
        eps=float(cfg.get("eps", 1.0e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    actor_params = [param for param in actor_model.parameters() if param.requires_grad]
    critic_params = [param for param in critic_model.parameters() if param.requires_grad]
    if not actor_params:
        raise RuntimeError("actor optimizer has no trainable parameters")
    if not critic_params:
        raise RuntimeError("critic optimizer has no trainable parameters")
    actor_optimizer = optim_cls(actor_params, lr=policy_lr, **common)
    critic_optimizer = optim_cls(critic_params, lr=value_lr, **common)
    return actor_optimizer, critic_optimizer


def ppo_update(actor_model, critic_model, actor_optimizer, critic_optimizer, batch, cfg, ref_model=None):
    """[V4 independent critic] Central learner update after actor rollout merge."""
    # PPO ratios compare learner log_prob against rollout old_log_prob.  Rollout
    # sampling uses actor.eval(), so dropout must stay disabled here too;
    # otherwise the "old vs new" KL is non-zero before any parameter update and
    # the first minibatch can be rejected forever by the KL early-stop gate.
    actor_model.eval()
    critic_model.train()
    if ref_model is not None:
        ref_model.eval()
    device = _model_device(actor_model)

    n = int(batch["actions"].shape[0])
    minibatch = min(int(cfg["minibatch_size"]), n)
    update_epochs = int(cfg["update_epochs"])
    clip_range = float(cfg["clip_range"])
    value_coef = float(cfg["value_coef"])
    entropy_coef = float(cfg["entropy_coef"])
    kl_coef = float(cfg.get("kl_coef", 0.0))
    max_grad_norm = float(cfg["max_grad_norm"])
    target_approx_kl = float(cfg.get("target_approx_kl", 0.0))
    shanten_loss_beta = float(cfg.get("shanten_loss_beta", 0.0))
    enable_shanten_loss = bool(cfg.get("enable_shanten_adaptive_loss", shanten_loss_beta != 0.0))
    enable_ref_kl = bool(cfg.get("enable_ref_kl", False)) and ref_model is not None
    ref_kl_coef = float(cfg.get("ref_kl_coef", 0.0))
    logprob_temperature = max(
        float(cfg.get("ppo_logprob_temperature", cfg.get("actor_temperature", 1.0))),
        1.0e-6,
    )
    target_key = "returns" if str(cfg.get("value_target", "returns")).lower() in ("return", "returns", "mc", "monte_carlo") else "td_targets"

    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "signed_kl": 0.0,
        "clipfrac": 0.0,
        "applied_update": 0.0,
        "updates": 0,
        "early_stop": 0.0,
        "shanten_loss": 0.0,
        "ref_kl": 0.0,
    }

    stop_update = False
    for _ in range(update_epochs):
        if stop_update:
            break
        perm = torch.randperm(n, device=batch["actions"].device)
        for start in range(0, n, minibatch):
            idx = perm[start:start + minibatch]
            obs = _batch_slice(batch, "obs", idx, device)
            history_obs = _batch_slice(batch, "history_obs", idx, device)
            history_actions = _batch_slice(batch, "history_actions", idx, device)
            masks = _batch_slice(batch, "masks", idx, device)
            critic_features = _batch_slice(batch, "critic_features", idx, device)
            actions = _batch_slice(batch, "actions", idx, device)
            old_log_probs = _batch_slice(batch, "old_log_probs", idx, device)
            value_targets = _batch_slice(batch, target_key, idx, device)
            advantages = _batch_slice(batch, "advantages", idx, device)
            shanten_delta = _batch_slice(batch, "shanten_delta", idx, device)

            logits = actor_model(
                obs,
                history_obs=history_obs,
                history_actions=history_actions,
            )
            # [V4 potential reward] PPO ratio must use the same action
            # distribution as rollout old_log_prob.  PPOEngine samples from
            # logits / actor_temperature, so the learner must do the same;
            # otherwise approx_kl/clipfrac are measuring a fake policy shift.
            dist, _ = categorical_from_logits(logits / logprob_temperature, masks)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ref_kl = torch.zeros((), dtype=logits.dtype, device=logits.device)

            log_ratio = log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            pg1 = -advantages * ratio
            pg2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = torch.max(pg1, pg2).mean()
            with torch.no_grad():
                # [V4 RL framework audit] Use Schulman's positive approximate
                # KL for PPO early-stop.  The old signed mean(old-new) can be
                # noisy or misleading on minibatches, especially after masking.
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                signed_kl = (-log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_range).float().mean()

            if target_approx_kl > 0.0 and float(approx_kl.detach().cpu()) > target_approx_kl:
                # [V4 stable PPO] Do not apply the minibatch that already
                # violates the trust-region budget.  The previous version
                # checked only after optimizer.step(), so every iteration could
                # still push one extra high-KL update before stopping.
                # Record the rejected minibatch diagnostics so TensorBoard does
                # not misleadingly show KL/loss as clean zeros when no update
                # was actually completed.
                if totals["updates"] <= 0:
                    totals["policy_loss"] += float(policy_loss.detach().cpu())
                    totals["entropy"] += float(entropy.detach().cpu())
                    totals["approx_kl"] += float(approx_kl.detach().cpu())
                    totals["signed_kl"] += float(signed_kl.detach().cpu())
                    totals["clipfrac"] += float(clipfrac.detach().cpu())
                totals["early_stop"] = 1.0
                stop_update = True
                break

            # [V4 potential reward] Legacy Eq. (4) hook.  Disabled by default
            # because dense guidance now comes from main-fan potential reward.
            #
            # [V4 MJ_RM formula alignment] Eq. (4) adds
            # beta * sign(Delta S) * |A_t|.  In practical PPO, A_t and Delta S
            # are fixed rollout quantities; multiplying by the current ratio
            # makes the term affect the policy instead of becoming a constant.
            if enable_shanten_loss:
                shanten_weight = shanten_loss_beta * torch.sign(shanten_delta) * advantages.detach().abs()
                shanten_loss = (shanten_weight * ratio).mean()
            else:
                shanten_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)

            # [V4 stable PPO] Adaptive KL penalty complements clipping.  It is
            # normally small, and is adjusted between rollout iterations.
            kl_loss = ((ratio - 1.0) - log_ratio).mean()
            if enable_ref_kl and ref_kl_coef > 0.0:
                # [V4 SL reference KL] Keep RL fine-tuning close to the
                # supervised policy that already plays well. PPO old/new KL
                # limits one update; this term prevents long-horizon drift.
                with torch.no_grad():
                    ref_logits = ref_model(
                        obs,
                        history_obs=history_obs,
                        history_actions=history_actions,
                    )
                _, new_masked_logits = categorical_from_logits(logits / logprob_temperature, masks)
                _, ref_masked_logits = categorical_from_logits(ref_logits / logprob_temperature, masks)
                new_logp = F.log_softmax(new_masked_logits.float(), dim=-1)
                ref_logp = F.log_softmax(ref_masked_logits.float(), dim=-1)
                new_prob = new_logp.exp()
                ref_kl = (new_prob * (new_logp - ref_logp)).sum(dim=-1).mean()

            # [V4 independent critic] Split actor and critic objectives.  The
            # logged total keeps the paper-style combined scalar, but backward
            # is intentionally separated to avoid value loss damaging policy.
            actor_loss = (
                policy_loss
                - entropy_coef * entropy
                + kl_coef * kl_loss
                + ref_kl_coef * ref_kl
                + shanten_loss
            )
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_grad_norm)
            actor_optimizer.step()

            values = critic_model(
                obs,
                history_obs=history_obs,
                history_actions=history_actions,
                critic_features=critic_features,
            )
            # [V4 independent critic] L_value is optimized only on the critic
            # network; actor gradients never pass through this value loss.
            value_loss = F.mse_loss(values, value_targets)
            critic_loss = value_coef * value_loss

            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_grad_norm)
            critic_optimizer.step()

            totals["loss"] += float(actor_loss.detach().cpu()) + float(critic_loss.detach().cpu())
            totals["policy_loss"] += float(policy_loss.detach().cpu())
            totals["value_loss"] += float(value_loss.detach().cpu())
            totals["entropy"] += float(entropy.detach().cpu())
            totals["approx_kl"] += float(approx_kl.detach().cpu())
            totals["signed_kl"] += float(signed_kl.detach().cpu())
            totals["clipfrac"] += float(clipfrac.detach().cpu())
            totals["shanten_loss"] += float(shanten_loss.detach().cpu())
            totals["ref_kl"] += float(ref_kl.detach().cpu())
            totals["updates"] += 1
            totals["applied_update"] += 1.0

    denom = max(1, totals.pop("updates"))
    early_stop = float(totals.pop("early_stop"))
    averaged = {key: value / denom for key, value in totals.items()}
    # [V4 RL framework audit] Keep early_stop as a boolean-like flag instead of
    # dividing it by the number of completed minibatch updates.
    averaged["early_stop"] = early_stop
    averaged["critic_warmup"] = 0.0
    return averaged


def adapt_regularizers(rl_cfg, train_stats):
    """[V4 stable PPO] Adaptive entropy and KL controller.

    This matches the paper's stabilizing intent without adding Redis/model-pool
    infrastructure: entropy avoids premature collapse, KL penalty becomes
    stronger only when the policy update is too large.
    """
    if bool(rl_cfg.get("adaptive_entropy", True)):
        entropy = float(train_stats.get("entropy", 0.0))
        target = float(rl_cfg.get("entropy_target", 0.20))
        rate = float(rl_cfg.get("entropy_adjust_rate", 1.05))
        coef = float(rl_cfg.get("entropy_coef", 0.0))
        if entropy > 0.0 and target > 0.0:
            if entropy < target:
                coef *= rate
            elif entropy > target * 1.5:
                coef /= rate
            coef = min(
                float(rl_cfg.get("entropy_coef_max", 0.003)),
                max(float(rl_cfg.get("entropy_coef_min", 0.0003)), coef),
            )
            rl_cfg["entropy_coef"] = coef

    if bool(rl_cfg.get("adaptive_kl", True)):
        approx_kl = float(train_stats.get("approx_kl", 0.0))
        target = float(rl_cfg.get("target_approx_kl", 0.012))
        rate = float(rl_cfg.get("kl_coef_adjust_rate", 1.5))
        coef = float(rl_cfg.get("kl_coef", 0.0))
        if approx_kl > 0.0 and target > 0.0:
            if approx_kl > target * 1.5:
                coef = coef * rate if coef > 0.0 else float(rl_cfg.get("kl_coef_min", 1.0e-4))
            elif approx_kl < target / 1.5:
                coef /= rate
            coef = min(float(rl_cfg.get("kl_coef_max", 0.05)), max(0.0, coef))
            rl_cfg["kl_coef"] = coef

    if bool(rl_cfg.get("enable_ref_kl", False)) and bool(rl_cfg.get("adaptive_ref_kl", True)):
        # [V4 SL reference KL] Adaptive long-term anchor to the supervised
        # initialization. This is intentionally separate from PPO old/new KL.
        ref_kl = float(train_stats.get("ref_kl", 0.0))
        target = float(rl_cfg.get("ref_kl_target", 0.03))
        rate = float(rl_cfg.get("ref_kl_adjust_rate", 1.5))
        coef = float(rl_cfg.get("ref_kl_coef", 0.0))
        if ref_kl > 0.0 and target > 0.0:
            if ref_kl > target * 1.5:
                coef = coef * rate if coef > 0.0 else float(rl_cfg.get("ref_kl_coef_min", 0.002))
            elif ref_kl < target / 1.5:
                coef /= rate
            coef = min(
                float(rl_cfg.get("ref_kl_coef_max", 0.20)),
                max(float(rl_cfg.get("ref_kl_coef_min", 0.002)), coef),
            )
            rl_cfg["ref_kl_coef"] = coef


def save_all(ckpt_dir, actor_model, critic_model, actor_optimizer, critic_optimizer, iteration, cfg, stats, best=False):
    # [V4 stable PPO] Preserve runtime-adjusted RL coefficients such as
    # adaptive entropy/KL when saving checkpoints.
    save_config = dict(config)
    save_config["rl"] = dict(cfg)
    save_ppo_checkpoint(
        ckpt_dir / "ppo_latest.pth",
        actor_model,
        critic_model,
        actor_optimizer,
        critic_optimizer,
        iteration,
        save_config,
        stats,
    )
    save_actor_checkpoint(ckpt_dir / "gbmj_policy_latest.pth", actor_model, iteration, save_config, stats)
    if best:
        save_ppo_checkpoint(
            ckpt_dir / "ppo_best.pth",
            actor_model,
            critic_model,
            actor_optimizer,
            critic_optimizer,
            iteration,
            save_config,
            stats,
        )
        save_actor_checkpoint(ckpt_dir / "gbmj_policy_best.pth", actor_model, iteration, save_config, stats)


def _weighted_merge(stats_list):
    # [V4 rank-aligned PPO] Merge hand diagnostics by hand count and full-game
    # rank metrics by game count so best_metric can target first/fourth rates.
    total_rounds = max(1.0, sum(float(s.get("rounds", 0)) for s in stats_list))
    total_ranked_rounds = max(1.0, sum(float(s.get("round_ranked", 0)) for s in stats_list))
    total_games = max(1.0, sum(float(s.get("games", 0)) for s in stats_list))
    merged = {
        "games": int(sum(int(s.get("games", 0)) for s in stats_list)),
        "rounds": int(sum(int(s.get("rounds", 0)) for s in stats_list)),
        "round_ranked": int(sum(int(s.get("round_ranked", 0)) for s in stats_list)),
        "transitions": int(sum(int(s.get("transitions", 0)) for s in stats_list)),
        "actor_shards": int(len(stats_list)),
    }
    for key in ("avg_score_delta", "win_rate", "zimo_rate", "ron_win_rate", "deal_in_rate", "other_zimo_rate", "other_ron_rate", "draw_rate", "outcome_utility"):
        merged[key] = float(sum(float(s.get(key, 0.0)) * float(s.get("rounds", 0)) for s in stats_list) / total_rounds)
    for key in ("round_rank_1", "round_rank_2", "round_rank_3", "round_rank_4", "avg_round_rank", "round_rank_pt", "round_rank_edge"):
        merged[key] = float(sum(float(s.get(key, 0.0)) * float(s.get("round_ranked", 0)) for s in stats_list) / total_ranked_rounds)
    for key in ("game_rank_1", "game_rank_2", "game_rank_3", "game_rank_4", "avg_game_rank", "game_rank_pt", "game_rank_edge"):
        merged[key] = float(sum(float(s.get(key, 0.0)) * float(s.get("games", 0)) for s in stats_list) / total_games)
    return merged


def collect_dppo_rollouts(actor_model, baseline_engine, rl_cfg, iteration, reward_shaper, device):
    """[V4 MJ_RM strict DPPO] Multi-actor rollout aggregation.

    This is DPPO-style actor/learner separation inside one process: each actor
    shard collects fresh games, then the learner aggregates all transitions and
    performs one PPO update. It can be split into real processes later.
    """
    actor_count = int(rl_cfg.get("dppo_num_actors", 8))
    seed_count = int(rl_cfg.get("rollout_seed_count", 2))
    seed_start = int(rl_cfg["seed_start"])
    seed_key = int(str(rl_cfg.get("seed_key", "0x20260514")), 0)
    game_length = int(rl_cfg.get("game_length", 16))
    rank_bonus = [float(x) for x in rl_cfg.get("rank_bonus", [0.0, 0.0, 0.0, 0.0])]
    rank_edge_fourth_penalty = float(rl_cfg.get("rank_edge_fourth_penalty", 1.25))

    all_transitions = []
    stats_list = []
    for actor_idx in range(actor_count):
        shard_seed = seed_start + (iteration * actor_count + actor_idx) * seed_count
        actor_engine = PPOEngine(
            actor_model,
            device=device,
            enable_amp=bool(rl_cfg.get("enable_amp_rollout", True)),
            enable_quick_eval=bool(rl_cfg.get("enable_quick_eval", True)),
            enable_rule_based_agari_guard=bool(rl_cfg.get("enable_rule_based_agari_guard", True)),
            temperature=float(rl_cfg.get("actor_temperature", 1.0)),
            name="mortal_actor_%02d" % actor_idx,
        )
        transitions, stats = collect_one_vs_three(
            actor_engine=actor_engine,
            baseline_engine=baseline_engine,
            seed_start=(shard_seed, seed_key),
            seed_count=seed_count,
            rank_bonus=rank_bonus,
            game_length=game_length,
            rank_edge_fourth_penalty=rank_edge_fourth_penalty,
            disable_progress_bar=bool(rl_cfg.get("disable_progress_bar", False)),
            reward_shaper=reward_shaper,
        )
        all_transitions.extend(transitions)
        stats_list.append(stats)
    return all_transitions, _weighted_merge(stats_list)


def run_training():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rl_cfg = dict(config["rl"])
    device = torch.device(str(rl_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    ckpt_dir = resolve_path(rl_cfg["checkpoint_dir"])
    log_dir = resolve_path(rl_cfg["tensorboard_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    latest_ppo = ckpt_dir / "ppo_latest.pth"
    init_checkpoint = resolve_path(rl_cfg["init_checkpoint"])
    baseline_checkpoint = resolve_path(rl_cfg.get("baseline_checkpoint", rl_cfg["init_checkpoint"]))

    resume = bool(rl_cfg.get("resume", True)) and latest_ppo.exists()
    load_path = latest_ppo if resume else init_checkpoint
    # [V4 asymmetric critic] Build the actor from the supervised model config,
    # then attach critic-only route-potential feature inputs for RL value
    # learning.  Actor-only exports strip these critic modules.
    critic_feature_dim = 0
    if bool(rl_cfg.get("enable_asymmetric_critic", True)):
        critic_feature_dim = int(rl_cfg.get("critic_feature_dim", CRITIC_FEATURE_DIM))
    rl_cfg["critic_feature_dim"] = critic_feature_dim
    model_cfg = dict(config.get("model", {}))
    critic_feature_hidden = int(rl_cfg.get("critic_feature_hidden", 64))
    actor_model, critic_model, _ = load_actor_and_critic_from_checkpoint(
        load_path,
        device=device,
        model_cfg=model_cfg,
        critic_feature_dim=critic_feature_dim,
        critic_feature_hidden=critic_feature_hidden,
        strict_actor=not resume,
    )
    bad_actor_param = first_nonfinite_parameter(actor_model)
    bad_critic_param = first_nonfinite_parameter(critic_model)
    if resume and (bad_actor_param or bad_critic_param):
        # [V4 RL NaN guard] A previous failed PPO run may have saved a poisoned
        # ppo_latest.pth.  Do not resume from it; restart from the supervised
        # policy checkpoint and a fresh independent critic instead.
        logging.warning(
            "ignored non-finite PPO checkpoint %s: actor=%s critic=%s; restarting from %s",
            latest_ppo,
            bad_actor_param,
            bad_critic_param,
            init_checkpoint,
        )
        resume = False
        load_path = init_checkpoint
        actor_model, critic_model, _ = load_actor_and_critic_from_checkpoint(
            load_path,
            device=device,
            model_cfg=model_cfg,
            critic_feature_dim=critic_feature_dim,
            critic_feature_hidden=critic_feature_hidden,
            strict_actor=True,
        )
        bad_actor_param = first_nonfinite_parameter(actor_model)
        bad_critic_param = first_nonfinite_parameter(critic_model)
    if bad_actor_param or bad_critic_param:
        raise RuntimeError(
            "loaded non-finite PPO model: actor=%s critic=%s checkpoint=%s"
            % (bad_actor_param, bad_critic_param, load_path)
        )
    if bool(rl_cfg.get("freeze_actor_backbone", False)):
        trainable, frozen = freeze_actor_backbone(actor_model)
        logging.info(
            "froze actor backbone for GRP PPO: trainable_head_params=%s frozen_params=%s",
            f"{trainable:,}",
            f"{frozen:,}",
        )
    logging.info(
        "independent critic: enabled=%s feature_dim=%s feature_hidden=%s",
        bool(critic_feature_dim > 0),
        critic_feature_dim,
        critic_feature_hidden,
    )
    actor_optimizer, critic_optimizer = build_optimizers(actor_model, critic_model, rl_cfg)

    start_iter = 0
    resume_stats = {}
    best_metric_name = str(rl_cfg.get("best_metric", "avg_score_delta"))
    best_metric_mode = str(rl_cfg.get("best_metric_mode", "max")).lower()
    best_metric_value = float("-inf") if best_metric_mode == "max" else float("inf")
    if resume:
        state = torch.load(str(latest_ppo), map_location=device)
        if "actor_optimizer" in state:
            actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            critic_optimizer.load_state_dict(state["critic_optimizer"])
        start_iter = int(state.get("iteration", 0))
        resume_stats = state.get("stats", {})
        best_metric_value = float(resume_stats.get("best_metric_value", best_metric_value))
        # [V4 stable PPO] Restore adaptive controller state on resume.
        saved_rl_cfg = state.get("config", {}).get("rl", {}) if isinstance(state.get("config", {}), dict) else {}
        for key in ("entropy_coef", "kl_coef", "ref_kl_coef"):
            if key in saved_rl_cfg:
                rl_cfg[key] = saved_rl_cfg[key]
        logging.info("resumed GRP PPO checkpoint: %s at iteration %s", latest_ppo, start_iter)
    else:
        logging.info("initialized GRP PPO actor from supervised checkpoint: %s", init_checkpoint)

    ref_model = None
    if bool(rl_cfg.get("enable_ref_kl", False)):
        # [V4 SL reference KL] Frozen supervised anchor.  Rollout and learner
        # are still allowed to improve, but this keeps fine-tuning from drifting
        # far away from the strong SL policy under noisy sparse rewards.
        ref_model, _ = load_actor_from_checkpoint(
            init_checkpoint,
            device=device,
            model_cfg=model_cfg,
            strict_actor=True,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
        logging.info(
            "enabled SL reference KL: checkpoint=%s target=%s coef=%s",
            init_checkpoint,
            rl_cfg.get("ref_kl_target", 0.03),
            rl_cfg.get("ref_kl_coef", 0.0),
        )

    baseline_engine = SupervisedEngine.from_checkpoint(
        checkpoint=str(baseline_checkpoint),
        device=str(device),
        name="baseline",
        enable_amp=bool(rl_cfg.get("enable_amp_rollout", True)),
        enable_quick_eval=bool(rl_cfg.get("enable_quick_eval", True)),
        enable_rule_based_agari_guard=bool(rl_cfg.get("enable_rule_based_agari_guard", True)),
        deterministic=bool(rl_cfg.get("baseline_deterministic", True)),
    )

    reward_shaper = MJRMRewardShaper.from_config(rl_cfg)
    grp_model = None
    if bool(rl_cfg.get("enable_grp_reward", False)):
        grp_checkpoint = resolve_path(rl_cfg.get("grp_checkpoint", ""))
        if not grp_checkpoint.exists():
            raise RuntimeError(
                "enable_grp_reward=true but GRP checkpoint does not exist: %s" % grp_checkpoint
            )
        grp_model = load_grp_checkpoint(
            grp_checkpoint,
            device=device,
            model_cfg=dict(config.get("model", {})),
        )
        logging.info(
            "enabled GRP reward: checkpoint=%s weight=%s gamma=%s clip=%s",
            grp_checkpoint,
            rl_cfg.get("grp_reward_weight", 0.2),
            rl_cfg.get("grp_reward_gamma", 1.0),
            rl_cfg.get("grp_reward_clip", 0.08),
        )
    actor_temp = float(rl_cfg.get("actor_temperature", 1.0))
    logprob_temp = float(rl_cfg.get("ppo_logprob_temperature", actor_temp))
    if abs(actor_temp - logprob_temp) > 1.0e-9:
        logging.warning(
            "actor_temperature=%s differs from ppo_logprob_temperature=%s; PPO ratios will be biased",
            actor_temp,
            logprob_temp,
        )
    if hasattr(reward_shaper, "potential"):
        table_dir = getattr(reward_shaper.potential, "table_dir", "")
        logging.info(
            "fan-potential tables: dir=%s loaded=%s cache_size=%s mode=%s",
            table_dir,
            len(getattr(reward_shaper.potential, "tables", {})),
            getattr(reward_shaper.potential, "cache_size", 0),
            getattr(reward_shaper.potential, "score_mode", ""),
        )
        if bool(rl_cfg.get("enable_potential_reward", True)) and table_dir and not Path(table_dir).exists():
            logging.warning("fan-potential table directory does not exist: %s", table_dir)
    iterations = int(rl_cfg["iterations"])
    # [V4 RL progress] Papers usually report training by environment steps.
    # Keep iteration as the PPO update index, and additionally track cumulative
    # actor decision samples plus hand count for apples-to-apples comparison.
    global_env_step = int(resume_stats.get("global_env_step", 0))
    global_round_step = int(resume_stats.get("global_round_step", 0))

    pbar = trange(start_iter, iterations, desc="ClosedLoop-PPO", initial=start_iter, total=iterations)
    for iteration in pbar:
        transitions, rollout_stats = collect_dppo_rollouts(
            actor_model=actor_model,
            baseline_engine=baseline_engine,
            rl_cfg=rl_cfg,
            iteration=iteration,
            reward_shaper=reward_shaper,
            device=device,
        )
        if not transitions:
            logging.warning("iteration %s produced no transitions; skipping update", iteration + 1)
            continue

        grp_stats = {}
        if grp_model is not None:
            grp_stats = apply_grp_potential_rewards(
                grp_model,
                transitions,
                device=device,
                weight=float(rl_cfg.get("grp_reward_weight", 0.2)),
                gamma=float(rl_cfg.get("grp_reward_gamma", 1.0)),
                reward_clip=float(rl_cfg.get("grp_reward_clip", 0.08)),
                batch_size=int(rl_cfg.get("grp_reward_batch_size", 4096)),
                enable_amp=bool(rl_cfg.get("enable_amp_rollout", True)),
            )

        refresh_transition_values(
            critic_model,
            transitions,
            device=device,
            critic_feature_dim=int(rl_cfg.get("critic_feature_dim", 0)),
            batch_size=int(rl_cfg.get("critic_value_eval_batch_size", 4096)),
            enable_amp=bool(rl_cfg.get("enable_amp_rollout", True)),
        )
        batch_device = torch.device("cpu") if bool(rl_cfg.get("keep_ppo_batch_on_cpu", True)) else device
        batch = tensor_batch(
            transitions,
            device=batch_device,
            gamma=float(rl_cfg["gamma"]),
            gae_lambda=float(rl_cfg["gae_lambda"]),
            critic_feature_dim=int(rl_cfg.get("critic_feature_dim", 0)),
            advantage_mode=str(rl_cfg.get("advantage_mode", "returns")),
        )
        # [V4 stable PPO] The actor starts from SL, but the critic head is new.
        # Warm up only value_head for a few rollout iterations before allowing
        # policy updates to consume TD/GAE advantages.
        if iteration < int(rl_cfg.get("critic_warmup_iterations", 0)):
            train_stats = critic_warmup_update(critic_model, critic_optimizer, batch, rl_cfg)
        else:
            train_stats = ppo_update(
                actor_model,
                critic_model,
                actor_optimizer,
                critic_optimizer,
                batch,
                rl_cfg,
                ref_model=ref_model,
            )
            adapt_regularizers(rl_cfg, train_stats)
        component_stats = reward_component_stats(transitions)

        step = iteration + 1
        global_env_step += int(rollout_stats.get("transitions", 0))
        global_round_step += int(rollout_stats.get("rounds", 0))
        tb_step = global_env_step
        for key, value in rollout_stats.items():
            writer.add_scalar("rollout/" + key, value, tb_step)
        for key, value in train_stats.items():
            writer.add_scalar("dppo/" + key, value, tb_step)
        for key, value in component_stats.items():
            writer.add_scalar("reward/" + key, value, tb_step)
        for key, value in grp_stats.items():
            writer.add_scalar("grp/" + key, value, tb_step)
        # [V4 MJ_RM strict DPPO] Paper-style rollout diagnostics: reward mean
        # is displayed together with win/deal-in rates in the terminal postfix.
        reward_mean = float(batch["rewards"].mean().detach().cpu())
        reward_std = float(batch["rewards"].std().detach().cpu())
        writer.add_scalar("rollout/reward_mean", reward_mean, tb_step)
        writer.add_scalar("rollout/reward_std", reward_std, tb_step)
        writer.add_scalar("progress/update_step", step, tb_step)
        writer.add_scalar("progress/env_step", global_env_step, tb_step)
        writer.add_scalar("progress/round_step", global_round_step, tb_step)
        writer.add_scalar("dppo/current_entropy_coef", float(rl_cfg.get("entropy_coef", 0.0)), tb_step)
        writer.add_scalar("dppo/current_kl_coef", float(rl_cfg.get("kl_coef", 0.0)), tb_step)
        writer.add_scalar("dppo/current_ref_kl_coef", float(rl_cfg.get("ref_kl_coef", 0.0)), tb_step)

        metric_value = float(rollout_stats.get(best_metric_name, rollout_stats.get("avg_score_delta", 0.0)))
        is_better = metric_value > best_metric_value if best_metric_mode == "max" else metric_value < best_metric_value
        stats = {
            **rollout_stats,
            **train_stats,
            **component_stats,
            **{"grp_" + key: value for key, value in grp_stats.items()},
            "best_metric": best_metric_name,
            "best_metric_mode": best_metric_mode,
            "best_metric_value": best_metric_value,
            "global_update_step": step,
            "global_env_step": global_env_step,
            "global_round_step": global_round_step,
        }
        if is_better:
            best_metric_value = metric_value
            stats["best_metric_value"] = best_metric_value
            save_all(
                ckpt_dir,
                actor_model,
                critic_model,
                actor_optimizer,
                critic_optimizer,
                step,
                rl_cfg,
                stats,
                best=True,
            )
        elif step % int(rl_cfg.get("save_every", 1)) == 0:
            save_all(
                ckpt_dir,
                actor_model,
                critic_model,
                actor_optimizer,
                critic_optimizer,
                step,
                rl_cfg,
                stats,
                best=False,
            )

        # [V4 potential reward] The potential cache is useful inside one rollout
        # batch, but keeping Python tuple/dict entries across hundreds of PPO
        # iterations risks unnecessary memory growth.  Clear it periodically.
        cache_clear_every = int(rl_cfg.get("potential_cache_clear_every", 1))
        if cache_clear_every > 0 and step % cache_clear_every == 0:
            reward_shaper.clear_cache()

        # [V4 MJ_RM strict DPPO] Terminal progress output now follows the
        # paper-style RL view: win/deal-in behavior + reward + entropy/loss.
        pbar.set_postfix({
            "iter": step,
            "step": global_env_step,
            "hand": global_round_step,
            "win": "%.3f" % rollout_stats.get("win_rate", 0.0),
            "zimo": "%.3f" % rollout_stats.get("zimo_rate", 0.0),
            "deal": "%.3f" % rollout_stats.get("deal_in_rate", 0.0),
            "oZ": "%.3f" % rollout_stats.get("other_zimo_rate", 0.0),
            "oR": "%.3f" % rollout_stats.get("other_ron_rate", 0.0),
            "pt": "%.2f" % rollout_stats.get("avg_score_delta", 0.0),
            "r1": "%.3f" % rollout_stats.get("round_rank_1", 0.0),
            "g1": "%.3f" % rollout_stats.get("game_rank_1", 0.0),
            "g4": "%.3f" % rollout_stats.get("game_rank_4", 0.0),
            "edge": "%.4f" % rollout_stats.get("game_rank_edge", 0.0),
            "util": "%.4f" % rollout_stats.get("outcome_utility", 0.0),
            "avgR": "%.4f" % reward_mean,
            "ent": "%.3f" % train_stats["entropy"],
            "kl": "%.4f" % train_stats["approx_kl"],
            "klc": "%.4f" % float(rl_cfg.get("kl_coef", 0.0)),
            "rkl": "%.4f" % train_stats.get("ref_kl", 0.0),
            "rklc": "%.4f" % float(rl_cfg.get("ref_kl_coef", 0.0)),
            "loss": "%.4f" % train_stats["loss"],
            "clip": "%.3f" % train_stats["clipfrac"],
            "stop": int(train_stats["early_stop"]),
            "warm": int(train_stats.get("critic_warmup", 0.0)),
            "games": rollout_stats.get("games", 0),
            "rounds": rollout_stats["rounds"],
        })

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_training()


if __name__ == "__main__":
    main()
