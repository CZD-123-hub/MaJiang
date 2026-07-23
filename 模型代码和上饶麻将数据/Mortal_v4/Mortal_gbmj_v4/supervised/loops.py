import torch
import torch.nn.functional as F

from mortal_part.consts import (
    ADDKONG_BASE,
    ADDKONG_COUNT,
    ANKANG_BASE,
    ANKANG_COUNT,
    CHOW_BASE,
    CHOW_COUNT,
    DISCARD_BASE,
    DISCARD_COUNT,
    MINGGANG_BASE,
    MINGGANG_COUNT,
    PUNG_BASE,
    PUNG_COUNT,
)


"""监督学习的损失、评估和指标。

当前训练目标是行为克隆：给定专家对局中的 ``obs``，用交叉熵让模型提高
专家动作 ``target`` 的概率。``mask`` 不参与反向传播，但会用于统计“模型
是否预测了非法动作”以及推理时屏蔽非法 logits。
"""


MELD_RANGES = (
    (MINGGANG_BASE, MINGGANG_COUNT),
    (ANKANG_BASE, ANKANG_COUNT),
    (ADDKONG_BASE, ADDKONG_COUNT),
    (PUNG_BASE, PUNG_COUNT),
    (CHOW_BASE, CHOW_COUNT),
)


def move_batch(batch, device):
    # Current dataloader may still yield history/value fields for backward
    # compatibility.  Direct-235 supervised training only consumes obs/mask/action.
    if len(batch) >= 6:
        obs, _, _, mask, target = batch[:5]
    else:
        obs, mask, target = batch[:3]
    return (
        obs.to(device, non_blocking=True),
        mask.to(device=device, non_blocking=True).bool(),
        target.to(device=device, non_blocking=True).long(),
    )


def build_meld_mask(target):
    meld_mask = torch.zeros_like(target, dtype=torch.bool)
    for base, count in MELD_RANGES:
        meld_mask |= (target >= base) & (target < base + count)
    return meld_mask


def build_kong_mask(target):
    return (
        ((target >= MINGGANG_BASE) & (target < MINGGANG_BASE + MINGGANG_COUNT))
        | ((target >= ANKANG_BASE) & (target < ANKANG_BASE + ANKANG_COUNT))
        | ((target >= ADDKONG_BASE) & (target < ADDKONG_BASE + ADDKONG_COUNT))
    )


def compute_policy_loss(
    model,
    batch,
    device,
    enable_amp,
    label_smoothing=0.0,
    value_loss_weight=0.0,
    subaction_loss_weight=1.0,
):
    # 一批样本的张量形状通常为：obs=[B,C,4,9]，mask=[B,235]，target=[B]。
    obs, mask, target = move_batch(batch, device)
    obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
    with torch.autocast(device_type=device.type, enabled=enable_amp):
        logits = model(obs)
        # logits 是未归一化分数；cross_entropy 内部会执行 log_softmax。
        # 训练标签是专家动作，而不是 mask 后的 argmax 结果。
        policy_loss = F.cross_entropy(
            logits.float(),
            target,
            label_smoothing=float(label_smoothing),
        )
        loss = policy_loss

    with torch.no_grad():
        logits_detached = logits.detach().float()
        pred = logits_detached.argmax(dim=-1)
        # masked_pred 只用于评估：将非法动作分数设为极小值后再取最大值。
        masked_pred = logits_detached.masked_fill(~mask, -1.0e9).argmax(dim=-1)
        correct = pred == target
        masked_correct = masked_pred == target
        discard_mask = (target >= DISCARD_BASE) & (target < DISCARD_BASE + DISCARD_COUNT)
        meld_mask = build_meld_mask(target)
        kong_mask = build_kong_mask(target)
        # legal_rate 表示原始 argmax 是否落在合法动作集合中。
        legal_rate = float(mask.gather(1, pred.unsqueeze(1)).float().mean().item())

    zero = logits.new_tensor(0.0).float()
    metrics = {
        "loss_sum": float(loss.detach()) * int(target.numel()),
        "policy_loss_sum": float(policy_loss.detach()) * int(target.numel()),
        # Kept as zero-valued keys so existing logging/checkpoint code does not
        # need a migration when moving back from the two-head experiment.
        "type_loss_sum": 0.0,
        "subaction_loss_sum": 0.0,
        "value_loss_sum": 0.0,
        "value_abs_error_sum": 0.0,
        "value_pred_sum": 0.0,
        "value_target_sum": 0.0,
        "correct": int(correct.sum()),
        "action_235_correct": int(correct.sum()),
        "masked_action_correct": int(masked_correct.sum()),
        "count": int(target.numel()),
        "action_type_correct": 0,
        "discard_correct": int(correct[discard_mask].sum()) if discard_mask.any() else 0,
        "discard_count": int(discard_mask.sum()),
        "meld_correct": int(correct[meld_mask].sum()) if meld_mask.any() else 0,
        "meld_count": int(meld_mask.sum()),
        "kong_correct": int(correct[kong_mask].sum()) if kong_mask.any() else 0,
        "kong_count": int(kong_mask.sum()),
        "discard_tile_correct": 0,
        "discard_tile_count": 0,
        "pong_tile_correct": 0,
        "pong_tile_count": 0,
        "chow_tile_correct": 0,
        "chow_tile_count": 0,
        "kong_tile_correct": 0,
        "kong_tile_count": 0,
        "legal_rate": legal_rate,
    }
    return loss + zero * 0.0, metrics


class RunningAverages:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.policy_loss_sum = 0.0
        self.type_loss_sum = 0.0
        self.subaction_loss_sum = 0.0
        self.value_loss_sum = 0.0
        self.value_abs_error_sum = 0.0
        self.value_pred_sum = 0.0
        self.value_target_sum = 0.0
        self.correct = 0
        self.action_235_correct = 0
        self.masked_action_correct = 0
        self.count = 0
        self.action_type_correct = 0
        self.discard_correct = 0
        self.discard_count = 0
        self.meld_correct = 0
        self.meld_count = 0
        self.kong_correct = 0
        self.kong_count = 0
        self.discard_tile_correct = 0
        self.discard_tile_count = 0
        self.pong_tile_correct = 0
        self.pong_tile_count = 0
        self.chow_tile_correct = 0
        self.chow_tile_count = 0
        self.kong_tile_correct = 0
        self.kong_tile_count = 0
        self.legal_rate_sum = 0.0
        self.batches = 0

    def update(self, metrics):
        self.loss_sum += float(metrics["loss_sum"])
        self.policy_loss_sum += float(metrics.get("policy_loss_sum", 0.0))
        self.type_loss_sum += float(metrics.get("type_loss_sum", 0.0))
        self.subaction_loss_sum += float(metrics.get("subaction_loss_sum", 0.0))
        self.value_loss_sum += float(metrics.get("value_loss_sum", 0.0))
        self.value_abs_error_sum += float(metrics.get("value_abs_error_sum", 0.0))
        self.value_pred_sum += float(metrics.get("value_pred_sum", 0.0))
        self.value_target_sum += float(metrics.get("value_target_sum", 0.0))
        self.correct += int(metrics["correct"])
        self.action_235_correct += int(metrics.get("action_235_correct", 0))
        self.masked_action_correct += int(metrics.get("masked_action_correct", 0))
        self.count += int(metrics["count"])
        self.action_type_correct += int(metrics.get("action_type_correct", 0))
        self.discard_correct += int(metrics["discard_correct"])
        self.discard_count += int(metrics["discard_count"])
        self.meld_correct += int(metrics["meld_correct"])
        self.meld_count += int(metrics["meld_count"])
        self.kong_correct += int(metrics.get("kong_correct", 0))
        self.kong_count += int(metrics.get("kong_count", 0))
        self.discard_tile_correct += int(metrics.get("discard_tile_correct", 0))
        self.discard_tile_count += int(metrics.get("discard_tile_count", 0))
        self.pong_tile_correct += int(metrics.get("pong_tile_correct", 0))
        self.pong_tile_count += int(metrics.get("pong_tile_count", 0))
        self.chow_tile_correct += int(metrics.get("chow_tile_correct", 0))
        self.chow_tile_count += int(metrics.get("chow_tile_count", 0))
        self.kong_tile_correct += int(metrics.get("kong_tile_correct", 0))
        self.kong_tile_count += int(metrics.get("kong_tile_count", 0))
        self.legal_rate_sum += float(metrics["legal_rate"])
        self.batches += 1

    def compute(self):
        return {
            "loss": self.loss_sum / max(1, self.count),
            "policy_loss": self.policy_loss_sum / max(1, self.count),
            "type_loss": self.type_loss_sum / max(1, self.count),
            "subaction_loss": self.subaction_loss_sum / max(1, self.count),
            "value_loss": self.value_loss_sum / max(1, self.count),
            "value_abs_error": self.value_abs_error_sum / max(1, self.count),
            "value_pred_mean": self.value_pred_sum / max(1, self.count),
            "value_target_mean": self.value_target_sum / max(1, self.count),
            "action_acc": self.correct / max(1, self.count),
            "action_235_acc": self.action_235_correct / max(1, self.count),
            "masked_action_acc": self.masked_action_correct / max(1, self.count),
            "action_type_acc": self.action_type_correct / max(1, self.count),
            "discard_acc": self.discard_correct / max(1, self.discard_count),
            "meld_acc": self.meld_correct / max(1, self.meld_count),
            "kong_acc": self.kong_correct / max(1, self.kong_count),
            "discard_tile_acc": self.discard_tile_correct / max(1, self.discard_tile_count),
            "pong_tile_acc": self.pong_tile_correct / max(1, self.pong_tile_count),
            "chow_tile_acc": self.chow_tile_correct / max(1, self.chow_tile_count),
            "kong_tile_acc": self.kong_tile_correct / max(1, self.kong_tile_count),
            "legal_rate": self.legal_rate_sum / max(1, self.batches),
            "count": self.count,
            "batches": self.batches,
        }

    def pop(self):
        metrics = self.compute()
        self.reset()
        return metrics


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    enable_amp=False,
    max_batches=None,
    value_loss_weight=0.0,
    subaction_loss_weight=1.0,
):
    model.eval()
    running = RunningAverages()
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        _, metrics = compute_policy_loss(
            model,
            batch,
            device,
            enable_amp,
            value_loss_weight=value_loss_weight,
            subaction_loss_weight=subaction_loss_weight,
        )
        running.update(metrics)
    return running.compute()
