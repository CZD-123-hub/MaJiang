import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from supervised.policy import masked_logits


def categorical_from_logits(logits, masks):
    # [V4 RL NaN guard] PPO rollout samples from a distribution, so logits
    # must be finite.  Keep distribution construction in fp32; half-precision
    # inference can turn saturated Transformer outputs into NaN on long runs.
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    logits = logits.clamp(min=-30.0, max=30.0)
    masked = masked_logits(logits, masks).float()
    empty_mask = ~masks.any(dim=-1)
    if torch.any(empty_mask):
        # A no-legal-action state should never happen, but Categorical with an
        # all-masked row is undefined.  Let PASS be the only fallback action.
        masked = masked.clone()
        masked[empty_mask] = torch.finfo(masked.dtype).min
        masked[empty_mask, 0] = 0.0
    return Categorical(logits=masked), masked


def kl_ref_to_new(ref_masked_logits, new_masked_logits):
    """[V4 PPO RL] KL(reference || new) over legal actions only."""
    ref_logp = F.log_softmax(ref_masked_logits.float(), dim=-1)
    new_logp = F.log_softmax(new_masked_logits.float(), dim=-1)
    ref_prob = ref_logp.exp()
    return (ref_prob * (ref_logp - new_logp)).sum(dim=-1)


class PPOEngine:
    """[V4 independent critic] Stochastic rollout engine for PPO actor."""

    def __init__(
        self,
        model,
        device,
        critic_model=None,
        enable_amp=True,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=True,
        name="ppo",
        temperature=1.0,
    ):
        self.engine_type = "ppo"
        self.model = model.to(device)
        self.critic_model = critic_model.to(device) if critic_model is not None else None
        self.device = torch.device(device)
        self.enable_amp = bool(enable_amp)
        self.enable_quick_eval = bool(enable_quick_eval)
        self.enable_rule_based_agari_guard = bool(enable_rule_based_agari_guard)
        self.name = name
        self.is_oracle = False
        self.version = 4
        self.temperature = float(temperature)
        # [V4 history-hierarchical] PPO rollout uses the same K-step own
        # decision history as supervised training/local inference.
        self.history_len = int(getattr(self.model, "history_len", 0))

    def react_batch(self, obs, masks, invisible_obs=None, history_obs=None, history_actions=None):
        self.model.eval()
        with torch.no_grad():
            # [V4 RL NaN guard] Rollout action selection is intentionally kept
            # in fp32.  PPO uses the sampled log-prob as a training target; AMP
            # instability here poisons the whole batch before learner updates.
            return self._react_batch(obs, masks, history_obs, history_actions)

    def _react_batch(self, obs, masks, history_obs=None, history_actions=None):
        obs_tensor = torch.as_tensor(np.stack(obs, axis=0), dtype=torch.float32, device=self.device)
        masks_tensor = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.bool, device=self.device)
        history_obs_tensor = None
        history_actions_tensor = None
        if history_obs is not None:
            history_obs_tensor = torch.as_tensor(np.stack(history_obs, axis=0), dtype=torch.float32, device=self.device)
        if history_actions is not None:
            history_actions_tensor = torch.as_tensor(np.stack(history_actions, axis=0), dtype=torch.long, device=self.device)
        logits = self.model(
            obs_tensor,
            history_obs=history_obs_tensor,
            history_actions=history_actions_tensor,
        )
        if isinstance(logits, tuple):
            logits, values = logits
        elif self.critic_model is not None:
            values = self.critic_model(
                obs_tensor,
                history_obs=history_obs_tensor,
                history_actions=history_actions_tensor,
            )
        else:
            # [V4 independent critic] Rollout values are refreshed by the
            # learner with critic-only features before GAE, so zeros here are
            # only a temporary placeholder.
            values = logits.new_zeros((logits.shape[0],))
        logits = logits / max(self.temperature, 1.0e-6)
        if not torch.isfinite(logits).all():
            # Do not silently propagate NaN into Categorical.  Replace only for
            # the current rollout decision; checkpoint-level NaN is checked in
            # train_ppo.py after loading/resume.
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
        dist, masked = categorical_from_logits(logits, masks_tensor)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return {
            "actions": actions.detach().cpu().tolist(),
            "logits": masked.detach().float().cpu().tolist(),
            "masks": masks_tensor.detach().cpu().tolist(),
            "is_greedy": [False] * int(actions.shape[0]),
            "log_probs": log_probs.detach().float().cpu().tolist(),
            "values": values.detach().float().cpu().tolist(),
        }
