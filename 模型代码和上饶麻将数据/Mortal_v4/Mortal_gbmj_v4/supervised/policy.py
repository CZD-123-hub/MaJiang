from pathlib import Path

import numpy as np
import torch
from torch import nn

from mortal_part.consts import ACTION_SPACE, PASS_INDEX, obs_shape
from supervised.config import PROJECT_ROOT, config


DEFAULT_MODEL_CFG = {
    "arch": "mahjong_v3_msres2_transformer",
    "channels": 256,
    "num_blocks": 10,
    "base_width": 26,
    "scale": 4,
    "transformer_dim": 512,
    "transformer_heads": 8,
    "transformer_layers": 3,
    "transformer_mlp_ratio": 4.0,
    "head_hidden": 512,
    "dropout": 0.15,
    "num_classes": ACTION_SPACE,
}

DIRECT235_MODEL_CFG = {
    "arch": "mahjong_v3_msres2_transformer",
    "channels": 192,
    "num_blocks": 10,
    "base_width": 26,
    "scale": 4,
    "transformer_dim": 384,
    "transformer_heads": 6,
    "transformer_layers": 3,
    "transformer_mlp_ratio": 4.0,
    "head_hidden": 0,
    "dropout": 0.1,
    "num_classes": ACTION_SPACE,
}


def _strip_module_prefix(state_dict):
    if not any(str(key).startswith("module.") for key in state_dict):
        return state_dict
    return {
        str(key)[7:] if str(key).startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _is_legacy_direct235_checkpoint(state_dict):
    # Direct-235 checkpoints use the v3-style "backbone.*" + "head.*" module
    # names.  Older two-head v4 checkpoints used "policy_head.*" and are not
    # treated as direct policies.
    keys = set(state_dict.keys())
    return (
        any(key.startswith("backbone.") for key in keys)
        and any(key.startswith("head.") for key in keys)
        and not any(key.startswith("policy_head.") for key in keys)
    )


def _filter_cfg(default_cfg, ckpt_cfg):
    cfg = dict(default_cfg)
    if isinstance(ckpt_cfg, dict):
        for key in default_cfg:
            if key in ckpt_cfg:
                cfg[key] = ckpt_cfg[key]
    return cfg


def _infer_direct235_cfg_from_state_dict(state_dict, cfg):
    # [V4 local-play v3-compat] Best effort fallback for older checkpoints
    # that may not have saved config.toml.  It keeps v3 opponent loading
    # shape-compatible instead of silently falling back to small defaults.
    stem_weight = state_dict.get("backbone.stem.branches.0.0.weight")
    if stem_weight is not None and getattr(stem_weight, "ndim", 0) == 4:
        cfg["channels"] = int(stem_weight.shape[0]) * 4
        cfg["input_channels"] = int(stem_weight.shape[1])

    block_ids = []
    for key in state_dict:
        if key.startswith("backbone.blocks."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                block_ids.append(int(parts[2]))
    if block_ids:
        cfg["num_blocks"] = max(block_ids) + 1

    pos_embed = state_dict.get("backbone.pos_embed")
    if pos_embed is not None and getattr(pos_embed, "ndim", 0) == 3:
        transformer_dim = int(pos_embed.shape[-1])
        cfg["transformer_dim"] = transformer_dim
        cfg["transformer_heads"] = 8 if transformer_dim % 8 == 0 else 6

    if "head.0.weight" in state_dict:
        cfg["head_hidden"] = int(state_dict["head.0.weight"].shape[0])
    elif "head.weight" in state_dict:
        cfg["head_hidden"] = 0
    return cfg


class Direct235PolicyModel(nn.Module):
    # [V4 local-play v3-compat] Legacy v3 model wrapper.  It intentionally
    # exposes the old "backbone/head" module names so v3 checkpoints load
    # with their original weights while still sharing v4's runtime engine.
    def __init__(
        self,
        num_classes=ACTION_SPACE,
        channels=192,
        num_blocks=10,
        base_width=26,
        scale=4,
        dropout=0.1,
        transformer_dim=384,
        transformer_heads=6,
        transformer_layers=3,
        transformer_mlp_ratio=4.0,
        head_hidden=0,
        input_channels=None,
        arch="mahjong_v3_msres2_transformer",
        **unused,
    ):
        super().__init__()
        from supervised.model import MahjongSpatialEncoder

        self.arch = arch
        self.input_channels = int(input_channels or obs_shape[0])
        self.history_len = 0
        self.enable_history = False
        self.backbone = MahjongSpatialEncoder(
            in_channels=self.input_channels,
            channels=int(channels),
            num_blocks=int(num_blocks),
            base_width=int(base_width),
            scale=int(scale),
            dropout=float(dropout),
            transformer_dim=int(transformer_dim),
            transformer_heads=int(transformer_heads),
            transformer_layers=int(transformer_layers),
            transformer_mlp_ratio=float(transformer_mlp_ratio),
        )
        self.dropout = nn.Dropout(float(dropout))
        if head_hidden and int(head_hidden) > 0:
            self.head = nn.Sequential(
                nn.Linear(self.backbone.out_dim, int(head_hidden)),
                nn.SiLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(int(head_hidden), int(num_classes)),
            )
        else:
            self.head = nn.Linear(self.backbone.out_dim, int(num_classes))
        self.num_classes = int(num_classes)

    def forward(self, obs, history_obs=None, history_actions=None):
        if obs.dim() != 4:
            raise ValueError(
                "expected obs shape [B, %d, %d, %d], got %s"
                % (self.input_channels, obs_shape[1], obs_shape[2], tuple(obs.shape))
            )
        if int(obs.shape[1]) != self.input_channels:
            raise ValueError(
                "checkpoint expects %d obs channels, got %d"
                % (self.input_channels, int(obs.shape[1]))
            )
        return self.head(self.dropout(self.backbone(obs)))


def _infer_input_channels_from_state_dict(state_dict):
    # [V4 local-play v3-compat] The first stem branch conv has shape
    # [out_channels, in_channels, kh, kw].  This lets one runtime load both
    # 194-channel v3 checkpoints and 205-channel v4 checkpoints correctly.
    for key in (
        "backbone.stem.branches.0.0.weight",
        "spatial_encoder.stem.branches.0.0.weight",
        "module.backbone.stem.branches.0.0.weight",
        "module.spatial_encoder.stem.branches.0.0.weight",
    ):
        weight = state_dict.get(key)
        if weight is not None and getattr(weight, "ndim", 0) == 4:
            return int(weight.shape[1])
    for key, weight in state_dict.items():
        if key.endswith("backbone.stem.branches.0.0.weight") and getattr(weight, "ndim", 0) == 4:
            return int(weight.shape[1])
    return None


def _obs_version_from_channels(input_channels):
    # 194-channel checkpoints use the v3 visible encoder layout.  Current v4
    # direct-235 deliberately keeps this layout so v3/v4 checkpoints can be
    # compared without an observation adapter.  205 is retained only for old
    # foresight checkpoints.
    if int(input_channels) == 194:
        return 3
    if int(input_channels) == 205:
        return 5
    return 3


def _obs_shape_for_version(version):
    # Version 3 and current version 4 both use the same 194-channel visible
    # encoder in this project.  Version 5 keeps compatibility with old
    # 205-channel foresight checkpoints.
    if int(version) >= 5:
        return (obs_shape[0] + 11, obs_shape[1], obs_shape[2])
    return obs_shape


def _compatible_obs_versions(requested, inferred):
    if requested is None:
        return True
    if int(requested) == int(inferred):
        return True
    return _obs_shape_for_version(requested) == _obs_shape_for_version(inferred)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_checkpoint_path(checkpoint="best"):
    # [V4 local-play] Resolve best/latest against v4's configured
    # checkpoint directory, while still allowing absolute paths for
    # cross-project matches such as v3 vs v4.
    if checkpoint in (None, "", "default"):
        checkpoint = "best"
    if checkpoint in ("best", "latest"):
        ckpt_name = "gbmj_policy_best.pth" if checkpoint == "best" else "gbmj_policy_latest.pth"
        return PROJECT_ROOT / config["control"]["checkpoint_dir"] / ckpt_name

    checkpoint_path = Path(str(checkpoint))
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    return checkpoint_path


def masked_logits(logits, mask):
    # [V4 local-play fix] AMP inference may produce float16 logits.
    # Filling illegal actions with -1e9 overflows half precision, so use
    # the current dtype's finite minimum instead.  This keeps argmax/softmax
    # behavior unchanged while making serial and parallel play AMP-safe.
    mask = mask.to(dtype=torch.bool, device=logits.device)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _sample_from_logits(logits):
    # [V4 local-play] Avoid torch.distributions dependency in runtime
    # inference; multinomial over softmax is enough for bot sampling.
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_top_p(logits, p):
    if p >= 1:
        return _sample_from_logits(logits)
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    remove_mask = probs_sum - probs_sort > p
    probs_sort = probs_sort.masked_fill(remove_mask, 0.0)
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled


class SupervisedPolicy:
    # [V4 local-play] Runtime wrapper for the 235-way supervised policy.
    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device
        self.input_channels = int(getattr(model, "input_channels", 194))
        self.obs_version = _obs_version_from_channels(self.input_channels)

    @classmethod
    def from_checkpoint(cls, checkpoint="best", device="cpu"):
        return cls.from_path(resolve_checkpoint_path(checkpoint), device=device)

    @classmethod
    def from_path(cls, checkpoint_path, device="cpu"):
        device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)
        state = _torch_load(str(checkpoint_path), map_location=device)

        ckpt_model_cfg = {}
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state_dict = state["model"]
            ckpt_cfg = state.get("config", {})
            if isinstance(ckpt_cfg, dict):
                ckpt_model_cfg = dict(ckpt_cfg.get("model", {}))
        elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state_dict = state["state_dict"]
            ckpt_cfg = state.get("config", {})
            if isinstance(ckpt_cfg, dict):
                ckpt_model_cfg = dict(ckpt_cfg.get("model", {}))
        else:
            state_dict = state
        state_dict = _strip_module_prefix(state_dict)

        inferred_channels = _infer_input_channels_from_state_dict(state_dict)

        if _is_legacy_direct235_checkpoint(state_dict):
            model_cfg = _filter_cfg(DIRECT235_MODEL_CFG, ckpt_model_cfg)
            model_cfg = _infer_direct235_cfg_from_state_dict(state_dict, model_cfg)
            model_cfg["num_classes"] = ACTION_SPACE
            if inferred_channels is not None:
                model_cfg["input_channels"] = inferred_channels
            model = Direct235PolicyModel(**model_cfg)
        else:
            from supervised.model import Res2NetPolicyModel

            model_cfg = dict(DEFAULT_MODEL_CFG)
            model_cfg.update(ckpt_model_cfg)
            model_cfg["num_classes"] = ACTION_SPACE
            if inferred_channels is not None:
                model_cfg["input_channels"] = inferred_channels
            model = Res2NetPolicyModel(**model_cfg)

        # [V4 local-play] Runtime inference only needs the policy path.
        # Older/newer checkpoints may carry value-head-only or optimizer
        # differences, so strict=False keeps cross-version local play usable.
        model.load_state_dict(state_dict, strict=False)
        policy = cls(model, device)
        policy.input_channels = int(model_cfg.get("input_channels", getattr(model, "input_channels", 0)))
        policy.obs_version = _obs_version_from_channels(policy.input_channels)
        return policy

    def logits(self, obs):
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        return self.model(obs_tensor)[0]

    def select_action(self, obs, legal_mask, deterministic=True, temperature=1.0):
        mask_np = np.asarray(legal_mask, dtype=np.bool_)
        if mask_np.shape[0] != ACTION_SPACE:
            raise ValueError("expected 235-way legal mask, got %s" % (mask_np.shape,))
        mask_tensor = torch.from_numpy(mask_np).to(self.device)
        logits = masked_logits(self.logits(obs), mask_tensor)
        if deterministic:
            return int(logits.argmax(-1).item())
        probs = torch.softmax(logits / max(float(temperature), 1.0e-6), dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


class SupervisedEngine:
    # [V4 local-play] Drop-in engine for the copied arena.  It exposes
    # the same react_batch surface as MortalEngine but runs one 235-way
    # supervised model directly.
    def __init__(
        self,
        policy,
        device=None,
        enable_amp=True,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=True,
        name="mortal",
        obs_version=None,
        deterministic=True,
        boltzmann_epsilon=0.0,
        boltzmann_temp=1.0,
        top_p=1.0,
    ):
        self.engine_type = "supervised"
        self.policy = policy
        self.model = policy.model
        self.device = torch.device(device or policy.device)
        self.supports_history = bool(getattr(self.model, "enable_history", False))
        self.history_len = int(getattr(self.model, "history_len", 0))
        self.enable_amp = bool(enable_amp)
        self.enable_quick_eval = bool(enable_quick_eval)
        self.enable_rule_based_agari_guard = bool(enable_rule_based_agari_guard)
        self.name = name
        self.is_oracle = False
        inferred_obs_version = int(getattr(policy, "obs_version", 4))
        requested_obs_version = None if obs_version is None else int(obs_version)
        if not _compatible_obs_versions(requested_obs_version, inferred_obs_version):
            raise ValueError(
                "checkpoint expects obs_version=%d, but local_play requested obs_version=%d"
                % (inferred_obs_version, requested_obs_version)
            )
        self.obs_version = int(requested_obs_version or inferred_obs_version)
        self.version = self.obs_version
        self.deterministic = bool(deterministic)
        self.boltzmann_epsilon = float(boltzmann_epsilon)
        self.boltzmann_temp = float(boltzmann_temp)
        self.top_p = float(top_p)

    @classmethod
    def from_checkpoint(cls, checkpoint="best", device="cpu", **kwargs):
        policy = SupervisedPolicy.from_checkpoint(checkpoint, device=device)
        return cls(policy, device=device, **kwargs)

    def react_batch(self, obs, masks, invisible_obs=None, history_obs=None, history_actions=None):
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(self.device.type, enabled=self.enable_amp):
                    return self._react_batch(obs, masks, history_obs, history_actions)
            return self._react_batch(obs, masks, history_obs, history_actions)

    def _react_batch(self, obs, masks, history_obs=None, history_actions=None):
        obs_tensor = torch.as_tensor(np.stack(obs, axis=0), dtype=torch.float32, device=self.device)
        masks_tensor = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.bool, device=self.device)
        hist_obs_tensor = None
        hist_actions_tensor = None
        if history_obs is not None:
            hist_obs_tensor = torch.as_tensor(np.stack(history_obs, axis=0), dtype=torch.float32, device=self.device)
        if history_actions is not None:
            hist_actions_tensor = torch.as_tensor(np.stack(history_actions, axis=0), dtype=torch.long, device=self.device)
        logits = self.model(obs_tensor, history_obs=hist_obs_tensor, history_actions=hist_actions_tensor)
        masked = masked_logits(logits, masks_tensor)
        actions, is_greedy = self._select_actions(masked)
        return (
            actions.tolist(),
            masked.detach().float().cpu().tolist(),
            masks_tensor.detach().cpu().tolist(),
            is_greedy.tolist(),
        )

    def _select_actions(self, logits):
        batch_size = logits.shape[0]
        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full((batch_size,), 1.0 - self.boltzmann_epsilon, device=self.device).bernoulli().bool()
            sampled = sample_top_p(logits / max(self.boltzmann_temp, 1.0e-6), self.top_p)
            actions = torch.where(is_greedy, logits.argmax(-1), sampled)
            return actions, is_greedy
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        if self.deterministic:
            return logits.argmax(-1), is_greedy
        sampled = _sample_from_logits(logits / max(self.boltzmann_temp, 1.0e-6))
        is_greedy[:] = False
        return sampled, is_greedy
