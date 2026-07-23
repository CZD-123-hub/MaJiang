from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from mortal_part.consts import obs_shape
from mortal_part.dataset.grp import GRP_FEATURE_DIM
from supervised.model import MahjongSpatialEncoder
from supervised.policy import DEFAULT_MODEL_CFG, _torch_load


def _clean_model_cfg(cfg: Optional[Dict]) -> Dict:
    model_cfg = dict(DEFAULT_MODEL_CFG)
    model_cfg.update(dict(cfg or {}))
    model_cfg["num_classes"] = int(model_cfg.get("num_classes", 235))
    return model_cfg


class GlobalRewardPredictor(nn.Module):
    """Predict final whole-game utility from the current decision state."""

    def __init__(
        self,
        feature_dim: int = GRP_FEATURE_DIM,
        feature_hidden: int = 64,
        value_hidden: int = 512,
        **model_cfg,
    ):
        super().__init__()
        cfg = _clean_model_cfg(model_cfg)
        self.input_channels = int(cfg.get("input_channels") or obs_shape[0])
        self.feature_dim = int(feature_dim)
        self.model_dim = int(cfg.get("transformer_dim", 512))

        self.spatial_encoder = MahjongSpatialEncoder(
            in_channels=self.input_channels,
            channels=int(cfg.get("channels", 256)),
            num_blocks=int(cfg.get("num_blocks", 10)),
            base_width=int(cfg.get("base_width", 26)),
            scale=int(cfg.get("scale", 4)),
            dropout=float(cfg.get("dropout", 0.15)),
            transformer_dim=int(cfg.get("transformer_dim", 512)),
            transformer_heads=int(cfg.get("transformer_heads", 8)),
            transformer_layers=int(cfg.get("transformer_layers", 3)),
            transformer_mlp_ratio=float(cfg.get("transformer_mlp_ratio", 4.0)),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(self.spatial_encoder.out_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.SiLU(inplace=True),
        )
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, int(feature_hidden)),
            nn.SiLU(inplace=True),
        )
        hidden = int(value_hidden or cfg.get("head_hidden", 512) or 512)
        self.head = nn.Sequential(
            nn.Linear(self.model_dim + int(feature_hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(float(cfg.get("dropout", 0.15))),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, grp_features):
        if obs.dim() != 4:
            raise ValueError("expected obs [B,C,4,9], got %s" % (tuple(obs.shape),))
        if int(obs.shape[1]) != self.input_channels:
            raise ValueError("expected %d obs channels, got %d" % (self.input_channels, int(obs.shape[1])))
        if grp_features is None:
            grp_features = torch.zeros(
                (obs.shape[0], self.feature_dim),
                dtype=obs.dtype,
                device=obs.device,
            )
        grp_features = grp_features.to(device=obs.device, dtype=obs.dtype)
        state_features = self.state_proj(self.spatial_encoder(obs))
        stage_features = self.feature_proj(self.feature_norm(grp_features))
        return self.head(torch.cat([state_features, stage_features], dim=-1)).squeeze(-1)


def _extract_state_dict(state):
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        if "actor" in state and isinstance(state["actor"], dict):
            return state["actor"]
    return state


def _shape_filtered_load(module, state_dict, prefix_map: Optional[Tuple[str, str]] = None) -> int:
    target_state = module.state_dict()
    copied = {}
    for key, value in state_dict.items():
        load_key = key
        if prefix_map is not None and key.startswith(prefix_map[0]):
            load_key = prefix_map[1] + key[len(prefix_map[0]):]
        target = target_state.get(load_key)
        if target is not None and tuple(target.shape) == tuple(value.shape):
            copied[load_key] = value
    module.load_state_dict(copied, strict=False)
    return len(copied)


def load_encoder_from_policy_checkpoint(model: GlobalRewardPredictor, checkpoint_path, device) -> int:
    state = _torch_load(str(checkpoint_path), map_location=device)
    state_dict = _extract_state_dict(state)
    return _shape_filtered_load(model, state_dict, prefix_map=("backbone.", "spatial_encoder."))


def load_grp_checkpoint(path, device, model_cfg: Optional[Dict] = None) -> GlobalRewardPredictor:
    state = _torch_load(str(path), map_location=device)
    ckpt_cfg = {}
    if isinstance(state, dict):
        ckpt_cfg = dict(state.get("model_cfg", {}))
        if not ckpt_cfg:
            ckpt_cfg = dict(state.get("config", {}).get("model", {})) if isinstance(state.get("config", {}), dict) else {}
    cfg = _clean_model_cfg(model_cfg or ckpt_cfg)
    grp_cfg = dict(state.get("grp", {})) if isinstance(state, dict) else {}
    model = GlobalRewardPredictor(
        feature_dim=int(grp_cfg.get("feature_dim", GRP_FEATURE_DIM)),
        feature_hidden=int(grp_cfg.get("feature_hidden", 64)),
        value_hidden=int(grp_cfg.get("value_hidden", cfg.get("head_hidden", 512))),
        **cfg,
    ).to(device)
    state_dict = state["model"] if isinstance(state, dict) and isinstance(state.get("model"), dict) else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _stack_grp_features(transitions: List[Dict], feature_dim: int) -> np.ndarray:
    zero = np.zeros((feature_dim,), dtype=np.float32)
    values = []
    for transition in transitions:
        features = transition.get("grp_features")
        if features is None:
            values.append(zero)
            continue
        arr = np.asarray(features, dtype=np.float32).reshape(-1)
        if arr.shape[0] != feature_dim:
            fixed = zero.copy()
            copy_len = min(feature_dim, arr.shape[0])
            fixed[:copy_len] = arr[:copy_len]
            arr = fixed
        values.append(arr)
    return np.stack(values, axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def predict_transition_values(
    model: GlobalRewardPredictor,
    transitions: List[Dict],
    device,
    batch_size: int = 4096,
    enable_amp: bool = True,
) -> np.ndarray:
    if not transitions:
        return np.zeros((0,), dtype=np.float32)
    values = []
    feature_dim = int(getattr(model, "feature_dim", GRP_FEATURE_DIM))
    for start in range(0, len(transitions), max(1, int(batch_size))):
        chunk = transitions[start:start + max(1, int(batch_size))]
        obs = torch.as_tensor(
            np.stack([t["obs"] for t in chunk], axis=0),
            dtype=torch.float32,
            device=device,
        )
        features = torch.as_tensor(
            _stack_grp_features(chunk, feature_dim),
            dtype=torch.float32,
            device=device,
        )
        if device.type == "cuda":
            with torch.autocast(device_type=device.type, enabled=bool(enable_amp)):
                pred = model(obs, features)
        else:
            pred = model(obs, features)
        values.extend(pred.detach().float().cpu().tolist())
    return np.asarray(values, dtype=np.float32)


def apply_grp_potential_rewards(
    model: GlobalRewardPredictor,
    transitions: List[Dict],
    device,
    weight: float = 0.2,
    gamma: float = 1.0,
    reward_clip: float = 0.08,
    batch_size: int = 4096,
    enable_amp: bool = True,
) -> Dict[str, float]:
    if not transitions or model is None or float(weight) == 0.0:
        return {
            "grp_reward": 0.0,
            "grp_delta": 0.0,
            "grp_value": 0.0,
            "grp_applied": 0.0,
        }

    values = predict_transition_values(
        model,
        transitions,
        device=device,
        batch_size=batch_size,
        enable_amp=enable_amp,
    )
    rewards = []
    deltas = []
    clip_value = max(0.0, float(reward_clip))
    for idx, transition in enumerate(transitions):
        transition["grp_value"] = float(values[idx])
        if bool(transition.get("done", False)) or idx + 1 >= len(transitions):
            continue
        delta = float(gamma) * float(values[idx + 1]) - float(values[idx])
        reward = float(weight) * delta
        if clip_value > 0.0:
            reward = max(-clip_value, min(clip_value, reward))
        transition["reward"] += reward
        comps = transition.setdefault("reward_components", {})
        comps["grp_reward"] = comps.get("grp_reward", 0.0) + float(reward)
        comps["grp_delta"] = comps.get("grp_delta", 0.0) + float(delta)
        comps["grp_value"] = comps.get("grp_value", 0.0) + float(values[idx])
        rewards.append(float(reward))
        deltas.append(float(delta))

    return {
        "grp_reward": float(np.mean(rewards)) if rewards else 0.0,
        "grp_delta": float(np.mean(deltas)) if deltas else 0.0,
        "grp_value": float(np.mean(values)) if len(values) else 0.0,
        "grp_applied": float(len(rewards)),
    }
