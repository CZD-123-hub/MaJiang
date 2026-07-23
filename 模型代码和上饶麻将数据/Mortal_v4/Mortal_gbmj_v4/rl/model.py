from pathlib import Path

import torch
from torch import nn

from mortal_part.consts import ACTION_SPACE, obs_shape
from supervised.model import MahjongSpatialEncoder, Res2NetPolicyModel
from supervised.policy import DEFAULT_MODEL_CFG, _torch_load


def _clean_model_cfg(cfg):
    """[V4 independent critic] Keep actor/critic construction on the same obs config."""
    model_cfg = dict(DEFAULT_MODEL_CFG)
    model_cfg.update(dict(cfg or {}))
    model_cfg["num_classes"] = ACTION_SPACE
    # RL critic is now independent; do not let old value-head checkpoints turn
    # the actor back into a shared ActorCritic model.
    model_cfg["enable_value_head"] = False
    return model_cfg


class CriticModel(nn.Module):
    """[V4 independent critic] Separate V(s) network for PPO.

    The critic uses its own spatial encoder and optimizer, so value regression
    can no longer back-propagate through the supervised policy network.  It is
    initialized from the actor encoder when possible, but after initialization
    the two modules are fully independent.
    """

    def __init__(
        self,
        critic_feature_dim=0,
        critic_feature_hidden=64,
        **model_cfg,
    ):
        super().__init__()
        cfg = _clean_model_cfg(model_cfg)
        self.input_channels = int(cfg.get("input_channels") or obs_shape[0])
        self.history_len = int(cfg.get("history_len", 0))
        self.enable_history = False
        self.model_dim = int(cfg.get("transformer_dim", 512))
        self.critic_feature_dim = int(critic_feature_dim)

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

        extra_dim = 0
        if self.critic_feature_dim > 0:
            hidden = int(critic_feature_hidden)
            self.critic_feature_norm = nn.LayerNorm(self.critic_feature_dim)
            self.critic_feature_proj = nn.Sequential(
                nn.Linear(self.critic_feature_dim, hidden),
                nn.SiLU(inplace=True),
            )
            extra_dim = hidden
        else:
            self.critic_feature_norm = None
            self.critic_feature_proj = None

        value_hidden = int(cfg.get("value_hidden", cfg.get("head_hidden", 512)))
        self.value_head = nn.Sequential(
            nn.Linear(self.model_dim + extra_dim, value_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(float(cfg.get("dropout", 0.15))),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, obs, history_obs=None, history_actions=None, critic_features=None):
        # [V4 independent critic] history args are accepted for API symmetry;
        # current v4 disables history, so critic evaluates the current state.
        features = self.state_proj(self.spatial_encoder(obs))
        if self.critic_feature_dim > 0:
            if critic_features is None:
                critic_features = torch.zeros(
                    (obs.shape[0], self.critic_feature_dim),
                    dtype=features.dtype,
                    device=features.device,
                )
            else:
                critic_features = critic_features.to(device=features.device, dtype=features.dtype)
            extra = self.critic_feature_proj(self.critic_feature_norm(critic_features))
            features = torch.cat([features, extra], dim=-1)
        return self.value_head(features).squeeze(-1)


def actor_state_dict(actor):
    """[V4 independent critic] Save actor-only weights for local play/Botzone."""
    return {
        key: value.detach().cpu()
        for key, value in actor.state_dict().items()
        if not key.startswith("value_head.")
    }


def model_cfg_from_checkpoint(state):
    cfg = dict(DEFAULT_MODEL_CFG)
    if isinstance(state, dict):
        ckpt_cfg = state.get("config", {})
        if isinstance(ckpt_cfg, dict):
            cfg.update(dict(ckpt_cfg.get("model", {})))
    return _clean_model_cfg(cfg)


def extract_state_dict(state):
    if isinstance(state, dict):
        if "actor" in state and isinstance(state["actor"], dict):
            return state["actor"]
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
    return state


def _shape_filtered_load(module, state_dict, strict=False, allowed_prefixes=None):
    target_state = module.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if allowed_prefixes and not any(key.startswith(prefix) for prefix in allowed_prefixes):
            continue
        target = target_state.get(key)
        if target is None or tuple(target.shape) != tuple(value.shape):
            continue
        filtered[key] = value
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    if strict:
        relevant_missing = [
            key for key in missing
            if not allowed_prefixes or any(key.startswith(prefix) for prefix in allowed_prefixes)
        ]
        if relevant_missing or unexpected:
            raise RuntimeError(
                "checkpoint mismatch: missing=%s unexpected=%s" % (relevant_missing, unexpected)
            )
    return missing, unexpected


def _actor_encoder_state(state_dict):
    """[V4 independent critic] Copy actor encoder init into critic without sharing."""
    copied = {}
    for key, value in state_dict.items():
        if key.startswith("spatial_encoder.") or key.startswith("state_proj."):
            copied[key] = value
        elif key.startswith("backbone."):
            copied["spatial_encoder." + key[len("backbone."):]] = value
    return copied


def _tensor_storage_key(tensor):
    try:
        ptr = tensor.untyped_storage().data_ptr()
    except Exception:
        ptr = tensor.data_ptr()
    return (str(tensor.device), int(ptr))


def assert_independent_modules(actor, critic):
    """Fail fast if actor and critic accidentally share trainable tensors."""
    actor_params = {
        _tensor_storage_key(param): name
        for name, param in actor.named_parameters()
    }
    shared = []
    for name, param in critic.named_parameters():
        actor_name = actor_params.get(_tensor_storage_key(param))
        if actor_name is not None:
            shared.append((actor_name, name))
    if shared:
        preview = ", ".join("%s <-> %s" % pair for pair in shared[:5])
        raise RuntimeError(
            "actor and critic share parameter storage; independent critic is broken: %s"
            % preview
        )


def load_actor_and_critic_from_checkpoint(
    path,
    device,
    model_cfg=None,
    critic_feature_dim=0,
    critic_feature_hidden=64,
    strict_actor=False,
):
    """[V4 independent critic] Load supervised/ppo checkpoint into actor+critic."""
    path = Path(path)
    state = _torch_load(str(path), map_location=device)
    cfg = _clean_model_cfg(model_cfg or model_cfg_from_checkpoint(state))

    actor = Res2NetPolicyModel(**cfg).to(device)
    critic = CriticModel(
        critic_feature_dim=int(critic_feature_dim),
        critic_feature_hidden=int(critic_feature_hidden),
        **cfg,
    ).to(device)

    actor_state = extract_state_dict(state)
    _shape_filtered_load(actor, actor_state, strict=strict_actor)

    if isinstance(state, dict) and "critic" in state and isinstance(state["critic"], dict):
        _shape_filtered_load(critic, state["critic"], strict=False)
    else:
        # [V4 independent critic] Supervised checkpoints have no value labels.
        # Initialize critic representation from the actor encoder, but keep its
        # value head random and trained only by PPO TD targets.
        _shape_filtered_load(
            critic,
            _actor_encoder_state(actor_state),
            strict=False,
            allowed_prefixes=("spatial_encoder.", "state_proj."),
        )

    assert_independent_modules(actor, critic)
    return actor, critic, cfg


def load_actor_from_checkpoint(path, device, model_cfg=None, strict_actor=False):
    """[V4 independent critic] Load a policy-only reference/baseline actor."""
    actor, _, cfg = load_actor_and_critic_from_checkpoint(
        path,
        device=device,
        model_cfg=model_cfg,
        critic_feature_dim=0,
        critic_feature_hidden=1,
        strict_actor=strict_actor,
    )
    return actor, cfg


def save_ppo_checkpoint(path, actor, critic, actor_optimizer, critic_optimizer, iteration, cfg, stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "iteration": int(iteration),
            "config": cfg,
            "stats": dict(stats or {}),
        },
        str(path),
    )


def save_actor_checkpoint(path, actor, iteration, cfg, stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": actor_state_dict(actor),
            "iteration": int(iteration),
            "config": cfg,
            "stats": dict(stats or {}),
            "inference_only": True,
            "source": "v4_ppo_actor_only_independent_critic",
        },
        str(path),
    )
