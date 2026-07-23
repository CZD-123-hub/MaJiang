import math
from typing import Optional

import torch
from torch import nn

from mortal_part.consts import (
    ACTION_SPACE,
    obs_shape,
)


PAD_ACTION = ACTION_SPACE


"""监督模型阅读提示

本文件定义的是一个“直接预测 235 个动作”的策略网络：
输入是麻将状态编码后的二维特征图 ``[B, C, 4, 9]``，输出是
``[B, 235]`` 的动作 logits。235 个输出位置对应弃牌、吃、碰、杠、
胡、过等离散动作；训练时交给交叉熵损失完成行为克隆。

网络主干的思路是：多尺度卷积提取局部牌型关系，Res2 残差块继续
融合空间特征，Transformer 在 4x9 的牌位 token 之间建模长距离关系，
最后用全连接层把状态特征映射为动作分数。
"""


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class MultiScaleStem(nn.Module):
    """Multi-scale local tile-pattern encoder that keeps the 4x9 grid intact."""

    def __init__(self, in_channels, channels):
        super().__init__()
        branch_channels = channels // 4
        if branch_channels <= 0 or branch_channels * 4 != channels:
            raise ValueError(f"channels must be divisible by 4, got {channels}")
        self.branches = nn.ModuleList([
            ConvBNAct(in_channels, branch_channels, kernel_size=(1, 3), padding=(0, 1)),
            ConvBNAct(in_channels, branch_channels, kernel_size=(1, 5), padding=(0, 2)),
            ConvBNAct(in_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0)),
            ConvBNAct(in_channels, branch_channels, kernel_size=(3, 3), padding=(1, 1)),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        # 四个分支分别关注横向连续牌、较宽横向模式、纵向关系和局部邻域。
        # 拼接后仍保持 [B, channels, 4, 9] 的空间布局，便于后续模块理解牌位。
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class SpatialRes2Block(nn.Module):
    """Res2Net-style residual block without spatial downsampling."""

    def __init__(self, channels, base_width=26, scale=4, dropout=0.0):
        super().__init__()
        if scale < 2:
            raise ValueError(f"scale must be >= 2, got {scale}")
        width = max(8, int(math.floor(channels * (base_width / 64.0))))
        self.width = width
        self.scale = scale
        self.nums = scale - 1

        self.conv1 = nn.Conv2d(channels, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv2d(width * scale, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 残差连接保留输入信息；Res2 分支让同一层同时看到不同粒度的局部模式。
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        splits = torch.split(out, self.width, dim=1)

        outputs = []
        branch = None
        for idx in range(self.nums):
            branch = splits[idx] if idx == 0 else branch + splits[idx]
            branch = self.act(self.bns[idx](self.convs[idx](branch)))
            outputs.append(branch)
        outputs.append(splits[self.nums])

        out = torch.cat(outputs, dim=1)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out = self.drop(out)
        return self.act(out + residual)


class MahjongSpatialEncoder(nn.Module):
    """Spatial encoder: multi-scale CNN + 4x9-token Transformer fusion."""

    def __init__(
        self,
        in_channels,
        channels=256,
        num_blocks=10,
        base_width=26,
        scale=4,
        dropout=0.0,
        transformer_dim=512,
        transformer_heads=8,
        transformer_layers=2,
        transformer_mlp_ratio=4.0,
    ):
        super().__init__()
        self.stem = MultiScaleStem(in_channels, channels)
        self.blocks = nn.Sequential(*[
            SpatialRes2Block(channels, base_width=base_width, scale=scale, dropout=dropout)
            for _ in range(int(num_blocks))
        ])
        self.transformer_layers = int(transformer_layers)
        self.grid_tokens = obs_shape[1] * obs_shape[2]

        if self.transformer_layers > 0:
            if transformer_dim % transformer_heads != 0:
                raise ValueError("transformer_dim must be divisible by transformer_heads")
            self.token_proj = nn.Linear(channels, transformer_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_tokens + 1, transformer_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=int(transformer_dim * transformer_mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
            self.out_dim = transformer_dim * 2
        else:
            self.token_proj = None
            self.cls_token = None
            self.pos_embed = None
            self.transformer = None
            self.out_dim = channels * 2

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # 卷积阶段只改变通道特征，不下采样，避免破坏 4x9 牌面坐标。
        x = self.blocks(self.stem(x))

        if self.transformer_layers <= 0:
            avg = torch.flatten(torch.mean(x, dim=(2, 3)), 1)
            max_values = torch.flatten(torch.amax(x, dim=(2, 3)), 1)
            return torch.cat([avg, max_values], dim=1)

        tokens = x.flatten(2).transpose(1, 2)
        # [B, C, 4, 9] -> [B, 36, C]：每个牌位变成一个 Transformer token。
        tokens = self.token_proj(tokens)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
        tokens = self.transformer(tokens)
        cls_feature = tokens[:, 0]
        mean_feature = tokens[:, 1:].mean(dim=1)
        # CLS 汇总全局信息，mean 汇总所有牌位信息；拼接后作为策略头输入。
        return torch.cat([cls_feature, mean_feature], dim=1)


class Res2NetPolicyModel(nn.Module):
    """直接 235 分类策略模型。

    ``obs`` 是特征工程后的状态；``logits[:, a]`` 是采取动作 ``a`` 的
    未归一化分数。注意这里不在模型内部应用合法动作 mask，mask 只在
    评估或实际出牌采样时使用，这样训练目标仍保持完整的 235 类空间。
    """

    def __init__(
        self,
        num_classes=ACTION_SPACE,
        channels=256,
        num_blocks=10,
        base_width=26,
        scale=4,
        dropout=0.1,
        transformer_dim=512,
        transformer_heads=8,
        transformer_layers=3,
        transformer_mlp_ratio=4.0,
        head_hidden=512,
        input_channels=None,
        history_len=0,
        history_encoder_layers=0,
        history_decoder_layers=0,
        history_heads=8,
        history_ffn_dim=1024,
        history_action_dim=128,
        enable_history=False,
        arch="mahjong_v3_msres2_transformer",
        **unused,
    ):
        super().__init__()
        if int(num_classes) != ACTION_SPACE:
            raise ValueError("direct-235 policy requires ACTION_SPACE=235")
        self.arch = arch
        self.input_channels = int(input_channels or obs_shape[0])
        # Keep these attributes so existing training/local-play/RL wrappers can
        # query them, but the direct v3-style policy ignores history inputs.
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
        self.dropout = nn.Dropout(dropout)
        if head_hidden and int(head_hidden) > 0:
            self.head = nn.Sequential(
                nn.Linear(self.backbone.out_dim, int(head_hidden)),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(int(head_hidden), int(num_classes)),
            )
        else:
            self.head = nn.Linear(self.backbone.out_dim, int(num_classes))
        self.num_classes = int(num_classes)

    def _encode_obs(self, obs):
        if obs.dim() != 4:
            raise ValueError(f"expected obs shape [B, {self.input_channels}, {obs_shape[1]}, {obs_shape[2]}], got {tuple(obs.shape)}")
        if obs.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} obs channels, got {obs.shape[1]}")
        return self.backbone(obs)

    def forward(
        self,
        obs,
        history_obs: Optional[torch.Tensor] = None,
        history_actions: Optional[torch.Tensor] = None,
        return_value: bool = False,
        return_dict: bool = False,
    ):
        # history_* 参数为了兼容旧版调用方而保留；当前 direct-235 模型只用当前状态。
        features = self._encode_obs(obs)
        logits = self.head(self.dropout(features))
        value = logits.new_zeros((logits.shape[0],))

        if return_dict:
            out = {
                "logits": logits,
                "features": features,
            }
            if return_value:
                out["value"] = value
            return out
        if return_value:
            return logits, value
        return logits
