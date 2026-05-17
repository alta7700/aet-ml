"""TCN-модели для new_arch.

Самодостаточная копия из scripts/v0400_lib.py — только PureTCN
(на этапе 5 включаем одну референс-архитектуру TCN1).

  SpatialDropout1d  — dropout по каналам, как Dropout2d.
  TemporalBlock     — residual TCN-блок: 2× WeightNorm(Conv1d) с дилатацией,
                      same-padding (k=3), ReLU после каждой свёртки.
  PureTCN           — стек TemporalBlock'ов → GAP → Linear.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout1d(nn.Module):
    """Dropout по каналам: обнуляет целые каналы (как Dropout2d).

    Вход: (B, C, T). Дропаут применяется по оси C — одинаково по всему T.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        return F.dropout2d(x.unsqueeze(-1), p=self.p, training=True).squeeze(-1)


class TemporalBlock(nn.Module):
    """Residual-блок TCN: два WeightNorm(Conv1d) с дилатацией, same-padding.

    Для нечётного kernel_size pad = (k-1)*d // 2 даёт выход той же длины T.
    Если in_ch != out_ch — добавляется 1×1 downsample для residual.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad))
        self.drop = SpatialDropout1d(dropout)
        self.act = nn.ReLU()
        self.downsample = (
            nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, 1))
            if in_ch != out_ch else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        h = self.act(self.drop(self.conv1(x)))
        h = self.act(self.drop(self.conv2(h)))
        return self.act(h + res)


class PureTCN(nn.Module):
    """Чистый dilated TCN: стек TemporalBlock → GlobalAvgPool → Linear.

    Вход: (B, seq_len, F) → permute → (B, F, seq_len).
    Для dilations=[1,2,4,8], kernel=3 receptive field = 1 + 2·(1+2+4+8) = 31.
    """

    def __init__(self, input_size: int, n_channels: int,
                 kernel_size: int, dilations: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_size
        for d in dilations:
            layers.append(TemporalBlock(in_ch, n_channels, kernel_size, d, dropout))
            in_ch = n_channels
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.permute(0, 2, 1))   # (B, n_ch, T)
        h = h.mean(dim=-1)                  # GAP → (B, n_ch)
        return self.head(h).squeeze(-1)     # (B,)
