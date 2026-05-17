"""TCN-модели для new_arch — все 4 архитектуры (PureTCN, DwtTCN, WaveNetTCN).

ВАЖНО: все свёртки **causal** — padding только слева, чтобы предсказание
в позиции t не использовало вход на позициях > t. В исходном v0400_lib
свёртки были non-causal (padding на обе стороны) — это эквивалентно
утечке "будущего" внутри окна. В new_arch исправлено.

  SpatialDropout1d  — dropout по каналам.
  causal_pad_left   — F.pad слева на (k-1)*dilation, без padding в Conv1d.
  TemporalBlock     — 2× WeightNorm(Conv1d) + causal padding + residual.
  HaarDWT           — Haar DWT по временной оси (depthwise, незаучиваемый).
  PureTCN           — стек TemporalBlock'ов → GAP → Linear.
  DwtTCN            — Haar DWT → 2 ветви TCN (approx + detail) → concat → head.
  WaveNetTCN        — WaveNet-стиль с causal padding, gated activation, skip-conn.
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


class _CausalConv1d(nn.Module):
    """Conv1d с causal padding: pad слева на (k-1)*dilation, без правого padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int = 1):
        super().__init__()
        self.pad_left = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      dilation=dilation, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x_padded = F.pad(x, (self.pad_left, 0))
        return self.conv(x_padded)


class TemporalBlock(nn.Module):
    """Residual-блок TCN: два причинных WeightNorm(Conv1d) с дилатацией.

    Causal padding: вход в позиции t зависит только от позиций ≤ t.
    Если in_ch != out_ch — добавляется 1×1 downsample для residual.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
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
    """Чистый dilated TCN (causal) → GAP → Linear.

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


class HaarDWT(nn.Module):
    """Однократное Haar-DWT вдоль временной оси.

    Фиксированные (незаучиваемые) depthwise conv1d с шагом 2.
    Вход: (B, F, T). Выход: cA, cD — каждый (B, F, T//2).

    Замечание о causality: Haar DWT с шагом 2 на парах (x[2i], x[2i+1])
    производит выход в позиции i, использующий обе исходные позиции.
    Это full-window операция — она не "видит будущее" по отношению
    к target в конце окна (GAP над всем cA/cD), потому что target определяется
    в позиции T-1 = последней паре.
    """

    def __init__(self, n_features: int):
        super().__init__()
        inv_sqrt2 = float(2 ** -0.5)
        w_lo = torch.full((n_features, 1, 2), inv_sqrt2)
        w_hi = torch.tensor([-inv_sqrt2, inv_sqrt2]).view(1, 1, 2).expand(
            n_features, 1, 2).clone()
        self.register_buffer("w_lo", w_lo)
        self.register_buffer("w_hi", w_hi)
        self.n_features = n_features

    def forward(self, x: torch.Tensor):
        cA = F.conv1d(x, self.w_lo, stride=2, padding=0, groups=self.n_features)
        cD = F.conv1d(x, self.w_hi, stride=2, padding=0, groups=self.n_features)
        return cA, cD


class DwtTCN(nn.Module):
    """DWT-TCN: Haar-DWT → две параллельные causal TCN-ветви → concat → head.

    Ветвь approx (cA) обрабатывает медленные тренды,
    ветвь detail (cD) — быстрые изменения.
    """

    def __init__(self, input_size: int, n_channels: int,
                 kernel_size: int, dilations: list[int], dropout: float):
        super().__init__()
        self.dwt = HaarDWT(input_size)
        branch_ch = max(n_channels // 2, 8)

        def _make_branch():
            layers: list[nn.Module] = []
            in_ch = input_size
            for d in dilations:
                layers.append(TemporalBlock(in_ch, branch_ch, kernel_size, d, dropout))
                in_ch = branch_ch
            return nn.Sequential(*layers)

        self.branch_a = _make_branch()
        self.branch_d = _make_branch()
        self.head = nn.Linear(branch_ch * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.permute(0, 2, 1)            # (B, F, T)
        cA, cD = self.dwt(xt)              # каждый (B, F, T//2)
        hA = self.branch_a(cA).mean(-1)    # GAP → (B, branch_ch)
        hD = self.branch_d(cD).mean(-1)
        return self.head(torch.cat([hA, hD], dim=-1)).squeeze(-1)


class WaveNetBlock(nn.Module):
    """WaveNet-блок: causal kernel=2 + gated activation (tanh·σ) + residual + skip.

    Causal padding слева на dilation; без правого padding.
    """

    def __init__(self, residual_ch: int, skip_ch: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.pad_left = dilation   # (k-1)*d = dilation для kernel=2
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(residual_ch, 2 * residual_ch, 2,
                      dilation=dilation, padding=0))
        self.skip_conv = nn.utils.weight_norm(
            nn.Conv1d(residual_ch, skip_ch, 1))
        self.res_conv = nn.utils.weight_norm(
            nn.Conv1d(residual_ch, residual_ch, 1))
        self.drop = SpatialDropout1d(dropout)

    def forward(self, x: torch.Tensor, skip_acc):
        x_padded = F.pad(x, (self.pad_left, 0))
        h = self.conv(x_padded)
        h = self.drop(h)
        R = h.shape[1] // 2
        h = torch.tanh(h[:, :R, :]) * torch.sigmoid(h[:, R:, :])
        skip = self.skip_conv(h)
        skip_acc = skip if skip_acc is None else skip_acc + skip
        return x + self.res_conv(h), skip_acc


class WaveNetTCN(nn.Module):
    """WaveNet-style TCN (causal): стек WaveNetBlock → ReLU-Conv-ReLU-Conv → GAP → скаляр."""

    def __init__(self, input_size: int, residual_ch: int, skip_ch: int,
                 dilations: list[int], dropout: float):
        super().__init__()
        self.input_proj = nn.utils.weight_norm(
            nn.Conv1d(input_size, residual_ch, 1))
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_ch, skip_ch, d, dropout) for d in dilations
        ])
        self.post = nn.Sequential(
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(skip_ch, skip_ch // 2, 1)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(skip_ch // 2, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))   # (B, R, T)
        skip_acc = None
        for blk in self.blocks:
            h, skip_acc = blk(h, skip_acc)
        out = self.post(skip_acc)                  # (B, 1, T)
        return out.mean(dim=-1).squeeze(-1)
