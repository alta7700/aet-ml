"""LSTM-модели и датасеты для new_arch.

  StatelessSeqDataset       — sliding window для stateless/attention runner'а.
  StatefulRecordDataset     — один элемент = одна запись (для stateful).
  LSTMRegressor             — простой stateless LSTM (forward → скаляр).
  AttentionLSTMRegressor    — LSTM + аддитивный attention.
  LSTMStatefulRegressor     — LSTM с пробросом (h, c) и return_all
                              (для TBPTT chunks в stateful runner).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ─── Датасеты ───────────────────────────────────────────────────────────────

class StatelessSeqDataset(Dataset):
    """Последовательность из seq_len подокон с внутренним шагом."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_len: int, internal_stride_rows: int,
                 outer_stride_rows: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.s = internal_stride_rows
        self.span = (seq_len - 1) * internal_stride_rows + 1
        last_start = len(X) - self.span
        if last_start < 0:
            self.starts = np.array([], dtype=np.int64)
        else:
            self.starts = np.arange(0, last_start + 1, outer_stride_rows,
                                    dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        st = int(self.starts[idx])
        sel = st + np.arange(self.seq_len) * self.s
        X_seq = torch.from_numpy(self.X[sel]).float()
        y_t = torch.tensor(self.y[sel[-1]], dtype=torch.float32)
        return X_seq, y_t


class StatefulRecordDataset(Dataset):
    """Один элемент = одна непрерывная запись subject'а после субсэмплирования.

    Внутри субъекта выбираются строки с шагом internal_stride_rows.
    Чанкинг и проброс (h, c) делается уже в train-loop'е.
    """

    def __init__(self, X_by_subj: list[np.ndarray], y_by_subj: list[np.ndarray],
                 internal_stride_rows: int):
        self.records: list[tuple[np.ndarray, np.ndarray]] = []
        for X, y in zip(X_by_subj, y_by_subj):
            sel = np.arange(0, len(X), internal_stride_rows, dtype=np.int64)
            if len(sel) < 2:
                continue
            self.records.append((X[sel], y[sel]))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        X, y = self.records[idx]
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()


# ─── Модели ──────────────────────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    """Stateless LSTM + линейная голова на последнем шаге.

    forward(x) → (B,) — предикт на последнем step.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


class AttentionLSTMRegressor(nn.Module):
    """LSTM + аддитивный attention над всеми скрытыми состояниями.

    forward(x) → (B,) — предикт.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 attn_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, attn_dim), nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                       # (B, T, H)
        w = self.attn(out).squeeze(-1)              # (B, T)
        a = torch.softmax(w, dim=1).unsqueeze(-1)   # (B, T, 1)
        ctx = (out * a).sum(dim=1)                  # (B, H)
        return self.fc(self.dropout(ctx)).squeeze(-1)


class LSTMStatefulRegressor(nn.Module):
    """LSTM с пробросом (h, c) и опциональным return_all (для TBPTT).

    forward(x, state=None, return_all=False):
      x:     (B, T, F)
      state: (h, c) опциональный
      return_all=True  → (B, T) — предикт на каждом шаге
      return_all=False → (B,)   — на последнем шаге
    Возвращает (y, (h, c)).
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, state=None, return_all: bool = False):
        out, (h, c) = self.lstm(x, state)
        if return_all:
            y = self.fc(self.dropout(out)).squeeze(-1)
        else:
            y = self.fc(self.dropout(out[:, -1, :])).squeeze(-1)
        return y, (h, c)
