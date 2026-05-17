"""LSTM-модель и датасет для new_arch.

Самодостаточная копия из scripts/v0300_lib.py (только то, что нужно
для stateless-LSTM с фиксированной длиной последовательности).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class StatelessSeqDataset(Dataset):
    """Последовательность из seq_len подокон с внутренним шагом.

    X[i] — (seq_len, n_features). Конец окна — позиция start + (seq_len-1)*internal_stride_rows.
    y[i] — таргет в конце окна.
    """

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


class LSTMRegressor(nn.Module):
    """LSTM + линейная голова на последнем шаге.

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
