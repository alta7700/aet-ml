"""Общие утилиты для NN-обучения в new_arch.

  • get_device()          — CUDA → MPS → CPU.
  • CwtCache              — обёртка над dataset/cwt_cache.npz.
  • prepare_X_for_fold()  — импутация + per-fold StandardScaler + опц. конкат CWT.

CwtCache используется только для архитектур с wavelet_mode='cwt'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from dataset_pipeline.common import DEFAULT_DATASET_DIR

CWT_CACHE_PATH = DEFAULT_DATASET_DIR / "cwt_cache.npz"


def get_device() -> str:
    """CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CwtCache:
    """Обёртка над dataset/cwt_cache.npz.

    Кеш хранит CWT сырого ЭМГ-сигнала (VL_dist, VL_prox) внутри каждого
    30-сек окна: по 7 шкалам × 2 статистики (mean, std).

    get(df_indices) возвращает (n_rows, n_channels * n_scales * 2) без нормализации.
    """

    def __init__(self, path: Path = CWT_CACHE_PATH):
        d = np.load(path, allow_pickle=True)
        self.cwt_mean = d["cwt_mean"]
        self.cwt_std = d["cwt_std"]
        self.row_ids = d["row_ids"]
        self.scales = d["scales"]
        self.channel_names = d["channel_names"].tolist()
        self.row_to_pos = {int(r): i for i, r in enumerate(self.row_ids)}

    @property
    def n_features(self) -> int:
        nc, ns = self.cwt_mean.shape[1], self.cwt_mean.shape[2]
        return nc * ns * 2

    def get(self, df_indices: np.ndarray) -> np.ndarray:
        pos = np.array([self.row_to_pos[int(i)] for i in df_indices])
        mean = self.cwt_mean[pos]
        std = self.cwt_std[pos]
        n = len(pos)
        flat = np.concatenate([mean.reshape(n, -1), std.reshape(n, -1)], axis=1)
        np.nan_to_num(flat, copy=False, nan=0.0)
        return flat.astype(np.float32)


def prepare_X_for_fold(df: pd.DataFrame, feat_cols: list[str],
                       train_idx: pd.Index, test_idx: pd.Index,
                       cwt: Optional[CwtCache] = None) -> tuple[np.ndarray, np.ndarray]:
    """Импутация + per-fold стандартизация + опциональный конкат CWT-фич.

    Все скейлеры обучаются только на train_idx — нет утечки на test.
    """
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr_raw = df.loc[train_idx, feat_cols].values
    X_te_raw = df.loc[test_idx, feat_cols].values
    X_tr = sc.fit_transform(imp.fit_transform(X_tr_raw))
    X_te = sc.transform(imp.transform(X_te_raw))
    if cwt is not None:
        cwt_tr = cwt.get(train_idx.values)
        cwt_te = cwt.get(test_idx.values)
        cwt_sc = StandardScaler()
        cwt_tr = cwt_sc.fit_transform(cwt_tr)
        cwt_te = cwt_sc.transform(cwt_te)
        X_tr = np.concatenate([X_tr, cwt_tr], axis=1)
        X_te = np.concatenate([X_te, cwt_te], axis=1)
    return X_tr.astype(np.float32), X_te.astype(np.float32)
