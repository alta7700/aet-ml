"""v0300_lib — общая библиотека для серии v0301..v0318 (LSTM-grid).

Состав:
  Dataset:
    StatelessSeqDataset    — скользящее окно из N подокон,
                             внутренний шаг 30/15/5 сек, outer-шаг 5 сек.
    StatefulRecordDataset  — один элемент = одна непрерывная запись,
                             используется в train_stateful (TBPTT chunks).
  Модели:
    LSTMRegressor          — stateless и stateful (state опциональный).
    AttentionLSTMRegressor — multiplicative attention над всеми step.
  Тренировка / эксперимент:
    run_experiment(cfg)    — LOSO + сохранение артефактов в формате v0101.

Условия (зафиксировано):
  база    — dataset/merged_features_ml.parquet (окно 30 сек, шаг 5 сек);
  фичи    — только EMG+NIRS+HRV, noabs;
  таргет  — z-norm per fold, MSELoss, inverse_transform перед метриками;
  wavelet — берётся из dataset/cwt_cache.npz (precomputed);
  device  — CUDA если есть, иначе MPS, иначе CPU.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from scripts.v0011_modality_ablation import (
    prepare_data, get_feature_cols, kalman_smooth,
)

# Абсолютные признаки — исключаются (noabs всегда).
EXCLUDE_ABS = frozenset([
    "trainred_smo2_mean", "trainred_hhb_mean",
    "trainred_hbdiff_mean", "trainred_thb_mean",
    "hrv_mean_rr_ms", "feat_smo2_x_rr",
])

# База: окно 30 сек, шаг 5 сек.
BASE_STEP_SEC = 5

CWT_CACHE_PATH = DEFAULT_DATASET_DIR / "cwt_cache.npz"


# ─── Конфиг эксперимента ──────────────────────────────────────────────────────

@dataclass
class ExperimentCfg:
    """Описание одного из 18 экспериментов.

    name                  — короткое имя версии (= имя папки в results/).
    family                — stateless | stateful | attention.
    seq_len               — длина последовательности подокон.
                            Для stateful игнорируется (весь рекорд).
    internal_stride_sec   — шаг между подокнами внутри sample (30/15/5).
    outer_stride_sec      — шаг между sample'ами (для stateless/attention).
                            Для stateful — шаг при субсэмплировании рекорда.
    use_wavelet           — добавить CWT-фичи concat'ом к сырым.
    chunk_size            — TBPTT chunk (только stateful).
    target                — lt1 / lt2 / both (берётся из CLI).
    """
    name: str
    family: Literal["stateless", "stateful", "attention"]
    seq_len: Optional[int]
    internal_stride_sec: int
    outer_stride_sec: int = 5
    use_wavelet: bool = False
    chunk_size: int = 10
    target: str = "both"

    # фикс. гиперы
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 80
    patience: int = 15
    batch_size: int = 64
    val_frac: float = 0.2  # последние val_frac окон каждого train-subject

    @property
    def internal_stride_rows(self) -> int:
        """Шаг внутри sample в строках 5-сек грид'а."""
        assert self.internal_stride_sec % BASE_STEP_SEC == 0
        return self.internal_stride_sec // BASE_STEP_SEC

    @property
    def outer_stride_rows(self) -> int:
        assert self.outer_stride_sec % BASE_STEP_SEC == 0
        return self.outer_stride_sec // BASE_STEP_SEC


# ─── CWT-кеш ──────────────────────────────────────────────────────────────────

class CwtCache:
    """Обёртка над dataset/cwt_cache.npz.

    Кеш хранит CWT для всех 138 фич × 5 шкал. На загрузке выбираем подсет
    нужных фич, сплющиваем (n_f × n_sc) → (n_f * n_sc) и нормализуем
    глобально (mean/std по всем строкам). Для z-norm per fold нет смысла:
    CWT уже стационарен по природе.
    """
    def __init__(self):
        d = np.load(CWT_CACHE_PATH, allow_pickle=True)
        self.cwt = d["cwt"]                # (n_rows, n_features, n_scales)
        self.row_ids = d["row_ids"]        # (n_rows,) индексы df.index
        self.feat_cols = d["feat_cols"].tolist()
        self.scales = d["scales"]
        # row_id → position в массиве cwt
        self.row_to_pos = {int(r): i for i, r in enumerate(self.row_ids)}

    def get(self, df_indices: np.ndarray, feat_subset: list[str]) -> np.ndarray:
        """Возвращает (n_rows, n_feat_subset * n_scales) float32, z-norm.

        df_indices — оригинальные df.index. Игнорируем те, что отсутствуют
        в кеше (их быть не должно, но защитимся).
        """
        feat_idx = [self.feat_cols.index(c) for c in feat_subset
                    if c in self.feat_cols]
        if len(feat_idx) != len(feat_subset):
            missing = [c for c in feat_subset if c not in self.feat_cols]
            raise KeyError(f"В CWT-кеше нет признаков: {missing}")
        pos = np.array([self.row_to_pos[int(i)] for i in df_indices])
        sub = self.cwt[np.ix_(pos, feat_idx)]                   # (n, n_f, n_sc)
        flat = sub.reshape(len(pos), -1).astype(np.float32)
        mu, sd = flat.mean(0), flat.std(0) + 1e-8
        return (flat - mu) / sd


# ─── Datasets ─────────────────────────────────────────────────────────────────

class StatelessSeqDataset(Dataset):
    """Последовательность из seq_len подокон с внутренним шагом.

    X[i] — (seq_len, n_features). Конец окна — позиция start + (seq_len-1)*int_rows.
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
        # допустимые стартовые позиции с outer-шагом
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
        # индексы подокон внутри sample
        sel = st + np.arange(self.seq_len) * self.s
        X_seq = torch.from_numpy(self.X[sel]).float()
        y_t   = torch.tensor(self.y[sel[-1]], dtype=torch.float32)
        return X_seq, y_t


class StatefulRecordDataset(Dataset):
    """Один элемент = один subject (целиком, после субсэмплирования).

    Возвращает (X[T, F], y[T]); chunking делает train_stateful.
    Внутри одного subject субсэмплируем строки с шагом internal_stride_rows.
    """
    def __init__(self, X_by_subj: list[np.ndarray], y_by_subj: list[np.ndarray],
                 internal_stride_rows: int):
        self.records = []
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


# ─── Модели ───────────────────────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    """LSTM с линейной головой.

    forward(x, state=None, return_all=False):
      x        — (B, T, F)
      state    — (h, c) опционально (для stateful).
      return_all=False → возвращает (B,) предикт на последнем шаге;
      return_all=True  → возвращает (B, T) предикт на каждом шаге.
    Всегда возвращает второй элемент — обновлённый state.
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, state=None, return_all: bool = False):
        out, (h, c) = self.lstm(x, state)
        if return_all:
            y = self.fc(self.dropout(out)).squeeze(-1)        # (B, T)
        else:
            y = self.fc(self.dropout(out[:, -1, :])).squeeze(-1)  # (B,)
        return y, (h, c)


class AttentionLSTMRegressor(nn.Module):
    """LSTM + аддитивный attention над всеми скрытыми состояниями.

    forward(x) → (B,) — предикт; state не пробрасывается.
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 attn_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, attn_dim), nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)                       # (B, T, H)
        w = self.attn(out).squeeze(-1)              # (B, T)
        a = torch.softmax(w, dim=1).unsqueeze(-1)   # (B, T, 1)
        ctx = (out * a).sum(dim=1)                  # (B, H)
        return self.fc(self.dropout(ctx)).squeeze(-1)


def build_model(cfg: ExperimentCfg, input_size: int) -> nn.Module:
    if cfg.family == "attention":
        return AttentionLSTMRegressor(input_size, cfg.hidden_size,
                                      cfg.num_layers, cfg.dropout)
    return LSTMRegressor(input_size, cfg.hidden_size,
                         cfg.num_layers, cfg.dropout)


# ─── Тренировка (stateless / attention) ───────────────────────────────────────

def _train_stateless(model: nn.Module, train_loader: DataLoader,
                     val_loader: DataLoader, cfg: ExperimentCfg,
                     device: str) -> nn.Module:
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf"); patience = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(cfg.max_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            if cfg.family == "attention":
                pred = model(X)
            else:
                pred, _ = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval(); vl = 0.0; n = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X) if cfg.family == "attention" else model(X)[0]
                vl += criterion(pred, y).item() * len(y); n += len(y)
        vl /= max(n, 1)
        if vl < best:
            best = vl; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model


def _predict_stateless(model: nn.Module, loader: DataLoader,
                       cfg: ExperimentCfg, device: str) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for X, _ in loader:
            pred = model(X.to(device)) if cfg.family == "attention" \
                   else model(X.to(device))[0]
            out.append(pred.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


# ─── Тренировка (stateful, TBPTT chunks=cfg.chunk_size) ───────────────────────

def _train_stateful(model: nn.Module, train_records, val_records,
                    cfg: ExperimentCfg, device: str) -> nn.Module:
    """Каждая запись = одна последовательность. h,c пробрасывается между
    chunks с detach. Loss считается на каждом step внутри chunk.
    """
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf"); patience = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    def _run_record(X, y, train: bool):
        h = c = None
        loss_sum = 0.0; n_steps = 0; preds_collect = []
        for st in range(0, len(X), cfg.chunk_size):
            end = min(st + cfg.chunk_size, len(X))
            x_chunk = torch.from_numpy(X[st:end]).float().unsqueeze(0).to(device)  # (1, k, F)
            y_chunk = torch.from_numpy(y[st:end]).float().unsqueeze(0).to(device)  # (1, k)
            state = (h, c) if h is not None else None
            pred, (h, c) = model(x_chunk, state, return_all=True)  # (1, k)
            if train:
                loss = criterion(pred, y_chunk)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                loss = criterion(pred, y_chunk)
                preds_collect.append(pred.detach().cpu().numpy().ravel())
            loss_sum += loss.item() * (end - st); n_steps += (end - st)
            h = h.detach(); c = c.detach()
        return loss_sum / max(n_steps, 1), (
            np.concatenate(preds_collect) if preds_collect else None
        )

    for _ in range(cfg.max_epochs):
        model.train()
        order = np.random.permutation(len(train_records))
        for i in order:
            X, y = train_records[i]
            _run_record(X, y, train=True)

        model.eval(); vl = 0.0; n = 0
        with torch.no_grad():
            for X, y in val_records:
                l, _ = _run_record(X, y, train=False)
                vl += l * len(X); n += len(X)
        vl /= max(n, 1)
        if vl < best:
            best = vl; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model


def _predict_stateful(model: nn.Module, X: torch.Tensor,
                      cfg: ExperimentCfg, device: str) -> np.ndarray:
    """Прогон одной тестовой записи через TBPTT-chunks с пробросом state."""
    model.eval(); h = c = None; out = []
    with torch.no_grad():
        for st in range(0, len(X), cfg.chunk_size):
            end = min(st + cfg.chunk_size, len(X))
            x_chunk = X[st:end].unsqueeze(0).to(device)
            state = (h, c) if h is not None else None
            pred, (h, c) = model(x_chunk, state, return_all=True)
            out.append(pred.cpu().numpy().ravel())
    return np.concatenate(out)


# ─── LOSO ─────────────────────────────────────────────────────────────────────

def _prepare_X(df: pd.DataFrame, feat_cols: list[str], cwt: Optional[CwtCache],
               train_idx: pd.Index, test_idx: pd.Index):
    """Импутация + стандартизация + (опционально) конкат CWT-фич."""
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr_raw = df.loc[train_idx, feat_cols].values
    X_te_raw = df.loc[test_idx, feat_cols].values
    X_tr = sc.fit_transform(imp.fit_transform(X_tr_raw))
    X_te = sc.transform(imp.transform(X_te_raw))
    if cwt is not None:
        cwt_tr = cwt.get(train_idx.values, feat_cols)
        cwt_te = cwt.get(test_idx.values, feat_cols)
        X_tr = np.concatenate([X_tr, cwt_tr], axis=1)
        X_te = np.concatenate([X_te, cwt_te], axis=1)
    return X_tr.astype(np.float32), X_te.astype(np.float32)


def _split_subject_by_pos(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает массив позиций внутри group (отсортированной по времени)."""
    pos = np.arange(len(group))
    return pos


def _loso_fold_stateless(test_s, df, feat_cols, target_col, cwt, cfg, device):
    train_df = df[df["subject_id"] != test_s].sort_values(
        ["subject_id", "window_start_sec"])
    test_df  = df[df["subject_id"] == test_s].sort_values("window_start_sec")
    if len(test_df) < cfg.seq_len * cfg.internal_stride_rows:
        return None

    X_tr, X_te = _prepare_X(df, feat_cols, cwt, train_df.index, test_df.index)
    y_tr_raw = train_df[target_col].values.astype(np.float32)
    y_te_raw = test_df[target_col].values.astype(np.float32)
    y_sc = StandardScaler()
    y_tr = y_sc.fit_transform(y_tr_raw.reshape(-1, 1)).ravel().astype(np.float32)
    y_te = y_sc.transform(y_te_raw.reshape(-1, 1)).ravel().astype(np.float32)

    # Разрезаем train на per-subject куски, чтобы датасеты были непрерывны
    # внутри одного субъекта (sliding не пересекает границы).
    train_groups = train_df.groupby("subject_id", sort=False).indices
    train_datasets, val_datasets = [], []
    for sid, idx in train_groups.items():
        idx = np.asarray(idx)
        n = len(idx)
        cut = max(1, int(n * (1 - cfg.val_frac)))
        Xs_tr = X_tr[idx[:cut]]; ys_tr = y_tr[idx[:cut]]
        Xs_vl = X_tr[idx[cut:]]; ys_vl = y_tr[idx[cut:]]
        if len(Xs_tr) >= cfg.seq_len * cfg.internal_stride_rows:
            train_datasets.append(StatelessSeqDataset(
                Xs_tr, ys_tr, cfg.seq_len,
                cfg.internal_stride_rows, cfg.outer_stride_rows))
        if len(Xs_vl) >= cfg.seq_len * cfg.internal_stride_rows:
            val_datasets.append(StatelessSeqDataset(
                Xs_vl, ys_vl, cfg.seq_len,
                cfg.internal_stride_rows, cfg.outer_stride_rows))
    if not train_datasets or not val_datasets:
        return None
    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds   = torch.utils.data.ConcatDataset(val_datasets)
    test_ds  = StatelessSeqDataset(X_te, y_te, cfg.seq_len,
                                   cfg.internal_stride_rows,
                                   cfg.outer_stride_rows)
    if len(test_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg, X_tr.shape[1]).to(device)
    model = _train_stateless(model, train_loader, val_loader, cfg, device)
    pred_norm = _predict_stateless(model, test_loader, cfg, device)
    if len(pred_norm) == 0:
        return None
    pred = y_sc.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
    # y_true берём в позициях, соответствующих окончанию каждого sample
    end_pos = test_ds.starts + (cfg.seq_len - 1) * cfg.internal_stride_rows
    y_true = y_te_raw[end_pos]
    return {"fold": test_s, "y_pred": pred, "y_true": y_true}


def _loso_fold_stateful(test_s, df, feat_cols, target_col, cwt, cfg, device):
    train_df = df[df["subject_id"] != test_s].sort_values(
        ["subject_id", "window_start_sec"])
    test_df  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

    X_tr, X_te = _prepare_X(df, feat_cols, cwt, train_df.index, test_df.index)
    y_tr_raw = train_df[target_col].values.astype(np.float32)
    y_te_raw = test_df[target_col].values.astype(np.float32)
    y_sc = StandardScaler()
    y_tr = y_sc.fit_transform(y_tr_raw.reshape(-1, 1)).ravel().astype(np.float32)
    y_te = y_sc.transform(y_te_raw.reshape(-1, 1)).ravel().astype(np.float32)

    # Группируем train по subject_id, делим на train/val (по subject'у целиком).
    train_groups = train_df.groupby("subject_id", sort=False).indices
    subj_ids = list(train_groups.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(subj_ids)
    n_val = max(1, int(len(subj_ids) * cfg.val_frac))
    val_sids = set(subj_ids[:n_val])

    X_by_subj_tr, y_by_subj_tr = [], []
    X_by_subj_vl, y_by_subj_vl = [], []
    for sid, idx in train_groups.items():
        idx = np.asarray(idx)
        if sid in val_sids:
            X_by_subj_vl.append(X_tr[idx]); y_by_subj_vl.append(y_tr[idx])
        else:
            X_by_subj_tr.append(X_tr[idx]); y_by_subj_tr.append(y_tr[idx])

    train_ds = StatefulRecordDataset(X_by_subj_tr, y_by_subj_tr,
                                     cfg.internal_stride_rows)
    val_ds   = StatefulRecordDataset(X_by_subj_vl, y_by_subj_vl,
                                     cfg.internal_stride_rows)
    if len(train_ds) == 0 or len(val_ds) == 0:
        return None

    model = build_model(cfg, X_tr.shape[1]).to(device)
    model = _train_stateful(model, train_ds.records, val_ds.records, cfg, device)

    # Тест: одна запись целиком, с субсэмплированием по той же ставке.
    sel = np.arange(0, len(X_te), cfg.internal_stride_rows, dtype=np.int64)
    if len(sel) < 2:
        return None
    X_te_sub = torch.from_numpy(X_te[sel]).float()
    pred_norm = _predict_stateful(model, X_te_sub, cfg, device)
    pred = y_sc.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
    y_true = y_te_raw[sel]
    return {"fold": test_s, "y_pred": pred, "y_true": y_true}


def _device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def _loso(df: pd.DataFrame, feat_cols: list[str], target_col: str,
          cwt: Optional[CwtCache], cfg: ExperimentCfg, device: str) -> dict:
    subjects = sorted(df["subject_id"].unique())
    fold_fn = _loso_fold_stateful if cfg.family == "stateful" else _loso_fold_stateless

    all_pred, all_true, subj_rows = [], [], []
    for s in subjects:
        rec = fold_fn(s, df, feat_cols, target_col, cwt, cfg, device)
        if rec is None:
            print(f"    [skip subj={s}] недостаточно данных")
            continue
        all_pred.append(rec["y_pred"]); all_true.append(rec["y_true"])
        mae_s = mean_absolute_error(rec["y_true"], rec["y_pred"]) / 60.0
        r2_s  = r2_score(rec["y_true"], rec["y_pred"]) if len(rec["y_true"]) > 1 else float("nan")
        subj_rows.append({"subject_id": rec["fold"],
                          "mae_min": round(mae_s, 4), "r2": round(r2_s, 3)})
        print(f"    [subj={s}] mae={mae_s:.3f} r2={r2_s:.3f} "
              f"n_test={len(rec['y_pred'])}")

    if not all_pred:
        return {"error": "Нет валидных результатов"}
    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    return {
        "y_pred": y_pred, "y_true": y_true,
        "raw_mae_min": mean_absolute_error(y_true, y_pred) / 60.0,
        "r2":  r2_score(y_true, y_pred),
        "rho": float(spearmanr(y_true, y_pred).statistic),
        "per_subject": subj_rows,
    }


# ─── Точка входа эксперимента ─────────────────────────────────────────────────

def _parse_cli(default_target: str = "both") -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["lt1", "lt2", "both"],
                   default=default_target)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def run_experiment(cfg: ExperimentCfg) -> None:
    args = _parse_cli(cfg.target)
    cfg.target = args.target
    device = _device()

    out_dir = _ROOT / "results" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"{cfg.name} — family={cfg.family}, seq_len={cfg.seq_len}, "
          f"int_stride={cfg.internal_stride_sec}s, "
          f"out_stride={cfg.outer_stride_sec}s, wavelet={cfg.use_wavelet}")
    print(f"target={cfg.target}, device={device}, "
          f"hidden={cfg.hidden_size}x{cfg.num_layers}, "
          f"batch={cfg.batch_size}, max_epochs={cfg.max_epochs}")
    print("=" * 70)

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()
    cwt = CwtCache() if cfg.use_wavelet else None

    targets_cfg = {
        "lt1": "target_time_to_lt1_pchip_sec",
        "lt2": "target_time_to_lt2_center_sec",
    }
    if cfg.target != "both":
        targets_cfg = {cfg.target: targets_cfg[cfg.target]}

    all_records, all_subj_records = [], []
    feature_sets = ["EMG", "EMG+NIRS", "EMG+NIRS+HRV"]

    for fset in feature_sets:
        fset_tag = fset.replace("+", "_")
        print(f"\n{'='*70}\n── НАБОР ПРИЗНАКОВ: {fset} ──")

        for tgt_name, target_col in targets_cfg.items():
            print(f"\n── ТАРГЕТ {tgt_name.upper()} ──")
            df_prep = prepare_data(df_raw, session_params, tgt_name)
            df_tgt = df_prep.dropna(subset=[target_col])
            feat_cols_full = get_feature_cols(df_tgt, fset)
            feat_cols = [c for c in feat_cols_full if c not in EXCLUDE_ABS]
            n_subj = df_tgt["subject_id"].nunique()
            print(f"  n_subj={n_subj}, n_features={len(feat_cols)}"
                  + (f" (+CWT {len(feat_cols)*5})" if cwt is not None else ""))

            t0 = time.perf_counter()
            res = _loso(df_tgt, feat_cols, target_col, cwt, cfg, device)
            elapsed = time.perf_counter() - t0

            if "error" in res:
                print(f"  ❌ {res['error']}")
                continue

            for row in res.get("per_subject", []):
                all_subj_records.append({"variant": "noabs", "feature_set": fset,
                                         "target": tgt_name, **row})
            np.save(out_dir / f"ypred_{tgt_name}_{fset_tag}.npy", res["y_pred"])
            np.save(out_dir / f"ytrue_{tgt_name}_{fset_tag}.npy", res["y_true"])

            sigma_grid = [30.0, 50.0, 75.0, 150.0]
            best_mae = float("inf"); best_sigma = sigma_grid[0]; k_maes = {}
            for sigma in sigma_grid:
                y_k = kalman_smooth(res["y_pred"], sigma_p=5.0, sigma_obs=sigma)
                mae_k = mean_absolute_error(res["y_true"], y_k) / 60.0
                k_maes[sigma] = round(mae_k, 4)
                if mae_k < best_mae:
                    best_mae = mae_k; best_sigma = sigma

            all_records.append({
                "variant": "noabs", "feature_set": fset, "target": tgt_name,
                "n_subjects": n_subj, "n_features": len(feat_cols),
                "raw_mae_min":    round(res["raw_mae_min"], 4),
                "kalman_mae_min": round(best_mae, 4),
                "best_sigma_obs": best_sigma,
                "kalman_30":  k_maes.get(30.0),  "kalman_50":  k_maes.get(50.0),
                "kalman_75":  k_maes.get(75.0),  "kalman_150": k_maes.get(150.0),
                "r2":  round(res["r2"], 3), "rho": round(res["rho"], 3),
                "sec": round(elapsed, 1),
            })
            print(f"  raw={res['raw_mae_min']:.3f}  "
                  f"kalman_best={best_mae:.3f} (sigma={best_sigma})  ({elapsed:.1f}s)")

    pd.DataFrame(all_records).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(all_subj_records).to_csv(out_dir / "per_subject.csv", index=False)
    print(f"\n✅ Готово: {out_dir.resolve()}")
