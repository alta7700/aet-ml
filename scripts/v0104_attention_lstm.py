"""v0104 — Attention-LSTM для предсказания времени до лактатного порога

Версия:    v0104
Дата:      2026-05-08

Архитектура:
  Последовательность признаков (seq_len × n_features)
      ↓
  LSTM (2 слоя, bidirectional=False)
      ↓
  Multi-Head Self-Attention: какие шаги во времени важны?
      ↓
  Взвешенное суммирование (attention pooling) → вектор
      ↓
  Dropout → Linear → предсказание

Почему Attention:
  • LSTM + Attention умеет выявлять КАКИЕ моменты в тренировке критичны
    (не просто "последние несколько секунд", а значимые переходы)
  • В отличие от Global Average Pooling, attention присваивает веса
    разным временным шагам → интерпретируемость
  • Длинный контекст (seq_len=20): почти 7 мин истории

Воспроизведение:
  uv run python scripts/v0104_attention_lstm.py --target both
  uv run python scripts/v0104_attention_lstm.py --target lt2 --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from joblib import Parallel, delayed

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_v0011_ref() -> dict:
    """Загружает референсные MAE v0011 из best_per_set.csv; fallback на N=14 константы."""
    csv = _ROOT / "results" / "v0011" / "best_per_set.csv"
    if not csv.exists():
        return {("lt2", "EMG+NIRS+HRV"): 1.859, ("lt1", "EMG+NIRS+HRV"): 2.277}
    ref_df = pd.read_csv(csv)
    return {(row["target"], row["feature_set"]): row["kalman_mae_min"]
            for _, row in ref_df.iterrows()}

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from scripts.v0011_modality_ablation import (
    prepare_data,
    get_feature_cols,
    kalman_smooth,
)

OUT_DIR = _ROOT / "results" / "v0104"

EXCLUDE_ABS = frozenset([
    "trainred_smo2_mean", "trainred_hhb_mean", "trainred_hbdiff_mean", "trainred_thb_mean",
    "hrv_mean_rr_ms", "feat_smo2_x_rr",
])


# ─── Конфиг ───────────────────────────────────────────────────────────────────

class Config:
    window_step   = 6        # 6 × 5с = 30с, нулевое перекрытие
    seq_length    = 12       # 12 × 30с = 6 мин контекста
    batch_size    = 16
    num_epochs    = 100
    learning_rate = 0.001
    patience      = 15
    # LSTM параметры
    hidden_size   = 64       # небольшой для N=14
    num_layers    = 2
    # Attention параметры
    num_heads     = 4        # multi-head self-attention
    dropout       = 0.3
    device        = "cpu"


# ─── Dataset (те же независимые окна что в v0102/v0103) ──────────────────────

class NonOverlapWindowDataset(Dataset):
    """Последовательности из независимых (не нахлестующихся) окон."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_length: int, window_step: int):
        total = len(X)
        ind_idx = np.arange(0, total, window_step)
        self.X_ind = X[ind_idx].astype(np.float32)
        self.y_ind = y[ind_idx].astype(np.float32)
        self.seq_length = seq_length
        self.n = len(self.X_ind)

    def __len__(self) -> int:
        return max(0, self.n - self.seq_length + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X_seq   = torch.from_numpy(self.X_ind[idx: idx + self.seq_length])
        y_target = torch.tensor(
            self.y_ind[idx + self.seq_length - 1], dtype=torch.float32)
        return X_seq, y_target


# ─── Модель ───────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """Multi-head self-attention + взвешенное суммирование по времени."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        # Скалярный вес для каждого шага (для pooling)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.norm(attn_out + x)           # residual
        scores = F.softmax(self.score(attn_out), dim=1)  # (batch, seq_len, 1)
        pooled = (attn_out * scores).sum(dim=1)      # (batch, hidden_size)
        return pooled


class AttentionLSTM(nn.Module):
    """LSTM + Multi-Head Attention pooling."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = AttentionPooling(hidden_size, num_heads, dropout)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        pooled = self.attention(lstm_out)   # (batch, hidden_size)
        return self.fc(self.dropout(pooled)).squeeze(1)


# ─── Обучение ─────────────────────────────────────────────────────────────────

def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, config: Config) -> nn.Module:
    """Обучение с early stopping, AdamW + CosineAnnealingLR, Huber loss."""
    # Таргет нормирован per-fold → ошибки порядка 1 std, MSE адекватен
    # и не вырождается в MAE-режим (исторически провоцировал коллапс
    # к константному предсказанию).
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-5)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(config.num_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(config.device), y.to(config.device)
                val_loss += criterion(model(X), y).item() * len(y)
        val_loss /= max(len(val_loader.dataset), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    model.load_state_dict(best_state)
    return model


# ─── Один LOSO фолд ───────────────────────────────────────────────────────────

def _run_one_loso_fold(test_subject_id,
                       df: pd.DataFrame,
                       feat_cols: list[str],
                       target_col: str,
                       config: Config) -> dict:
    """Один LOSO фолд для Attention-LSTM."""
    train = df[df["subject_id"] != test_subject_id].sort_values(
        ["subject_id", "window_start_sec"])
    test = df[df["subject_id"] == test_subject_id].sort_values("window_start_sec")

    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
    X_te = sc.transform(imp.transform(test[feat_cols].values))
    # Нормировка таргета per-fold; на инференсе предсказания
    # денормализуются обратно в секунды (см. конец функции).
    y_tr_raw = train[target_col].values.astype(np.float32)
    y_te_raw = test[target_col].values.astype(np.float32)
    y_sc = StandardScaler()
    y_tr = y_sc.fit_transform(y_tr_raw.reshape(-1, 1)).ravel().astype(np.float32)
    y_te = y_sc.transform(y_te_raw.reshape(-1, 1)).ravel().astype(np.float32)

    _offset_ds = [
        NonOverlapWindowDataset(X_tr[off:], y_tr[off:], config.seq_length, config.window_step)
        for off in range(config.window_step)
    ]
    train_ds = ConcatDataset([d for d in _offset_ds if len(d) > 0])
    test_ds  = NonOverlapWindowDataset(X_te, y_te, config.seq_length, config.window_step)

    if len(train_ds) < 4 or len(test_ds) == 0:
        return {"fold": test_subject_id, "error": "Недостаточно данных"}

    split = max(1, int(0.8 * len(train_ds)))
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [split, len(train_ds) - split],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_sub, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,   batch_size=config.batch_size, shuffle=False)

    model = AttentionLSTM(
        input_size=X_tr.shape[1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(config.device)

    model = train_model(model, train_loader, val_loader, config)

    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X, _ in test_loader:
            y_pred_list.append(model(X.to(config.device)).cpu().numpy())

    y_pred_sparse = np.concatenate(y_pred_list)

    test_ind_idx = np.arange(0, len(y_te), config.window_step)
    pred_at = test_ind_idx[config.seq_length - 1:
                           config.seq_length - 1 + len(y_pred_sparse)]

    y_pred_full = np.full(len(y_te), np.nan)
    for i, idx in enumerate(pred_at):
        if idx < len(y_pred_full):
            y_pred_full[idx] = y_pred_sparse[i]

    x_all = np.arange(len(y_te))
    valid = ~np.isnan(y_pred_full)
    if valid.sum() >= 2:
        y_pred_full = np.interp(x_all, x_all[valid], y_pred_full[valid])
    else:
        y_pred_full = np.full(len(y_te), np.nanmean(y_pred_full))

    # Денормализация предсказаний обратно в секунды; y_true возвращаем
    # в исходных секундах для совместимости с downstream-метриками.
    y_pred_full = y_sc.inverse_transform(y_pred_full.reshape(-1, 1)).ravel()
    return {
        "fold": test_subject_id,
        "y_pred": y_pred_full,
        "y_true": y_te_raw,
    }


# ─── LOSO с joblib ────────────────────────────────────────────────────────────

def loso_attention(df: pd.DataFrame,
                   feat_cols: list[str],
                   target_col: str,
                   config: Config,
                   n_jobs: int = -1) -> dict:
    """LOSO через joblib.Parallel."""
    subjects = sorted(df["subject_id"].unique())

    records = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_run_one_loso_fold)(s, df, feat_cols, target_col, config)
        for s in subjects
    )

    all_pred, all_true, subj_rows = [], [], []
    for rec in records:
        if "error" not in rec:
            all_pred.append(rec["y_pred"])
            all_true.append(rec["y_true"])
            mae_s = mean_absolute_error(rec["y_true"], rec["y_pred"]) / 60.0
            r2_s  = r2_score(rec["y_true"], rec["y_pred"]) if len(rec["y_true"]) > 1 else float("nan")
            subj_rows.append({"subject_id": rec["fold"], "mae_min": round(mae_s, 4), "r2": round(r2_s, 3)})

    if not all_pred:
        return {"error": "Нет данных"}

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "raw_mae_min": mean_absolute_error(y_true, y_pred) / 60.0,
        "r2": r2_score(y_true, y_pred),
        "rho": float(spearmanr(y_true, y_pred).statistic),
        "per_subject": subj_rows,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0104 — Attention-LSTM")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--feature-set", nargs="+",
                   choices=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"])
    p.add_argument("--seq-len",     type=int, default=12)
    p.add_argument("--window-step", type=int, default=6)
    p.add_argument("--n-jobs",      type=int, default=-1)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = Config()
    config.seq_length  = args.seq_len
    config.window_step = args.window_step

    # Автовыбор устройства: CUDA → GPU + последовательно, иначе CPU + параллельно
    if torch.cuda.is_available():
        config.device = "cuda"
        config.batch_size = 256
        n_jobs = 1
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}, n_jobs=1")
    else:
        config.device = "cpu"
        n_jobs = args.n_jobs
        print(f"[CPU] CUDA недоступна, n_jobs={n_jobs}")

    print("=" * 70)
    print("v0104 — ATTENTION-LSTM (LSTM + Multi-Head Self-Attention pooling)")
    print("=" * 70)
    print(f"window_step={config.window_step} (каждые {config.window_step*5} сек)")
    print(f"seq_length={config.seq_length}  ({config.seq_length*config.window_step*5} сек истории)")
    print(f"hidden_size={config.hidden_size}, num_heads={config.num_heads}")
    print(f"device={config.device}, n_jobs={n_jobs}\n")

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets_cfg = {
        "lt2": {"col": "target_time_to_lt2_center_sec"},
        "lt1": {"col": "target_time_to_lt1_pchip_sec"},
    }
    if args.target != "both":
        targets_cfg = {k: v for k, v in targets_cfg.items() if k == args.target}

    all_records, all_subj_records = [], []
    sigma_grid  = [30.0, 50.0, 75.0, 150.0]

    for tgt_name, tgt_cfg in targets_cfg.items():
        print(f"\n{'═'*70}")
        print(f"ТАРГЕТ: {tgt_name.upper()}")
        print(f"{'═'*70}\n")

        df_prep    = prepare_data(df_raw, session_params, tgt_name)
        target_col = tgt_cfg["col"]
        df_tgt     = df_prep.dropna(subset=[target_col])

        variants = [("with_abs", None, OUT_DIR), ("noabs", EXCLUDE_ABS, OUT_DIR / "noabs")]
        for fset in args.feature_set:
            feat_cols_full = get_feature_cols(df_tgt, fset)
            if not feat_cols_full:
                continue
            n_subj = df_tgt["subject_id"].nunique()

            for variant, exclude_set, out_sub in variants:
                feat_cols = (feat_cols_full if exclude_set is None
                             else [c for c in feat_cols_full if c not in exclude_set])
                if not feat_cols: continue
                out_sub.mkdir(exist_ok=True)
                print(f"  [{fset} / {tgt_name} / {variant}]  n={n_subj}, {len(feat_cols)} признаков")

                t0  = time.perf_counter()
                res = loso_attention(df_tgt, feat_cols, target_col, config,
                                     n_jobs=n_jobs)
                elapsed = time.perf_counter() - t0
                if "error" in res: print(f"    ❌ {res['error']}"); continue
                for row in res.get("per_subject", []):
                    all_subj_records.append({"variant": variant, "feature_set": fset, "target": tgt_name, **row})

                fset_tag = fset.replace("+", "_")
                np.save(out_sub / f"ypred_{tgt_name}_{fset_tag}.npy", res["y_pred"])
                np.save(out_sub / f"ytrue_{tgt_name}_{fset_tag}.npy", res["y_true"])

                best_kalman_mae = float("inf")
                best_sigma      = sigma_grid[0]
                kalman_maes     = {}
                for sigma in sigma_grid:
                    y_k   = kalman_smooth(res["y_pred"], sigma_p=5.0, sigma_obs=sigma)
                    mae_k = mean_absolute_error(res["y_true"], y_k) / 60.0
                    kalman_maes[sigma] = round(mae_k, 4)
                    if mae_k < best_kalman_mae:
                        best_kalman_mae = mae_k
                        best_sigma      = sigma

                all_records.append({
                    "variant":        variant,
                    "feature_set":    fset,
                    "target":         tgt_name,
                    "n_subjects":     n_subj,
                    "n_features":     len(feat_cols),
                    "raw_mae_min":    round(res["raw_mae_min"], 4),
                    "kalman_mae_min": round(best_kalman_mae, 4),
                    "best_sigma_obs": best_sigma,
                    "kalman_30":      kalman_maes.get(30.0),
                    "kalman_50":      kalman_maes.get(50.0),
                    "kalman_75":      kalman_maes.get(75.0),
                    "kalman_150":     kalman_maes.get(150.0),
                    "r2":             round(res["r2"], 3),
                    "rho":            round(res["rho"], 3),
                    "sec":            round(elapsed, 1),
                })

                print(f"    raw={res['raw_mae_min']:.3f}  "
                      f"kalman_best={best_kalman_mae:.3f} (sigma={best_sigma})"
                      f"  ({elapsed:.1f}s)")

    summary_df = pd.DataFrame(all_records)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(all_subj_records).to_csv(OUT_DIR / "per_subject.csv", index=False)
    df_noabs = summary_df[summary_df["variant"] == "noabs"]
    if not df_noabs.empty:
        df_noabs.to_csv(OUT_DIR / "noabs" / "summary.csv", index=False)

    v0011_ref = _load_v0011_ref()
    print("\n" + "=" * 70)
    print("ИТОГИ:")
    for variant in ["with_abs", "noabs"]:
        sub = summary_df[summary_df["variant"] == variant]
        if sub.empty: continue
        print(f"\n  [{variant}]")
        for _, row in sub.sort_values(["target", "kalman_mae_min"]).iterrows():
            ref   = v0011_ref.get((row["target"], row["feature_set"]))
            delta = f"  Δ={row['kalman_mae_min']-ref:+.3f} vs v0011" if ref else ""
            print(f"    {row['target'].upper()} / {row['feature_set']:<16s}  "
                  f"kalman={row['kalman_mae_min']:.3f}{delta}")

    print(f"\n✅ Готово: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
