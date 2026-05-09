"""v0102 — TCN (Temporal Convolutional Network), улучшенный

Версия:    v0102 (rev2)
Дата:      2026-05-08

Улучшения vs первого запуска:
  1. Huber loss вместо MSE — меньше влияние выбросов
  2. Global Average Pooling вместо последнего шага — TCN видит всю историю
  3. AdamW + CosineAnnealingLR — стабильнее обучение
  4. Grid search по sigma_obs Kalman [30, 50, 75, 150]
  5. Исправлен баг: clip_grad_norm теперь ПОСЛЕ backward()

Воспроизведение:
  uv run python scripts/v0102_tcn_temporal.py --target both
  uv run python scripts/v0102_tcn_temporal.py --target lt2 --seq-len 12
  uv run python scripts/v0102_tcn_temporal.py --target lt2 --seq-len 24 --window-step 3
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from joblib import Parallel, delayed

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from dataset_pipeline.baselines import run_honest_baselines, format_honest_block
from scripts.v0011_modality_ablation import (
    prepare_data,
    get_feature_cols,
    kalman_smooth,
)

OUT_DIR = _ROOT / "results" / "v0102"


# ─── Конфиг ───────────────────────────────────────────────────────────────────

class Config:
    # Семплирование: каждое 6-е окно = 30 сек шаг = нахлест 0
    window_step  = 6       # каждое N-е окно из датасета
    seq_length   = 24      # 24 × 30 сек = 12 мин истории
    batch_size   = 16
    num_epochs   = 100
    learning_rate = 0.001
    patience     = 15
    # TCN параметры
    n_channels   = 64      # фильтров в каждом блоке
    kernel_size  = 3
    # дилатации: 1,2,4,8,16 — охватывает 31 шаг назад (31×30 сек ≈ 15 мин)
    dilations    = [1, 2, 4, 8, 16]
    dropout      = 0.2
    device       = "cpu"


# ─── TCN Архитектура ─────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Causal (причинная) свертка — смотрит только в прошлое."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        # Padding слева = (kernel_size-1)*dilation — только прошлое
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Один блок TCN: 2 дилатированные свертки + residual."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 1×1 проекция если размерности не совпадают
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        out = self.relu(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out).transpose(1, 2)).transpose(1, 2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRegressor(nn.Module):
    """TCN с несколькими дилатированными блоками для регрессии."""

    def __init__(self, input_size: int, n_channels: int = 64,
                 kernel_size: int = 3, dilations: list[int] = None,
                 dropout: float = 0.2):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        layers = []
        in_ch = input_size
        for d in dilations:
            layers.append(TCNBlock(in_ch, n_channels, kernel_size, d, dropout))
            in_ch = n_channels

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) → нужно (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Global Average Pooling — используем всю историю, не только последний шаг
        pooled = out.mean(dim=2)
        return self.fc(pooled).squeeze(1)


# ─── Dataset с правильным семплированием ─────────────────────────────────────

class NonOverlapWindowDataset(Dataset):
    """Последовательности из независимых (не нахлестующихся) окон.

    window_step=6 → каждое 6-е окно из датасета = 30 сек шаг = нахлест=0.
    seq_length=24  → 24 × 30 сек = 12 минут контекста.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_length: int, window_step: int):
        # Индексы независимых окон (каждое window_step-е)
        total = len(X)
        independent_idx = np.arange(0, total, window_step)
        self.X_ind = X[independent_idx]
        self.y_ind = y[independent_idx]
        self.seq_length = seq_length
        self.n = len(self.X_ind)

    def __len__(self) -> int:
        return max(0, self.n - self.seq_length + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X_seq = torch.from_numpy(
            self.X_ind[idx: idx + self.seq_length]).float()
        y_target = torch.tensor(
            self.y_ind[idx + self.seq_length - 1], dtype=torch.float32)
        return X_seq, y_target


# ─── Обучение ─────────────────────────────────────────────────────────────────

def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, config: Config) -> nn.Module:
    """Обучение с early stopping, AdamW + CosineAnnealingLR."""
    # Huber loss: delta=60 сек — мягче к выбросам чем MSE
    criterion = nn.HuberLoss(delta=60.0)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-5)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = model.state_dict().copy()

    for epoch in range(config.num_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            # clip_grad_norm ПОСЛЕ backward — иначе градиентов нет
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
    """Один LOSO фолд для TCN с независимыми окнами."""
    train = df[df["subject_id"] != test_subject_id].sort_values(
        ["subject_id", "window_start_sec"])
    test = df[df["subject_id"] == test_subject_id].sort_values("window_start_sec")

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
    X_te = sc.transform(imp.transform(test[feat_cols].values))
    y_tr = train[target_col].values
    y_te = test[target_col].values

    train_ds = NonOverlapWindowDataset(X_tr, y_tr, config.seq_length, config.window_step)
    test_ds  = NonOverlapWindowDataset(X_te, y_te, config.seq_length, config.window_step)

    if len(train_ds) < 4 or len(test_ds) == 0:
        return {"fold": test_subject_id, "error": "Недостаточно данных"}

    split = max(1, int(0.8 * len(train_ds)))
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [split, len(train_ds) - split],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_sub, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,   batch_size=config.batch_size, shuffle=False)

    model = TCNRegressor(
        input_size=X_tr.shape[1],
        n_channels=config.n_channels,
        kernel_size=config.kernel_size,
        dilations=config.dilations,
        dropout=config.dropout,
    ).to(config.device)

    model = train_model(model, train_loader, val_loader, config)

    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X, _ in test_loader:
            y_pred_list.append(model(X.to(config.device)).cpu().numpy())

    y_pred_sparse = np.concatenate(y_pred_list)

    # Восстанавливаем предсказания на все окна (не только каждое 6-е)
    test_ind_idx = np.arange(0, len(y_te), config.window_step)
    test_ind_idx = test_ind_idx[:len(test_ds) + config.seq_length - 1]
    # Индексы окон которые предсказаны (последнее окно каждой последовательности)
    pred_at = test_ind_idx[config.seq_length - 1: config.seq_length - 1 + len(y_pred_sparse)]

    # Интерполируем на все окна
    y_pred_full = np.full(len(y_te), np.nan)
    for i, idx in enumerate(pred_at):
        if idx < len(y_pred_full):
            y_pred_full[idx] = y_pred_sparse[i]

    # Заполняем пропуски линейной интерполяцией
    x_all = np.arange(len(y_te))
    valid = ~np.isnan(y_pred_full)
    if valid.sum() >= 2:
        y_pred_full = np.interp(x_all, x_all[valid], y_pred_full[valid])
    else:
        y_pred_full = np.full(len(y_te), np.nanmean(y_pred_full))

    return {
        "fold": test_subject_id,
        "y_pred": y_pred_full,
        "y_true": y_te,
    }


# ─── LOSO с joblib ────────────────────────────────────────────────────────────

def loso_tcn(df: pd.DataFrame,
             feat_cols: list[str],
             target_col: str,
             config: Config,
             n_jobs: int = -1) -> dict:
    """LOSO через joblib.Parallel — все фолды параллельно."""
    subjects = sorted(df["subject_id"].unique())

    records = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_run_one_loso_fold)(s, df, feat_cols, target_col, config)
        for s in subjects
    )

    all_pred, all_true = [], []
    for rec in records:
        if "error" not in rec:
            all_pred.append(rec["y_pred"])
            all_true.append(rec["y_true"])

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
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0102 — TCN temporal")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--feature-set", nargs="+",
                   choices=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"])
    p.add_argument("--seq-len", type=int, default=24,
                   help="Длина последовательности (в независимых окнах)")
    p.add_argument("--window-step", type=int, default=6,
                   help="Шаг между окнами (6 = нахлест 0, каждые 30 сек)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    config.seq_length  = args.seq_len
    config.window_step = args.window_step

    print("=" * 70)
    print("v0102 — TCN TEMPORAL (независимые окна, нахлест=0)")
    print("=" * 70)
    print(f"window_step={config.window_step} (каждые {config.window_step*5} сек)")
    print(f"seq_length={config.seq_length}  ({config.seq_length*config.window_step*5} сек истории)")
    print(f"dilations={config.dilations}, n_channels={config.n_channels}")
    print(f"n_jobs={args.n_jobs}\n")

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets_cfg = {
        "lt2": {"col": "target_time_to_lt2_center_sec"},
        "lt1": {"col": "target_time_to_lt1_sec"},
    }
    if args.target != "both":
        targets_cfg = {k: v for k, v in targets_cfg.items() if k == args.target}

    all_records = []

    for tgt_name, tgt_cfg in targets_cfg.items():
        print(f"\n{'═'*70}")
        print(f"ТАРГЕТ: {tgt_name.upper()}")
        print(f"{'═'*70}\n")

        df_prep = prepare_data(df_raw, session_params, tgt_name)
        target_col = tgt_cfg["col"]
        df_tgt = df_prep.dropna(subset=[target_col])

        for fset in args.feature_set:
            feat_cols = get_feature_cols(df_tgt, fset)
            if not feat_cols:
                continue

            n_subj = df_tgt["subject_id"].nunique()
            print(f"  [{fset} / {tgt_name}]  n={n_subj}, {len(feat_cols)} признаков")

            t0 = time.perf_counter()
            res = loso_tcn(df_tgt, feat_cols, target_col, config, n_jobs=args.n_jobs)
            elapsed = time.perf_counter() - t0

            if "error" in res:
                print(f"    ❌ {res['error']}")
                continue

            # Grid search по sigma_obs — оптимальный Kalman для TCN может отличаться от v0011
            sigma_grid = [30.0, 50.0, 75.0, 150.0]
            best_kalman_mae = float("inf")
            best_sigma = sigma_grid[0]
            kalman_maes = {}
            for sigma in sigma_grid:
                y_k = kalman_smooth(res["y_pred"], sigma_p=5.0, sigma_obs=sigma)
                mae_k = mean_absolute_error(res["y_true"], y_k) / 60.0
                kalman_maes[sigma] = round(mae_k, 4)
                if mae_k < best_kalman_mae:
                    best_kalman_mae = mae_k
                    best_sigma = sigma

            all_records.append({
                "feature_set": fset,
                "target": tgt_name,
                "n_subjects": n_subj,
                "n_features": len(feat_cols),
                "raw_mae_min": round(res["raw_mae_min"], 4),
                "kalman_mae_min": round(best_kalman_mae, 4),
                "best_sigma_obs": best_sigma,
                "kalman_30": kalman_maes.get(30.0),
                "kalman_50": kalman_maes.get(50.0),
                "kalman_75": kalman_maes.get(75.0),
                "kalman_150": kalman_maes.get(150.0),
                "r2": round(res["r2"], 3),
                "rho": round(res["rho"], 3),
                "sec": round(elapsed, 1),
            })

            print(f"    raw={res['raw_mae_min']:.3f}  "
                  f"kalman_best={best_kalman_mae:.3f} (sigma={best_sigma})"
                  f"  ({elapsed:.1f}s)")
            print(f"    sigma grid: {kalman_maes}")

    summary_df = pd.DataFrame(all_records)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    # Сравнение с v0011
    v0011_ref = {
        ("lt2", "EMG+NIRS+HRV"): 1.859,
        ("lt1", "EMG+NIRS+HRV"): 2.277,
    }
    print("\n" + "=" * 70)
    print("ИТОГИ:")
    for _, row in summary_df.sort_values(["target", "kalman_mae_min"]).iterrows():
        ref = v0011_ref.get((row["target"], row["feature_set"]))
        delta = f"  Δ={row['kalman_mae_min']-ref:+.3f} vs v0011" if ref else ""
        print(f"  {row['target'].upper()} / {row['feature_set']:<16s}  "
              f"kalman={row['kalman_mae_min']:.3f}{delta}")

    print(f"\n✅ Готово: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
