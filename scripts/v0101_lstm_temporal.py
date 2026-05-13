"""v0101 — Temporal LSTM/GRU модели для tracking лактатного порога

Версия:    v0101
Дата:      2026-05-08
Предыдущая версия: v0011 (ElasticNet/GBM, MAE LT2=1.859 мин)

Что делает:
  Baseline LSTM модели для capturing динамики внутри теста.
  LOSO валидация + Kalman сглаживание.
  Параллелизация на все ядра через joblib.

  Наборы признаков:
    EMG          — 77 признаков
    NIRS         — 18 признаков
    EMG+NIRS     — 95 признаков
    EMG+NIRS+HRV — 105 признаков

  Архитектуры (на каждый LOSO fold):
    LSTM1   — 1 слой LSTM (hidden=32)
    LSTM2   — 2 слоя LSTM (hidden=32)
    GRU1    — 1 слой GRU (hidden=32)

  Выход:
    results/v0101/summary.csv         — все конфиги × MAE
    results/v0101/best_per_set.csv    — лучший per (набор, таргет)
    results/v0101/honest_baselines.csv
    results/v0101/report.md

  Воспроизведение:
    uv run python scripts/v0101_lstm_temporal.py --target both
    uv run python scripts/v0101_lstm_temporal.py --n-jobs 4
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Optional
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
from sklearn.linear_model import Ridge
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

OUT_DIR = _ROOT / "results" / "v0101"


# ─── Конфиг ───────────────────────────────────────────────────────────────────

class Config:
    """Оптимизированные параметры LSTM на CPU."""
    seq_length = 15          # 75 сек контекста (было 5)
    batch_size = 16          # точнее градиент (было 32)
    num_epochs = 100         # больше обучения (было 30)
    learning_rate = 0.0005   # медленнее, стабильнее (было 0.001)
    hidden_size = 128        # в 4 раза больше (было 32)
    dropout = 0.4            # больше регуляризации (было 0.3)
    patience = 20            # больше терпения (было 10)
    device = "cpu"


# ─── Dataset ───────────────────────────────────────────────────────────────────

class TemporalWindowDataset(Dataset):
    """Создаёт последовательности окон для LSTM."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.valid_indices = list(range(len(X) - seq_length + 1))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.valid_indices[idx]
        end = start + self.seq_length
        X_seq = torch.from_numpy(self.X[start:end]).float()
        y_target = torch.tensor(self.y[end - 1], dtype=torch.float32)
        return X_seq, y_target


# ─── Модели ───────────────────────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.fc(h_n[-1]).squeeze(1)


# ─── Обучение ─────────────────────────────────────────────────────────────────

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: Config) -> nn.Module:
    """Обучает модель с early stopping."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(config.device), y.to(config.device)
                y_pred = model(X)
                val_loss += criterion(y_pred, y).item() * len(y)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            model.load_state_dict(best_state)
            break

    return model


def _run_one_loso_fold(test_subject_id: int,
                       df: pd.DataFrame,
                       feat_cols: list[str],
                       target_col: str,
                       models_specs: list[dict],
                       config: Config) -> dict:
    """Запускает обучение для одного LOSO фолда (один test subject)."""
    train = df[df["subject_id"] != test_subject_id].sort_values(
        ["subject_id", "window_start_sec"])
    test = df[df["subject_id"] == test_subject_id].sort_values("window_start_sec")

    if len(train) < config.seq_length or len(test) < config.seq_length:
        return {"fold": test_subject_id, "error": "Недостаточно данных"}

    # Препроцессинг
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
    X_te = sc.transform(imp.transform(test[feat_cols].values))
    y_tr = train[target_col].values
    y_te = test[target_col].values

    # Dataset и DataLoader
    train_ds = TemporalWindowDataset(X_tr, y_tr, config.seq_length)
    test_ds = TemporalWindowDataset(X_te, y_te, config.seq_length)

    if len(train_ds) == 0 or len(test_ds) == 0:
        return {"fold": test_subject_id, "error": "Нет валидных последовательностей"}

    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds_split, val_ds_split = torch.utils.data.random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds_split, batch_size=config.batch_size,
                               shuffle=True)
    val_loader = DataLoader(val_ds_split, batch_size=config.batch_size,
                             shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # Обучаем модели и собираем предсказания
    fold_preds = []
    fold_times = []

    for spec in models_specs:
        t0 = time.perf_counter()
        model = spec["factory"](X_tr.shape[1]).to(config.device)
        model = train_model(model, train_loader, val_loader, config)

        model.eval()
        with torch.no_grad():
            y_pred_list = []
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(config.device)
                y_pred = model(X_batch).cpu().numpy()
                y_pred_list.append(y_pred)

        y_pred = np.concatenate(y_pred_list)
        if len(y_pred) < len(y_te):
            pad = np.full(len(y_te) - len(y_pred), y_pred[0])
            y_pred = np.concatenate([pad, y_pred])

        fold_preds.append(y_pred)
        fold_times.append(time.perf_counter() - t0)

    # Усредняем предсказания от всех моделей (ensemble)
    y_pred_ensemble = np.mean(fold_preds, axis=0)

    return {
        "fold": test_subject_id,
        "y_pred": y_pred_ensemble,
        "y_true": y_te,
        "time_sec": np.mean(fold_times),
    }


# ─── Main LOSO с joblib ────────────────────────────────────────────────────────

def loso_temporal(df: pd.DataFrame,
                  feat_cols: list[str],
                  target_col: str,
                  models_specs: list[dict],
                  config: Config,
                  n_jobs: int = -1) -> dict:
    """LOSO через joblib.Parallel для всех фолдов."""
    subjects = sorted(df["subject_id"].unique())

    records = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_run_one_loso_fold)(
            test_s, df, feat_cols, target_col, models_specs, config
        )
        for test_s in subjects
    )

    # Собираем результаты
    all_preds, all_trues = [], []
    for rec in records:
        if "error" not in rec:
            all_preds.append(rec["y_pred"])
            all_trues.append(rec["y_true"])

    if not all_preds:
        return {"error": "Нет валидных результатов"}

    y_pred_all = np.concatenate(all_preds)
    y_true_all = np.concatenate(all_trues)

    return {
        "y_pred": y_pred_all,
        "y_true": y_true_all,
        "raw_mae_min": mean_absolute_error(y_true_all, y_pred_all) / 60.0,
        "r2": r2_score(y_true_all, y_pred_all),
        "rho": float(spearmanr(y_true_all, y_pred_all).statistic),
    }


# ─── Зоопарк моделей ──────────────────────────────────────────────────────────

def build_model_specs(config: Config) -> list[dict]:
    """Спецификации моделей (3 архитектуры)."""
    return [
        {"name": f"LSTM2(h={config.hidden_size})",
         "factory": lambda inp_sz, h=config.hidden_size, d=config.dropout: LSTMRegressor(
             input_size=inp_sz, hidden_size=h, num_layers=2, dropout=d)},
        {"name": f"LSTM3(h={config.hidden_size})",
         "factory": lambda inp_sz, h=config.hidden_size, d=config.dropout: LSTMRegressor(
             input_size=inp_sz, hidden_size=h, num_layers=3, dropout=d)},
        {"name": f"GRU2(h={config.hidden_size})",
         "factory": lambda inp_sz, h=config.hidden_size, d=config.dropout: GRURegressor(
             input_size=inp_sz, hidden_size=h, num_layers=2, dropout=d)},
    ]


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(summary: pd.DataFrame, best: pd.DataFrame, out_dir: Path) -> None:
    """Пишет report.md."""
    lines = [
        "# v0101 — Temporal LSTM/GRU модели\n\n",
        f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n",
        "LSTM/GRU ensemble на всех 3 модели, LOSO, Kalman σ_obs=150.\n\n",
    ]

    for tgt in ["lt2", "lt1"]:
        sub = best[best["target"] == tgt].copy()
        if sub.empty:
            continue
        lines.append(f"## {tgt.upper()}\n\n")
        lines.append("| Набор | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for _, row in sub.sort_values("kalman_mae_min").iterrows():
            lines.append(
                f"| **{row['feature_set']}** | {row['n_subjects']} | "
                f"{row['n_features']} | {row['raw_mae_min']:.3f} | "
                f"**{row['kalman_mae_min']:.3f}** | {row['r2']:.3f} | {row['rho']:.3f} |\n"
            )

    # Сравнение с v0011
    lines.append("\n## vs v0011 (ElasticNet/GBM)\n\n")
    lines.append("| Набор | v0011 | v0101 | Δ |\n")
    lines.append("|---|---|---|---|\n")
    v0011_ref = {
        ("lt2", "EMG"): 3.198,
        ("lt2", "EMG+NIRS"): 3.117,
        ("lt2", "EMG+NIRS+HRV"): 1.859,
        ("lt1", "EMG"): 2.896,
        ("lt1", "EMG+NIRS"): 2.750,
        ("lt1", "EMG+NIRS+HRV"): 2.277,
    }
    for key, v0011_mae in v0011_ref.items():
        tgt, fset = key
        best_row = best[(best["target"] == tgt) & (best["feature_set"] == fset)]
        if not best_row.empty:
            v0101_mae = best_row.iloc[0]["kalman_mae_min"]
            delta = v0101_mae - v0011_mae
            lines.append(f"| {tgt.upper()}/{fset:<15s} | {v0011_mae:.3f} | {v0101_mae:.3f} | {delta:+.3f} |\n")

    p = out_dir / "report.md"
    p.write_text("".join(lines), encoding="utf-8")
    print(f"\n✅ {p.name}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0101 — LSTM temporal с joblib")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--feature-set", nargs="+",
                   choices=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"])
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Число параллельных процессов LOSO (-1 = все ядра)")
    p.add_argument("--sigma-obs", type=float, default=150.0)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("v0101 — LSTM TEMPORAL С JOBLIB PARALLELIZATION")
    print("=" * 70)

    config = Config()
    print(f"Config: hidden={config.hidden_size}, batch={config.batch_size}, "
          f"epochs={config.num_epochs}, device={config.device}")
    print(f"n_jobs={args.n_jobs} (параллельные LOSO фолды)\n")

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets_cfg = {
        "lt2": {"col": "target_time_to_lt2_center_sec", "label": "lt2"},
        "lt1": {"col": "target_time_to_lt1_pchip_sec", "label": "lt1"},
    }
    if args.target != "both":
        targets_cfg = {k: v for k, v in targets_cfg.items() if k == args.target}

    model_specs = build_model_specs(config)
    all_records = []

    for tgt_name, tgt_cfg in targets_cfg.items():
        print(f"\n{'═'*70}")
        print(f"ТАРГЕТ: {tgt_name.upper()}")
        print(f"{'═'*70}\n")

        df_prep = prepare_data(df_raw, session_params, tgt_name)
        target_col = tgt_cfg["col"]
        df_prep_tgt = df_prep.dropna(subset=[target_col])

        for fset in args.feature_set:
            feat_cols = get_feature_cols(df_prep_tgt, fset)
            if not feat_cols:
                continue

            n_subj = df_prep_tgt["subject_id"].nunique()
            n_feat = len(feat_cols)
            print(f"  [{fset} / {tgt_name}]  n={n_subj}, {n_feat} признаков")

            # LOSO с joblib параллелизацией
            t0 = time.perf_counter()
            res = loso_temporal(df_prep_tgt, feat_cols, target_col,
                               model_specs, config, n_jobs=args.n_jobs)
            elapsed = time.perf_counter() - t0

            if "error" in res:
                print(f"    ❌ {res['error']}")
                continue

            # Kalman
            y_pred_k = kalman_smooth(res["y_pred"], sigma_p=5.0,
                                     sigma_obs=args.sigma_obs)
            kalman_mae = mean_absolute_error(res["y_true"], y_pred_k) / 60.0

            all_records.append({
                "feature_set": fset,
                "target": tgt_name,
                "n_subjects": n_subj,
                "n_features": n_feat,
                "raw_mae_min": res["raw_mae_min"],
                "kalman_mae_min": round(kalman_mae, 4),
                "r2": round(res["r2"], 3),
                "rho": round(res["rho"], 3),
                "sec": round(elapsed, 1),
            })

            print(f"    raw={res['raw_mae_min']:.3f}  kalman={kalman_mae:.3f}  ({elapsed:.1f}s)")

    # Сохраняем результаты
    summary_df = pd.DataFrame(all_records)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\n✅ summary.csv ({len(summary_df)} строк)")

    best_df = (summary_df
               .sort_values("kalman_mae_min")
               .groupby(["feature_set", "target"], sort=False)
               .first()
               .reset_index())
    best_df.to_csv(OUT_DIR / "best_per_set.csv", index=False)

    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ (Kalman MAE мин):")
    for tgt in ["lt2", "lt1"]:
        sub = best_df[best_df["target"] == tgt].sort_values("kalman_mae_min")
        if sub.empty:
            continue
        print(f"\n  {tgt.upper()}:")
        for _, row in sub.iterrows():
            print(f"    {row['feature_set']:<16s}  {row['kalman_mae_min']:.3f} мин")

    write_report(summary_df, best_df, OUT_DIR)
    print(f"\n✅ Готово: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
