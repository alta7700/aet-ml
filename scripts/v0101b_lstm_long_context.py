"""v0101b — LSTM с длинным контекстом без перекрытия

Версия:    v0101b
Базируется на: v0101 (LSTM, после ynorm+MSE-фикса)
Дата:      2026-05-16

Отличия от v0101:
  1) Прореживание окон: окна датасета имеют шаг 5 сек, но в одну
     последовательность берём каждое window_step-е окно (по умолчанию 6 →
     30 сек между шагами); итоговый контекст = seq_length × 30 сек.
  2) Без перекрытия внутри обучающего сэмпла (NonOverlapWindowDataset),
     с offset-аугментацией (train: window_step независимых датасетов с
     разным начальным сдвигом, как в v0102).
  3) Контекст конфигурируется CLI: --seq-length 6  (3 мин) или 12 (6 мин).

Цель: проверить, поднимет ли расширенный контекст точность LSTM по
сравнению с 75-сек окном из v0101 на этой задаче.

ОРИГИНАЛЬНЫЙ ЗАГОЛОВОК v0101:
v0101 — LSTM/GRU temporal ensemble

Версия:    v0101 (rev2)
Дата:      2026-05-13

Архитектуры (ensemble на каждый LOSO фолд):
  LSTM2(h=128) — 2 слоя LSTM
  LSTM3(h=128) — 3 слоя LSTM
  GRU2(h=128)  — 2 слоя GRU
  Финальный предикт = среднее трёх моделей.

Fixes vs rev1:
  1. CUDA/MPS → n_jobs=1, CPU → n_jobs=args.n_jobs (устраняет loky-deadlock)
  2. Добавлены варианты with_abs/noabs (EXCLUDE_ABS — те же что v0102)
  3. Per-subject tracking в loso_temporal
  4. ypred/ytrue .npy сохраняются по всем конфигам
  5. Kalman grid [30, 50, 75, 150] вместо фиксированного sigma=150
  6. Корректное клонирование best_state + загрузка в конце train_model

Воспроизведение:
  python scripts/v0101_lstm_temporal.py --target lt1
  python scripts/v0101_lstm_temporal.py --target lt2
  python scripts/v0101_lstm_temporal.py --target both
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
    """Загружает референсные MAE v0011 из best_per_set.csv; fallback на константы."""
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

# Базовое имя; реальный OUT_DIR с суффиксом контекста (_ctx3min / _ctx6min)
# собирается в main() после парсинга CLI, чтобы прогоны с разным seq_length
# не затирали друг друга.
OUT_DIR_BASE = _ROOT / "results" / "v0101b"
OUT_DIR = OUT_DIR_BASE  # перезаписывается в main()

# Абсолютные признаки — исключаются в варианте noabs (идентично v0102)
EXCLUDE_ABS = frozenset([
    "trainred_smo2_mean", "trainred_hhb_mean", "trainred_hbdiff_mean", "trainred_thb_mean",
    "hrv_mean_rr_ms", "feat_smo2_x_rr",
])


# ─── Конфиг ───────────────────────────────────────────────────────────────────

class Config:
    """Параметры LSTM/GRU ансамбля.

    seq_length × window_step × 5 сек = длина контекста.
    Дефолт: 6 × 6 × 5 = 180 сек = 3 мин (6 шагов по 30 сек).
    """
    seq_length    = 6        # число шагов в последовательности (CLI override)
    window_step   = 6        # каждое 6-е окно датасета → 30 сек между шагами
    batch_size    = 16
    num_epochs    = 100
    learning_rate = 0.0005
    hidden_size   = 128
    dropout       = 0.4
    patience      = 20
    device        = "cpu"    # переопределяется в main()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SlidingSubsampledDataset(Dataset):
    """Sliding stride=1 в исходной (5-сек) сетке, выборка внутри окна — с
    шагом window_step (30 сек при window_step=6).

    Для каждого стартового индекса i ∈ [0, len(X) - context]:
      X_seq = X[i : i + context : window_step]  # seq_length шагов
      y     = y[i + context - 1]
    где context = seq_length * window_step.

    Это объединяет преимущества v0101 (плотное покрытие, sliding-1) и
    нового длинного контекста: число обучающих пар сохраняется ~ N − context,
    а внутри окна сохраняется 30-сек прореживание для расширенного охвата.

    Альтернативное толкование старой схемы:
      v0101b «non-overlap + 6 фазовых сдвигов» давало window_step параллельных
      гребёнок (1,6,12,…; 2,7,13,… итд) — всего ≈ N/seq_length × window_step
      сэмплов. Это в seq_length раз меньше, чем настоящий sliding-1, что и
      приводило к коллапсу на длинных контекстах.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_length: int, window_step: int):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.window_step = window_step
        self.context = seq_length * window_step
        self.n = max(0, len(X) - self.context + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = idx + self.context
        X_seq = torch.from_numpy(
            self.X[idx:end:self.window_step]
        ).float()
        y_target = torch.tensor(self.y[end - 1], dtype=torch.float32)
        return X_seq, y_target


# Тестовый датасет совпадает с обучающим — sliding-1 на тесте корректен
# (предсказание для каждой плотной 5-сек точки, начиная с context-1).
StridedTestDataset = SlidingSubsampledDataset


# ─── Модели ───────────────────────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.fc(h_n[-1]).squeeze(1)


# ─── Обучение ─────────────────────────────────────────────────────────────────

def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, config: Config) -> nn.Module:
    """Обучение с early stopping. Загружает best_state в конце."""
    # Таргет нормирован per-fold (см. _run_one_loso_fold) → ошибки порядка 1
    # std, MSE адекватен и не вырождается в MAE-режим, который провоцировал
    # коллапс к константному предсказанию.
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    patience_counter = 0
    # Корректное глубокое копирование весов
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(config.num_epochs):
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

    # Всегда восстанавливаем лучшие веса
    model.load_state_dict(best_state)
    return model


# ─── Один LOSO фолд ───────────────────────────────────────────────────────────

def _run_one_loso_fold(test_subject_id,
                       df: pd.DataFrame,
                       feat_cols: list[str],
                       target_col: str,
                       model_specs: list[dict],
                       config: Config) -> dict:
    """Обучает ensemble LSTM/GRU для одного LOSO фолда."""
    train = df[df["subject_id"] != test_subject_id].sort_values(
        ["subject_id", "window_start_sec"])
    test = df[df["subject_id"] == test_subject_id].sort_values("window_start_sec")

    # Нужна длина >= seq_length × window_step строк (датасет с шагом 5 сек):
    # для seq=6, step=6 это 36 строк (3 мин); для seq=12, step=6 — 72 строки.
    min_rows = config.seq_length * config.window_step
    if len(train) < min_rows or len(test) < min_rows:
        return {"fold": test_subject_id, "error": "Недостаточно данных"}

    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
    X_te = sc.transform(imp.transform(test[feat_cols].values))
    # Нормировка таргета per-fold: модель учится в стандартизированных
    # единицах, на инференсе предсказания денормализуются обратно в секунды.
    # Без этого фикса (исторически таргет шёл в секундах) NN сваливалась
    # к константному предсказанию среднего.
    y_tr_raw = train[target_col].values.astype(np.float32)
    y_te_raw = test[target_col].values.astype(np.float32)
    y_sc = StandardScaler()
    y_tr = y_sc.fit_transform(y_tr_raw.reshape(-1, 1)).ravel().astype(np.float32)
    y_te = y_sc.transform(y_te_raw.reshape(-1, 1)).ravel().astype(np.float32)

    # Sliding stride=1 в исходной 5-сек сетке: каждое следующее окно
    # сдвигается на 1 строку (5 сек), внутри окна выборка прорежена с
    # шагом window_step (=30 сек). Это сохраняет ~ N − context обучающих
    # пар на субъекта вместо N / (seq_length × window_step), как было
    # в предыдущей non-overlap версии v0101b.
    train_ds = SlidingSubsampledDataset(X_tr, y_tr,
                                        config.seq_length, config.window_step)
    test_ds  = SlidingSubsampledDataset(X_te, y_te,
                                        config.seq_length, config.window_step)

    if len(train_ds) == 0 or len(test_ds) == 0:
        return {"fold": test_subject_id, "error": "Нет валидных последовательностей"}

    split = max(1, int(0.8 * len(train_ds)))
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [split, len(train_ds) - split],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_sub, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,   batch_size=config.batch_size, shuffle=False)

    # Обучаем каждую архитектуру и усредняем (ensemble)
    fold_preds = []
    for spec in model_specs:
        model = spec["factory"](X_tr.shape[1]).to(config.device)
        model = train_model(model, train_loader, val_loader, config)

        model.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                preds.append(model(X_batch.to(config.device)).cpu().numpy())
        fold_preds.append(np.concatenate(preds))

    # Sliding stride=1 → предсказания идут подряд для позиций
    # [context-1, context, ..., len(y_te)-1]. Дополняем начало первым
    # значением (первые context-1 точек без предсказания).
    y_pred_sparse = np.mean(fold_preds, axis=0)
    n_pad = len(y_te) - len(y_pred_sparse)
    if n_pad > 0:
        y_pred_full = np.concatenate(
            [np.full(n_pad, y_pred_sparse[0]), y_pred_sparse]
        )
    else:
        y_pred_full = y_pred_sparse[:len(y_te)]

    # Денормализация предсказаний обратно в секунды; y_true возвращаем в
    # исходных секундах для совместимости с downstream-метриками.
    y_pred_full = y_sc.inverse_transform(y_pred_full.reshape(-1, 1)).ravel()
    return {
        "fold":   test_subject_id,
        "y_pred": y_pred_full,
        "y_true": y_te_raw,
    }


# ─── LOSO ─────────────────────────────────────────────────────────────────────

def loso_temporal(df: pd.DataFrame,
                  feat_cols: list[str],
                  target_col: str,
                  model_specs: list[dict],
                  config: Config,
                  n_jobs: int = 1) -> dict:
    """LOSO для LSTM/GRU ensemble. n_jobs=1 на GPU (нет loky-deadlock)."""
    subjects = sorted(df["subject_id"].unique())

    records = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_run_one_loso_fold)(s, df, feat_cols, target_col, model_specs, config)
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
        return {"error": "Нет валидных результатов"}

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    return {
        "y_pred":       y_pred,
        "y_true":       y_true,
        "raw_mae_min":  mean_absolute_error(y_true, y_pred) / 60.0,
        "r2":           r2_score(y_true, y_pred),
        "rho":          float(spearmanr(y_true, y_pred).statistic),
        "per_subject":  subj_rows,
    }


# ─── Зоопарк моделей ──────────────────────────────────────────────────────────

def build_model_specs(config: Config) -> list[dict]:
    """3 архитектуры: LSTM2, LSTM3, GRU2 — усредняются в ensemble."""
    h, d = config.hidden_size, config.dropout
    return [
        {"name": f"LSTM2(h={h})",
         "factory": lambda inp, h=h, d=d: LSTMRegressor(inp, h, num_layers=2, dropout=d)},
        {"name": f"LSTM3(h={h})",
         "factory": lambda inp, h=h, d=d: LSTMRegressor(inp, h, num_layers=3, dropout=d)},
        {"name": f"GRU2(h={h})",
         "factory": lambda inp, h=h, d=d: GRURegressor(inp, h, num_layers=2, dropout=d)},
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v0101b — LSTM с длинным контекстом без перекрытия")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--feature-set", nargs="+",
                   choices=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"])
    p.add_argument("--seq-length", type=int, default=6,
                   help="Число шагов (по 30 сек) в последовательности: "
                        "6 → 3 мин контекста, 12 → 6 мин. По умолчанию 6.")
    p.add_argument("--window-step", type=int, default=6,
                   help="Шаг прореживания (× 5 сек). По умолчанию 6 → 30 сек.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Число параллельных LOSO фолдов (только для CPU; на GPU всегда 1)")
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    config.seq_length = args.seq_length
    config.window_step = args.window_step

    # OUT_DIR с суффиксом длины контекста, чтобы прогоны с разным
    # seq_length писались в разные папки.
    global OUT_DIR
    ctx_min = config.seq_length * config.window_step * 5 // 60
    OUT_DIR = OUT_DIR_BASE.with_name(f"{OUT_DIR_BASE.name}_ctx{ctx_min}min")

    # Автовыбор устройства: CUDA → n_jobs=1 (иначе loky-deadlock)
    if torch.cuda.is_available():
        config.device = "cuda"
        config.batch_size = 256
        n_jobs = 1
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}, n_jobs=1")
    elif torch.backends.mps.is_available():
        config.device = "mps"
        n_jobs = 1
        print(f"[GPU] Apple MPS, n_jobs=1")
    else:
        config.device = "cpu"
        n_jobs = args.n_jobs
        print(f"[CPU] CUDA недоступна, n_jobs={n_jobs}")

    print("=" * 70)
    print("v0101b — LSTM с длинным контекстом без перекрытия")
    print("=" * 70)
    ctx_sec = config.seq_length * config.window_step * 5
    print(f"seq_length={config.seq_length}, window_step={config.window_step} "
          f"→ контекст {ctx_sec} сек ({ctx_sec/60:.1f} мин)")
    print(f"hidden={config.hidden_size}, dropout={config.dropout}, batch={config.batch_size}")
    print(f"ensemble: LSTM2 + LSTM3 + GRU2")
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

    model_specs = build_model_specs(config)
    all_records, all_subj_records = [], []

    for tgt_name, tgt_cfg in targets_cfg.items():
        print(f"\n{'═'*70}")
        print(f"ТАРГЕТ: {tgt_name.upper()}")
        print(f"{'═'*70}\n")

        df_prep = prepare_data(df_raw, session_params, tgt_name)
        target_col = tgt_cfg["col"]
        df_tgt = df_prep.dropna(subset=[target_col])

        variants = [("with_abs", None, OUT_DIR), ("noabs", EXCLUDE_ABS, OUT_DIR / "noabs")]

        for fset in args.feature_set:
            feat_cols_full = get_feature_cols(df_tgt, fset)
            if not feat_cols_full:
                continue
            n_subj = df_tgt["subject_id"].nunique()

            for variant, exclude_set, out_sub in variants:
                feat_cols = (feat_cols_full if exclude_set is None
                             else [c for c in feat_cols_full if c not in exclude_set])
                if not feat_cols:
                    continue
                out_sub.mkdir(exist_ok=True)
                print(f"  [{fset} / {tgt_name} / {variant}]  n={n_subj}, {len(feat_cols)} признаков")

                t0 = time.perf_counter()
                res = loso_temporal(df_tgt, feat_cols, target_col,
                                    model_specs, config, n_jobs=n_jobs)
                elapsed = time.perf_counter() - t0

                if "error" in res:
                    print(f"    ❌ {res['error']}")
                    continue

                for row in res.get("per_subject", []):
                    all_subj_records.append({"variant": variant, "feature_set": fset,
                                             "target": tgt_name, **row})

                # Сохраняем предсказания и истину по всем окнам
                fset_tag = fset.replace("+", "_")
                np.save(out_sub / f"ypred_{tgt_name}_{fset_tag}.npy", res["y_pred"])
                np.save(out_sub / f"ytrue_{tgt_name}_{fset_tag}.npy", res["y_true"])

                # Kalman grid search
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
        if sub.empty:
            continue
        print(f"\n  [{variant}]")
        for _, row in sub.sort_values(["target", "kalman_mae_min"]).iterrows():
            ref = v0011_ref.get((row["target"], row["feature_set"]))
            delta = f"  Δ={row['kalman_mae_min']-ref:+.3f} vs v0011" if ref else ""
            print(f"    {row['target'].upper()} / {row['feature_set']:<16s}  "
                  f"kalman={row['kalman_mae_min']:.3f}{delta}")

    print(f"\n✅ Готово: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
