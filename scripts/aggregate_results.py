"""Агрегатор результатов всех версий моделей.

Читает ypred/ytrue .npy из results/vXXXX/ и results/vXXXX/noabs/,
считает метрики и пишет единый summary_all_versions.csv.

Использование:
  python scripts/aggregate_results.py
  python scripts/aggregate_results.py --versions v0102 v0103 v0107
  python scripts/aggregate_results.py --out results/my_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.v0011_modality_ablation import kalman_smooth

RESULTS_DIR = _ROOT / "results"

# Все версии в порядке от baseline к сложным
ALL_VERSIONS = [
    "v0011", "v0101", "v0102", "v0103", "v0104",
    "v0105", "v0106a", "v0106b", "v0106c", "v0107",
]

TARGETS = ["lt1", "lt2"]

# Наборы признаков: (тег в имени файла, отображаемое имя)
FSETS = [
    ("EMG",           "EMG"),
    ("NIRS",          "NIRS"),
    ("HRV",           "HRV"),           # только v0011
    ("EMG_NIRS",      "EMG+NIRS"),
    ("EMG_NIRS_HRV",  "EMG+NIRS+HRV"),
]

SIGMA_GRID = [30.0, 50.0, 75.0, 150.0]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE (raw), Kalman grid, R², ρ. Всё в минутах."""
    raw_mae = mean_absolute_error(y_true, y_pred) / 60.0

    best_kalman = float("inf")
    best_sigma  = SIGMA_GRID[0]
    kalman_vals = {}
    for sigma in SIGMA_GRID:
        y_k   = kalman_smooth(y_pred, sigma_p=5.0, sigma_obs=sigma)
        mae_k = mean_absolute_error(y_true, y_k) / 60.0
        kalman_vals[sigma] = round(mae_k, 4)
        if mae_k < best_kalman:
            best_kalman = mae_k
            best_sigma  = sigma

    r2  = r2_score(y_true, y_pred)
    rho = float(spearmanr(y_true, y_pred).statistic)

    return {
        "raw_mae_min":    round(raw_mae, 4),
        "kalman_mae_min": round(best_kalman, 4),
        "best_sigma_obs": best_sigma,
        "kalman_30":      kalman_vals.get(30.0),
        "kalman_50":      kalman_vals.get(50.0),
        "kalman_75":      kalman_vals.get(75.0),
        "kalman_150":     kalman_vals.get(150.0),
        "r2":             round(r2, 3),
        "rho":            round(rho, 3),
        "n_windows":      len(y_true),
    }


def process_version(version: str) -> list[dict]:
    """Собирает записи для одной версии из npy-файлов."""
    vdir = RESULTS_DIR / version
    if not vdir.exists():
        return []

    variants = [
        ("with_abs", vdir),
        ("noabs",    vdir / "noabs"),
    ]

    records = []
    for variant, subdir in variants:
        if not subdir.exists():
            continue
        for fset_tag, fset_name in FSETS:
            for tgt in TARGETS:
                pred_path = subdir / f"ypred_{tgt}_{fset_tag}.npy"
                true_path = subdir / f"ytrue_{tgt}_{fset_tag}.npy"
                if not pred_path.exists() or not true_path.exists():
                    continue

                y_pred = np.load(pred_path)
                y_true = np.load(true_path)

                if len(y_pred) != len(y_true) or len(y_true) == 0:
                    print(f"  ⚠ {version}/{variant}/{tgt}/{fset_name}: "
                          f"shape mismatch ({len(y_pred)} vs {len(y_true)}), пропуск")
                    continue

                metrics = compute_metrics(y_true, y_pred)
                records.append({
                    "version":     version,
                    "variant":     variant,
                    "target":      tgt,
                    "feature_set": fset_name,
                    **metrics,
                })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Агрегатор результатов моделей")
    parser.add_argument("--versions", nargs="+", default=ALL_VERSIONS,
                        help="Версии для обработки (по умолчанию все)")
    parser.add_argument("--out", type=Path,
                        default=RESULTS_DIR / "summary_all_versions.csv",
                        help="Путь к выходному CSV")
    args = parser.parse_args()

    all_records = []
    for version in args.versions:
        recs = process_version(version)
        if recs:
            print(f"  {version}: {len(recs)} конфигов")
        else:
            print(f"  {version}: нет данных")
        all_records.extend(recs)

    if not all_records:
        print("❌ Нет данных для агрегации")
        return

    df = pd.DataFrame(all_records)

    # Порядок колонок
    col_order = [
        "version", "variant", "target", "feature_set",
        "raw_mae_min", "kalman_mae_min", "best_sigma_obs",
        "kalman_30", "kalman_50", "kalman_75", "kalman_150",
        "r2", "rho", "n_windows",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n✅ Сохранено: {args.out} ({len(df)} строк)")

    # Быстрый дайджест: лучшее per (version, target) по EMG+NIRS+HRV with_abs
    print("\n── Дайджест: kalman MAE (EMG+NIRS+HRV / with_abs) ──────────────────")
    digest = df[
        (df["feature_set"] == "EMG+NIRS+HRV") & (df["variant"] == "with_abs")
    ].sort_values(["target", "version"])

    for tgt in ["lt1", "lt2"]:
        sub = digest[digest["target"] == tgt]
        if sub.empty:
            continue
        print(f"\n  {tgt.upper()}:")
        for _, row in sub.iterrows():
            print(f"    {row['version']:<8s}  kalman={row['kalman_mae_min']:.3f}  "
                  f"raw={row['raw_mae_min']:.3f}  R²={row['r2']:.3f}  ρ={row['rho']:.3f}")


if __name__ == "__main__":
    main()
