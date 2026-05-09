"""v0011b — Подбор параметров фильтра Калмана (σ_p, σ_obs) по сетке.

Версия:    v0011b
Дата:      2026-05-08
Предыдущая версия: v0011_modality_ablation.py

Что делает:
  1. Запускает LOSO для лучшей модели каждого (feature_set, target) из v0011
     → сохраняет сырые предсказания per-subject
  2. Для каждой пары (σ_p, σ_obs) применяет Kalman к сохранённым предсказаниям
     → не нужно перефитить модели, grid очень быстрый
  3. Строит тепловую карту MAE(σ_p, σ_obs) и отчёт

Сетка:
  σ_p   ∈ [5, 10, 15, 25, 40, 60, 90]       (с/фолд, скорость изменения состояния)
  σ_obs ∈ [30, 60, 100, 150, 200, 300, 500]  (с, шум наблюдений)

Базовые модели из v0011 (лучшие per set/target):
  EMG/lt2:          EN(α=1.0, l1=0.2)
  EMG/lt1:          Ridge(α=1000)
  NIRS/lt2:         SVR(C=10, ε=0.1)
  NIRS/lt1:         SVR(C=10, ε=1.0)
  EMG+NIRS/lt2:     EN(α=1.0, l1=0.2)
  EMG+NIRS/lt1:     Ridge(α=1000)
  EMG+NIRS+HRV/lt2: EN(α=0.1, l1=0.5)
  EMG+NIRS+HRV/lt1: GBM(n=50, d=2)

Запуск:
  uv run python scripts/v0011b_kalman_grid.py
  uv run python scripts/v0011b_kalman_grid.py --target lt2
  uv run python scripts/v0011b_kalman_grid.py --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
sys.path.insert(0, str(_ROOT / "scripts"))
from v0011_modality_ablation import (
    prepare_data,
    get_feature_cols,
)

OUT_DIR = _ROOT / "results" / "v0011b"

# ─── Сетка σ ─────────────────────────────────────────────────────────────────

SIGMA_P_GRID   = [5, 10, 15, 25, 40, 60, 90]
SIGMA_OBS_GRID = [30, 60, 100, 150, 200, 300, 500]

# ─── Baseline v0011 ───────────────────────────────────────────────────────────

V0011_BASELINE = {
    ("EMG",          "lt2"): 3.1979,
    ("EMG",          "lt1"): 2.8957,
    ("NIRS",         "lt2"): 4.2508,
    ("NIRS",         "lt1"): 4.0629,
    ("EMG+NIRS",     "lt2"): 3.1172,
    ("EMG+NIRS",     "lt1"): 2.7499,
    ("EMG+NIRS+HRV", "lt2"): 1.8586,
    ("EMG+NIRS+HRV", "lt1"): 2.2774,
}

TARGET_CFG = {
    "lt2": "target_time_to_lt2_center_sec",
    "lt1": "target_time_to_lt1_sec",
}

# ─── Лучшие модели из v0011 ───────────────────────────────────────────────────

def _best_factory(feature_set: str, target: str):
    """Возвращает factory лучшей модели из v0011."""
    key = (feature_set, target)
    factories = {
        ("EMG",          "lt2"): lambda: ElasticNet(alpha=1.0, l1_ratio=0.2, max_iter=5000, random_state=42),
        ("EMG",          "lt1"): lambda: Ridge(alpha=1000),
        ("NIRS",         "lt2"): lambda: SVR(kernel="rbf", C=10, epsilon=0.1),
        ("NIRS",         "lt1"): lambda: SVR(kernel="rbf", C=10, epsilon=1.0),
        ("EMG+NIRS",     "lt2"): lambda: ElasticNet(alpha=1.0, l1_ratio=0.2, max_iter=5000, random_state=42),
        ("EMG+NIRS",     "lt1"): lambda: Ridge(alpha=1000),
        ("EMG+NIRS+HRV", "lt2"): lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42),
        ("EMG+NIRS+HRV", "lt1"): lambda: GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42),
    }
    if key not in factories:
        raise ValueError(f"Нет конфига для {key}")
    return factories[key]


# ─── LOSO — получение сырых предсказаний ─────────────────────────────────────

def loso_raw(df: pd.DataFrame, feat_cols: list[str],
             target_col: str, model_factory) -> dict:
    """LOSO без Kalman. Возвращает y_pred, y_true, subjects per row."""
    subjects = sorted(df["subject_id"].unique())
    preds, trues, subjs = [], [], []

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        mdl = model_factory()

        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        mdl.fit(X_tr, train[target_col].values)
        y_pred = mdl.predict(X_te)

        preds.append(y_pred)
        trues.append(test[target_col].values)
        subjs.append(np.full(len(y_pred), test_s))

    return {
        "y_pred": np.concatenate(preds),
        "y_true": np.concatenate(trues),
        "subjects": np.concatenate(subjs),
    }


# ─── Kalman ───────────────────────────────────────────────────────────────────

def kalman_smooth(y_pred: np.ndarray,
                  sigma_p: float, sigma_obs: float,
                  dt: float = 5.0) -> np.ndarray:
    """Одномерный фильтр Калмана. Модель: x[t] = x[t-1] - dt."""
    n = len(y_pred)
    x = float(y_pred[0])
    p = sigma_obs ** 2
    smoothed = np.empty(n)
    for i in range(n):
        x -= dt
        p += sigma_p ** 2
        k  = p / (p + sigma_obs ** 2)
        x  = x + k * (y_pred[i] - x)
        p  = (1 - k) * p
        smoothed[i] = x
    return smoothed


def apply_kalman_grid(loso_raw_result: dict,
                      sigma_p_grid: list[float],
                      sigma_obs_grid: list[float]) -> pd.DataFrame:
    """Применяет Kalman per-subject для каждой пары (σ_p, σ_obs).

    Возвращает DataFrame: sigma_p, sigma_obs, kalman_mae_min, r2, rho.
    """
    y_pred_all = loso_raw_result["y_pred"]
    y_true_all = loso_raw_result["y_true"]
    subj_all   = loso_raw_result["subjects"]
    subjects   = sorted(np.unique(subj_all))

    rows = []
    for sp in sigma_p_grid:
        for so in sigma_obs_grid:
            preds_k, trues_k = [], []
            for s in subjects:
                mask = subj_all == s
                if mask.sum() == 0:
                    continue
                yk = kalman_smooth(y_pred_all[mask], sp, so)
                preds_k.append(yk)
                trues_k.append(y_true_all[mask])

            y_pk = np.concatenate(preds_k)
            y_tk = np.concatenate(trues_k)
            mae  = mean_absolute_error(y_tk, y_pk) / 60.0
            r2   = r2_score(y_tk, y_pk)
            rho  = float(spearmanr(y_tk, y_pk).statistic)
            rows.append({
                "sigma_p":   sp,
                "sigma_obs": so,
                "kalman_mae": round(mae, 4),
                "r2":         round(r2, 3),
                "rho":        round(rho, 3),
            })

    return pd.DataFrame(rows)


# ─── Тепловая карта ───────────────────────────────────────────────────────────

def plot_heatmap(grid_df: pd.DataFrame, feature_set: str, target: str,
                 baseline: float | None, best_sp: float, best_so: float,
                 out_dir: Path) -> None:
    """Сохраняет тепловую карту MAE(σ_p, σ_obs)."""
    pivot = grid_df.pivot(index="sigma_obs", columns="sigma_p",
                          values="kalman_mae")
    pivot = pivot.sort_index(ascending=False)  # σ_obs по убыванию сверху

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="MAE (мин)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("σ_p (с)")
    ax.set_ylabel("σ_obs (с)")

    title = f"Kalman MAE: {feature_set} / {target.upper()}"
    if baseline:
        title += f"\nv0011 baseline (σ_p=15, σ_obs=150): {baseline:.4f} мин"
    ax.set_title(title, fontsize=11)

    # Аннотации значений
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color="black")

    # Отметить baseline (σ_p=15, σ_obs=150)
    if 15 in pivot.columns and 150 in pivot.index:
        jb = list(pivot.columns).index(15)
        ib = list(pivot.index[::-1]).index(150) if 150 in pivot.index else None
        if ib is not None:
            ax.add_patch(plt.Rectangle(
                (jb - 0.5, len(pivot.index) - 1 - list(pivot.index).index(150) - 0.5),
                1, 1, fill=False, edgecolor="blue", linewidth=2.5, label="v0011 baseline"))

    # Отметить лучшую ячейку
    jbest = list(pivot.columns).index(best_sp) if best_sp in pivot.columns else None
    ibest = list(pivot.index).index(best_so) if best_so in pivot.index else None
    if jbest is not None and ibest is not None:
        ax.add_patch(plt.Rectangle(
            (jbest - 0.5, ibest - 0.5),
            1, 1, fill=False, edgecolor="green", linewidth=2.5, label="лучшая"))

    plt.tight_layout()
    fname = f"heatmap_{feature_set.replace('+', '_')}_{target}.png"
    fig.savefig(out_dir / fname, dpi=120)
    plt.close(fig)
    print(f"  → тепловая карта: {fname}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v0011b — Kalman σ grid")
    parser.add_argument("--target",      choices=["lt1", "lt2", "both"], default="both")
    parser.add_argument("--feature-set", nargs="+",
                        choices=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"],
                        default=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"])
    parser.add_argument("--dataset",
                        default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "heatmaps").mkdir(exist_ok=True)

    print(f"Загружаю {args.dataset}")
    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    targets = ["lt2", "lt1"] if args.target == "both" else [args.target]

    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for tgt_name in targets:
        target_col = TARGET_CFG[tgt_name]
        df_prep = prepare_data(df_raw, session_params, tgt_name)
        if target_col not in df_prep.columns:
            print(f"[skip] нет {target_col}")
            continue
        df_prep = df_prep.dropna(subset=[target_col])
        n_subj = df_prep["subject_id"].nunique()

        print(f"\n{'═'*70}")
        print(f"ТАРГЕТ: {tgt_name.upper()}  ({n_subj} субъектов)")
        print(f"{'═'*70}")

        for fset in args.feature_set:
            feat_cols = get_feature_cols(df_prep, fset)
            if not feat_cols:
                continue
            baseline = V0011_BASELINE.get((fset, tgt_name))
            print(f"\n  [{fset}]  {len(feat_cols)} признаков"
                  + (f"  | v0011: {baseline:.4f} мин" if baseline else ""))

            # ── LOSO (одна модель) ────────────────────────────────────────────
            factory = _best_factory(fset, tgt_name)
            print(f"  Запуск LOSO...", end=" ", flush=True)
            raw_result = loso_raw(df_prep, feat_cols, target_col, factory)

            raw_mae = mean_absolute_error(
                raw_result["y_true"], raw_result["y_pred"]) / 60.0
            print(f"raw MAE={raw_mae:.4f} мин")

            # ── Kalman grid ───────────────────────────────────────────────────
            grid_df = apply_kalman_grid(raw_result, SIGMA_P_GRID, SIGMA_OBS_GRID)
            grid_df["feature_set"] = fset
            grid_df["target"]      = tgt_name
            all_rows.append(grid_df)

            best_row = grid_df.loc[grid_df["kalman_mae"].idxmin()]
            best_sp  = best_row["sigma_p"]
            best_so  = best_row["sigma_obs"]
            best_mae = best_row["kalman_mae"]
            delta    = best_mae - baseline if baseline else None

            print(f"  Лучший: σ_p={best_sp:.0f}, σ_obs={best_so:.0f}"
                  f"  → MAE={best_mae:.4f} мин"
                  + (f"  Δ={delta:+.4f} мин" if delta is not None else ""))

            # Baseline ячейка (σ_p=15, σ_obs=150)
            bl_cell = grid_df[(grid_df.sigma_p == 15) & (grid_df.sigma_obs == 150)]
            if not bl_cell.empty:
                bl_mae = bl_cell.iloc[0]["kalman_mae"]
                print(f"  Сетка (σ_p=15, σ_obs=150): {bl_mae:.4f} мин"
                      f"  (v0011: {baseline:.4f})" if baseline else "")

            summary_rows.append({
                "feature_set":      fset,
                "target":           tgt_name,
                "best_sigma_p":     best_sp,
                "best_sigma_obs":   best_so,
                "kalman_mae_best":  best_mae,
                "kalman_mae_v0011": baseline,
                "delta":            round(delta, 4) if delta else None,
                "r2":               best_row["r2"],
                "rho":              best_row["rho"],
                "raw_mae":          round(raw_mae, 4),
            })

            # ── Тепловая карта ────────────────────────────────────────────────
            plot_heatmap(grid_df, fset, tgt_name, baseline,
                         best_sp, best_so, OUT_DIR / "heatmaps")

    # ── Сохранение ────────────────────────────────────────────────────────────
    if all_rows:
        full_grid = pd.concat(all_rows, ignore_index=True)
        full_grid.to_csv(OUT_DIR / "kalman_grid_full.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "kalman_grid_summary.csv", index=False)

    # ── Отчёт ─────────────────────────────────────────────────────────────────
    _write_report(summary_df, OUT_DIR)

    print(f"\n✓ Сохранено в {OUT_DIR}/")
    print(f"  kalman_grid_full.csv:    {len(full_grid)} строк")
    print(f"  kalman_grid_summary.csv: {len(summary_df)} строк")


def _write_report(summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# v0011b — Подбор параметров Калмана (σ_p, σ_obs)",
        f"Дата: {pd.Timestamp.now().date()}",
        "",
        f"Сетка: σ_p ∈ {SIGMA_P_GRID}",
        f"       σ_obs ∈ {SIGMA_OBS_GRID}",
        "",
    ]

    for tgt in ["lt2", "lt1"]:
        rows = summary[summary["target"] == tgt]
        if rows.empty:
            continue
        lines.append(f"## {tgt.upper()}")
        lines.append("")
        lines.append("| Набор | σ_p* | σ_obs* | MAE_best | MAE_v0011 | Δ | R² | ρ |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in rows.sort_values("kalman_mae_best").iterrows():
            delta_str = f"{r['delta']:+.4f}" if pd.notna(r["delta"]) else "—"
            v0011_str = f"{r['kalman_mae_v0011']:.4f}" if pd.notna(r["kalman_mae_v0011"]) else "—"
            lines.append(
                f"| **{r['feature_set']}** "
                f"| {r['best_sigma_p']:.0f} "
                f"| {r['best_sigma_obs']:.0f} "
                f"| **{r['kalman_mae_best']:.4f}** "
                f"| {v0011_str} "
                f"| {delta_str} "
                f"| {r['r2']:.3f} "
                f"| {r['rho']:.3f} |"
            )
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
