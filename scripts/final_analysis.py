"""Финальный анализ: дельта-признаки + nested LOSO + ансамбль + визуализации.

Запуск:
  uv run python scripts/final_analysis.py

Что делает:
  1. Строит дельта-признаки (Δ за 30 с и 120 с) для SmO2, HHb, HRV
  2. Nested LOSO: подбор гиперпараметров честно внутри каждого внешнего фолда
  3. Ансамбль: ElasticNet + Ridge → усреднение предсказаний
  4. 7 канонических визуализаций → results/final_analysis/
"""

from __future__ import annotations

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATASET_PATH = _ROOT / "dataset" / "merged_features_ml.parquet"
OUT_DIR = _ROOT / "results" / "final_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Цвета ───────────────────────────────────────────────────────────────────
C_NIRS = "#2ca02c"
C_HRV  = "#9467bd"
C_EMG  = "#1f77b4"
C_KIN  = "#17becf"
C_DELTA = "#ff7f0e"

SUBJ_PALETTE = plt.cm.tab20.colors  # 20 цветов

# ─────────────────────────────────────────────────────────────────────────────
# 1. ЗАГРУЗКА И ДЕЛЬТА-ПРИЗНАКИ
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    df = df[df["window_valid_all_required"] == 1].copy()
    df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет дельта-признаки (изменение за N шагов назад) внутри субъекта.

    Шаг окна = 5 с → 6 шагов = 30 с, 24 шага = 120 с.
    Дельта вычисляется каузально: только прошлые значения.
    """
    df = df.copy()

    # Ключевые признаки для дельт
    delta_targets = [
        "trainred_smo2_mean",
        "trainred_hhb_mean",
        "trainred_thb_mean",
        "hrv_mean_rr_ms",
        "hrv_dfa_alpha1",
        "hrv_rmssd_ms",
    ]

    for feat in delta_targets:
        if feat not in df.columns:
            continue
        fname = feat.replace("trainred_", "").replace("hrv_", "")
        for lag, label in [(6, "30s"), (24, "120s")]:
            col = f"delta_{fname}_{label}"
            df[col] = df.groupby("subject_id")[feat].transform(
                lambda s: s - s.shift(lag)
            )

    return df


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    nirs  = [c for c in df.columns if c.startswith("trainred_")]
    hrv   = [c for c in df.columns if c.startswith("hrv_")
              and not c.endswith(("_valid", "_fraction", "_count"))]
    delta = [c for c in df.columns if c.startswith("delta_")]
    return {
        "nirs_hrv":       nirs + hrv,
        "nirs_hrv_delta": nirs + hrv + delta,
        "delta_only":     delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. МЕТРИКИ
# ─────────────────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred) / 60.0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / 60.0
    r2   = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {"mae_min": mae, "rmse_min": rmse, "r2": r2, "spearman": rho}


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOSO ОДНОГО PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

TARGET = "target_time_to_lt2_center_sec"

def loso_predict(df: pd.DataFrame, feat_cols: list[str], pipe: Pipeline,
                 return_preds: bool = False):
    """LOSO CV: возвращает агрегированные метрики или (метрики + предсказания)."""
    X = df[feat_cols].values
    y = df[TARGET].values
    groups = df["subject_id"].values
    logo = LeaveOneGroupOut()

    fold_rows = []
    preds_all = np.full(len(df), np.nan)

    for tr, te in logo.split(X, y, groups=groups):
        p = clone(pipe)
        p.fit(X[tr], y[tr])
        y_pred = p.predict(X[te])
        preds_all[te] = y_pred
        m = metrics(y[te], y_pred)
        m["subject_id"] = groups[te][0]
        m["n_test"] = len(te)
        m["train_mae_min"] = mean_absolute_error(y[tr], p.predict(X[tr])) / 60
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    agg = {
        "mae_min":  fold_df["mae_min"].mean(),
        "mae_std":  fold_df["mae_min"].std(),
        "r2":       fold_df["r2"].mean(),
        "spearman": fold_df["spearman"].mean(),
    }
    if return_preds:
        return agg, fold_df, preds_all
    return agg, fold_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. NESTED LOSO
# ─────────────────────────────────────────────────────────────────────────────

# Сетка гиперпараметров для ElasticNet
_ALPHAS   = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
_L1RATIOS = [0.1, 0.5, 0.9]
_PARAM_GRID = list(product(_ALPHAS, _L1RATIOS))


def _inner_loso_score(X_inner: np.ndarray, y_inner: np.ndarray,
                      groups_inner: np.ndarray, alpha: float, l1: float) -> float:
    """MAE для одной комбинации гиперпараметров на внутренних 12 субъектах."""
    logo = LeaveOneGroupOut()
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=3000)),
    ])
    maes = []
    for tr, te in logo.split(X_inner, y_inner, groups=groups_inner):
        p = clone(pipe)
        p.fit(X_inner[tr], y_inner[tr])
        maes.append(mean_absolute_error(y_inner[te], p.predict(X_inner[te])))
    return float(np.mean(maes))


def nested_loso_predict(df: pd.DataFrame, feat_cols: list[str]) -> tuple[dict, pd.DataFrame, np.ndarray, list]:
    """Nested LOSO: подбирает гиперпараметры честно внутри каждого фолда."""
    X = df[feat_cols].values
    y = df[TARGET].values
    groups = df["subject_id"].values
    logo = LeaveOneGroupOut()

    fold_rows = []
    preds_all = np.full(len(df), np.nan)
    best_params_per_fold = []

    for tr, te in logo.split(X, y, groups=groups):
        subj = groups[te][0]
        X_inner, y_inner, g_inner = X[tr], y[tr], groups[tr]

        # Параллельный перебор гиперпараметров на внутренних субъектах
        scores = Parallel(n_jobs=-1)(
            delayed(_inner_loso_score)(X_inner, y_inner, g_inner, a, l)
            for a, l in _PARAM_GRID
        )
        best_idx = int(np.argmin(scores))
        best_alpha, best_l1 = _PARAM_GRID[best_idx]
        best_params_per_fold.append({"subject_id": subj, "alpha": best_alpha, "l1_ratio": best_l1,
                                      "inner_mae_min": scores[best_idx] / 60})

        # Финальное обучение с лучшими параметрами на всех train-субъектах
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=3000)),
        ])
        pipe.fit(X[tr], y[tr])
        y_pred = pipe.predict(X[te])
        preds_all[te] = y_pred

        m = metrics(y[te], y_pred)
        m["subject_id"] = subj
        m["n_test"] = len(te)
        m["best_alpha"] = best_alpha
        m["best_l1"] = best_l1
        fold_rows.append(m)
        print(f"    {subj}: best α={best_alpha}, l1={best_l1} → MAE={m['mae_min']:.3f} мин", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    agg = {
        "mae_min":  fold_df["mae_min"].mean(),
        "mae_std":  fold_df["mae_min"].std(),
        "r2":       fold_df["r2"].mean(),
        "spearman": fold_df["spearman"].mean(),
    }
    return agg, fold_df, preds_all, best_params_per_fold


# ─────────────────────────────────────────────────────────────────────────────
# 5. АНСАМБЛЬ
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_loso_predict(df: pd.DataFrame, feat_cols: list[str],
                           weights: tuple[float, float] = (0.5, 0.5)) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Ансамбль: ElasticNet(0.5, 0.9) + Ridge(100) с весами."""
    X = df[feat_cols].values
    y = df[TARGET].values
    groups = df["subject_id"].values
    logo = LeaveOneGroupOut()

    pipe_en = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=3000)),
    ])
    pipe_r = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", Ridge(alpha=100.0)),
    ])

    fold_rows = []
    preds_all = np.full(len(df), np.nan)

    for tr, te in logo.split(X, y, groups=groups):
        en = clone(pipe_en); en.fit(X[tr], y[tr])
        r  = clone(pipe_r);  r.fit(X[tr], y[tr])
        y_pred = weights[0] * en.predict(X[te]) + weights[1] * r.predict(X[te])
        preds_all[te] = y_pred
        m = metrics(y[te], y_pred)
        m["subject_id"] = groups[te][0]
        m["n_test"] = len(te)
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    agg = {"mae_min": fold_df["mae_min"].mean(), "mae_std": fold_df["mae_min"].std(),
           "r2": fold_df["r2"].mean(), "spearman": fold_df["spearman"].mean()}
    return agg, fold_df, preds_all


# ─────────────────────────────────────────────────────────────────────────────
# 6. БАЗОВЫЕ НАИВНЫЕ МОДЕЛИ
# ─────────────────────────────────────────────────────────────────────────────

def naive_baselines(df: pd.DataFrame) -> dict[str, dict]:
    """Вычисляет наивные baseline-ы через LOSO."""
    y = df[TARGET].values
    groups = df["subject_id"].values
    logo = LeaveOneGroupOut()

    results: dict[str, list] = {n: [] for n in ["zero", "mean", "elapsed", "power"]}

    for tr, te in logo.split(df, groups=groups):
        y_tr, y_te = y[tr], y[te]

        # Предсказать 0 (сейчас у LT2)
        results["zero"].append(np.abs(y_te).mean() / 60)

        # Предсказать среднее тренировочной выборки
        results["mean"].append(np.abs(y_te - y_tr.mean()).mean() / 60)

        # Линейная регрессия на elapsed_sec
        x_el = df["elapsed_sec"].values
        c = np.polyfit(x_el[tr], y_tr, 1)
        results["elapsed"].append(np.abs(y_te - np.polyval(c, x_el[te])).mean() / 60)

        # Линейная регрессия на current_power_w
        x_pw = df["current_power_w"].values
        c2 = np.polyfit(x_pw[tr], y_tr, 1)
        results["power"].append(np.abs(y_te - np.polyval(c2, x_pw[te])).mean() / 60)

    return {name: {"mae_min": np.mean(v), "mae_std": np.std(v)} for name, v in results.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 7. ВИЗУАЛИЗАЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(df: pd.DataFrame, preds: np.ndarray,
                              fold_df: pd.DataFrame, model_name: str) -> None:
    """График 1: predicted vs actual, цвет = субъект."""
    y_true = df[TARGET].values / 60
    y_pred = preds / 60

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Predicted vs Actual — {model_name} (LOSO, N=13)", fontsize=13, fontweight="bold")

    # Scatter
    ax = axes[0]
    subjects = sorted(df["subject_id"].unique())
    for i, subj in enumerate(subjects):
        mask = df["subject_id"].values == subj
        ax.scatter(y_true[mask], y_pred[mask],
                   color=SUBJ_PALETTE[i % len(SUBJ_PALETTE)],
                   alpha=0.4, s=8, label=subj)
    lim = (min(y_true.min(), y_pred[~np.isnan(y_pred)].min()) - 1,
           max(y_true.max(), y_pred[~np.isnan(y_pred)].max()) + 1)
    ax.plot(lim, lim, "k--", lw=1.5, label="Идеал")
    ax.axhline(0, color="red", lw=0.8, alpha=0.5, ls=":")
    ax.axvline(0, color="red", lw=0.8, alpha=0.5, ls=":")
    ax.set_xlabel("Истинное время до LT2 (мин)")
    ax.set_ylabel("Предсказанное время до LT2 (мин)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    mae_all = fold_df["mae_min"].mean()
    r2_all  = fold_df["r2"].mean()
    rho_all = fold_df["spearman"].mean()
    ax.set_title(f"MAE={mae_all:.2f} мин | R²={r2_all:.3f} | ρ={rho_all:.3f}")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    # Per-subject MAE + train MAE
    ax2 = axes[1]
    fold_s = fold_df.sort_values("mae_min", ascending=True)
    colors_s = [SUBJ_PALETTE[subjects.index(s) % len(SUBJ_PALETTE)] for s in fold_s["subject_id"]]
    bars = ax2.barh(fold_s["subject_id"], fold_s["mae_min"], color=colors_s, alpha=0.8, label="Test MAE")
    if "train_mae_min" in fold_s.columns:
        ax2.barh(fold_s["subject_id"], fold_s["train_mae_min"],
                 color=colors_s, alpha=0.3, hatch="///", label="Train MAE")
    ax2.axvline(mae_all, color="black", lw=1.5, ls="--", label=f"Среднее {mae_all:.2f}")
    ax2.set_xlabel("MAE (мин)")
    ax2.set_title("MAE по субъектам (train vs test)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "1_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → 1_predicted_vs_actual.png")


def plot_model_comparison(all_models: dict[str, dict]) -> None:
    """График 2: сравнение всех моделей по MAE."""
    names = list(all_models.keys())
    maes  = [all_models[n]["mae_min"] for n in names]
    stds  = [all_models[n].get("mae_std", 0) for n in names]

    # Цвет по типу
    def color(name: str) -> str:
        if "Наивн" in name or "Baseline" in name or "Elapsed" in name or "Power" in name or "Zero" in name or "Mean" in name:
            return "#d62728"
        if "Delta" in name or "delta" in name:
            return C_DELTA
        if "Nested" in name:
            return "#8c564b"
        if "Ensemble" in name or "Ансамбль" in name:
            return "#e377c2"
        return "#1f77b4"

    colors = [color(n) for n in names]

    order = np.argsort(maes)[::-1]  # от худшей к лучшей (горизонтальный бар)
    names_s  = [names[i]  for i in order]
    maes_s   = [maes[i]   for i in order]
    stds_s   = [stds[i]   for i in order]
    colors_s = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.45)))
    bars = ax.barh(names_s, maes_s, xerr=stds_s, color=colors_s, alpha=0.85,
                   capsize=4, error_kw={"elinewidth": 1.5})

    for bar, mae in zip(bars, maes_s):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{mae:.2f}", va="center", ha="left", fontsize=9)

    best_mae = min(maes)
    ax.axvline(best_mae, color="green", lw=1.5, ls="--", alpha=0.7, label=f"Лучшее: {best_mae:.2f} мин")
    ax.set_xlabel("MAE (мин) — LOSO CV, меньше = лучше")
    ax.set_title("Сравнение моделей: LOSO MAE (± std по фолдам)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color="#d62728", label="Наивные baseline"),
        mpatches.Patch(color="#1f77b4", label="ElasticNet / Ridge"),
        mpatches.Patch(color=C_DELTA,   label="+ Дельта-признаки"),
        mpatches.Patch(color="#8c564b", label="Nested LOSO"),
        mpatches.Patch(color="#e377c2", label="Ансамбль"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "2_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → 2_model_comparison.png")


def plot_residuals(df: pd.DataFrame, preds: np.ndarray, model_name: str) -> None:
    """График 3: анализ остатков."""
    y_true = df[TARGET].values / 60
    y_pred = preds / 60
    resid  = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Анализ остатков — {model_name}", fontsize=13, fontweight="bold")

    # Residuals vs fitted
    ax = axes[0]
    ax.scatter(y_pred, resid, alpha=0.3, s=6, color="#1f77b4")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Предсказанное (мин)")
    ax.set_ylabel("Остаток (мин)")
    ax.set_title("Остатки vs Предсказанное")
    ax.grid(alpha=0.3)

    # Распределение остатков
    ax = axes[1]
    ax.hist(resid, bins=40, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.axvline(resid.mean(), color="orange", lw=1.5, ls="-", label=f"mean={resid.mean():.2f}")
    ax.set_xlabel("Остаток (мин)")
    ax.set_ylabel("Частота")
    ax.set_title("Распределение остатков")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Residuals vs time_to_LT2 (есть ли гетероскедастичность)
    ax = axes[2]
    subjects = sorted(df["subject_id"].unique())
    for i, subj in enumerate(subjects):
        mask = df["subject_id"].values == subj
        ax.scatter(y_true[mask], resid[mask],
                   color=SUBJ_PALETTE[i % len(SUBJ_PALETTE)],
                   alpha=0.35, s=7, label=subj)
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Истинное время до LT2 (мин)")
    ax.set_ylabel("Остаток (мин)")
    ax.set_title("Остатки vs Истинное время (гетероскедастичность?)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "3_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → 3_residuals.png")


def plot_trajectories(df: pd.DataFrame, preds: np.ndarray, n_subjects: int = 6) -> None:
    """График 4: временны́е ряды предсказаний vs истинных значений."""
    subjects = sorted(df["subject_id"].unique())[:n_subjects]
    cols = 3
    rows = (len(subjects) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    fig.suptitle("Траектории предсказаний по субъектам (LOSO)", fontsize=13, fontweight="bold")

    for idx, subj in enumerate(subjects):
        mask = df["subject_id"].values == subj
        sub_df = df[mask].reset_index(drop=True)
        t_min = sub_df["elapsed_sec"].values / 60
        y_true = sub_df[TARGET].values / 60
        y_pred = preds[mask] / 60

        ax = axes[idx]
        ax.plot(t_min, y_true, color="black", lw=2, label="Истинное")
        ax.plot(t_min, y_pred, color=SUBJ_PALETTE[idx % len(SUBJ_PALETTE)],
                lw=1.8, ls="--", label="Предсказанное")
        ax.axhline(0, color="red", lw=1.2, ls=":", alpha=0.8, label="LT2")
        ax.fill_between(t_min, y_true, y_pred, alpha=0.15,
                        color=SUBJ_PALETTE[idx % len(SUBJ_PALETTE)])
        mae_s = np.abs(y_true - y_pred).mean()
        ax.set_title(f"{subj} | MAE={mae_s:.2f} мин", fontsize=10)
        ax.set_xlabel("Время протокола (мин)")
        ax.set_ylabel("Время до LT2 (мин)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for idx in range(len(subjects), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "4_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → 4_trajectories.png")


def plot_coefficients(df: pd.DataFrame, feat_cols: list[str]) -> None:
    """График 5: коэффициенты ElasticNet, обученного на всех данных."""
    X = df[feat_cols].values
    y = df[TARGET].values

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=3000)),
    ])
    pipe.fit(X, y)
    coef = pipe.named_steps["mdl"].coef_
    nonzero = coef != 0
    if nonzero.sum() == 0:
        print("  ⚠️  Все коэффициенты нулевые — пропускаем plot_coefficients")
        return

    feat_arr = np.array(feat_cols)
    coef_nz  = coef[nonzero]
    feat_nz  = feat_arr[nonzero]
    order    = np.argsort(coef_nz)

    def feat_color(name: str) -> str:
        if name.startswith("trainred_"):
            return C_NIRS
        if name.startswith("hrv_"):
            return C_HRV
        if name.startswith("delta_"):
            return C_DELTA
        return C_EMG

    colors = [feat_color(f) for f in feat_nz[order]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(coef_nz) * 0.35)))
    ax.barh(feat_nz[order], coef_nz[order], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Коэффициент ElasticNet (стандартизованное пространство)")
    ax.set_title(f"Ненулевые коэффициенты ElasticNet (α=0.5, l1=0.9)\n"
                 f"Обучено на всех данных | {nonzero.sum()} из {len(coef)} ненулевых",
                 fontweight="bold")
    legend_handles = [
        mpatches.Patch(color=C_NIRS,  label="NIRS"),
        mpatches.Patch(color=C_HRV,   label="HRV"),
        mpatches.Patch(color=C_DELTA, label="Δ-признаки"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "5_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → 5_coefficients.png")


def plot_nested_params(best_params: list[dict]) -> None:
    """График 6: какие гиперпараметры выбирал nested LOSO для каждого субъекта."""
    bdf = pd.DataFrame(best_params)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Nested LOSO: выбранные гиперпараметры по субъектам", fontweight="bold")

    ax = axes[0]
    ax.bar(bdf["subject_id"], np.log10(bdf["alpha"]),
           color="#8c564b", alpha=0.8)
    ax.set_ylabel("log₁₀(α)")
    ax.set_title("Оптимальная α (ElasticNet)")
    ax.set_xticklabels(bdf["subject_id"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(np.log10(0.5), color="blue", ls="--", lw=1.2, label="α=0.5 (фиксированный)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(bdf["subject_id"], bdf["l1_ratio"], color="#e377c2", alpha=0.8)
    ax.set_ylabel("L1 ratio")
    ax.set_ylim(0, 1.1)
    ax.set_title("Оптимальный L1 ratio")
    ax.set_xticklabels(bdf["subject_id"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.9, color="blue", ls="--", lw=1.2, label="l1=0.9 (фиксированный)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "6_nested_params.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → 6_nested_params.png")


def plot_learning_curve(df: pd.DataFrame, feat_cols: list[str]) -> None:
    """График 7: кривая обучения — MAE в зависимости от числа train-субъектов."""
    subjects = sorted(df["subject_id"].unique())
    N = len(subjects)
    X = df[feat_cols].values
    y = df[TARGET].values
    groups = df["subject_id"].values

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=3000)),
    ])

    # Для каждого тестового субъекта: обучаем на k=1..N-1 других субъектов
    # Берём средний MAE по всем тестовым субъектам при каждом k
    curve_mean = []
    curve_std  = []
    ks = list(range(1, N))

    for k in ks:
        fold_maes = []
        for test_subj in subjects:
            train_subjs = [s for s in subjects if s != test_subj]
            # берём первые k тренировочных субъектов (по алфавиту)
            train_k = train_subjs[:k]
            tr_mask = np.isin(groups, train_k)
            te_mask = groups == test_subj
            if tr_mask.sum() < 5 or te_mask.sum() == 0:
                continue
            p = clone(pipe)
            p.fit(X[tr_mask], y[tr_mask])
            fold_maes.append(mean_absolute_error(y[te_mask], p.predict(X[te_mask])) / 60)
        curve_mean.append(np.mean(fold_maes))
        curve_std.append(np.std(fold_maes))

    curve_mean = np.array(curve_mean)
    curve_std  = np.array(curve_std)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, curve_mean, "o-", color="#1f77b4", lw=2, ms=7, label="Test MAE (среднее)")
    ax.fill_between(ks, curve_mean - curve_std, curve_mean + curve_std,
                    alpha=0.2, color="#1f77b4", label="±std")
    ax.axhline(curve_mean[-1], color="green", ls="--", lw=1.2,
               label=f"Полный LOSO ({N-1} субъектов): {curve_mean[-1]:.2f} мин")

    # Наивные baseline
    ax.axhline(4.28, color="red", ls=":", lw=1.2, label="Baseline: elapsed_sec (4.28 мин)")
    ax.axhline(5.22, color="orange", ls=":", lw=1.2, label="Baseline: predict mean (5.22 мин)")

    ax.set_xlabel("Количество обучающих субъектов")
    ax.set_ylabel("MAE (мин)")
    ax.set_title("Кривая обучения: ElasticNet(NIRS+HRV), LOSO", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "7_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → 7_learning_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("ФИНАЛЬНЫЙ АНАЛИЗ")
    print("=" * 70)

    # ── Данные ──
    print("\n[1] Загрузка и дельта-признаки...")
    df_raw = load_data()
    df = add_delta_features(df_raw)
    feat_sets = get_feature_sets(df)
    n_delta = len(feat_sets["delta_only"])
    print(f"    Окон: {len(df)}, субъектов: {df['subject_id'].nunique()}")
    print(f"    Дельта-признаков добавлено: {n_delta} → {feat_sets['delta_only']}")

    # ── Naive baselines ──
    print("\n[2] Наивные baseline-ы...")
    baselines = naive_baselines(df)
    for n, v in baselines.items():
        print(f"    {n:10s}: MAE={v['mae_min']:.3f} ± {v['mae_std']:.3f}")

    # ── Фиксированный ElasticNet (nirs_hrv) — точка отсчёта ──
    pipe_en = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=3000)),
    ])
    pipe_ridge = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", Ridge(alpha=100.0)),
    ])

    print("\n[3] ElasticNet(0.5, 0.9) на nirs_hrv...")
    agg_en, fold_en, preds_en = loso_predict(df, feat_sets["nirs_hrv"], pipe_en, return_preds=True)
    print(f"    MAE={agg_en['mae_min']:.3f} ± {agg_en['mae_std']:.3f} | R²={agg_en['r2']:.3f}")

    print("\n[4] ElasticNet на nirs_hrv + delta...")
    agg_en_d, fold_en_d, preds_en_d = loso_predict(df, feat_sets["nirs_hrv_delta"], pipe_en, return_preds=True)
    print(f"    MAE={agg_en_d['mae_min']:.3f} ± {agg_en_d['mae_std']:.3f} | R²={agg_en_d['r2']:.3f}")

    print("\n[5] Ridge(100) на nirs_hrv + delta...")
    agg_r_d, fold_r_d, _ = loso_predict(df, feat_sets["nirs_hrv_delta"], pipe_ridge, return_preds=True)
    print(f"    MAE={agg_r_d['mae_min']:.3f} ± {agg_r_d['mae_std']:.3f} | R²={agg_r_d['r2']:.3f}")

    print("\n[6] Ансамбль ElasticNet + Ridge (nirs_hrv)...")
    agg_ens, fold_ens, preds_ens = ensemble_loso_predict(df, feat_sets["nirs_hrv"])
    print(f"    MAE={agg_ens['mae_min']:.3f} ± {agg_ens['mae_std']:.3f} | R²={agg_ens['r2']:.3f}")

    print("\n[7] Ансамбль ElasticNet + Ridge (nirs_hrv + delta)...")
    agg_ens_d, fold_ens_d, preds_ens_d = ensemble_loso_predict(df, feat_sets["nirs_hrv_delta"])
    print(f"    MAE={agg_ens_d['mae_min']:.3f} ± {agg_ens_d['mae_std']:.3f} | R²={agg_ens_d['r2']:.3f}")

    print("\n[8] Nested LOSO (nirs_hrv + delta)...")
    t0 = time.perf_counter()
    agg_nested, fold_nested, preds_nested, best_params = nested_loso_predict(df, feat_sets["nirs_hrv_delta"])
    print(f"    Время: {time.perf_counter()-t0:.0f}с")
    print(f"    MAE={agg_nested['mae_min']:.3f} ± {agg_nested['mae_std']:.3f} | R²={agg_nested['r2']:.3f}")

    print("\n[9] Nested LOSO (nirs_hrv только)...")
    agg_nested_nh, fold_nested_nh, preds_nested_nh, best_params_nh = nested_loso_predict(df, feat_sets["nirs_hrv"])
    print(f"    MAE={agg_nested_nh['mae_min']:.3f} ± {agg_nested_nh['mae_std']:.3f} | R²={agg_nested_nh['r2']:.3f}")

    # ── Финальный лидерборд ──
    all_models = {
        "Baseline: predict 0":       {"mae_min": baselines["zero"]["mae_min"],    "mae_std": baselines["zero"]["mae_std"]},
        "Baseline: predict mean":     {"mae_min": baselines["mean"]["mae_min"],    "mae_std": baselines["mean"]["mae_std"]},
        "Baseline: power (linear)":   {"mae_min": baselines["power"]["mae_min"],   "mae_std": baselines["power"]["mae_std"]},
        "Baseline: elapsed (linear)": {"mae_min": baselines["elapsed"]["mae_min"], "mae_std": baselines["elapsed"]["mae_std"]},
        "ElasticNet (nirs_hrv)":      {**agg_en,    "mae_std": agg_en["mae_std"]},
        "ElasticNet (nirs_hrv+Δ)":    {**agg_en_d,  "mae_std": agg_en_d["mae_std"]},
        "Ridge (nirs_hrv+Δ)":         {**agg_r_d,   "mae_std": agg_r_d["mae_std"]},
        "Ensemble (nirs_hrv)":         {**agg_ens,   "mae_std": agg_ens["mae_std"]},
        "Ensemble (nirs_hrv+Δ)":       {**agg_ens_d, "mae_std": agg_ens_d["mae_std"]},
        "Nested LOSO (nirs_hrv)":      {**agg_nested_nh, "mae_std": agg_nested_nh["mae_std"]},
        "Nested LOSO (nirs_hrv+Δ)":   {**agg_nested,    "mae_std": agg_nested["mae_std"]},
    }

    print("\n" + "═" * 65)
    print("ИТОГОВЫЙ ЛИДЕРБОРД")
    print("═" * 65)
    sorted_models = sorted(all_models.items(), key=lambda x: x[1]["mae_min"])
    for rank, (name, v) in enumerate(sorted_models, 1):
        tag = " ← ЛУЧШАЯ" if rank == 1 else ""
        print(f"  {rank:2d}. {name:<30s} MAE={v['mae_min']:.3f} ± {v['mae_std']:.3f}{tag}")

    # Сохраняем лидерборд
    lb_df = pd.DataFrame([{"model": n, **v} for n, v in all_models.items()])
    lb_df.to_csv(OUT_DIR / "leaderboard_final.csv", index=False)

    # ── Определяем лучшую модель для визуализаций ──
    best_name = sorted_models[0][0]
    if "Nested" in best_name and "Δ" in best_name:
        best_preds, best_fold = preds_nested, fold_nested
    elif "Nested" in best_name:
        best_preds, best_fold = preds_nested_nh, fold_nested_nh
    elif "Ensemble" in best_name and "Δ" in best_name:
        best_preds, best_fold = preds_ens_d, fold_ens_d
    elif "Ensemble" in best_name:
        best_preds, best_fold = preds_ens, fold_ens
    elif "Δ" in best_name:
        best_preds, best_fold = preds_en_d, fold_en_d
    else:
        best_preds, best_fold = preds_en, fold_en

    # Добавляем train_mae в fold_en (уже есть)
    if "train_mae_min" not in best_fold.columns:
        best_fold["train_mae_min"] = np.nan

    # ── Визуализации ──
    print(f"\n[10] Строим визуализации ({best_name})...")
    plot_predicted_vs_actual(df, best_preds, best_fold, best_name)
    plot_model_comparison(all_models)
    plot_residuals(df, best_preds, best_name)
    plot_trajectories(df, best_preds, n_subjects=6)
    plot_coefficients(df, feat_sets["nirs_hrv_delta"])
    plot_nested_params(best_params)
    plot_learning_curve(df, feat_sets["nirs_hrv"])

    print(f"\n✅ Готово. Всё в: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
