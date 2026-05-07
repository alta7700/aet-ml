"""v0001 — ElasticNet baseline: LT1 + LT2

Версия:    v0001
Дата:      2026-05-07
Предыдущая версия: нет
Результаты: results/v0001/

Что делает:
  Первая рабочая мультимодальная модель для предсказания времени до LT1 и LT2.
  Session-z нормировка ЭМГ, interaction-признаки SmO2×RR, running NIRS для LT1.

Гиперпараметры:
  Модель (LT1 и LT2): ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=5000, random_state=42)
  Imputer:            SimpleImputer(strategy="median")
  Scaler:             StandardScaler(with_mean=True, with_std=True)
  Session-z EMG:      z = (x - mu_subj) / (std_subj + 1e-8)  [per-subject, per-feature]
  CV:                 LOSO (Leave-One-Subject-Out), нет вложенного CV

Ожидаемые результаты:
  LT2: MAE = 2.110 мин, R² = 0.846, ρ = 0.920  (конфигурация: NIRS+HRV+interaction+z_EMG)
  LT1: MAE = 2.030 мин, R² = 0.838, ρ = 0.932  (конфигурация: z_EMG+HRV)

Воспроизведение:
  uv run python scripts/v0001_train_elasticnet.py
  uv run python scripts/v0001_train_elasticnet.py --target lt1
  uv run python scripts/v0001_train_elasticnet.py --no-plots

Замечания:
  - ElasticNet с random_state не зафиксированным в ранних запусках мог давать
    незначительно другие числа. Здесь random_state=42 зафиксирован.
  - Супершёдший вариант: v0004_train_ridge_huber.py (Ridge/Huber, чуть лучше).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR

# Результаты пишутся в results/v0001/
OUT_DIR = _ROOT / "results" / "v0001"

# ─── Гиперпараметры модели ────────────────────────────────────────────────────

def make_model() -> ElasticNet:
    """Фабрика модели v0001: ElasticNet с зафиксированными гиперпараметрами."""
    return ElasticNet(
        alpha=0.5,
        l1_ratio=0.9,
        max_iter=5000,
        tol=1e-4,           # sklearn default
        fit_intercept=True,  # sklearn default
        random_state=42,     # зафиксирован для воспроизведения
    )


# ─── Наборы исходных признаков ────────────────────────────────────────────────

NIRS_FEATURES = [
    "trainred_smo2_mean", "trainred_smo2_drop", "trainred_smo2_slope",
    "trainred_smo2_std", "trainred_dsmo2_dt",
    "trainred_hhb_mean", "trainred_hhb_slope", "trainred_hhb_std", "trainred_dhhb_dt",
    "trainred_hbdiff_mean", "trainred_hbdiff_slope", "trainred_hbdiff_std",
    "trainred_thb_mean", "trainred_thb_slope", "trainred_thb_std",
]

HRV_FEATURES = [
    "hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms",
    "hrv_sd1_ms", "hrv_sd2_ms", "hrv_sd1_sd2_ratio",
    "hrv_dfa_alpha1",
]

EMG_RAW_PREFIX = "vl_"


# ─── Feature engineering ──────────────────────────────────────────────────────

def add_session_z_emg(df: pd.DataFrame) -> pd.DataFrame:
    """Session-z нормировка ЭМГ-признаков (per-subject, per-feature)."""
    df = df.copy()
    emg_cols = [c for c in df.columns if c.startswith(EMG_RAW_PREFIX)]
    for col in emg_cols:
        subj_mean = df.groupby("subject_id")[col].transform("mean")
        subj_std = df.groupby("subject_id")[col].transform("std")
        df[f"z_{col}"] = (df[col] - subj_mean) / (subj_std + 1e-8)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Произведения сигналов из разных физиологических систем."""
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    df["feat_smo2_x_rr"] = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
    df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    df["feat_hhb_x_rr"] = df["trainred_hhb_mean"] * df["hrv_mean_rr_ms"] / 1e3
    df["feat_rr_per_watt"] = df["hrv_mean_rr_ms"] / pw
    df["feat_smo2_per_watt"] = df["trainred_smo2_mean"] / pw
    return df


def add_running_nirs_features(df: pd.DataFrame, session_params: pd.DataFrame) -> pd.DataFrame:
    """Каузальные running-признаки NIRS для LT1."""
    df = df.copy()
    sp = session_params.set_index("subject_id")
    for col in ["smo2_from_running_max", "hhb_from_running_min",
                "smo2_rel_drop_pct", "hhb_rel_rise_pct", "hbdiff_from_running_max"]:
        df[col] = np.nan
    for subj, g in df.groupby("subject_id"):
        g = g.sort_values("window_center_sec")
        idx = g.index
        smo2 = g["trainred_smo2_mean"].values
        hhb = g["trainred_hhb_mean"].values
        hbdiff = g["trainred_hbdiff_mean"].values
        smo2_rmax = np.maximum.accumulate(np.where(np.isfinite(smo2), smo2, -np.inf))
        hhb_rmin = np.minimum.accumulate(np.where(np.isfinite(hhb), hhb, np.inf))
        hbdiff_rmax = np.maximum.accumulate(np.where(np.isfinite(hbdiff), hbdiff, -np.inf))
        smo2_rmax = np.where(np.isinf(smo2_rmax), np.nan, smo2_rmax)
        hhb_rmin = np.where(np.isinf(hhb_rmin), np.nan, hhb_rmin)
        hbdiff_rmax = np.where(np.isinf(hbdiff_rmax), np.nan, hbdiff_rmax)
        df.loc[idx, "smo2_from_running_max"] = smo2_rmax - smo2
        df.loc[idx, "hhb_from_running_min"] = hhb - hhb_rmin
        df.loc[idx, "hbdiff_from_running_max"] = hbdiff_rmax - hbdiff
        if subj in sp.index:
            baseline_smo2 = float(sp.loc[subj, "nirs_smo2_baseline_mean"])
            if np.isfinite(baseline_smo2) and baseline_smo2 > 0:
                df.loc[idx, "smo2_rel_drop_pct"] = (baseline_smo2 - smo2) / baseline_smo2 * 100.0
            first_hhb = g["trainred_hhb_mean"].iloc[:3].mean()
            if np.isfinite(first_hhb) and first_hhb != 0:
                df.loc[idx, "hhb_rel_rise_pct"] = (hhb - first_hhb) / abs(first_hhb) * 100.0
    return df


# ─── LOSO CV ──────────────────────────────────────────────────────────────────

def loso(df: pd.DataFrame, features: list[str], target: str) -> dict:
    """LOSO CV с make_model(). Возвращает метрики, предсказания и коэффициенты."""
    subjects = sorted(df["subject_id"].unique())
    feat_cols = [f for f in features if f in df.columns]
    preds, trues, subjs, coefs_list = [], [], [], []

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test = df[df["subject_id"] == test_s]
        X_tr = train[feat_cols].values
        y_tr = train[target].values
        X_te = test[feat_cols].values
        y_te = test[target].values

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        model = make_model()

        X_tr = sc.fit_transform(imp.fit_transform(X_tr))
        X_te = sc.transform(imp.transform(X_te))
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        preds.append(y_pred)
        trues.append(y_te)
        subjs.append(np.full(len(y_te), test_s))
        if hasattr(model, "coef_"):
            coefs_list.append(model.coef_)

    y_pred_all = np.concatenate(preds)
    y_true_all = np.concatenate(trues)
    subj_all = np.concatenate(subjs)

    mae_min = float(mean_absolute_error(y_true_all, y_pred_all)) / 60.0
    r2 = float(r2_score(y_true_all, y_pred_all))
    rho = float(spearmanr(y_true_all, y_pred_all).statistic)
    per_subj = {
        s: float(mean_absolute_error(y_true_all[subj_all == s],
                                     y_pred_all[subj_all == s])) / 60.0
        for s in subjects
    }
    return {
        "mae_min": mae_min,
        "mae_std": float(np.std(list(per_subj.values()))),
        "r2": r2,
        "rho": rho,
        "per_subj_mae_min": per_subj,
        "y_pred": y_pred_all,
        "y_true": y_true_all,
        "subjects": subj_all,
        "features_used": feat_cols,
        "coef_mean": np.mean(coefs_list, axis=0) if coefs_list else None,
        "coef_std": np.std(coefs_list, axis=0) if coefs_list else None,
    }


def baseline_mae(df: pd.DataFrame, target: str) -> float:
    """LOSO baseline: предсказываем среднее по тренировке."""
    subjects = sorted(df["subject_id"].unique())
    errors = []
    for s in subjects:
        y_tr = df[df["subject_id"] != s][target].values
        y_te = df[df["subject_id"] == s][target].values
        errors.append(float(mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))))
    return float(np.mean(errors)) / 60.0


def print_feature_importance(result: dict, top_n: int = 15) -> None:
    """Топ признаков по LOSO-усреднённым коэффициентам."""
    coef_mean = result.get("coef_mean")
    coef_std = result.get("coef_std")
    feats = result.get("features_used", [])
    if coef_mean is None or len(feats) == 0:
        return
    order = np.argsort(np.abs(coef_mean))[::-1][:top_n]
    print(f"\n  {'Признак':<42s} {'Коэф (μ±σ)':<22s} {'Направление'}")
    print("  " + "─" * 80)
    for i in order:
        mu = coef_mean[i]
        sd = coef_std[i] if coef_std is not None else 0.0
        sign = "→ позже (дальше)" if mu > 0 else "→ раньше (ближе)"
        print(f"  {feats[i]:<42s}  {mu:+.1f} ± {sd:.1f}     {sign}")


# ─── Визуализация ─────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(result: dict, df: pd.DataFrame, title: str, path: Path) -> None:
    """Scatter predicted vs actual + MAE по участникам."""
    subjects = sorted(df["subject_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}
    y_true = result["y_true"] / 60.0
    y_pred = result["y_pred"] / 60.0
    subj_labels = result["subjects"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    for s in subjects:
        m = subj_labels == s
        ax.scatter(y_true[m], y_pred[m], color=subj_color[s], alpha=0.3, s=8, label=s)
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Истинное время до порога, мин")
    ax.set_ylabel("Предсказанное время до порога, мин")
    ax.set_title(f"MAE={result['mae_min']:.3f} | R²={result['r2']:.3f} | ρ={result['rho']:.3f}")
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    per = result["per_subj_mae_min"]
    subs_sorted = sorted(per.keys(), key=lambda s: per[s])
    vals = [per[s] for s in subs_sorted]
    bars = ax2.barh(subs_sorted, vals, color=[subj_color[s] for s in subs_sorted], alpha=0.85)
    ax2.axvline(result["mae_min"], color="black", lw=1.5, ls="--")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=8)
    ax2.set_xlabel("MAE, мин")
    ax2.set_title("MAE по участникам (LOSO)")
    ax2.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")


def plot_trajectories(result: dict, df: pd.DataFrame, target: str, title: str, path: Path) -> None:
    """Временные траектории предсказаний vs истина."""
    subjects = sorted(df["subject_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}
    ncols = 3
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle(title, fontsize=12, fontweight="bold")
    y_pred = result["y_pred"] / 60.0
    subj_labels = result["subjects"]
    for i, s in enumerate(subjects):
        ax = axes_flat[i]
        sub = df[df["subject_id"] == s].sort_values("elapsed_sec")
        t = sub["elapsed_sec"].values / 60.0
        ax.plot(t, sub[target].values / 60.0, "k-", lw=1.8, label="Истина")
        ax.plot(t, y_pred[subj_labels == s], color=subj_color[s], lw=1.5, ls="--",
                label="Предсказание")
        ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.8)
        ax.set_title(f"{s} | MAE={result['per_subj_mae_min'][s]:.2f} мин", fontsize=9)
        ax.set_xlabel("Прошло, мин", fontsize=8)
        ax.set_ylabel("До порога, мин", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")


# ─── LT2 ─────────────────────────────────────────────────────────────────────

def run_lt2(df_full: pd.DataFrame, no_plots: bool = False) -> dict:
    """LT2: NIRS + HRV + interaction + session-z EMG, ElasticNet v0001."""
    print("\n" + "═" * 65)
    print("LT2 — v0001 ElasticNet(α=0.5, l1=0.9, random_state=42)")
    print("═" * 65)

    df = df_full[df_full["window_valid_all_required"] == 1].copy()
    df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    print(f"Окон: {len(df)}, участников: {df['subject_id'].nunique()}")

    df = add_interaction_features(df)
    df = add_session_z_emg(df)

    target = "target_time_to_lt2_center_sec"
    z_emg_key = [c for c in [
        "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
        "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
        "z_vl_prox_rest_rms", "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
    ] if c in df.columns]
    inter_key = [c for c in ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"]
                 if c in df.columns]

    configs = {
        "Baseline (NIRS+HRV)": NIRS_FEATURES + HRV_FEATURES,
        "NIRS+HRV+interaction": NIRS_FEATURES + HRV_FEATURES + inter_key,
        "NIRS+HRV+interaction+z_EMG": NIRS_FEATURES + HRV_FEATURES + inter_key + z_emg_key,
        "z_EMG только": z_emg_key,
        "z_EMG+HRV": z_emg_key + HRV_FEATURES,
    }

    print(f"\nBaseline (predict-mean): {baseline_mae(df, target):.3f} мин")
    results = {}
    for name, feats in configs.items():
        avail = [f for f in feats if f in df.columns]
        if len(avail) < 2:
            continue
        r = loso(df, avail, target)
        print(f"  {name:<40s} MAE={r['mae_min']:.3f}±{r['mae_std']:.3f}  "
              f"R²={r['r2']:.3f}  ρ={r['rho']:.3f}")
        results[name] = r

    best_name = min(results, key=lambda k: results[k]["mae_min"])
    best = results[best_name]
    print(f"\n★ {best_name}  MAE={best['mae_min']:.3f}  R²={best['r2']:.3f}  ρ={best['rho']:.3f}")
    for s, m in sorted(best["per_subj_mae_min"].items()):
        print(f"    {s}: {m:.2f} мин")
    print_feature_importance(best)

    if not no_plots:
        out = OUT_DIR / "lt2"
        plot_predicted_vs_actual(best, df, f"LT2 v0001: {best_name}\nElasticNet(α=0.5, l1=0.9)",
                                 out / "lt2_predicted_vs_actual.png")
        plot_trajectories(best, df, target, f"LT2 v0001: Траектории ({best_name})",
                          out / "lt2_trajectories.png")

    return {"best_name": best_name, "results": results, "df": df}


# ─── LT1 ─────────────────────────────────────────────────────────────────────

def run_lt1(df_full: pd.DataFrame, no_plots: bool = False) -> dict:
    """LT1: session-z EMG + HRV, ElasticNet v0001."""
    print("\n" + "═" * 65)
    print("LT1 — v0001 ElasticNet(α=0.5, l1=0.9, random_state=42)")
    print("═" * 65)

    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    df = df_full[df_full["target_time_to_lt1_usable"] == 1].copy()
    df = df[df["nirs_valid"] == 1].copy()
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    print(f"Окон: {len(df)}, участников: {df['subject_id'].nunique()}")
    print(f"Участники: {sorted(df['subject_id'].unique())}")

    df = add_session_z_emg(df)
    if not session_params.empty:
        df = add_running_nirs_features(df, session_params)

    target = "target_time_to_lt1_sec"
    z_emg_key = [c for c in [
        "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
        "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
        "z_vl_prox_rest_rms", "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
    ] if c in df.columns]
    running_nirs = [c for c in [
        "smo2_from_running_max", "hhb_from_running_min", "smo2_rel_drop_pct", "hhb_rel_rise_pct",
    ] if c in df.columns]

    configs = {
        "Baseline (NIRS+HRV)": NIRS_FEATURES + HRV_FEATURES,
        "HRV только": HRV_FEATURES,
        "z_EMG только": z_emg_key,
        "z_EMG+HRV": z_emg_key + HRV_FEATURES,
        "Running NIRS+HRV": running_nirs + HRV_FEATURES,
        "z_EMG+HRV+running NIRS": z_emg_key + HRV_FEATURES + running_nirs,
        "z_EMG+running NIRS": z_emg_key + running_nirs,
    }

    print(f"\nBaseline (predict-mean): {baseline_mae(df, target):.3f} мин")
    results = {}
    for name, feats in configs.items():
        avail = [f for f in feats if f in df.columns]
        if len(avail) < 2:
            continue
        r = loso(df, avail, target)
        print(f"  {name:<40s} MAE={r['mae_min']:.3f}±{r['mae_std']:.3f}  "
              f"R²={r['r2']:.3f}  ρ={r['rho']:.3f}")
        results[name] = r

    best_name = min(results, key=lambda k: results[k]["mae_min"])
    best = results[best_name]
    print(f"\n★ {best_name}  MAE={best['mae_min']:.3f}  R²={best['r2']:.3f}  ρ={best['rho']:.3f}")
    for s, m in sorted(best["per_subj_mae_min"].items()):
        print(f"    {s}: {m:.2f} мин")
    print_feature_importance(best)

    if not no_plots:
        out = OUT_DIR / "lt1"
        plot_predicted_vs_actual(best, df, f"LT1 v0001: {best_name}\nElasticNet(α=0.5, l1=0.9)",
                                 out / "lt1_predicted_vs_actual.png")
        plot_trajectories(best, df, target, f"LT1 v0001: Траектории ({best_name})",
                          out / "lt1_trajectories.png")

    return {"best_name": best_name, "results": results, "df": df}


# ─── Сохранение сводки ────────────────────────────────────────────────────────

def save_summary(lt2_out: dict, lt1_out: dict) -> None:
    """Записывает сводный CSV с результатами всех конфигураций."""
    rows = []
    for task, out in [("LT2", lt2_out), ("LT1", lt1_out)]:
        for name, r in out["results"].items():
            rows.append({
                "task": task, "config": name,
                "mae_min": r["mae_min"], "mae_std": r["mae_std"],
                "r2": r["r2"], "rho": r["rho"],
                "best": name == out["best_name"],
            })
    df_out = pd.DataFrame(rows).sort_values(["task", "mae_min"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "summary.csv"
    df_out.to_csv(path, index=False)
    print(f"\n✅ Сводка: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Аргументы командной строки."""
    p = argparse.ArgumentParser(description="v0001: ElasticNet baseline LT1 + LT2.")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    """Точка входа."""
    args = parse_args()
    print("=" * 65)
    print("v0001 — ElasticNet(α=0.5, l1=0.9, random_state=42)")
    print("=" * 65)

    df_full = pd.read_parquet(args.dataset)
    print(f"Датасет: {df_full.shape[0]} окон, {df_full['subject_id'].nunique()} участников")

    lt2_out = lt1_out = None
    if args.target in ("lt2", "both"):
        lt2_out = run_lt2(df_full, no_plots=args.no_plots)
    if args.target in ("lt1", "both"):
        lt1_out = run_lt1(df_full, no_plots=args.no_plots)
    if lt2_out and lt1_out:
        save_summary(lt2_out, lt1_out)

    print("\n✅ Готово.")


if __name__ == "__main__":
    main()
