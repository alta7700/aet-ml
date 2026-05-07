"""v0004 — Ridge(LT2) + HuberRegressor(LT1)

Версия:    v0004
Дата:      2026-05-07
Предыдущая версия: v0001_train_elasticnet.py
Результаты: results/v0004/

Что изменено по сравнению с v0001:
  - LT2: ElasticNet → Ridge(alpha=100) — детерминированная, чуть лучше
  - LT1: ElasticNet → HuberRegressor(epsilon=1.35) — устойчива к выбросу S003
  - Добавлен вывод LOSO-усреднённых коэффициентов (важность признаков)

Гиперпараметры:
  Модель LT2: Ridge(alpha=100.0, fit_intercept=True, solver="auto")  [детерминирован]
  Модель LT1: HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500,
              fit_intercept=True, warm_start=False)  [детерминирован]
  Imputer:    SimpleImputer(strategy="median")
  Scaler:     StandardScaler(with_mean=True, with_std=True)
  Session-z EMG: z = (x - mu_subj) / (std_subj + 1e-8)  [per-subject, per-feature]

Ожидаемые результаты:
  LT2: MAE = 2.097 мин, R² = 0.846, ρ = 0.919  (NIRS+HRV+interaction+z_EMG)
  LT1: MAE = 2.021 мин, R² = 0.842, ρ = 0.935  (z_EMG+HRV)

Воспроизведение:
  uv run python scripts/v0004_train_ridge_huber.py
  uv run python scripts/v0004_train_ridge_huber.py --target lt1
  uv run python scripts/v0004_train_ridge_huber.py --no-plots
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
from sklearn.linear_model import ElasticNet, Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR

OUT_DIR = _ROOT / "results" / "v0004"

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

# EMG: все «сырые» признаки с prefixом vl_ (68 штук)
# После session-z у них появятся дублёры с prefixом z_vl_
EMG_RAW_PREFIX = "vl_"


# ─── Нормировка: session-z для ЭМГ ───────────────────────────────────────────

def add_session_z_emg(df: pd.DataFrame) -> pd.DataFrame:
    """Session-z нормировка ЭМГ-признаков.

    Для каждого участника отдельно вычитаем среднее и делим на std.
    Убирает межсубъектные различия абсолютных значений ЭМГ (мкВ),
    которые зависят от электрода, кожи, анатомии — но сохраняет
    форму траектории (как она меняется с нагрузкой).

    Применяется ТОЛЬКО к ЭМГ, не к NIRS (там % — физическая шкала).
    """
    df = df.copy()
    emg_cols = [c for c in df.columns if c.startswith(EMG_RAW_PREFIX)]

    for col in emg_cols:
        subj_mean = df.groupby("subject_id")[col].transform("mean")
        subj_std = df.groupby("subject_id")[col].transform("std")
        df[f"z_{col}"] = (df[col] - subj_mean) / (subj_std + 1e-8)

    return df


# ─── Interaction признаки ─────────────────────────────────────────────────────

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


# ─── Running NIRS признаки для LT1 ───────────────────────────────────────────

def add_running_nirs_features(
    df: pd.DataFrame,
    session_params: pd.DataFrame,
) -> pd.DataFrame:
    """Каузальные running-признаки NIRS для LT1.

    Проблема: у разных участников LT1 наступает при разном абсолютном SmO₂
    (у одного при 75%, у другого при 62%), поэтому абсолютный SmO₂ не работает
    кросс-субъектно. Но внутри сессии ρ(SmO₂_drop, time_to_LT1) = 0.977.

    Решение: smo2_from_running_max = max(SmO₂ в сессии до сих пор) − текущий.
    Каузально: используем только прошлое (expanding window).
    """
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
                df.loc[idx, "smo2_rel_drop_pct"] = (
                    (baseline_smo2 - smo2) / baseline_smo2 * 100.0
                )
            first_hhb = g["trainred_hhb_mean"].iloc[:3].mean()
            if np.isfinite(first_hhb) and first_hhb != 0:
                df.loc[idx, "hhb_rel_rise_pct"] = (
                    (hhb - first_hhb) / abs(first_hhb) * 100.0
                )

    return df


# ─── LOSO CV ──────────────────────────────────────────────────────────────────

def loso(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_factory=None,
) -> dict:
    """LOSO CV. Возвращает метрики, предсказания и усреднённые коэффициенты.

    Args:
        model_factory: callable без аргументов, возвращающий sklearn-модель.
            По умолчанию ElasticNet(alpha=0.5, l1_ratio=0.9).
            Поддерживаются модели с атрибутами .coef_ (Ridge, ElasticNet, Huber).
    """
    if model_factory is None:
        # random_state=42 зафиксирован для воспроизведения (ElasticNet использует
        # случайность во внутренней оптимизации coordinate descent)
        model_factory = lambda: ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=5000,
                                           random_state=42)

    subjects = sorted(df["subject_id"].unique())
    feat_cols = [f for f in features if f in df.columns]
    preds, trues, subjs = [], [], []
    coefs_list = []  # коэффициенты по каждому фолду

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test = df[df["subject_id"] == test_s]

        X_tr = train[feat_cols].values
        y_tr = train[target].values
        X_te = test[feat_cols].values
        y_te = test[target].values

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        model = model_factory()

        X_tr = sc.fit_transform(imp.fit_transform(X_tr))
        X_te = sc.transform(imp.transform(X_te))
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        preds.append(y_pred)
        trues.append(y_te)
        subjs.append(np.full(len(y_te), test_s))

        # Сбор коэффициентов (для Ridge, ElasticNet, HuberRegressor)
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
    mae_std = float(np.std(list(per_subj.values())))

    # Усреднённые LOSO-коэффициенты (знак = направление эффекта)
    coef_mean = np.mean(coefs_list, axis=0) if coefs_list else None
    coef_std = np.std(coefs_list, axis=0) if coefs_list else None

    return {
        "mae_min": mae_min,
        "mae_std": mae_std,
        "r2": r2,
        "rho": rho,
        "per_subj_mae_min": per_subj,
        "y_pred": y_pred_all,
        "y_true": y_true_all,
        "subjects": subj_all,
        "features_used": feat_cols,
        "coef_mean": coef_mean,
        "coef_std": coef_std,
    }


def print_feature_importance(result: dict, top_n: int = 15) -> None:
    """Печатает таблицу LOSO-усреднённых коэффициентов (важность признаков).

    Коэффициенты стандартизированы (входные данные прошли StandardScaler),
    поэтому их абсолютные значения сравнимы между собой.
    Знак: «+» → чем больше признак, тем дальше до порога (позже);
           «-» → чем больше признак, тем ближе к порогу.
    """
    coef_mean = result.get("coef_mean")
    coef_std = result.get("coef_std")
    feats = result.get("features_used", [])
    if coef_mean is None or len(feats) == 0:
        return

    order = np.argsort(np.abs(coef_mean))[::-1][:top_n]
    print(f"\n  {'Признак':<42s} {'Коэф (μ±σ)':<22s} {'Направление'}")
    print("  " + "─" * 80)
    for i in order:
        name = feats[i]
        mu = coef_mean[i]
        sd = coef_std[i] if coef_std is not None else 0.0
        sign = "→ позже (дальше)" if mu > 0 else "→ раньше (ближе)"
        print(f"  {name:<42s}  {mu:+.1f} ± {sd:.1f}     {sign}")


def baseline_mae(df: pd.DataFrame, target: str) -> float:
    """LOSO baseline: предсказываем среднее по тренировке."""
    subjects = sorted(df["subject_id"].unique())
    errors = []
    for s in subjects:
        y_tr = df[df["subject_id"] != s][target].values
        y_te = df[df["subject_id"] == s][target].values
        errors.append(float(mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))))
    return float(np.mean(errors)) / 60.0


# ─── Визуализация ─────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(
    result: dict,
    df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Scatter predicted vs actual + траектории по участникам."""
    subjects = sorted(df["subject_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}

    y_true = result["y_true"] / 60.0
    y_pred = result["y_pred"] / 60.0
    subj_labels = result["subjects"]
    mae = result["mae_min"]
    r2 = result["r2"]
    rho = result["rho"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Scatter ──
    ax = axes[0]
    for s in subjects:
        m = subj_labels == s
        ax.scatter(y_true[m], y_pred[m], color=subj_color[s],
                   alpha=0.3, s=8, label=s)
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="идеал")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Истинное время до порога, мин")
    ax.set_ylabel("Предсказанное время до порога, мин")
    ax.set_title(f"MAE={mae:.3f} мин | R²={r2:.3f} | ρ={rho:.3f}")
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # ── MAE по участникам ──
    ax2 = axes[1]
    per = result["per_subj_mae_min"]
    subs_sorted = sorted(per.keys(), key=lambda s: per[s])
    vals = [per[s] for s in subs_sorted]
    colors = [subj_color[s] for s in subs_sorted]
    bars = ax2.barh(subs_sorted, vals, color=colors, alpha=0.85)
    ax2.axvline(mae, color="black", lw=1.5, ls="--", label=f"Среднее {mae:.2f}")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=8)
    ax2.set_xlabel("MAE, мин")
    ax2.set_title("MAE по участникам (LOSO)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path.name}")


def plot_trajectories(
    result: dict,
    df: pd.DataFrame,
    target: str,
    title: str,
    output_path: Path,
) -> None:
    """Временны́е траектории предсказаний vs истина по участникам."""
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
        y_true_s = sub[target].values / 60.0
        y_pred_s = y_pred[subj_labels == s]

        ax.plot(t, y_true_s, "k-", lw=1.8, label="Истина")
        ax.plot(t, y_pred_s, color=subj_color[s], lw=1.5, ls="--", label="Предсказание")
        ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.8)
        mae_s = result["per_subj_mae_min"][s]
        ax.set_title(f"{s} | MAE={mae_s:.2f} мин", fontsize=9)
        ax.set_xlabel("Прошло, мин", fontsize=8)
        ax.set_ylabel("До порога, мин", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path.name}")


# ─── LT2 ─────────────────────────────────────────────────────────────────────

def run_lt2(df_full: pd.DataFrame, no_plots: bool = False) -> dict:
    """Финальная модель LT2: NIRS + HRV + interaction + session-z EMG."""
    print("\n" + "═" * 65)
    print("LT2 — финальная модель")
    print("═" * 65)

    # Фильтрация: только окна со всеми тремя модальностями
    df = df_full[df_full["window_valid_all_required"] == 1].copy()
    df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    print(f"Окон: {len(df)}, участников: {df['subject_id'].nunique()}")

    # Признаки
    df = add_interaction_features(df)
    df = add_session_z_emg(df)

    target = "target_time_to_lt2_center_sec"

    # Ключевые z_EMG признаки (самые физиологически осмысленные)
    z_emg_key = [
        "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
        "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
        "z_vl_prox_rest_rms",
        "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
    ]
    z_emg_key = [c for c in z_emg_key if c in df.columns]

    inter_key = ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"]
    inter_key = [c for c in inter_key if c in df.columns]

    # Наборы для сравнения
    configs = {
        "Baseline (NIRS+HRV)": NIRS_FEATURES + HRV_FEATURES,
        "NIRS+HRV+interaction": NIRS_FEATURES + HRV_FEATURES + inter_key,
        "NIRS+HRV+interaction+z_EMG": NIRS_FEATURES + HRV_FEATURES + inter_key + z_emg_key,
        "z_EMG только": z_emg_key,
        "z_EMG+HRV": z_emg_key + HRV_FEATURES,
    }

    b_mae = baseline_mae(df, target)
    print(f"\nBaseline (predict-mean): {b_mae:.3f} мин")

    # Лучшая модель для LT2: Ridge(alpha=100) — стабильнее ElasticNet на 13 фолдах
    # Ridge детерминирован (не использует случайность), seed не нужен
    ridge_factory = lambda: Ridge(alpha=100.0, fit_intercept=True, solver="auto")

    results = {}
    for name, feats in configs.items():
        avail = [f for f in feats if f in df.columns]
        if len(avail) < 2:
            continue
        r = loso(df, avail, target, model_factory=ridge_factory)
        std = float(np.std(list(r["per_subj_mae_min"].values())))
        print(f"  {name:<40s} MAE={r['mae_min']:.3f}±{std:.3f}  R²={r['r2']:.3f}  ρ={r['rho']:.3f}")
        results[name] = r

    best_name = min(results, key=lambda k: results[k]["mae_min"])
    best = results[best_name]
    print(f"\n★ Лучшая конфигурация: {best_name}  [Ridge(α=100)]")
    print(f"  MAE  = {best['mae_min']:.3f} мин")
    print(f"  R²   = {best['r2']:.3f}")
    print(f"  ρ    = {best['rho']:.3f}")
    print("  По участникам:")
    for s, m in sorted(best["per_subj_mae_min"].items()):
        print(f"    {s}: {m:.2f} мин")

    print("\n  Важность признаков (LOSO-усреднённые коэффициенты Ridge):")
    print_feature_importance(best)

    if not no_plots:
        out = OUT_DIR / "lt2"
        plot_predicted_vs_actual(
            best, df,
            f"LT2: {best_name}\nLOSO Ridge(α=100)",
            out / "lt2_predicted_vs_actual.png",
        )
        plot_trajectories(
            best, df, target,
            f"LT2: Траектории предсказаний ({best_name})",
            out / "lt2_trajectories.png",
        )

    return {"best_name": best_name, "results": results, "df": df}


# ─── LT1 ─────────────────────────────────────────────────────────────────────

def run_lt1(df_full: pd.DataFrame, no_plots: bool = False) -> dict:
    """Финальная модель LT1: session-z EMG + HRV + running NIRS."""
    print("\n" + "═" * 65)
    print("LT1 — финальная модель")
    print("═" * 65)

    # Загружаем session_params для running NIRS
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    # Фильтрация: только usable LT1 (high/medium quality, lt1 ≠ lt2)
    df = df_full[df_full["target_time_to_lt1_usable"] == 1].copy()
    df = df[df["nirs_valid"] == 1].copy()
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    print(f"Окон: {len(df)}, участников: {df['subject_id'].nunique()}")
    print(f"Участники: {sorted(df['subject_id'].unique())}")

    # Признаки
    df = add_session_z_emg(df)
    if not session_params.empty:
        df = add_running_nirs_features(df, session_params)

    target = "target_time_to_lt1_sec"

    z_emg_key = [
        "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
        "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
        "z_vl_prox_rest_rms",
        "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
    ]
    z_emg_key = [c for c in z_emg_key if c in df.columns]

    running_nirs = [
        "smo2_from_running_max", "hhb_from_running_min",
        "smo2_rel_drop_pct", "hhb_rel_rise_pct",
    ]
    running_nirs = [c for c in running_nirs if c in df.columns]

    configs = {
        "Baseline (NIRS+HRV)": NIRS_FEATURES + HRV_FEATURES,
        "HRV только": HRV_FEATURES,
        "z_EMG только": z_emg_key,
        "z_EMG+HRV": z_emg_key + HRV_FEATURES,
        "Running NIRS+HRV": running_nirs + HRV_FEATURES,
        "z_EMG+HRV+running NIRS": z_emg_key + HRV_FEATURES + running_nirs,
        "z_EMG+running NIRS": z_emg_key + running_nirs,
    }

    b_mae = baseline_mae(df, target)
    print(f"\nBaseline (predict-mean): {b_mae:.3f} мин")

    # Лучшая модель для LT1: HuberRegressor(epsilon=1.35)
    # Причина: S003 — элитный атлет с нестандартной физиологией, сильный выброс.
    # Huber снижает его вес без исключения из датасета.
    # HuberRegressor детерминирован, seed не нужен.
    huber_factory = lambda: HuberRegressor(
        epsilon=1.35, alpha=0.0001, max_iter=500,
        fit_intercept=True, warm_start=False,
    )

    results = {}
    for name, feats in configs.items():
        avail = [f for f in feats if f in df.columns]
        if len(avail) < 2:
            continue
        r = loso(df, avail, target, model_factory=huber_factory)
        std = float(np.std(list(r["per_subj_mae_min"].values())))
        print(f"  {name:<40s} MAE={r['mae_min']:.3f}±{std:.3f}  R²={r['r2']:.3f}  ρ={r['rho']:.3f}")
        results[name] = r

    best_name = min(results, key=lambda k: results[k]["mae_min"])
    best = results[best_name]
    print(f"\n★ Лучшая конфигурация: {best_name}  [HuberRegressor(ε=1.35)]")
    print(f"  MAE  = {best['mae_min']:.3f} мин")
    print(f"  R²   = {best['r2']:.3f}")
    print(f"  ρ    = {best['rho']:.3f}")
    print("  По участникам:")
    for s, m in sorted(best["per_subj_mae_min"].items()):
        print(f"    {s}: {m:.2f} мин")

    print("\n  Важность признаков (LOSO-усреднённые коэффициенты Huber):")
    print_feature_importance(best)

    if not no_plots:
        out = OUT_DIR / "lt1"
        plot_predicted_vs_actual(
            best, df,
            f"LT1: {best_name}\nLOSO HuberRegressor(ε=1.35)",
            out / "lt1_predicted_vs_actual.png",
        )
        plot_trajectories(
            best, df, target,
            f"LT1: Траектории предсказаний ({best_name})",
            out / "lt1_trajectories.png",
        )

    return {"best_name": best_name, "results": results, "df": df}


# ─── Сохранение сводки ────────────────────────────────────────────────────────

def save_summary(lt2_out: dict, lt1_out: dict) -> None:
    """Записывает сводный CSV с результатами всех конфигураций."""
    rows = []
    for task, out in [("LT2", lt2_out), ("LT1", lt1_out)]:
        for name, r in out["results"].items():
            rows.append({
                "task": task,
                "config": name,
                "mae_min": r["mae_min"],
                "mae_std": r["mae_std"],
                "r2": r["r2"],
                "rho": r["rho"],
                "best": name == out["best_name"],
            })
    df = pd.DataFrame(rows).sort_values(["task", "mae_min"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "summary.csv"
    df.to_csv(path, index=False)
    print(f"\n✅ Сводка сохранена: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    p = argparse.ArgumentParser(description="Финальные модели LT1 и LT2.")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet",
    )
    return p.parse_args()


def main() -> None:
    """Точка входа."""
    args = parse_args()
    print("=" * 65)
    print("ФИНАЛЬНЫЕ МОДЕЛИ — LT1 и LT2")
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
