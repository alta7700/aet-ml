"""v0006 — Нормировка HRV/NIRS относительно первой ступени (stage 0)

Версия:    v0006
Дата:      2026-05-07
Предыдущая версия: v0004_train_ridge_huber.py
Результаты: results/v0006/

Проблема, решаемая в этой версии:
  hrv_mean_rr_ms — доминирующий признак (SHAP, коэффициенты), но несёт
  межсубъектный конфаунд: более тренированные спортсмены имеют ниже ЧСС покоя
  и выше пороги, что не является причинно-следственной связью для одного субъекта.
  Ρ_within = +0.997 (хороший сигнал), ρ_between = +0.753 (конфаунд).

Идея:
  Брать первую ступень протокола (stage_index == 0, ~первые 3 минуты, ~60 Вт)
  как индивидуальный anchor. Нормировать HRV и NIRS относительно него.
  Это убирает абсолютный уровень (= фитнес-конфаунд), сохраняет динамику
  (насколько ЧСС выросла с нагрузкой для данного человека).

Что изменено по сравнению с v0004:
  - Добавлена функция add_stage0_baselines(): per-subject baseline из stage 0
  - Новые признаки: rr_pct_change, rr_delta, rmssd_pct_change, sdnn_pct_change,
    sd1_pct_change, sd2_pct_change, dfa_delta, smo2_delta, hhb_delta,
    hbdiff_delta, rr_slope_per_stage
  - Тестируем несколько вариантов замены hrv_mean_rr_ms:
      A: hrv_mean_rr_ms → rr_pct_change + rr_delta
      B: все абсолютные HRV-ms → pct_change + dfa_delta; + smo2_delta
      C: вариант B + rr_slope_per_stage (скорость роста ЧСС)
      D: вариант A + rr_slope_per_stage (промежуточный)
  - Контрольный "ref": признаки v0004 как есть (для сравнения)

Гиперпараметры:
  Модель LT2: Ridge(alpha=100.0, fit_intercept=True, solver="auto")  [детерминирован]
  Модель LT1: HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500,
              fit_intercept=True, warm_start=False)  [детерминирован]
  Imputer:    SimpleImputer(strategy="median")
  Scaler:     StandardScaler(with_mean=True, with_std=True)
  Session-z EMG: z = (x - mu_subj) / (std_subj + 1e-8)  [per-subject, per-feature]
  Stage-0 baseline: mean(feature | stage_index==0, subject_id==s)

Воспроизведение:
  uv run python scripts/v0006_train_stage0_norm.py
  uv run python scripts/v0006_train_stage0_norm.py --target lt1
  uv run python scripts/v0006_train_stage0_norm.py --no-plots
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

OUT_DIR = _ROOT / "results" / "v0006"

# ─── Базовые наборы признаков (исходные, без нормировки) ─────────────────────

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

# HRV без абсолютных мс-признаков (оставляем только scale-free)
HRV_SCALEFREE = ["hrv_sd1_sd2_ratio", "hrv_dfa_alpha1"]

EMG_RAW_PREFIX = "vl_"

Z_EMG_KEY = [
    "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
    "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
    "z_vl_prox_rest_rms",
    "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
]

INTER_KEY_REF = ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"]


# ─── Feature engineering: stage-0 normalization ──────────────────────────────

# HRV-признаки, для которых считаем pct_change (мс-шкала)
_HRV_MS_COLS = ["hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_sd1_ms", "hrv_sd2_ms"]
# HRV-признаки, для которых считаем delta (безразмерные)
_HRV_DELTA_COLS = ["hrv_dfa_alpha1"]
# NIRS-признаки, для которых считаем delta (уже в %)
_NIRS_DELTA_COLS = ["trainred_smo2_mean", "trainred_hhb_mean", "trainred_hbdiff_mean", "trainred_thb_mean"]


def add_stage0_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признаки, нормированные относительно первой ступени (stage_index == 0).

    Для каждого участника:
      baseline[col] = mean(col | stage_index == 0)
      pct_change[col] = (col - baseline) / baseline * 100   [для мс-признаков]
      delta[col]      = col - baseline                       [для безразмерных/NIRS]

    Физиологический смысл:
      pct_change снимает межсубъектный конфаунд уровня тренированности,
      сохраняя индивидуальную динамику ЧСС/HRV в ответ на нагрузку.
    """
    df = df.copy()

    # Базовые значения stage 0 для каждого участника
    stage0 = df[df["stage_index"] == 0].groupby("subject_id")

    # mс-признаки HRV → pct_change и delta
    for col in _HRV_MS_COLS:
        if col not in df.columns:
            continue
        s0_mean = stage0[col].mean().rename(f"_s0_{col}")
        df = df.join(s0_mean, on="subject_id")
        df[f"{col}_pct_change"] = (df[col] - df[f"_s0_{col}"]) / (df[f"_s0_{col}"].abs() + 1e-8) * 100
        df[f"{col}_delta"] = df[col] - df[f"_s0_{col}"]
        df = df.drop(columns=[f"_s0_{col}"])

    # Безразмерные HRV → delta
    for col in _HRV_DELTA_COLS:
        if col not in df.columns:
            continue
        s0_mean = stage0[col].mean().rename(f"_s0_{col}")
        df = df.join(s0_mean, on="subject_id")
        df[f"{col}_delta"] = df[col] - df[f"_s0_{col}"]
        df = df.drop(columns=[f"_s0_{col}"])

    # NIRS → delta (уже в %, абсолютная разница осмысленна)
    for col in _NIRS_DELTA_COLS:
        if col not in df.columns:
            continue
        s0_mean = stage0[col].mean().rename(f"_s0_{col}")
        df = df.join(s0_mean, on="subject_id")
        df[f"{col}_delta"] = df[col] - df[f"_s0_{col}"]
        df = df.drop(columns=[f"_s0_{col}"])

    # Скорость роста ЧСС: (rr_pct_change) / (stage_index + 1)
    # Отвечает на вопрос: насколько быстро падает RR за каждую ступень?
    # Осмысленно только со 2-й ступени (stage_index >= 1)
    if "hrv_mean_rr_ms_pct_change" in df.columns:
        df["rr_slope_per_stage"] = df["hrv_mean_rr_ms_pct_change"] / (df["stage_index"] + 1)

    return df


def add_session_z_emg(df: pd.DataFrame) -> pd.DataFrame:
    """Session-z нормировка ЭМГ-признаков (per-subject)."""
    df = df.copy()
    for col in [c for c in df.columns if c.startswith(EMG_RAW_PREFIX)]:
        m = df.groupby("subject_id")[col].transform("mean")
        s = df.groupby("subject_id")[col].transform("std")
        df[f"z_{col}"] = (df[col] - m) / (s + 1e-8)
    return df


def add_interaction_features_ref(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction-признаки как в v0004 (для контрольного варианта ref)."""
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    df["feat_smo2_x_rr"] = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
    df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    df["feat_rr_per_watt"] = df["hrv_mean_rr_ms"] / pw
    return df


def add_interaction_features_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction-признаки на нормированных признаках (варианты A-D)."""
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    # Используем pct_change вместо абсолютного RR
    if "hrv_mean_rr_ms_pct_change" in df.columns:
        df["feat_smo2_x_rr_pct"] = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms_pct_change"] / 1e2
        df["feat_rr_pct_per_watt"] = df["hrv_mean_rr_ms_pct_change"] / pw
    if "hrv_dfa_alpha1" in df.columns:
        df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    return df


# ─── LOSO CV ──────────────────────────────────────────────────────────────────

def loso(df: pd.DataFrame, features: list[str], target: str, model_factory=None) -> dict:
    """LOSO CV с LOSO-усреднёнными коэффициентами."""
    if model_factory is None:
        model_factory = lambda: ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=5000, random_state=42)

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
        model = model_factory()
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
    per_subj = {s: float(mean_absolute_error(y_true_all[subj_all == s],
                                              y_pred_all[subj_all == s])) / 60.0
                for s in subjects}
    mae_std = float(np.std(list(per_subj.values())))

    return {
        "mae_min": mae_min, "mae_std": mae_std, "r2": r2, "rho": rho,
        "per_subj_mae_min": per_subj,
        "y_pred": y_pred_all, "y_true": y_true_all, "subjects": subj_all,
        "features_used": feat_cols,
        "coef_mean": np.mean(coefs_list, axis=0) if coefs_list else None,
        "coef_std": np.std(coefs_list, axis=0) if coefs_list else None,
    }


def baseline_mae(df: pd.DataFrame, target: str) -> float:
    """Наивный baseline: предсказываем среднее по train."""
    subjects = sorted(df["subject_id"].unique())
    errors = []
    for s in subjects:
        y_tr = df[df["subject_id"] != s][target].values
        y_te = df[df["subject_id"] == s][target].values
        errors.append(float(mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))))
    return float(np.mean(errors)) / 60.0


def print_feature_importance(result: dict, top_n: int = 15) -> None:
    """Печатает таблицу LOSO-усреднённых коэффициентов."""
    coef_mean = result.get("coef_mean")
    coef_std = result.get("coef_std")
    feats = result.get("features_used", [])
    if coef_mean is None or len(feats) == 0:
        return
    order = np.argsort(np.abs(coef_mean))[::-1][:top_n]
    print(f"\n  {'Признак':<45s} {'Коэф (μ±σ)':<22s} {'Направление'}")
    print("  " + "─" * 85)
    for i in order:
        name = feats[i]
        mu = coef_mean[i]
        sd = coef_std[i] if coef_std is not None else 0.0
        sign = "→ позже" if mu > 0 else "→ раньше"
        print(f"  {name:<45s}  {mu:+.1f} ± {sd:.1f}     {sign}")


def print_results_table(results: dict, b_mae: float, v004_mae: float | None = None) -> None:
    """Сводная таблица всех конфигураций."""
    print(f"\n  {'Конфигурация':<45s} {'MAE±std':>12s}  {'R²':>6s}  {'ρ':>6s}  {'vs v004':>8s}")
    print("  " + "─" * 90)
    for name, r in sorted(results.items(), key=lambda x: x[1]["mae_min"]):
        delta = ""
        if v004_mae is not None:
            d = r["mae_min"] - v004_mae
            delta = f"  {d:+.3f}"
        mark = " ★" if name == min(results, key=lambda k: results[k]["mae_min"]) else ""
        print(f"  {name:<45s}  {r['mae_min']:.3f}±{r['mae_std']:.3f}{delta}  {r['r2']:.3f}  {r['rho']:.3f}{mark}")
    print(f"  {'[Baseline predict-mean]':<45s}  {b_mae:.3f}")
    if v004_mae is not None:
        print(f"  {'[v004 best (Ridge, hrv_mean_rr_ms)]':<45s}  {v004_mae:.3f}")


# ─── Визуализация ─────────────────────────────────────────────────────────────

def plot_results(result: dict, df: pd.DataFrame, target: str, title: str, out_dir: Path) -> None:
    """Scatter + MAE по участникам + траектории."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(df["subject_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}

    y_true = result["y_true"] / 60.0
    y_pred = result["y_pred"] / 60.0
    subj_labels = result["subjects"]
    mae = result["mae_min"]
    r2 = result["r2"]
    rho = result["rho"]
    per = result["per_subj_mae_min"]

    # Scatter + MAE bar
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax = axes[0]
    for s in subjects:
        m = subj_labels == s
        ax.scatter(y_true[m], y_pred[m], color=subj_color[s], alpha=0.3, s=8, label=s)
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Истинное, мин"); ax.set_ylabel("Предсказанное, мин")
    ax.set_title(f"MAE={mae:.3f} | R²={r2:.3f} | ρ={rho:.3f}")
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    subs_sorted = sorted(per.keys(), key=lambda s: per[s])
    vals = [per[s] for s in subs_sorted]
    bars = ax2.barh(subs_sorted, vals, color=[subj_color[s] for s in subs_sorted], alpha=0.85)
    ax2.axvline(mae, color="black", lw=1.5, ls="--", label=f"Среднее {mae:.2f}")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=8)
    ax2.set_xlabel("MAE, мин"); ax2.set_title("MAE по участникам (LOSO)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    p = out_dir / "predicted_vs_actual.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {p.name}")

    # Траектории
    ncols = 3
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle(f"Траектории: {title}", fontsize=11, fontweight="bold")
    for i, s in enumerate(subjects):
        ax = axes_flat[i]
        sub = df[df["subject_id"] == s].sort_values("elapsed_sec")
        t = sub["elapsed_sec"].values / 60.0
        y_true_s = sub[target].values / 60.0
        y_pred_s = y_pred[subj_labels == s]
        ax.plot(t, y_true_s, "k-", lw=1.8, label="Истина")
        ax.plot(t, y_pred_s, color=subj_color[s], lw=1.5, ls="--", label="Предсказание")
        ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.8)
        ax.set_title(f"{s} | MAE={per[s]:.2f} мин", fontsize=9)
        ax.set_xlabel("Прошло, мин", fontsize=8); ax.set_ylabel("До порога, мин", fontsize=8)
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    p = out_dir / "trajectories.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {p.name}")


# ─── LT2 ─────────────────────────────────────────────────────────────────────

def run_lt2(df_full: pd.DataFrame, no_plots: bool = False) -> dict:
    """LT2 с разными вариантами нормировки HRV."""
    print("\n" + "═" * 65)
    print("LT2 — stage-0 normalization эксперимент")
    print("═" * 65)

    df = df_full[df_full["window_valid_all_required"] == 1].copy()
    df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)

    df = add_stage0_baselines(df)
    df = add_session_z_emg(df)
    df = add_interaction_features_ref(df)
    df = add_interaction_features_norm(df)

    target = "target_time_to_lt2_center_sec"
    z_emg = [c for c in Z_EMG_KEY if c in df.columns]

    # Признаки нормированного HRV
    hrv_pct = [f"{c}_pct_change" for c in _HRV_MS_COLS if f"{c}_pct_change" in df.columns]
    hrv_delta = [f"{c}_delta" for c in _HRV_MS_COLS if f"{c}_delta" in df.columns]
    dfa_delta = ["hrv_dfa_alpha1_delta"] if "hrv_dfa_alpha1_delta" in df.columns else []
    nirs_delta = [f"{c}_delta" for c in _NIRS_DELTA_COLS if f"{c}_delta" in df.columns]
    inter_norm = [c for c in ["feat_smo2_x_rr_pct", "feat_rr_pct_per_watt", "feat_smo2_x_dfa"] if c in df.columns]
    slope = ["rr_slope_per_stage"] if "rr_slope_per_stage" in df.columns else []

    # Конфигурации для сравнения
    # ref: v0004 best (с hrv_mean_rr_ms)
    inter_ref = [c for c in INTER_KEY_REF if c in df.columns]
    configs = {
        "ref_v004 (hrv_mean_rr_ms, как в v004)": NIRS_FEATURES + HRV_FEATURES + inter_ref + z_emg,
        "A: rr_pct+delta вместо rr_ms":
            NIRS_FEATURES + HRV_SCALEFREE + hrv_pct[:1] + hrv_delta[:1] +
            ["hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_sd1_ms", "hrv_sd2_ms"] + inter_norm + z_emg,
        "B: все HRV→pct + smo2_delta":
            NIRS_FEATURES + HRV_SCALEFREE + hrv_pct + dfa_delta + nirs_delta + inter_norm + z_emg,
        "B2: все HRV→pct+delta + smo2_delta":
            NIRS_FEATURES + HRV_SCALEFREE + hrv_pct + hrv_delta + dfa_delta + nirs_delta + inter_norm + z_emg,
        "C: B + slope":
            NIRS_FEATURES + HRV_SCALEFREE + hrv_pct + dfa_delta + nirs_delta + inter_norm + z_emg + slope,
        "D: A + slope":
            NIRS_FEATURES + HRV_SCALEFREE + hrv_pct[:1] + hrv_delta[:1] +
            ["hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_sd1_ms", "hrv_sd2_ms"] +
            inter_norm + z_emg + slope,
        "E: только pct+scale-free (нет абс. NIRS)":
            [c for c in NIRS_FEATURES if "mean" not in c or "slope" in c or "std" in c or "dt" in c] +
            ["trainred_smo2_drop", "trainred_dsmo2_dt", "trainred_hhb_slope",
             "trainred_dhhb_dt", "trainred_hbdiff_slope"] +
            nirs_delta + HRV_SCALEFREE + hrv_pct + hrv_delta + dfa_delta + inter_norm + z_emg + slope,
    }

    # v0004 reference MAE
    v004_mae = 2.097

    b_mae = baseline_mae(df, target)
    print(f"Baseline (predict-mean): {b_mae:.3f} мин")
    print(f"Участников: {df['subject_id'].nunique()}, окон: {len(df)}")

    ridge_factory = lambda: Ridge(alpha=100.0, fit_intercept=True, solver="auto")

    results = {}
    for name, feats in configs.items():
        avail = sorted(set(f for f in feats if f in df.columns))
        if len(avail) < 2:
            print(f"  {name}: пропуск (мало признаков)")
            continue
        r = loso(df, avail, target, model_factory=ridge_factory)
        results[name] = r

    print_results_table(results, b_mae, v004_mae)

    best_name = min(results, key=lambda k: results[k]["mae_min"])
    best = results[best_name]
    print(f"\n★ Лучшая: {best_name}")
    print(f"  MAE  = {best['mae_min']:.3f} мин")
    print(f"  R²   = {best['r2']:.3f}")
    print(f"  ρ    = {best['rho']:.3f}")
    print("  По участникам:")
    for s, m in sorted(best["per_subj_mae_min"].items()):
        print(f"    {s}: {m:.2f} мин")
    print_feature_importance(best)

    if not no_plots:
        plot_results(best, df, target,
                     f"LT2: {best_name}\nRidge(α=100), stage-0 norm",
                     OUT_DIR / "lt2")

    # Сохраняем сравнительную таблицу
    rows = []
    for name, r in results.items():
        for s, m in r["per_subj_mae_min"].items():
            rows.append({"config": name, "subject_id": s, "mae_min": m})
    pd.DataFrame(rows).to_csv(OUT_DIR / "lt2_variant_comparison.csv", index=False)

    return {"best_name": best_name, "results": results, "df": df}


# ─── LT1 ─────────────────────────────────────────────────────────────────────

def run_lt1(df_full: pd.DataFrame, no_plots: bool = False) -> dict:
    """LT1 с разными вариантами нормировки HRV."""
    print("\n" + "═" * 65)
    print("LT1 — stage-0 normalization эксперимент")
    print("═" * 65)

    df = df_full[df_full["target_time_to_lt1_usable"] == 1].copy()
    df = df[df["nirs_valid"] == 1].copy()
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)

    df = add_stage0_baselines(df)
    df = add_session_z_emg(df)
    df = add_interaction_features_norm(df)

    target = "target_time_to_lt1_sec"
    z_emg = [c for c in Z_EMG_KEY if c in df.columns]

    hrv_pct = [f"{c}_pct_change" for c in _HRV_MS_COLS if f"{c}_pct_change" in df.columns]
    hrv_delta = [f"{c}_delta" for c in _HRV_MS_COLS if f"{c}_delta" in df.columns]
    dfa_delta = ["hrv_dfa_alpha1_delta"] if "hrv_dfa_alpha1_delta" in df.columns else []
    nirs_delta = [f"{c}_delta" for c in _NIRS_DELTA_COLS if f"{c}_delta" in df.columns]
    inter_norm = [c for c in ["feat_smo2_x_rr_pct", "feat_rr_pct_per_watt", "feat_smo2_x_dfa"] if c in df.columns]
    slope = ["rr_slope_per_stage"] if "rr_slope_per_stage" in df.columns else []

    configs = {
        "ref_v004 (hrv_mean_rr_ms, как в v004)": z_emg + HRV_FEATURES,
        "A: rr_pct+delta вместо rr_ms":
            z_emg + HRV_SCALEFREE + hrv_pct[:1] + hrv_delta[:1] +
            ["hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_sd1_ms", "hrv_sd2_ms"],
        "B: все HRV→pct + smo2_delta":
            z_emg + HRV_SCALEFREE + hrv_pct + dfa_delta + nirs_delta + inter_norm,
        "B2: все HRV→pct+delta":
            z_emg + HRV_SCALEFREE + hrv_pct + hrv_delta + dfa_delta + nirs_delta,
        "C: B + slope":
            z_emg + HRV_SCALEFREE + hrv_pct + dfa_delta + nirs_delta + inter_norm + slope,
        "D: A + slope":
            z_emg + HRV_SCALEFREE + hrv_pct[:1] + hrv_delta[:1] +
            ["hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_sd1_ms", "hrv_sd2_ms"] + slope,
        "E: pct+scale-free+z_EMG+slope":
            z_emg + HRV_SCALEFREE + hrv_pct + hrv_delta + dfa_delta + inter_norm + slope,
    }

    v004_mae = 2.021

    b_mae = baseline_mae(df, target)
    print(f"Baseline (predict-mean): {b_mae:.3f} мин")
    print(f"Участников: {df['subject_id'].nunique()}, окон: {len(df)}")

    huber_factory = lambda: HuberRegressor(
        epsilon=1.35, alpha=0.0001, max_iter=500, fit_intercept=True, warm_start=False,
    )

    results = {}
    for name, feats in configs.items():
        avail = sorted(set(f for f in feats if f in df.columns))
        if len(avail) < 2:
            print(f"  {name}: пропуск (мало признаков)")
            continue
        r = loso(df, avail, target, model_factory=huber_factory)
        results[name] = r

    print_results_table(results, b_mae, v004_mae)

    best_name = min(results, key=lambda k: results[k]["mae_min"])
    best = results[best_name]
    print(f"\n★ Лучшая: {best_name}")
    print(f"  MAE  = {best['mae_min']:.3f} мин")
    print(f"  R²   = {best['r2']:.3f}")
    print(f"  ρ    = {best['rho']:.3f}")
    print("  По участникам:")
    for s, m in sorted(best["per_subj_mae_min"].items()):
        print(f"    {s}: {m:.2f} мин")
    print_feature_importance(best)

    if not no_plots:
        plot_results(best, df, target,
                     f"LT1: {best_name}\nHuber(ε=1.35), stage-0 norm",
                     OUT_DIR / "lt1")

    rows = []
    for name, r in results.items():
        for s, m in r["per_subj_mae_min"].items():
            rows.append({"config": name, "subject_id": s, "mae_min": m})
    pd.DataFrame(rows).to_csv(OUT_DIR / "lt1_variant_comparison.csv", index=False)

    return {"best_name": best_name, "results": results, "df": df}


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0006: нормировка относительно stage 0")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("v0006 — Stage-0 normalization эксперимент")
    print("=" * 65)
    print(f"Датасет: {args.dataset}")

    df_full = pd.read_parquet(args.dataset)
    print(f"Всего окон: {len(df_full)}, участников: {df_full['subject_id'].nunique()}")

    lt2_out = lt1_out = None

    if args.target in ("lt2", "both"):
        lt2_out = run_lt2(df_full, no_plots=args.no_plots)

    if args.target in ("lt1", "both"):
        lt1_out = run_lt1(df_full, no_plots=args.no_plots)

    print("\n" + "=" * 65)
    print("ИТОГ v0006")
    print("=" * 65)
    if lt2_out:
        best = lt2_out["results"][lt2_out["best_name"]]
        print(f"LT2 лучшая: {lt2_out['best_name']}")
        print(f"     MAE={best['mae_min']:.3f} мин  R²={best['r2']:.3f}  ρ={best['rho']:.3f}")
        print(f"     vs v004: {best['mae_min'] - 2.097:+.3f} мин")
    if lt1_out:
        best = lt1_out["results"][lt1_out["best_name"]]
        print(f"LT1 лучшая: {lt1_out['best_name']}")
        print(f"     MAE={best['mae_min']:.3f} мин  R²={best['r2']:.3f}  ρ={best['rho']:.3f}")
        print(f"     vs v004: {best['mae_min'] - 2.021:+.3f} мин")


if __name__ == "__main__":
    main()
