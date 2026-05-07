"""v0005 — SHAP + Conformal Prediction (Ridge LT2, Huber LT1)

Версия:    v0005
Дата:      2026-05-07
Предыдущая версия: v0004_train_ridge_huber.py
Результаты: results/v0005/

Что делает:
  SHAP-анализ важности признаков и conformal prediction (доверительные интервалы).
  Использует те же модели, что v0004: Ridge(LT2), HuberRegressor(LT1).

Гиперпараметры:
  Модель LT2: Ridge(alpha=100.0, fit_intercept=True, solver="auto")  [детерминирован]
  Модель LT1: HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500)  [детерминирован]
  SHAP:       LinearExplainer, background = train mean (стандартизованное пространство)
              Обучение глобальной модели — на всех данных (не LOSO)
  Conformal:  cross-conformal LOSO, nonconformity = |y_true - y_pred|
              Квантиль: ceil((n_cal+1)*(1-alpha)) / n_cal, clamped to [0,1]
              alpha уровни: 0.05, 0.10, 0.20

Ожидаемые результаты (conformal, alpha=0.10):
  LT2 (n=13): покрытие 89.3%, медианная ширина 9.39 мин (±4.70 мин)
  LT1 (n=9):  покрытие 89.7%, медианная ширина 9.04 мин (±4.52 мин)

Воспроизведение:
  uv run python scripts/v0005_train_shap_conformal_ridge_huber.py
  uv run python scripts/v0005_train_shap_conformal_ridge_huber.py --target lt2
  uv run python scripts/v0005_train_shap_conformal_ridge_huber.py --no-plots

SHAP (SHapley Additive exPlanations):
  Для линейной модели SHAP-значение признака j для наблюдения i:
    phi_j(i) = coef_j * (x_j(i) - E[x_j])
  Знак: положительный -> признак толкает предсказание вверх (дальше до порога).
  Модели:
    LT2 -> Ridge(alpha=100)         -- стабильнее на 13 фолдах
    LT1 -> HuberRegressor(eps=1.35) -- устойчива к S003 (элитный атлет, выброс)

Conformal Prediction (cross-conformal, LOSO-версия):
  Для каждого тестового субъекта s:
    - для каждого тренировочного субъекта s' строим модель на {все} без {s, s'}
    - предсказываем s' → получаем |ошибку| (nonconformity score)
    - n-1 таких ошибок → калибровочный квантиль Q(α)
  Интервал предсказания: [ŷ − Q(α), ŷ + Q(α)]
  Гарантия: при n→∞ покрытие = 1−α (здесь n=12 или 8 — небольшое, но оценка честная).

Запуск:
  uv run python scripts/train_shap_conformal.py
  uv run python scripts/train_shap_conformal.py --target lt2
  uv run python scripts/train_shap_conformal.py --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR

OUT_DIR = _ROOT / "results" / "v0005"

# ─── Признаки ─────────────────────────────────────────────────────────────────

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
Z_EMG_KEY = [
    "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
    "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
    "z_vl_prox_rest_rms",
    "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
]
INTER_KEY = ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"]


# ─── Feature engineering ──────────────────────────────────────────────────────

def prepare_lt2(df_full: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Готовит датафрейм и список признаков для LT2."""
    df = df_full[df_full["window_valid_all_required"] == 1].copy()
    df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    df = _add_session_z_emg(df)
    df = _add_interactions(df)
    feats = NIRS_FEATURES + HRV_FEATURES + INTER_KEY + Z_EMG_KEY
    feats = [f for f in feats if f in df.columns]
    return df, feats


def prepare_lt1(df_full: pd.DataFrame, session_params: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Готовит датафрейм и список признаков для LT1."""
    df = df_full[df_full["target_time_to_lt1_usable"] == 1].copy()
    df = df[df["nirs_valid"] == 1].copy()
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    df = _add_session_z_emg(df)
    if not session_params.empty:
        df = _add_running_nirs(df, session_params)
    feats = Z_EMG_KEY + HRV_FEATURES
    feats = [f for f in feats if f in df.columns]
    return df, feats


def _add_session_z_emg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [c for c in df.columns if c.startswith("vl_")]:
        m = df.groupby("subject_id")[col].transform("mean")
        s = df.groupby("subject_id")[col].transform("std")
        df[f"z_{col}"] = (df[col] - m) / (s + 1e-8)
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    df["feat_smo2_x_rr"] = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
    df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    df["feat_rr_per_watt"] = df["hrv_mean_rr_ms"] / pw
    return df


def _add_running_nirs(df: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sp_idx = sp.set_index("subject_id")
    for col in ["smo2_from_running_max", "hhb_from_running_min", "smo2_rel_drop_pct"]:
        df[col] = np.nan
    for subj, g in df.groupby("subject_id"):
        g = g.sort_values("window_center_sec")
        idx = g.index
        smo2 = g["trainred_smo2_mean"].values
        hhb = g["trainred_hhb_mean"].values
        rmax = np.maximum.accumulate(np.where(np.isfinite(smo2), smo2, -np.inf))
        rmin = np.minimum.accumulate(np.where(np.isfinite(hhb), hhb, np.inf))
        rmax = np.where(np.isinf(rmax), np.nan, rmax)
        rmin = np.where(np.isinf(rmin), np.nan, rmin)
        df.loc[idx, "smo2_from_running_max"] = rmax - smo2
        df.loc[idx, "hhb_from_running_min"] = hhb - rmin
        if subj in sp_idx.index:
            b = float(sp_idx.loc[subj, "nirs_smo2_baseline_mean"])
            if np.isfinite(b) and b > 0:
                df.loc[idx, "smo2_rel_drop_pct"] = (b - smo2) / b * 100.0
    return df


# ─── LOSO ─────────────────────────────────────────────────────────────────────

def loso(df, features, target, model_factory=None):
    """Стандартный LOSO, возвращает предсказания и список обученных моделей."""
    if model_factory is None:
        model_factory = lambda: ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=5000,
                                           random_state=42)

    subjects = sorted(df["subject_id"].unique())
    feat_cols = [f for f in features if f in df.columns]
    preds, trues, subjs, models_list = [], [], [], []

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test = df[df["subject_id"] == test_s].sort_values("window_start_sec")
        X_tr = train[feat_cols].values
        y_tr = train[target].values
        X_te = test[feat_cols].values
        y_te = test[target].values

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        model = model_factory()
        X_tr_s = sc.fit_transform(imp.fit_transform(X_tr))
        X_te_s = sc.transform(imp.transform(X_te))
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        preds.append(y_pred)
        trues.append(y_te)
        subjs.append(np.full(len(y_te), test_s))
        models_list.append({"model": model, "imp": imp, "sc": sc, "test_s": test_s})

    y_pred_all = np.concatenate(preds)
    y_true_all = np.concatenate(trues)
    subj_all = np.concatenate(subjs)
    mae_min = mean_absolute_error(y_true_all, y_pred_all) / 60.0
    r2 = r2_score(y_true_all, y_pred_all)
    rho = float(spearmanr(y_true_all, y_pred_all).statistic)
    per_subj = {s: mean_absolute_error(y_true_all[subj_all == s],
                                       y_pred_all[subj_all == s]) / 60.0
                for s in subjects}

    return {
        "mae_min": mae_min, "r2": r2, "rho": rho,
        "per_subj_mae_min": per_subj,
        "y_pred": y_pred_all, "y_true": y_true_all,
        "subjects": subj_all, "feat_cols": feat_cols,
        "models": models_list,
    }


# ─── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    task_name: str,
    model_factory=None,
    model_label: str = "ElasticNet",
    no_plots: bool = False,
) -> pd.DataFrame:
    """Вычисляет SHAP-значения: глобальная модель.

    Для линейных моделей (Ridge, ElasticNet, Huber) используем shap.LinearExplainer:
      φ_j(i) = coef_j × (x_j(i) − E[x_j])
    где E[x_j] — среднее по обучающей выборке.

    Args:
        model_factory: callable → sklearn LinearModel с атрибутом .coef_
        model_label: строка для заголовка графика (например "Ridge(α=100)")
    Возвращает Series mean_abs_shap по признакам (для таблицы в диссертации).
    """
    if model_factory is None:
        model_factory = lambda: ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=5000,
                                           random_state=42)

    feat_cols = [f for f in features if f in df.columns]
    out = OUT_DIR / task_name.lower()
    out.mkdir(parents=True, exist_ok=True)

    # ── Глобальная модель (обучение на всех данных) ──────────────────────────
    X_all = df[feat_cols].values
    y_all = df[target].values

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    model_global = model_factory()
    X_imp = imp.fit_transform(X_all)
    X_sc = sc.fit_transform(X_imp)
    model_global.fit(X_sc, y_all)

    # LinearExplainer: background = train mean (уже в стандартизованном пространстве)
    explainer = shap.LinearExplainer(model_global, X_sc, feature_names=feat_cols)
    shap_values = explainer(X_sc)  # shape: (n_samples, n_features)

    sv = shap_values.values  # numpy array

    # Создаём DataFrame с SHAP и мета-информацией
    shap_df = pd.DataFrame(sv, columns=feat_cols)
    shap_df["subject_id"] = df["subject_id"].values
    shap_df["elapsed_sec"] = df["elapsed_sec"].values if "elapsed_sec" in df.columns else np.nan
    shap_df["y_true"] = y_all

    mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=feat_cols).sort_values(ascending=False)

    print(f"\n  Топ-10 признаков по |SHAP| ({task_name}):")
    for feat, val in mean_abs_shap.head(10).items():
        val_min = val / 60.0
        print(f"    {feat:<40s} {val_min:.3f} мин")

    if not no_plots:
        # ── График 1: Bar plot топ-20 признаков ──────────────────────────────
        top20 = mean_abs_shap.head(20)

        def feat_color(name: str) -> str:
            if name.startswith("trainred_"):   return "#2ca02c"
            if name.startswith("hrv_"):        return "#9467bd"
            if name.startswith("z_"):          return "#1f77b4"
            if name.startswith("feat_"):       return "#8c564b"
            if "running" in name or "drop" in name: return "#d62728"
            return "#7f7f7f"

        colors = [feat_color(f) for f in top20.index]

        fig, ax = plt.subplots(figsize=(9, max(5, len(top20) * 0.4)))
        bars = ax.barh(top20.index[::-1], top20.values[::-1] / 60.0,
                       color=colors[::-1], alpha=0.85)
        for bar, val in zip(bars, top20.values[::-1] / 60.0):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

        legend_handles = [
            mpatches.Patch(color="#2ca02c", label="NIRS"),
            mpatches.Patch(color="#9467bd", label="HRV"),
            mpatches.Patch(color="#1f77b4", label="Session-z EMG"),
            mpatches.Patch(color="#8c564b", label="Interaction"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
        ax.set_xlabel("mean |SHAP|, мин")
        ax.set_title(f"{task_name}: Важность признаков (SHAP)\n{model_label}, обучен на всех данных",
                     fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        p = out / f"{task_name.lower()}_shap_importance.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {p.name}")

        # ── График 2: Beeswarm (SHAP summary plot) ───────────────────────────
        top_k = min(15, len(feat_cols))
        top_feats = mean_abs_shap.head(top_k).index.tolist()
        sv_top = sv[:, [feat_cols.index(f) for f in top_feats]]
        X_top = X_sc[:, [feat_cols.index(f) for f in top_feats]]

        fig, ax = plt.subplots(figsize=(10, 6))
        for j in range(top_k - 1, -1, -1):
            vals = sv_top[:, j] / 60.0
            feat_vals = X_top[:, j]
            # нормируем цвет по значению признака
            feat_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-8)
            sc_plot = ax.scatter(
                vals,
                np.full(len(vals), j) + np.random.uniform(-0.3, 0.3, len(vals)),
                c=feat_norm, cmap="coolwarm", alpha=0.4, s=5,
            )
        ax.axvline(0, color="black", lw=1.2)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_feats, fontsize=8)
        ax.set_xlabel("SHAP-значение, мин (+ = дольше до порога)")
        ax.set_title(f"{task_name}: SHAP beeswarm (топ-{top_k}) — {model_label}\n"
                     f"Цвет: синий = низкое значение признака, красный = высокое",
                     fontweight="bold")
        plt.colorbar(sc_plot, ax=ax, label="Значение признака (норм.)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        p = out / f"{task_name.lower()}_shap_beeswarm.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {p.name}")

        # ── График 3: SHAP по участникам (среднее |SHAP| топ-8) ─────────────
        top8 = mean_abs_shap.head(8).index.tolist()
        subjects = sorted(df["subject_id"].unique())
        palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes_flat = axes.flatten()
        fig.suptitle(f"{task_name}: SHAP траектории топ-8 признаков", fontsize=12, fontweight="bold")

        for j, feat in enumerate(top8):
            ax = axes_flat[j]
            feat_idx = feat_cols.index(feat)
            for i, s in enumerate(subjects):
                mask = df["subject_id"].values == s
                t = df.loc[mask, "elapsed_sec"].values / 60.0
                sv_s = sv[mask, feat_idx] / 60.0
                ax.plot(t, sv_s, color=palette[i], lw=1.0, alpha=0.7, label=s)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_title(feat.replace("trainred_", "").replace("hrv_", "").replace("z_vl_dist_load_", "z_")[:30],
                         fontsize=8)
            ax.set_xlabel("мин", fontsize=7)
            ax.set_ylabel("SHAP, мин", fontsize=7)
            ax.grid(alpha=0.2)

        axes_flat[-1].legend([plt.Line2D([0], [0], color=palette[i], lw=1.5)
                               for i in range(len(subjects))],
                              subjects, fontsize=7, ncol=2, loc="center")
        axes_flat[-1].axis("off")
        plt.tight_layout()
        p = out / f"{task_name.lower()}_shap_trajectories.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {p.name}")

    return mean_abs_shap


# ─── Conformal Prediction ─────────────────────────────────────────────────────

def cross_conformal_loso(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    task_name: str,
    model_factory=None,
    alpha_levels: tuple[float, ...] = (0.05, 0.10, 0.20),
    no_plots: bool = False,
) -> pd.DataFrame:
    """Cross-conformal prediction в LOSO-схеме.

    Для каждого тестового субъекта s:
      1. Для каждого обучающего субъекта s' ≠ s:
           - Обучаем на {все} без {s, s'}
           - Предсказываем s' → nonconformity score = |y_true(s') − ŷ(s')|
      2. Набираем n-1 calibration scores (по одному на каждый s' ≠ s)
      3. Квантиль Q(1-α) этих scores = ширина интервала (в секундах)
      4. Предсказываем s с моделью, обученной на {все} без {s}
         Интервал: [ŷ(s) − Q, ŷ(s) + Q]

    Эмпирическое покрытие: доля окон, для которых y_true ∈ [ŷ−Q, ŷ+Q].
    При n→∞ должно быть ≥ 1−α. При n=12-8 — приближённо.

    Возвращает DataFrame с предсказаниями, интервалами и метриками покрытия.
    """
    if model_factory is None:
        model_factory = lambda: ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=5000,
                                           random_state=42)

    feat_cols = [f for f in features if f in df.columns]
    subjects = sorted(df["subject_id"].unique())
    n = len(subjects)

    out = OUT_DIR / task_name.lower()
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for test_s in subjects:
        train_subjects = [s for s in subjects if s != test_s]

        # ── Шаг 1: Calibration scores ────────────────────────────────────────
        cal_scores_sec: list[float] = []
        for cal_s in train_subjects:
            # Обучаем на {все} \ {test_s, cal_s}
            inner_train = df[~df["subject_id"].isin([test_s, cal_s])]
            cal_test = df[df["subject_id"] == cal_s]
            if len(inner_train) < 30 or len(cal_test) == 0:
                continue

            imp = SimpleImputer(strategy="median")
            sc = StandardScaler()
            model = model_factory()
            X_tr = sc.fit_transform(imp.fit_transform(inner_train[feat_cols].values))
            model.fit(X_tr, inner_train[target].values)
            X_cal = sc.transform(imp.transform(cal_test[feat_cols].values))
            y_pred_cal = model.predict(X_cal)
            scores = np.abs(cal_test[target].values - y_pred_cal)
            cal_scores_sec.extend(scores.tolist())

        cal_scores_sec = np.array(cal_scores_sec)

        # ── Шаг 2: Основная модель для test_s ────────────────────────────────
        train = df[df["subject_id"] != test_s]
        test = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        model = model_factory()
        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        model.fit(X_tr, train[target].values)
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        y_pred = model.predict(X_te)
        y_true = test[target].values

        # ── Шаг 3: Интервалы для каждого α ───────────────────────────────────
        for alpha in alpha_levels:
            # Квантиль уровня (1-α) × (n_cal+1) / n_cal (поправка на конечную выборку)
            q_level = np.ceil((len(cal_scores_sec) + 1) * (1 - alpha)) / len(cal_scores_sec)
            q_level = min(q_level, 1.0)
            Q = float(np.quantile(cal_scores_sec, q_level))

            covered = np.abs(y_true - y_pred) <= Q
            coverage = float(covered.mean())
            interval_width_min = 2 * Q / 60.0

            for i, (yt, yp, cov) in enumerate(zip(y_true, y_pred, covered)):
                all_rows.append({
                    "subject_id": test_s,
                    "window_idx": i,
                    "y_true_sec": yt,
                    "y_pred_sec": yp,
                    "y_true_min": yt / 60.0,
                    "y_pred_min": yp / 60.0,
                    "alpha": alpha,
                    "Q_sec": Q,
                    "Q_min": Q / 60.0,
                    "lower_min": (yp - Q) / 60.0,
                    "upper_min": (yp + Q) / 60.0,
                    "covered": int(cov),
                    "coverage_subj": coverage,
                    "interval_width_min": interval_width_min,
                })

    result_df = pd.DataFrame(all_rows)

    # ── Сводка покрытия ───────────────────────────────────────────────────────
    print(f"\n  Conformal Prediction — {task_name} (n={n} участников):")
    print(f"  {'α':>6}  {'Цел. покрытие':>14}  {'Факт. покрытие':>14}  {'Ширина интервала, мин':>22}")
    for alpha in alpha_levels:
        sub = result_df[result_df["alpha"] == alpha]
        actual_cov = sub["covered"].mean()
        median_width = sub["interval_width_min"].median()
        print(f"  {alpha:>6.2f}  {1-alpha:>14.0%}  {actual_cov:>14.1%}  {median_width:>22.2f}")

    if not no_plots:
        _plot_conformal(result_df, df, target, alpha_levels, task_name, out)

    return result_df


def _plot_conformal(
    result_df: pd.DataFrame,
    df: pd.DataFrame,
    target: str,
    alpha_levels: tuple,
    task_name: str,
    out: Path,
) -> None:
    """Визуализации conformal prediction."""
    subjects = sorted(df["subject_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}

    # ── График 1: Предсказания с доверительными полосами (α=0.10) ───────────
    alpha_main = 0.10
    sub = result_df[result_df["alpha"] == alpha_main]

    n_subj = len(subjects)
    ncols = 3
    nrows = int(np.ceil(n_subj / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    actual_cov = sub["covered"].mean()
    fig.suptitle(
        f"{task_name}: Conformal интервалы (α={alpha_main}, цель ≥{1-alpha_main:.0%}, факт {actual_cov:.1%})",
        fontsize=11, fontweight="bold",
    )

    for i, s in enumerate(subjects):
        ax = axes_flat[i]
        s_data = sub[sub["subject_id"] == s].sort_values("window_idx")
        subj_df = df[df["subject_id"] == s].sort_values("window_start_sec")
        t = subj_df["elapsed_sec"].values / 60.0

        ax.plot(t, s_data["y_true_min"].values, "k-", lw=1.8, label="Истина")
        ax.plot(t, s_data["y_pred_min"].values, color=subj_color[s], lw=1.4,
                ls="--", label="Предсказание")
        ax.fill_between(
            t,
            s_data["lower_min"].values,
            s_data["upper_min"].values,
            color=subj_color[s], alpha=0.2, label=f"90% интервал",
        )
        ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.8)
        cov_s = s_data["covered"].mean()
        width_s = s_data["interval_width_min"].median()
        ax.set_title(f"{s} | покрытие={cov_s:.0%} | ±{width_s/2:.1f}мин", fontsize=9)
        ax.set_xlabel("Прошло, мин", fontsize=8)
        ax.set_ylabel("До порога, мин", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    p = out / f"{task_name.lower()}_conformal_trajectories.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

    # ── График 2: Покрытие vs α (calibration plot) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    ax = axes[0]
    alphas = sorted(result_df["alpha"].unique())
    target_covs = [1 - a for a in alphas]
    actual_covs = [result_df[result_df["alpha"] == a]["covered"].mean() for a in alphas]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Идеальная калибровка")
    ax.plot(target_covs, actual_covs, "o-", color="#1f77b4", lw=2, ms=8, label=task_name)
    for t_cov, a_cov, a in zip(target_covs, actual_covs, alphas):
        ax.annotate(f"α={a:.2f}", (t_cov, a_cov), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Целевое покрытие (1−α)")
    ax.set_ylabel("Фактическое покрытие")
    ax.set_title(f"Calibration curve — {task_name}", fontweight="bold")
    ax.set_xlim(0.7, 1.02); ax.set_ylim(0.7, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Ширина интервала vs α
    ax = axes[1]
    widths = [result_df[result_df["alpha"] == a]["interval_width_min"].median() for a in alphas]
    ax.plot([1 - a for a in alphas], widths, "s-", color="#ff7f0e", lw=2, ms=8)
    for t_cov, w in zip(target_covs, widths):
        ax.annotate(f"{w:.1f} мин", (t_cov, w), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Целевое покрытие (1−α)")
    ax.set_ylabel("Медианная ширина интервала, мин")
    ax.set_title(f"Ширина интервала vs покрытие — {task_name}", fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out / f"{task_name.lower()}_conformal_calibration.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

    # ── График 3: Scatter с цветовым кодированием covered/not ────────────────
    alpha_main2 = 0.10
    sub2 = result_df[result_df["alpha"] == alpha_main2]
    covered_mask = sub2["covered"].astype(bool).values
    y_true = sub2["y_true_min"].values
    y_pred = sub2["y_pred_min"].values

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true[covered_mask], y_pred[covered_mask],
               color="#2ca02c", alpha=0.3, s=7, label=f"Покрыт ({covered_mask.mean():.0%})")
    ax.scatter(y_true[~covered_mask], y_pred[~covered_mask],
               color="#d62728", alpha=0.6, s=10, label="Не покрыт")
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Истинное, мин")
    ax.set_ylabel("Предсказанное, мин")
    ax.set_title(f"{task_name}: Covered vs not-covered (α={alpha_main2})\n"
                 f"Цель: {1-alpha_main2:.0%}, факт: {covered_mask.mean():.1%}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out / f"{task_name.lower()}_conformal_scatter.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Аргументы командной строки."""
    p = argparse.ArgumentParser(description="SHAP + Conformal Prediction для LT1 и LT2.")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--no-conformal", action="store_true")
    p.add_argument(
        "--dataset", type=Path,
        default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet",
    )
    return p.parse_args()


def main() -> None:
    """Точка входа."""
    args = parse_args()
    print("=" * 65)
    print("SHAP + CONFORMAL PREDICTION")
    print("=" * 65)

    df_full = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    # Фабрики лучших моделей
    ridge_factory = lambda: Ridge(alpha=100.0)
    huber_factory = lambda: HuberRegressor(epsilon=1.35, max_iter=500)

    # ── LT2 ──────────────────────────────────────────────────────────────────
    if args.target in ("lt2", "both"):
        print("\n" + "═" * 65)
        print("LT2  [Ridge(α=100)]")
        print("═" * 65)
        df_lt2, feats_lt2 = prepare_lt2(df_full)
        print(f"Окон: {len(df_lt2)}, участников: {df_lt2['subject_id'].nunique()}, признаков: {len(feats_lt2)}")

        # LOSO для проверки
        r_lt2 = loso(df_lt2, feats_lt2, "target_time_to_lt2_center_sec",
                     model_factory=ridge_factory)
        print(f"LOSO MAE = {r_lt2['mae_min']:.3f} мин | R² = {r_lt2['r2']:.3f} | ρ = {r_lt2['rho']:.3f}")

        if not args.no_shap:
            print("\n[SHAP] LT2...")
            shap_lt2 = compute_shap(
                df_lt2, feats_lt2, "target_time_to_lt2_center_sec", "LT2",
                model_factory=ridge_factory, model_label="Ridge(α=100)",
                no_plots=args.no_plots,
            )
            results["lt2_shap"] = shap_lt2

        if not args.no_conformal:
            print("\n[Conformal] LT2...")
            conf_lt2 = cross_conformal_loso(
                df_lt2, feats_lt2, "target_time_to_lt2_center_sec", "LT2",
                model_factory=ridge_factory, no_plots=args.no_plots,
            )
            conf_lt2.to_csv(OUT_DIR / "lt2" / "lt2_conformal_predictions.csv", index=False)
            results["lt2_conformal"] = conf_lt2

    # ── LT1 ──────────────────────────────────────────────────────────────────
    if args.target in ("lt1", "both"):
        print("\n" + "═" * 65)
        print("LT1  [HuberRegressor(ε=1.35)]")
        print("═" * 65)
        df_lt1, feats_lt1 = prepare_lt1(df_full, session_params)
        print(f"Окон: {len(df_lt1)}, участников: {df_lt1['subject_id'].nunique()}, признаков: {len(feats_lt1)}")

        r_lt1 = loso(df_lt1, feats_lt1, "target_time_to_lt1_sec",
                     model_factory=huber_factory)
        print(f"LOSO MAE = {r_lt1['mae_min']:.3f} мин | R² = {r_lt1['r2']:.3f} | ρ = {r_lt1['rho']:.3f}")

        if not args.no_shap:
            print("\n[SHAP] LT1...")
            shap_lt1 = compute_shap(
                df_lt1, feats_lt1, "target_time_to_lt1_sec", "LT1",
                model_factory=huber_factory, model_label="HuberRegressor(ε=1.35)",
                no_plots=args.no_plots,
            )
            results["lt1_shap"] = shap_lt1

        if not args.no_conformal:
            print("\n[Conformal] LT1...")
            conf_lt1 = cross_conformal_loso(
                df_lt1, feats_lt1, "target_time_to_lt1_sec", "LT1",
                model_factory=huber_factory, no_plots=args.no_plots,
            )
            conf_lt1.to_csv(OUT_DIR / "lt1" / "lt1_conformal_predictions.csv", index=False)
            results["lt1_conformal"] = conf_lt1

    print("\n✅ Готово. Результаты:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
