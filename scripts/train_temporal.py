"""Эксперименты с временно́й структурой поверх лучших текущих моделей.

Два подхода:
  1. Лаговые признаки: добавляем значения из окон t-1, t-2 и дельту (скорость изменения).
     ElasticNet обнуляет бесполезные лаги через L1-регуляризацию.

  2. Kalman-сглаживание предсказаний: применяется ПОСЛЕ модели как пост-обработка.
     Использует физиологический prior: время до порога убывает монотонно.
     state[t+1] = state[t] − 30 + шум_процесса
     наблюдение[t] = state[t] + шум_измерения (= предсказание модели)

Запуск:
  uv run python scripts/train_temporal.py
  uv run python scripts/train_temporal.py --target lt1
  uv run python scripts/train_temporal.py --no-kalman
  uv run python scripts/train_temporal.py --no-plots
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

OUT_DIR = _ROOT / "results" / "temporal"

# ─── Параметры Kalman ─────────────────────────────────────────────────────────
# sigma_process: ожидаемая вариация «истинного» времени до порога между окнами
# (помимо гарантированного убывания на 30 с)
KALMAN_SIGMA_PROCESS_SEC = 30.0   # ≈ 0.5 мин неопределённости в скачке порога
KALMAN_SIGMA_OBS_SEC = 120.0      # ≈ 2 мин — ошибка наблюдения (из LOSO MAE)

# ─── Исходные признаки (из train_best_models.py) ─────────────────────────────
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

def add_session_z_emg(df: pd.DataFrame) -> pd.DataFrame:
    """Session-z нормировка ЭМГ (внутри каждого участника)."""
    df = df.copy()
    for col in [c for c in df.columns if c.startswith("vl_")]:
        m = df.groupby("subject_id")[col].transform("mean")
        s = df.groupby("subject_id")[col].transform("std")
        df[f"z_{col}"] = (df[col] - m) / (s + 1e-8)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Произведения признаков из разных физиологических систем."""
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    df["feat_smo2_x_rr"] = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
    df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    df["feat_rr_per_watt"] = df["hrv_mean_rr_ms"] / pw
    return df


def add_lag_features(
    df: pd.DataFrame,
    base_features: list[str],
    lags: tuple[int, ...] = (1, 2),
    add_delta: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Добавляет лаговые признаки (t-1, t-2) и дельты (t − t-1).

    Внутри каждого участника, отсортировано по времени.
    Первые max(lags) окон участника будут NaN в лаговых признаках.
    SimpleImputer заполнит их медианой при обучении.

    Возвращает датафрейм с новыми колонками и список их имён.
    """
    df = df.copy().sort_values(["subject_id", "window_start_sec"])
    new_cols: list[str] = []

    available = [f for f in base_features if f in df.columns]

    # Собираем все новые столбцы в словарь и делаем один pd.concat в конце —
    # иначе многократный df[col] = ... фрагментирует DataFrame (PerformanceWarning)
    extra: dict[str, pd.Series] = {}

    for feat in available:
        for lag in lags:
            col_name = f"{feat}_lag{lag}"
            extra[col_name] = df.groupby("subject_id")[feat].shift(lag)
            new_cols.append(col_name)

        if add_delta and 1 in lags:
            delta_name = f"{feat}_delta"
            extra[delta_name] = df[feat] - df.groupby("subject_id")[feat].shift(1)
            new_cols.append(delta_name)

    df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
    return df, new_cols


def add_running_nirs_features(df: pd.DataFrame, session_params: pd.DataFrame) -> pd.DataFrame:
    """Running NIRS признаки (для LT1)."""
    df = df.copy()
    sp = session_params.set_index("subject_id")

    for col in ["smo2_from_running_max", "hhb_from_running_min", "smo2_rel_drop_pct"]:
        df[col] = np.nan

    for subj, g in df.groupby("subject_id"):
        g = g.sort_values("window_center_sec")
        idx = g.index
        smo2 = g["trainred_smo2_mean"].values
        hhb = g["trainred_hhb_mean"].values

        smo2_rmax = np.maximum.accumulate(np.where(np.isfinite(smo2), smo2, -np.inf))
        hhb_rmin = np.minimum.accumulate(np.where(np.isfinite(hhb), hhb, np.inf))
        smo2_rmax = np.where(np.isinf(smo2_rmax), np.nan, smo2_rmax)
        hhb_rmin = np.where(np.isinf(hhb_rmin), np.nan, hhb_rmin)

        df.loc[idx, "smo2_from_running_max"] = smo2_rmax - smo2
        df.loc[idx, "hhb_from_running_min"] = hhb - hhb_rmin

        if subj in sp.index:
            baseline_smo2 = float(sp.loc[subj, "nirs_smo2_baseline_mean"])
            if np.isfinite(baseline_smo2) and baseline_smo2 > 0:
                df.loc[idx, "smo2_rel_drop_pct"] = (
                    (baseline_smo2 - smo2) / baseline_smo2 * 100.0
                )
    return df


# ─── LOSO CV ──────────────────────────────────────────────────────────────────

def loso(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float = 0.5,
    l1_ratio: float = 0.9,
) -> dict:
    """LOSO CV. Возвращает метрики + предсказания для Kalman."""
    subjects = sorted(df["subject_id"].unique())
    preds, trues, subjs, window_ids = [], [], [], []
    coef_accumulator: list[np.ndarray] = []

    feat_cols = [f for f in features if f in df.columns]

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        X_tr = train[feat_cols].values
        y_tr = train[target].values
        X_te = test[feat_cols].values
        y_te = test[target].values

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

        X_tr = sc.fit_transform(imp.fit_transform(X_tr))
        X_te = sc.transform(imp.transform(X_te))
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        coef_accumulator.append(model.coef_)
        preds.append(y_pred)
        trues.append(y_te)
        subjs.append(np.full(len(y_te), test_s))
        window_ids.append(test.index.values)

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

    # Средние коэффициенты по фолдам (для анализа)
    mean_coef = np.mean(coef_accumulator, axis=0) if coef_accumulator else None

    return {
        "mae_min": mae_min,
        "mae_std": float(np.std(list(per_subj.values()))),
        "r2": r2, "rho": rho,
        "per_subj_mae_min": per_subj,
        "y_pred": y_pred_all,
        "y_true": y_true_all,
        "subjects": subj_all,
        "feat_cols": feat_cols,
        "mean_coef": mean_coef,
    }


def baseline_mae(df: pd.DataFrame, target: str) -> float:
    """LOSO predict-mean baseline."""
    subjects = sorted(df["subject_id"].unique())
    errors = []
    for s in subjects:
        y_tr = df[df["subject_id"] != s][target].values
        y_te = df[df["subject_id"] == s][target].values
        errors.append(float(mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))))
    return float(np.mean(errors)) / 60.0


# ─── Kalman-сглаживание ───────────────────────────────────────────────────────

def kalman_smooth_subject(
    predictions_sec: np.ndarray,
    window_step_sec: float = 5.0,
    sigma_process: float = KALMAN_SIGMA_PROCESS_SEC,
    sigma_obs: float = KALMAN_SIGMA_OBS_SEC,
) -> np.ndarray:
    """1D Kalman-сглаживатель для одного участника.

    Модель состояния (в секундах):
      τ_{t+1} = τ_t − step + w_t,  w_t ~ N(0, σ_process²)
    Наблюдение:
      z_t = τ_t + v_t,             v_t ~ N(0, σ_obs²)

    Физиологический смысл: истинное время до порога убывает примерно
    на step секунд за каждый шаг, но с шумом (вариации физиологии).
    Модель является «шумным датчиком» с ошибкой σ_obs.

    Возвращает сглаженные предсказания (те же единицы, что вход).
    """
    n = len(predictions_sec)
    if n == 0:
        return predictions_sec.copy()

    Q = sigma_process ** 2   # дисперсия процесса
    R = sigma_obs ** 2        # дисперсия наблюдения

    # Инициализация первым наблюдением
    tau_est = predictions_sec[0]
    P_est = R  # начальная неопределённость = шум наблюдения

    smoothed = np.empty(n)
    smoothed[0] = tau_est

    for t in range(1, n):
        # Предсказание (predict step)
        tau_pred = tau_est - window_step_sec
        P_pred = P_est + Q

        # Обновление (update step)
        z = predictions_sec[t]
        K = P_pred / (P_pred + R)          # коэффициент Kalman
        tau_est = tau_pred + K * (z - tau_pred)
        P_est = (1 - K) * P_pred

        smoothed[t] = tau_est

    return smoothed


def apply_kalman(
    result: dict,
    df: pd.DataFrame,
    sigma_process: float = KALMAN_SIGMA_PROCESS_SEC,
    sigma_obs: float = KALMAN_SIGMA_OBS_SEC,
) -> dict:
    """Применяет Kalman-сглаживание к предсказаниям LOSO.

    Для каждого участника отдельно: предсказания отсортированы по времени,
    прогоняются через Kalman-фильтр.

    Возвращает новый dict с теми же ключами, но обновлёнными предсказаниями.
    """
    subjects = sorted(df["subject_id"].unique())
    y_pred_raw = result["y_pred"].copy()
    y_pred_smooth = y_pred_raw.copy()
    subj_all = result["subjects"]

    for s in subjects:
        mask = subj_all == s
        sub_df = df[df["subject_id"] == s].sort_values("window_start_sec")

        # Находим индексы этого участника в массиве предсказаний
        # (они соответствуют порядку строк в sub_df после сортировки)
        subj_indices = np.where(mask)[0]  # позиции в y_pred_raw

        preds_subj = y_pred_raw[subj_indices]
        smoothed = kalman_smooth_subject(preds_subj, sigma_process=sigma_process, sigma_obs=sigma_obs)
        y_pred_smooth[subj_indices] = smoothed

    y_true_all = result["y_true"]
    mae_min = float(mean_absolute_error(y_true_all, y_pred_smooth)) / 60.0
    r2 = float(r2_score(y_true_all, y_pred_smooth))
    rho = float(spearmanr(y_true_all, y_pred_smooth).statistic)
    per_subj = {
        s: float(mean_absolute_error(y_true_all[subj_all == s],
                                     y_pred_smooth[subj_all == s])) / 60.0
        for s in subjects
    }

    return {
        **result,
        "mae_min": mae_min,
        "mae_std": float(np.std(list(per_subj.values()))),
        "r2": r2, "rho": rho,
        "per_subj_mae_min": per_subj,
        "y_pred": y_pred_smooth,
        "kalman_sigma_process": sigma_process,
        "kalman_sigma_obs": sigma_obs,
    }


# ─── Визуализация ─────────────────────────────────────────────────────────────

def plot_before_after_kalman(
    result_raw: dict,
    result_smooth: dict,
    df: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    """Scatter predicted vs actual до и после Kalman + сравнение MAE."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    subjects = sorted(df["subject_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}

    for ax, res, title in zip(
        axes[:2],
        [result_raw, result_smooth],
        ["До Kalman", f"После Kalman (σ_p={result_smooth['kalman_sigma_process']:.0f}с, σ_obs={result_smooth['kalman_sigma_obs']:.0f}с)"],
    ):
        y_true = res["y_true"] / 60.0
        y_pred = res["y_pred"] / 60.0
        subj_labels = res["subjects"]
        for s in subjects:
            m = subj_labels == s
            ax.scatter(y_true[m], y_pred[m], color=subj_color[s], alpha=0.3, s=7, label=s)
        lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Истинное, мин")
        ax.set_ylabel("Предсказанное, мин")
        ax.set_title(f"{title}\nMAE={res['mae_min']:.3f} | R²={res['r2']:.3f} | ρ={res['rho']:.3f}")
        ax.legend(markerscale=2, fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    # MAE по участникам: сравнение
    ax = axes[2]
    x = np.arange(len(subjects))
    w = 0.38
    vals_raw = [result_raw["per_subj_mae_min"].get(s, np.nan) for s in subjects]
    vals_smooth = [result_smooth["per_subj_mae_min"].get(s, np.nan) for s in subjects]
    ax.bar(x - w/2, vals_raw, w, label=f"Сырая  ({result_raw['mae_min']:.3f} мин)", alpha=0.8, color="#1f77b4")
    ax.bar(x + w/2, vals_smooth, w, label=f"Kalman ({result_smooth['mae_min']:.3f} мин)", alpha=0.8, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("MAE, мин")
    ax.set_title("MAE по участникам: до/после Kalman")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path.name}")


def plot_trajectory_kalman(
    result_raw: dict,
    result_smooth: dict,
    df: pd.DataFrame,
    target: str,
    output_path: Path,
    n_show: int = 6,
) -> None:
    """Траектории: истина, сырые предсказания, сглаженные Kalman."""
    subjects = sorted(df["subject_id"].unique())[:n_show]
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}

    ncols = 3
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle("Kalman-сглаживание: траектории предсказаний", fontsize=12, fontweight="bold")

    for i, s in enumerate(subjects):
        ax = axes_flat[i]
        sub = df[df["subject_id"] == s].sort_values("window_start_sec")
        t = sub["elapsed_sec"].values / 60.0
        y_true_s = sub[target].values / 60.0

        mask_raw = result_raw["subjects"] == s
        mask_sm = result_smooth["subjects"] == s
        y_raw_s = result_raw["y_pred"][mask_raw] / 60.0
        y_sm_s = result_smooth["y_pred"][mask_sm] / 60.0

        ax.plot(t, y_true_s, "k-", lw=2.0, label="Истина")
        ax.plot(t, y_raw_s, color=subj_color[s], lw=1.3, ls="--", alpha=0.7, label="Сырое")
        ax.plot(t, y_sm_s, color=subj_color[s], lw=1.8, ls="-", label="Kalman")
        ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.8)

        mae_raw = result_raw["per_subj_mae_min"][s]
        mae_sm = result_smooth["per_subj_mae_min"][s]
        ax.set_title(f"{s} | raw={mae_raw:.2f} → kalman={mae_sm:.2f} мин", fontsize=9)
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


def plot_lag_feature_importance(result: dict, output_path: Path) -> None:
    """Столбчатая диаграмма средних коэффициентов ElasticNet с разбивкой по типу признака."""
    coef = result.get("mean_coef")
    feat_cols = result.get("feat_cols", [])
    if coef is None or len(feat_cols) == 0:
        return

    nonzero = np.abs(coef) > 1e-6
    if nonzero.sum() == 0:
        return

    feat_arr = np.array(feat_cols)
    coef_nz = coef[nonzero]
    feat_nz = feat_arr[nonzero]
    order = np.argsort(np.abs(coef_nz))[::-1][:30]  # топ-30

    def feat_type(name: str) -> str:
        if "_lag1" in name or "_lag2" in name:
            return "lag"
        if "_delta" in name:
            return "delta"
        if name.startswith("z_"):
            return "z_emg"
        if name.startswith("trainred_"):
            return "nirs"
        if name.startswith("hrv_"):
            return "hrv"
        if name.startswith("feat_"):
            return "interaction"
        return "other"

    type_colors = {
        "nirs": "#2ca02c", "hrv": "#9467bd", "z_emg": "#1f77b4",
        "lag": "#ff7f0e", "delta": "#d62728", "interaction": "#8c564b", "other": "#7f7f7f"
    }
    colors = [type_colors[feat_type(f)] for f in feat_nz[order]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(order) * 0.33)))
    ax.barh(feat_nz[order], np.abs(coef_nz[order]), color=colors, alpha=0.85)
    ax.set_xlabel("|Коэффициент ElasticNet| (среднее по фолдам, стандартиз. пространство)")
    ax.set_title("Топ-30 признаков по значимости (ElasticNet с лагами)")

    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=c, label=k) for k, c in type_colors.items() if k != "other"]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path.name}")


# ─── Эксперимент для одной задачи ─────────────────────────────────────────────

def run_experiment(
    df: pd.DataFrame,
    target: str,
    base_features: list[str],
    task_name: str,
    no_kalman: bool = False,
    no_plots: bool = False,
    out_subdir: str = "",
) -> dict:
    """Запускает полный эксперимент: лаги + Kalman + сравнение."""
    out = OUT_DIR / (out_subdir or task_name.lower())
    out.mkdir(parents=True, exist_ok=True)

    b_mae = baseline_mae(df, target)
    print(f"\n{'─' * 60}")
    print(f"{task_name}")
    print(f"{'─' * 60}")
    print(f"Окон: {len(df)}, участников: {df['subject_id'].nunique()}")
    print(f"Baseline (predict-mean): {b_mae:.3f} мин")

    results: dict[str, dict] = {}

    # ── 1. Baseline (без лагов) ──
    r_base = loso(df, base_features, target)
    results["Без лагов (baseline)"] = r_base
    print(f"  {'Без лагов (baseline)':42s} MAE={r_base['mae_min']:.3f}±{r_base['mae_std']:.3f}  R²={r_base['r2']:.3f}  ρ={r_base['rho']:.3f}")

    # ── 2. + лаги t-1 ──
    df_l1, lag1_cols = add_lag_features(df, base_features, lags=(1,), add_delta=True)
    r_l1 = loso(df_l1, base_features + lag1_cols, target)
    results["+ лаг t-1 + delta"] = r_l1
    print(f"  {'+ лаг t-1 + delta':42s} MAE={r_l1['mae_min']:.3f}±{r_l1['mae_std']:.3f}  R²={r_l1['r2']:.3f}  ρ={r_l1['rho']:.3f}")

    # ── 3. + лаги t-1, t-2 ──
    df_l2, lag2_cols = add_lag_features(df, base_features, lags=(1, 2), add_delta=True)
    r_l2 = loso(df_l2, base_features + lag2_cols, target)
    results["+ лаги t-1, t-2 + delta"] = r_l2
    print(f"  {'+ лаги t-1, t-2 + delta':42s} MAE={r_l2['mae_min']:.3f}±{r_l2['mae_std']:.3f}  R²={r_l2['r2']:.3f}  ρ={r_l2['rho']:.3f}")

    # ── 4. + лаги t-1, t-2, t-3 ──
    df_l3, lag3_cols = add_lag_features(df, base_features, lags=(1, 2, 3), add_delta=True)
    r_l3 = loso(df_l3, base_features + lag3_cols, target)
    results["+ лаги t-1, t-2, t-3 + delta"] = r_l3
    print(f"  {'+ лаги t-1, t-2, t-3 + delta':42s} MAE={r_l3['mae_min']:.3f}±{r_l3['mae_std']:.3f}  R²={r_l3['r2']:.3f}  ρ={r_l3['rho']:.3f}")

    # Выбираем лучший для Kalman
    best_name = min(results, key=lambda k: results[k]["mae_min"])
    best_r = results[best_name]
    print(f"\n★ Лучший набор признаков: {best_name}")
    print(f"  MAE = {best_r['mae_min']:.3f} мин | R² = {best_r['r2']:.3f} | ρ = {best_r['rho']:.3f}")

    # ── 5. Kalman-сглаживание ──
    kalman_results: dict[str, dict] = {}
    if not no_kalman:
        print(f"\nKalman-сглаживание (σ_process={KALMAN_SIGMA_PROCESS_SEC:.0f}с, σ_obs={KALMAN_SIGMA_OBS_SEC:.0f}с):")
        for cfg_name, r in list(results.items()):
            r_k = apply_kalman(r, df)
            kalman_results[f"Kalman: {cfg_name}"] = r_k
            delta = r_k["mae_min"] - r["mae_min"]
            sign = "+" if delta > 0 else ""
            print(f"  {cfg_name:42s} → {r_k['mae_min']:.3f} ({sign}{delta:.3f} мин)")

        # Лучший Kalman
        best_kalman_name = min(kalman_results, key=lambda k: kalman_results[k]["mae_min"])
        best_kalman = kalman_results[best_kalman_name]
        print(f"\n★ Лучший с Kalman: {best_kalman_name}")
        print(f"  MAE = {best_kalman['mae_min']:.3f} мин | R² = {best_kalman['r2']:.3f} | ρ = {best_kalman['rho']:.3f}")

    # ── По участникам (лучшая конфигурация) ──
    overall_best_r = (
        min(list(results.values()) + list(kalman_results.values()),
            key=lambda r: r["mae_min"])
        if kalman_results else best_r
    )
    print("\n  По участникам (лучшая конф.):")
    for s, m in sorted(overall_best_r["per_subj_mae_min"].items()):
        print(f"    {s}: {m:.2f} мин")

    # ── Визуализации ──
    if not no_plots and not no_kalman:
        # Берём df с лагами из лучшего набора
        df_for_plots = {"Без лагов (baseline)": df, "+ лаг t-1 + delta": df_l1,
                        "+ лаги t-1, t-2 + delta": df_l2, "+ лаги t-1, t-2, t-3 + delta": df_l3}
        df_plot = df_for_plots.get(best_name, df)

        best_kalman_r = min(kalman_results.values(), key=lambda r: r["mae_min"])
        plot_before_after_kalman(
            best_r, best_kalman_r, df_plot, target,
            out / f"{task_name.lower()}_kalman_scatter.png",
        )
        plot_trajectory_kalman(
            best_r, best_kalman_r, df_plot, target,
            out / f"{task_name.lower()}_kalman_trajectories.png",
        )
        plot_lag_feature_importance(
            best_r, out / f"{task_name.lower()}_feature_importance.png",
        )

    all_results = {**results, **kalman_results}
    return {"results": all_results, "base_mae": b_mae, "task": task_name}


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Аргументы командной строки."""
    p = argparse.ArgumentParser(description="Лаговые признаки + Kalman-сглаживание.")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--no-kalman", action="store_true", help="Пропустить Kalman-сглаживание.")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--dataset", type=Path,
        default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet",
    )
    return p.parse_args()


def main() -> None:
    """Точка входа."""
    args = parse_args()
    print("=" * 65)
    print("ВРЕМЕННЫ́Е ЭКСПЕРИМЕНТЫ: Лаги + Kalman")
    print("=" * 65)

    df_full = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    all_exp_results: list[dict] = []

    # ── LT2 ──────────────────────────────────────────────────────────────────
    if args.target in ("lt2", "both"):
        df_lt2 = df_full[df_full["window_valid_all_required"] == 1].copy()
        df_lt2 = df_lt2.dropna(subset=["target_time_to_lt2_center_sec"])
        df_lt2 = df_lt2.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
        df_lt2 = add_session_z_emg(df_lt2)
        df_lt2 = add_interaction_features(df_lt2)

        z_emg = [c for c in Z_EMG_KEY if c in df_lt2.columns]
        inter = [c for c in INTER_KEY if c in df_lt2.columns]
        lt2_base_feats = NIRS_FEATURES + HRV_FEATURES + inter + z_emg

        exp = run_experiment(
            df_lt2, target="target_time_to_lt2_center_sec",
            base_features=lt2_base_feats,
            task_name="LT2",
            no_kalman=args.no_kalman,
            no_plots=args.no_plots,
            out_subdir="lt2",
        )
        all_exp_results.append(exp)

    # ── LT1 ──────────────────────────────────────────────────────────────────
    if args.target in ("lt1", "both"):
        df_lt1 = df_full[df_full["target_time_to_lt1_usable"] == 1].copy()
        df_lt1 = df_lt1[df_lt1["nirs_valid"] == 1].copy()
        df_lt1 = df_lt1.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
        df_lt1 = add_session_z_emg(df_lt1)
        if not session_params.empty:
            df_lt1 = add_running_nirs_features(df_lt1, session_params)

        z_emg = [c for c in Z_EMG_KEY if c in df_lt1.columns]
        running = [c for c in ["smo2_from_running_max", "hhb_from_running_min", "smo2_rel_drop_pct"]
                   if c in df_lt1.columns]
        lt1_base_feats = z_emg + HRV_FEATURES

        exp = run_experiment(
            df_lt1, target="target_time_to_lt1_sec",
            base_features=lt1_base_feats,
            task_name="LT1",
            no_kalman=args.no_kalman,
            no_plots=args.no_plots,
            out_subdir="lt1",
        )
        all_exp_results.append(exp)

    # ── Сводный CSV ──────────────────────────────────────────────────────────
    rows = []
    for exp in all_exp_results:
        for cfg_name, r in exp["results"].items():
            rows.append({
                "task": exp["task"],
                "config": cfg_name,
                "mae_min": r["mae_min"],
                "mae_std": r["mae_std"],
                "r2": r["r2"],
                "rho": r["rho"],
                "baseline_mae": exp["base_mae"],
                "vs_baseline": r["mae_min"] - exp["base_mae"],
            })
    summary = pd.DataFrame(rows).sort_values(["task", "mae_min"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "summary.csv"
    summary.to_csv(path, index=False)
    print(f"\n✅ Сводка: {path}")
    print("\n✅ Готово.")


if __name__ == "__main__":
    main()
