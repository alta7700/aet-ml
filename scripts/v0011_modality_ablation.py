"""v0011 — Ablation по модальностям: EMG / NIRS / EMG+NIRS / EMG+NIRS+HRV

Версия:    v0011
Дата:      2026-05-08
Предыдущая версия: v0009_kalman_postprocess.py

Что делает:
  Сравнивает четыре набора признаков (модальности) для LT1 и LT2.
  Для каждого набора × каждой конфигурации модели: LOSO → Kalman → MAE.
  SHAP для лучшей модели каждого набора.

Наборы признаков:
  EMG          — vl_* (68) + кинематика (9) → per-subject z-norm
  NIRS         — trainred_* (15) + running NIRS (3)
  EMG+NIRS     — EMG + NIRS (без interaction, нет HRV)
  EMG+NIRS+HRV — всё выше + HRV (7) + interaction (3) [референс ≈ v0004/v0009]

Зоопарк:
  Ridge(alpha ∈ [1,10,100,1000])
  HuberRegressor(epsilon ∈ [1.1,1.35,1.5,2.0])
  ElasticNet(alpha ∈ [0.01,0.1,1.0] × l1_ratio ∈ [0.2,0.5,0.9])
  GradientBoosting(n_estimators ∈ [50,100,200] × max_depth ∈ [2,3])
  SVR(rbf, C ∈ [1,10,100] × epsilon ∈ [0.1,1.0])

Выход:
  results/v0011/summary.csv        — все конфиги × MAE
  results/v0011/best_per_set.csv   — лучший per (набор, таргет)
  results/v0011/shap/              — SHAP для лучшей модели каждого набора
  results/v0011/report.md          — итоговый отчёт

Воспроизведение:
  uv run python scripts/v0011_modality_ablation.py
  uv run python scripts/v0011_modality_ablation.py --target lt2
  uv run python scripts/v0011_modality_ablation.py --feature-set EMG NIRS
  uv run python scripts/v0011_modality_ablation.py --no-shap
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from dataset_pipeline.baselines import (
    run_honest_baselines,
    format_honest_block,
)

OUT_DIR = _ROOT / "results" / "v0011"

# ─── Определения признаков ────────────────────────────────────────────────────

# Сырые ЭМГ-колонки (до z-norm) — 68 штук
_EMG_RAW_PREFIX = "vl_"

# Кинематические признаки (тот же сенсорный блок, что ЭМГ)
KINEMATICS_FEATURES = [
    "cadence_mean_rpm", "cadence_cv",
    "load_duration_ms", "rest_duration_ms", "load_rest_ratio",
    "load_duration_cv", "rest_duration_cv",
    "load_trend_cv_ratio_30s", "rest_trend_cv_ratio_30s",
    "load_sampen_30s", "rest_sampen_30s",    # ~15% nan → SimpleImputer
]

# NIRS — Train.Red, 15 признаков
NIRS_FEATURES = [
    "trainred_smo2_mean", "trainred_smo2_drop", "trainred_smo2_slope",
    "trainred_smo2_std", "trainred_dsmo2_dt",
    "trainred_hhb_mean", "trainred_hhb_slope", "trainred_hhb_std", "trainred_dhhb_dt",
    "trainred_hbdiff_mean", "trainred_hbdiff_slope", "trainred_hbdiff_std",
    "trainred_thb_mean", "trainred_thb_slope", "trainred_thb_std",
]

# Running NIRS — накопленные от начала теста (вычисляются динамически)
RUNNING_NIRS_FEATURES = [
    "smo2_from_running_max",
    "hhb_from_running_min",
    "smo2_rel_drop_pct",
]

# HRV — 7 признаков из сырого RR (без QC-колонок)
HRV_FEATURES = [
    "hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms",
    "hrv_sd1_ms", "hrv_sd2_ms", "hrv_sd1_sd2_ratio",
    "hrv_dfa_alpha1",
]

# Interaction (только когда есть и NIRS, и HRV)
INTERACTION_FEATURES = ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"]

# ─── Зоопарк моделей ──────────────────────────────────────────────────────────

def _build_zoo() -> list[dict]:
    """Возвращает список конфигураций моделей для перебора."""
    configs = []

    for alpha in [1, 10, 100, 1000]:
        configs.append({
            "name": f"Ridge(α={alpha})",
            "factory": lambda a=alpha: Ridge(alpha=a),
            "linear": True,
        })

    for eps in [1.1, 1.35, 1.5, 2.0]:
        configs.append({
            "name": f"Huber(ε={eps})",
            "factory": lambda e=eps: HuberRegressor(epsilon=e, max_iter=2000),
            "linear": True,
        })

    for alpha, l1 in itertools.product([0.01, 0.1, 1.0], [0.2, 0.5, 0.9]):
        configs.append({
            "name": f"EN(α={alpha},l1={l1})",
            "factory": lambda a=alpha, l=l1: ElasticNet(alpha=a, l1_ratio=l,
                                                         max_iter=5000, random_state=42),
            "linear": True,
        })

    for n_est, depth in itertools.product([50, 100, 200], [2, 3]):
        configs.append({
            "name": f"GBM(n={n_est},d={depth})",
            "factory": lambda n=n_est, d=depth: GradientBoostingRegressor(
                n_estimators=n, max_depth=d, random_state=42),
            "linear": False,
        })

    for C, eps in itertools.product([1, 10, 100], [0.1, 1.0]):
        configs.append({
            "name": f"SVR(C={C},ε={eps})",
            "factory": lambda c=C, e=eps: SVR(kernel="rbf", C=c, epsilon=e),
            "linear": False,
        })

    return configs


# ─── Feature engineering ──────────────────────────────────────────────────────

def _get_emg_raw_cols(df: pd.DataFrame) -> list[str]:
    """Возвращает все сырые vl_-колонки из датафрейма."""
    return [c for c in df.columns if c.startswith(_EMG_RAW_PREFIX)]


def _add_subject_z(df: pd.DataFrame, cols: list[str],
                   prefix: str = "z_") -> tuple[pd.DataFrame, list[str]]:
    """Per-subject z-нормировка колонок, создаёт новые колонки с prefix."""
    df = df.copy()
    z_cols = []
    for col in cols:
        if col not in df.columns:
            continue
        m = df.groupby("subject_id")[col].transform("mean")
        s = df.groupby("subject_id")[col].transform("std")
        z_col = f"{prefix}{col}"
        df[z_col] = (df[col] - m) / (s + 1e-8)
        z_cols.append(z_col)
    return df, z_cols


def _add_running_nirs(df: pd.DataFrame,
                      session_params: pd.DataFrame) -> pd.DataFrame:
    """Добавляет running NIRS признаки (накопленные от начала теста)."""
    df = df.copy()
    sp_idx = session_params.set_index("subject_id") if not session_params.empty else pd.DataFrame()
    for col in RUNNING_NIRS_FEATURES:
        df[col] = np.nan
    for subj, g in df.groupby("subject_id"):
        g = g.sort_values("window_start_sec")
        idx = g.index
        smo2 = g["trainred_smo2_mean"].values
        hhb  = g["trainred_hhb_mean"].values
        rmax = np.maximum.accumulate(np.where(np.isfinite(smo2), smo2, -np.inf))
        rmin = np.minimum.accumulate(np.where(np.isfinite(hhb),  hhb,   np.inf))
        rmax = np.where(np.isinf(rmax), np.nan, rmax)
        rmin = np.where(np.isinf(rmin), np.nan, rmin)
        df.loc[idx, "smo2_from_running_max"] = rmax - smo2
        df.loc[idx, "hhb_from_running_min"]  = hhb  - rmin
        if not sp_idx.empty and subj in sp_idx.index:
            b = float(sp_idx.loc[subj, "nirs_smo2_baseline_mean"])
            if np.isfinite(b) and b > 0:
                df.loc[idx, "smo2_rel_drop_pct"] = (b - smo2) / b * 100.0
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет interaction-признаки (только для набора с HRV+NIRS)."""
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    df["feat_smo2_x_rr"]  = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
    df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    df["feat_rr_per_watt"] = df["hrv_mean_rr_ms"] / pw
    return df


def prepare_data(df_raw: pd.DataFrame,
                 session_params: pd.DataFrame,
                 target: str) -> pd.DataFrame:
    """Базовая подготовка: фильтрация, z-norm EMG+kin, running NIRS, interactions.

    Возвращает датафрейм со всеми доступными признаками;
    конкретный набор выбирается в get_feature_cols().
    """
    # Фильтр по таргету
    if target == "lt2":
        df = df_raw[df_raw["window_valid_all_required"] == 1].copy()
        df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    else:
        df = df_raw[df_raw["target_time_to_lt1_usable"] == 1].copy()

    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)

    # Per-subject z-norm: EMG
    emg_raw = _get_emg_raw_cols(df)
    df, _ = _add_subject_z(df, emg_raw, prefix="z_")

    # Per-subject z-norm: кинематика
    kin_present = [c for c in KINEMATICS_FEATURES if c in df.columns]
    df, _ = _add_subject_z(df, kin_present, prefix="z_")

    # Running NIRS
    df = _add_running_nirs(df, session_params)

    # Interaction (нужен NIRS + HRV)
    if "trainred_smo2_mean" in df.columns and "hrv_mean_rr_ms" in df.columns:
        df = _add_interactions(df)

    return df


def get_feature_cols(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Возвращает список колонок для заданного набора признаков."""
    all_cols = set(df.columns)

    # z-norm EMG + z-norm кинематика
    emg_cols = [c for c in df.columns if c.startswith("z_vl_")]
    kin_cols  = [c for c in df.columns
                 if c.startswith("z_") and not c.startswith("z_vl_")]

    nirs_cols  = [c for c in NIRS_FEATURES if c in all_cols]
    run_nirs   = [c for c in RUNNING_NIRS_FEATURES if c in all_cols]
    hrv_cols   = [c for c in HRV_FEATURES if c in all_cols]
    inter_cols = [c for c in INTERACTION_FEATURES if c in all_cols]

    sets = {
        "EMG":          emg_cols + kin_cols,
        "NIRS":         nirs_cols + run_nirs,
        "HRV":          hrv_cols,
        "EMG+NIRS":     emg_cols + kin_cols + nirs_cols + run_nirs,
        "EMG+NIRS+HRV": emg_cols + kin_cols + nirs_cols + run_nirs + hrv_cols + inter_cols,
    }

    if feature_set not in sets:
        raise ValueError(f"Неизвестный набор: {feature_set}. Доступны: {list(sets)}")

    return [c for c in sets[feature_set] if c in all_cols]


# ─── LOSO ─────────────────────────────────────────────────────────────────────

def loso_predict(df: pd.DataFrame,
                 feat_cols: list[str],
                 target_col: str,
                 model_factory) -> dict:
    """LOSO: возвращает предсказания, истинные значения и список моделей."""
    subjects = sorted(df["subject_id"].unique())
    preds, trues, subjs, models_list = [], [], [], []

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
        models_list.append({"model": mdl, "imp": imp, "sc": sc, "test_s": test_s})

    y_pred_all = np.concatenate(preds)
    y_true_all = np.concatenate(trues)
    subj_all   = np.concatenate(subjs)

    return {
        "y_pred": y_pred_all, "y_true": y_true_all,
        "subjects": subj_all, "feat_cols": feat_cols,
        "models": models_list,
        "raw_mae_min": mean_absolute_error(y_true_all, y_pred_all) / 60.0,
        "r2":   r2_score(y_true_all, y_pred_all),
        "rho":  float(spearmanr(y_true_all, y_pred_all).statistic),
    }


# ─── Kalman постпроцессинг ────────────────────────────────────────────────────

def kalman_smooth(y_pred: np.ndarray,
                  sigma_p: float = 15.0,
                  sigma_obs: float = 150.0) -> np.ndarray:
    """Одномерный фильтр Калмана для монотонизации предсказания.

    Модель: x[t] = x[t-1] - dt (время до порога убывает)
    dt = шаг окна = 5 с.
    """
    n = len(y_pred)
    x = y_pred[0]
    p = sigma_obs ** 2
    smoothed = np.empty(n)
    dt = 5.0  # шаг окна

    for i in range(n):
        # Предсказание
        x -= dt
        p += sigma_p ** 2
        # Обновление
        k = p / (p + sigma_obs ** 2)
        x = x + k * (y_pred[i] - x)
        p = (1 - k) * p
        smoothed[i] = x

    return smoothed


def apply_kalman_loso(loso_result: dict,
                      df: pd.DataFrame,
                      sigma_p: float = 15.0,
                      sigma_obs: float = 150.0) -> float:
    """Применяет Kalman per-subject и возвращает итоговый MAE (мин)."""
    y_pred_all = loso_result["y_pred"]
    y_true_all = loso_result["y_true"]
    subj_all   = loso_result["subjects"]
    subjects   = sorted(df["subject_id"].unique())

    preds_k, trues_k = [], []
    for s in subjects:
        mask = subj_all == s
        if mask.sum() == 0:
            continue
        yk = kalman_smooth(y_pred_all[mask], sigma_p, sigma_obs)
        preds_k.append(yk)
        trues_k.append(y_true_all[mask])

    y_pred_k = np.concatenate(preds_k)
    y_true_k = np.concatenate(trues_k)
    return mean_absolute_error(y_true_k, y_pred_k) / 60.0


# ─── Grid search по зоопарку ──────────────────────────────────────────────────

def _run_one(cfg: dict,
             df: pd.DataFrame,
             feat_cols: list[str],
             target_col: str,
             feature_set: str,
             target_name: str,
             n_subj: int,
             n_feat: int,
             sigma_p: float,
             sigma_obs: float) -> dict:
    """Один конфиг: LOSO → Kalman → запись. Вызывается параллельно."""
    t0 = time.perf_counter()
    try:
        res = loso_predict(df, feat_cols, target_col, cfg["factory"])
        kalman_mae = apply_kalman_loso(res, df, sigma_p, sigma_obs)
        elapsed = time.perf_counter() - t0
        return {
            "feature_set":    feature_set,
            "target":         target_name,
            "model":          cfg["name"],
            "n_subjects":     n_subj,
            "n_features":     n_feat,
            "raw_mae_min":    round(res["raw_mae_min"], 4),
            "kalman_mae_min": round(kalman_mae, 4),
            "r2":             round(res["r2"], 3),
            "rho":            round(res["rho"], 3),
            "sec":            round(elapsed, 1),
            "_loso":          res,
        }
    except Exception as exc:
        return {
            "feature_set": feature_set, "target": target_name,
            "model": cfg["name"], "kalman_mae_min": np.nan,
            "raw_mae_min": np.nan, "_loso": None,
            "_error": str(exc),
        }


def run_zoo(df: pd.DataFrame,
            feat_cols: list[str],
            target_col: str,
            feature_set: str,
            target_name: str,
            zoo: list[dict],
            sigma_p: float = 15.0,
            sigma_obs: float = 150.0,
            n_jobs: int = -1) -> list[dict]:
    """Прогоняет весь зоопарк параллельно (joblib).

    Возвращает список записей для summary.csv.
    """
    n_subj = df["subject_id"].nunique()
    n_feat = len(feat_cols)
    print(f"\n  [{feature_set} / {target_name}]  n={n_subj} субъектов, "
          f"{n_feat} признаков, {len(zoo)} конфигов...")

    records = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_run_one)(
            cfg, df, feat_cols, target_col,
            feature_set, target_name, n_subj, n_feat,
            sigma_p, sigma_obs,
        )
        for cfg in zoo
    )

    # Вывод результатов после завершения всех
    for r in sorted(records, key=lambda x: x.get("kalman_mae_min") or 99):
        if r.get("_error"):
            print(f"    {r['model']:<30s}  ОШИБКА: {r['_error']}")
        else:
            print(f"    {r['model']:<30s}  raw={r['raw_mae_min']:.3f}  "
                  f"kalman={r['kalman_mae_min']:.3f}  ({r['sec']:.1f}s)")

    return records


# ─── SHAP ─────────────────────────────────────────────────────────────────────

def _feat_color(name: str) -> str:
    """Цвет по модальности признака."""
    if name.startswith("z_vl_"):      return "#1f77b4"   # EMG
    if name.startswith("z_cadence") or name.startswith("z_load") or \
       name.startswith("z_rest") or name.startswith("z_load_rest"):
        return "#17becf"                                  # кинематика
    if name.startswith("trainred_"):  return "#2ca02c"   # NIRS
    if name.startswith("smo2_from") or name.startswith("hhb_from") or \
       name.startswith("smo2_rel"):   return "#98df8a"   # running NIRS
    if name.startswith("hrv_"):       return "#9467bd"   # HRV
    if name.startswith("feat_"):      return "#8c564b"   # interaction
    return "#7f7f7f"


def compute_shap(df: pd.DataFrame,
                 feat_cols: list[str],
                 target_col: str,
                 feature_set: str,
                 target_name: str,
                 best_loso: dict,
                 out_dir: Path) -> None:
    """SHAP для лучшей модели набора.

    Использует первую модель из LOSO (обучена на всех кроме test_s=subjects[0]).
    Для глобального SHAP — дообучаем на всех данных.
    Поддерживаем только линейные модели (LinearExplainer).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Глобальная модель — на всех данных
    first_mdl_cfg = best_loso["models"][0]
    factory = type(first_mdl_cfg["model"])   # Ridge / Huber / EN
    # Воссоздаём с теми же гиперпараметрами
    global_model = first_mdl_cfg["model"].__class__(**{
        k: v for k, v in first_mdl_cfg["model"].get_params().items()
    })

    X_all = df[feat_cols].values
    y_all = df[target_col].values
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X_sc = sc.fit_transform(imp.fit_transform(X_all))
    global_model.fit(X_sc, y_all)

    # Проверяем что это линейная модель
    if not hasattr(global_model, "coef_"):
        print(f"    [SHAP] Пропуск: {type(global_model).__name__} не линейная.")
        return

    import shap
    explainer  = shap.LinearExplainer(global_model, X_sc, feature_names=feat_cols)
    shap_vals  = explainer(X_sc).values
    mean_abs   = pd.Series(np.abs(shap_vals).mean(0), index=feat_cols).sort_values(ascending=False)

    print(f"\n  Топ-10 признаков по |SHAP| ({feature_set}/{target_name}):")
    for feat, val in mean_abs.head(10).items():
        print(f"    {feat:<45s}  {val/60:.4f} мин")

    # Bar plot топ-20
    top20 = mean_abs.head(20)
    colors = [_feat_color(f) for f in top20.index]

    fig, ax = plt.subplots(figsize=(9, max(5, len(top20) * 0.4)))
    bars = ax.barh(top20.index[::-1], top20.values[::-1] / 60.0,
                   color=colors[::-1], alpha=0.85)
    for bar, val in zip(bars, top20.values[::-1] / 60.0):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    legend_handles = [
        mpatches.Patch(color="#1f77b4", label="EMG (z-norm)"),
        mpatches.Patch(color="#17becf", label="Кинематика (z-norm)"),
        mpatches.Patch(color="#2ca02c", label="NIRS"),
        mpatches.Patch(color="#98df8a", label="Running NIRS"),
        mpatches.Patch(color="#9467bd", label="HRV"),
        mpatches.Patch(color="#8c564b", label="Interaction"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right")
    ax.set_xlabel("mean |SHAP|, мин")
    ax.set_title(f"{target_name.upper()} / {feature_set}: Важность признаков (SHAP)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = out_dir / f"{target_name}_{feature_set.replace('+','_')}_shap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")


# ─── Honest baselines helper ─────────────────────────────────────────────────

def _loso_to_per_subject(df: pd.DataFrame,
                          feat_cols: list[str],
                          target_col: str,
                          model_factory) -> dict[str, dict]:
    """Запускает LOSO и возвращает per-subject dict для honest baselines."""
    subjects = sorted(df["subject_id"].unique())
    result: dict[str, dict] = {}
    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test  = df[df["subject_id"] == test_s].sort_values("window_start_sec")
        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        mdl = model_factory()
        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        mdl.fit(X_tr, train[target_col].values)
        result[test_s] = {
            "t_sec":  test["window_start_sec"].values,
            "y_pred": mdl.predict(X_te),
            "y_true": test[target_col].values,
        }
    return result


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(summary: pd.DataFrame, best: pd.DataFrame, out_dir: Path,
                 honest_md: str = "") -> None:
    """Пишет report.md с таблицами сравнения."""
    lines = [
        "# v0011 — Ablation по модальностям\n",
        f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d')}  \n",
        "Метрика: MAE (мин) после фильтра Калмана, LOSO CV.\n\n",
    ]

    for tgt in ["lt2", "lt1"]:
        sub = best[best["target"] == tgt].copy()
        if sub.empty:
            continue
        lines.append(f"## {tgt.upper()}\n\n")
        lines.append("| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for _, row in sub.sort_values("kalman_mae_min").iterrows():
            lines.append(
                f"| **{row['feature_set']}** | {row['model']} | {row['n_subjects']} | "
                f"{row['n_features']} | {row['raw_mae_min']:.3f} | **{row['kalman_mae_min']:.3f}** | "
                f"{row['r2']:.3f} | {row['rho']:.3f} |\n"
            )
        lines.append("\n")

    # Полная таблица по обоим таргетам
    lines.append("## Полная таблица (все конфиги)\n\n")
    for tgt in ["lt2", "lt1"]:
        sub = summary[summary["target"] == tgt].sort_values("kalman_mae_min")
        if sub.empty:
            continue
        lines.append(f"### {tgt.upper()}\n\n")
        lines.append("| Набор | Модель | Kalman MAE |\n")
        lines.append("|---|---|---|\n")
        for _, row in sub.head(20).iterrows():
            lines.append(f"| {row['feature_set']} | {row['model']} | {row['kalman_mae_min']:.3f} |\n")
        lines.append("\n")

    if honest_md:
        lines.append("## Honest baselines\n\n")
        lines.append(honest_md + "\n")

    p = out_dir / "report.md"
    p.write_text("".join(lines), encoding="utf-8")
    print(f"\n  → {p.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0011 — ablation по модальностям")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--feature-set", nargs="+",
                   choices=["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default=["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"],
                   help="Какие наборы признаков прогонять")
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--sigma-p",   type=float, default=15.0)
    p.add_argument("--sigma-obs", type=float, default=150.0)
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Число параллельных процессов (-1 = все ядра)")
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("v0011 — ABLATION ПО МОДАЛЬНОСТЯМ")
    print("=" * 70)

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shap_dir = OUT_DIR / "shap"

    zoo = _build_zoo()
    print(f"Зоопарк: {len(zoo)} конфигураций")

    targets_cfg = {
        "lt2": {
            "col": "target_time_to_lt2_center_sec",
            "label": "lt2",
        },
        "lt1": {
            "col": "target_time_to_lt1_sec",
            "label": "lt1",
        },
    }
    if args.target != "both":
        targets_cfg = {k: v for k, v in targets_cfg.items() if k == args.target}

    all_records: list[dict] = []
    honest_records: list[dict] = []
    honest_md_blocks: list[str] = []

    for tgt_name, tgt_cfg in targets_cfg.items():
        print(f"\n{'═'*70}")
        print(f"ТАРГЕТ: {tgt_name.upper()}")
        print(f"{'═'*70}")

        df_prep = prepare_data(df_raw, session_params, tgt_name)
        target_col = tgt_cfg["col"]
        if target_col not in df_prep.columns:
            continue
        df_prep_tgt = df_prep.dropna(subset=[target_col])

        for fset in args.feature_set:
            feat_cols = get_feature_cols(df_prep_tgt, fset)
            if not feat_cols:
                print(f"  [{fset}] — нет признаков, пропуск")
                continue

            records = run_zoo(df_prep_tgt, feat_cols, target_col,
                              fset, tgt_name, zoo,
                              args.sigma_p, args.sigma_obs,
                              n_jobs=args.n_jobs)
            all_records.extend(records)

            # ── Честные baselines для лучшей модели ──────────────────────────
            best_rec = min(
                (r for r in records if r.get("kalman_mae_min") is not None),
                key=lambda r: r["kalman_mae_min"],
                default=None,
            )
            if best_rec is not None:
                # Сохраняем предсказания лучшей модели для анализа per-window
                _loso_best = best_rec.get("_loso")
                if _loso_best is not None:
                    fset_tag = fset.replace("+", "_")
                    np.save(OUT_DIR / f"ypred_{tgt_name}_{fset_tag}.npy", _loso_best["y_pred"])
                    np.save(OUT_DIR / f"ytrue_{tgt_name}_{fset_tag}.npy", _loso_best["y_true"])

                # Находим factory лучшей модели из зоопарка
                best_factory = next(
                    (c["factory"] for c in zoo if c["name"] == best_rec["model"]),
                    None,
                )
                if best_factory is not None:
                    print(f"  [{fset}] honest baselines...", end=" ", flush=True)
                    raw_ps = _loso_to_per_subject(
                        df_prep_tgt, feat_cols, target_col, best_factory)
                    hb = run_honest_baselines(
                        df_prep_tgt, feat_cols, target_col,
                        raw_ps, best_factory,
                        kalman_fn=kalman_smooth,
                        sigma_p=args.sigma_p,
                        sigma_obs_ref=args.sigma_obs,
                    )
                    print(f"gap={hb['gap_raw_vs_fw']:+.3f}  {hb['verdict']}")
                    honest_records.append({
                        "feature_set":      fset,
                        "target":           tgt_name,
                        "best_model":       best_rec["model"],
                        "mae_mean_pred":    hb["mean_pred"]["mae_min"],
                        "mae_first_win":    hb["first_win"]["mae_min"],
                        "mae_raw":          hb["raw"]["mae_min"],
                        "mae_kalman_ref":   hb["kalman_ref"]["mae_min"],
                        "gap_raw_vs_fw":    hb["gap_raw_vs_fw"],
                        "verdict":          hb["verdict"],
                        "stability_raw":    hb["stability_raw"],
                        "stability_fw":     hb["stability_fw"],
                    })
                    honest_md_blocks.append(
                        format_honest_block(hb, fset, tgt_name))

    # Сводная таблица
    summary_rows = [{k: v for k, v in r.items() if k != "_loso"}
                    for r in all_records]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.dropna(subset=["kalman_mae_min"])
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\n  → summary.csv ({len(summary_df)} строк)")

    # Per-subject MAE по всем моделям и наборам признаков
    subj_rows = []
    for r in all_records:
        loso = r.get("_loso")
        if loso is None or loso.get("subjects") is None:
            continue
        subj_arr = loso["subjects"]
        y_pred   = loso["y_pred"]
        y_true   = loso["y_true"]
        for s in np.unique(subj_arr):
            mask = subj_arr == s
            if mask.sum() < 2:
                continue
            mae_s = mean_absolute_error(y_true[mask], y_pred[mask]) / 60.0
            r2_s  = r2_score(y_true[mask], y_pred[mask])
            subj_rows.append({
                "feature_set": r["feature_set"],
                "target":      r["target"],
                "model":       r["model"],
                "subject_id":  s,
                "mae_min":     round(mae_s, 4),
                "r2":          round(r2_s, 3),
            })
    pd.DataFrame(subj_rows).to_csv(OUT_DIR / "per_subject.csv", index=False)
    print(f"  → per_subject.csv ({len(subj_rows)} строк)")

    # Лучший на набор
    best_df = (summary_df
               .sort_values("kalman_mae_min")
               .groupby(["feature_set", "target"], sort=False)
               .first()
               .reset_index())
    best_df.to_csv(OUT_DIR / "best_per_set.csv", index=False)
    print(f"  → best_per_set.csv")

    # Вывод итогов
    print("\n" + "═" * 70)
    print("ИТОГИ (лучший per набор, Kalman MAE мин):")
    print("═" * 70)
    for tgt in ["lt2", "lt1"]:
        sub = best_df[best_df["target"] == tgt].sort_values("kalman_mae_min")
        if sub.empty:
            continue
        print(f"\n  {tgt.upper()}:")
        for _, row in sub.iterrows():
            print(f"    {row['feature_set']:<16s}  {row['model']:<30s}  "
                  f"kalman={row['kalman_mae_min']:.3f} мин")

    # SHAP для лучшего per набор
    if not args.no_shap:
        print("\n" + "─" * 70)
        print("SHAP для лучших моделей:")
        for _, best_row in best_df.iterrows():
            fset = best_row["feature_set"]
            tgt_name = best_row["target"]
            tgt_col = targets_cfg[tgt_name]["col"]
            model_name = best_row["model"]

            # Находим loso-результат из all_records
            loso_res = next(
                (r["_loso"] for r in all_records
                 if r.get("feature_set") == fset
                 and r.get("target") == tgt_name
                 and r.get("model") == model_name
                 and r.get("_loso") is not None),
                None,
            )
            if loso_res is None:
                continue

            # Проверяем что лучшая модель линейная (для LinearExplainer)
            first_model = loso_res["models"][0]["model"]
            if not hasattr(first_model, "coef_"):
                print(f"  [{fset}/{tgt_name}] SHAP пропущен: {type(first_model).__name__} нелинейная")
                continue

            df_prep = prepare_data(df_raw, session_params, tgt_name)
            feat_cols = get_feature_cols(df_prep, fset)
            df_tgt = df_prep.dropna(subset=[tgt_col])

            compute_shap(df_tgt, feat_cols, tgt_col, fset, tgt_name, loso_res, shap_dir)

    # Честные baselines → CSV
    if honest_records:
        pd.DataFrame(honest_records).to_csv(
            OUT_DIR / "honest_baselines.csv", index=False)
        print(f"  → honest_baselines.csv ({len(honest_records)} строк)")

    # Report (с честными блоками)
    write_report(summary_df, best_df, OUT_DIR,
                 honest_md="\n".join(honest_md_blocks))
    print(f"\n✅ Готово. Результаты: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
