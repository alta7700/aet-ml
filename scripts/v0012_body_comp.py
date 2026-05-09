"""v0012 — Аблация модальностей + состав тела (body composition)

Версия:    v0012
Дата:      2026-05-08
База:      v0011_modality_ablation.py

Что добавляет к v0011:
  Шесть статических признаков из биоимпеданса (subjects.parquet):
    bmi, body_fat_pct, phase_angle, dominant_leg_circumference,
    leg_fat_pct, muscle_to_fat_leg

  Новые наборы признаков:
    BodyComp        — только статика (baseline: что умеет тело само по себе)
    EMG+BC          — EMG + состав тела
    EMG+NIRS+BC     — EMG+NIRS + состав тела
    EMG+NIRS+HRV+BC — полная модель + состав тела

Защита от переобучения:
  1. BodyComp-baseline: если EMG+BC ≤ BodyComp → EMG добавляет нуль
  2. Честные baselines (MeanPredictor, FirstWindowPredictor) для каждого набора
  3. gap = raw_MAE − FirstWindow_MAE: если > +0.2 мин → online-ценности нет
  4. Per-subject Δ MAE: добавка BC стабильна или только на 1–2 субъектах?
  5. Сравнение с v0011: delta_mae = mae_v0012 − mae_v0011

Внимание: статические признаки одинаковы для всех окон субъекта.
  При малом N (14 субъектов LT2) риск переобучения высок.
  Итоговый вердикт: смотреть на per-subject Δ и честные baselines.

Выход:
  results/v0012/summary.csv          — все конфиги × MAE
  results/v0012/best_per_set.csv     — лучший per (набор, таргет)
  results/v0012/honest_baselines.csv — честные baselines
  results/v0012/body_comp_delta.csv  — per-subject Δ MAE (v0012 − v0011)
  results/v0012/report.md            — итоговый отчёт

Воспроизведение:
  uv run python scripts/v0012_body_comp.py
  uv run python scripts/v0012_body_comp.py --target lt2
  uv run python scripts/v0012_body_comp.py --feature-set EMG+BC EMG+NIRS+HRV+BC
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

OUT_DIR = _ROOT / "results" / "v0012"

# ─── Признаки (те же, что v0011) ─────────────────────────────────────────────

_EMG_RAW_PREFIX = "vl_"

KINEMATICS_FEATURES = [
    "cadence_mean_rpm", "cadence_cv",
    "load_duration_ms", "rest_duration_ms", "load_rest_ratio",
    "load_duration_cv", "rest_duration_cv",
    "load_trend_cv_ratio_30s", "rest_trend_cv_ratio_30s",
    "load_sampen_30s", "rest_sampen_30s",
]

NIRS_FEATURES = [
    "trainred_smo2_mean", "trainred_smo2_drop", "trainred_smo2_slope",
    "trainred_smo2_std", "trainred_dsmo2_dt",
    "trainred_hhb_mean", "trainred_hhb_slope", "trainred_hhb_std", "trainred_dhhb_dt",
    "trainred_hbdiff_mean", "trainred_hbdiff_slope", "trainred_hbdiff_std",
    "trainred_thb_mean", "trainred_thb_slope", "trainred_thb_std",
]

RUNNING_NIRS_FEATURES = [
    "smo2_from_running_max",
    "hhb_from_running_min",
    "smo2_rel_drop_pct",
]

HRV_FEATURES = [
    "hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms",
    "hrv_sd1_ms", "hrv_sd2_ms", "hrv_sd1_sd2_ratio",
    "hrv_dfa_alpha1",
]

INTERACTION_FEATURES = ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"]

# Новые: статические признаки состава тела
BODY_COMP_FEATURES = [
    "bc_bmi",
    "bc_body_fat_pct",
    "bc_phase_angle",
    "bc_dominant_leg_circumference",
    "bc_leg_fat_pct",
    "bc_muscle_to_fat_leg",
]


# ─── Зоопарк (тот же, что v0011) ─────────────────────────────────────────────

def _build_zoo() -> list[dict]:
    configs = []
    for alpha in [1, 10, 100, 1000]:
        configs.append({"name": f"Ridge(α={alpha})",
                         "factory": lambda a=alpha: Ridge(alpha=a),
                         "linear": True})
    for eps in [1.1, 1.35, 1.5, 2.0]:
        configs.append({"name": f"Huber(ε={eps})",
                         "factory": lambda e=eps: HuberRegressor(epsilon=e, max_iter=2000),
                         "linear": True})
    for alpha, l1 in itertools.product([0.01, 0.1, 1.0], [0.2, 0.5, 0.9]):
        configs.append({"name": f"EN(α={alpha},l1={l1})",
                         "factory": lambda a=alpha, l=l1: ElasticNet(
                             alpha=a, l1_ratio=l, max_iter=5000, random_state=42),
                         "linear": True})
    for n_est, depth in itertools.product([50, 100, 200], [2, 3]):
        configs.append({"name": f"GBM(n={n_est},d={depth})",
                         "factory": lambda n=n_est, d=depth: GradientBoostingRegressor(
                             n_estimators=n, max_depth=d, random_state=42),
                         "linear": False})
    for C, eps in itertools.product([1, 10, 100], [0.1, 1.0]):
        configs.append({"name": f"SVR(C={C},ε={eps})",
                         "factory": lambda c=C, e=eps: SVR(kernel="rbf", C=c, epsilon=e),
                         "linear": False})
    return configs


# ─── Body composition: загрузка и вычисление производных ─────────────────────

def _load_body_comp(dataset_dir: Path) -> pd.DataFrame:
    """Загружает subjects.parquet и вычисляет производные признаки.

    Возвращает DataFrame с колонками subject_id + BODY_COMP_FEATURES.
    Значения одинаковы для всех окон субъекта (статика).
    """
    subj = pd.read_parquet(dataset_dir / "subjects.parquet")

    bc = pd.DataFrame()
    bc["subject_id"] = subj["subject_id"]

    w = subj["weight"].replace(0, np.nan)
    h = subj["height"].replace(0, np.nan) / 100.0  # см → м

    bc["bc_bmi"] = w / (h ** 2)
    bc["bc_body_fat_pct"] = subj["body_fat_mass"] / w * 100.0

    total_leg = subj["dominant_leg_lean_mass"] + subj["dominant_leg_fat_mass"]
    bc["bc_leg_fat_pct"] = subj["dominant_leg_fat_mass"] / total_leg.replace(0, np.nan) * 100.0
    bc["bc_muscle_to_fat_leg"] = (
        subj["dominant_leg_lean_mass"] / subj["dominant_leg_fat_mass"].replace(0, np.nan)
    )

    bc["bc_phase_angle"] = subj["phase_angle"]
    bc["bc_dominant_leg_circumference"] = subj["dominant_leg_circumference"]

    n_ok = bc[BODY_COMP_FEATURES].notna().all(axis=1).sum()
    print(f"  Body comp: {n_ok}/{len(bc)} субъектов без пропусков")
    return bc


def _join_body_comp(df: pd.DataFrame, bc: pd.DataFrame) -> pd.DataFrame:
    """Присоединяет статические BC-признаки к windows датафрейму."""
    return df.merge(bc[["subject_id"] + BODY_COMP_FEATURES],
                    on="subject_id", how="left")


# ─── Feature engineering (идентично v0011) ───────────────────────────────────

def _get_emg_raw_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(_EMG_RAW_PREFIX)]


def _add_subject_z(df: pd.DataFrame, cols: list[str],
                   prefix: str = "z_") -> tuple[pd.DataFrame, list[str]]:
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
        rmin = np.minimum.accumulate(np.where(np.isfinite(hhb), hhb, np.inf))
        rmax = np.where(np.isinf(rmax), np.nan, rmax)
        rmin = np.where(np.isinf(rmin), np.nan, rmin)
        df.loc[idx, "smo2_from_running_max"] = rmax - smo2
        df.loc[idx, "hhb_from_running_min"]  = hhb - rmin
        if not sp_idx.empty and subj in sp_idx.index:
            b = float(sp_idx.loc[subj, "nirs_smo2_baseline_mean"])
            if np.isfinite(b) and b > 0:
                df.loc[idx, "smo2_rel_drop_pct"] = (b - smo2) / b * 100.0
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    df["feat_smo2_x_rr"]  = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
    df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    df["feat_rr_per_watt"] = df["hrv_mean_rr_ms"] / pw
    return df


def prepare_data(df_raw: pd.DataFrame,
                 session_params: pd.DataFrame,
                 target: str,
                 bc: pd.DataFrame) -> pd.DataFrame:
    """Базовая подготовка + присоединение BC-признаков."""
    if target == "lt2":
        df = df_raw[df_raw["window_valid_all_required"] == 1].copy()
        df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    else:
        df = df_raw[df_raw["target_time_to_lt1_usable"] == 1].copy()

    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)

    emg_raw = _get_emg_raw_cols(df)
    df, _ = _add_subject_z(df, emg_raw, prefix="z_")

    kin_present = [c for c in KINEMATICS_FEATURES if c in df.columns]
    df, _ = _add_subject_z(df, kin_present, prefix="z_")

    df = _add_running_nirs(df, session_params)

    if "trainred_smo2_mean" in df.columns and "hrv_mean_rr_ms" in df.columns:
        df = _add_interactions(df)

    df = _join_body_comp(df, bc)

    return df


def get_feature_cols(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Возвращает список признаков для заданного набора."""
    all_cols = set(df.columns)

    emg_cols = [c for c in df.columns if c.startswith("z_vl_")]
    kin_cols  = [c for c in df.columns
                 if c.startswith("z_") and not c.startswith("z_vl_")]
    nirs_cols  = [c for c in NIRS_FEATURES if c in all_cols]
    run_nirs   = [c for c in RUNNING_NIRS_FEATURES if c in all_cols]
    hrv_cols   = [c for c in HRV_FEATURES if c in all_cols]
    inter_cols = [c for c in INTERACTION_FEATURES if c in all_cols]
    bc_cols    = [c for c in BODY_COMP_FEATURES if c in all_cols]

    sets = {
        # Оригинальные (без BC) — для сравнения с v0011
        "EMG":          emg_cols + kin_cols,
        "NIRS":         nirs_cols + run_nirs,
        "EMG+NIRS":     emg_cols + kin_cols + nirs_cols + run_nirs,
        "EMG+NIRS+HRV": emg_cols + kin_cols + nirs_cols + run_nirs + hrv_cols + inter_cols,
        # Новые с BC
        "BodyComp":         bc_cols,
        "EMG+BC":           emg_cols + kin_cols + bc_cols,
        "EMG+NIRS+BC":      emg_cols + kin_cols + nirs_cols + run_nirs + bc_cols,
        "EMG+NIRS+HRV+BC":  emg_cols + kin_cols + nirs_cols + run_nirs + hrv_cols + inter_cols + bc_cols,
    }

    if feature_set not in sets:
        raise ValueError(f"Неизвестный набор: {feature_set}. Доступны: {list(sets)}")

    return [c for c in sets[feature_set] if c in all_cols]


# ─── LOSO ─────────────────────────────────────────────────────────────────────

def loso_predict(df: pd.DataFrame,
                 feat_cols: list[str],
                 target_col: str,
                 model_factory) -> dict:
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


def _loso_to_per_subject(df: pd.DataFrame,
                          feat_cols: list[str],
                          target_col: str,
                          model_factory) -> dict[str, dict]:
    """LOSO → dict[subject_id → {t_sec, y_pred, y_true}] для honest baselines."""
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


# ─── Kalman ───────────────────────────────────────────────────────────────────

def kalman_smooth(y_pred: np.ndarray,
                  sigma_p: float = 15.0,
                  sigma_obs: float = 150.0) -> np.ndarray:
    n = len(y_pred)
    x = y_pred[0]
    p = sigma_obs ** 2
    smoothed = np.empty(n)
    dt = 5.0
    for i in range(n):
        x -= dt
        p += sigma_p ** 2
        k = p / (p + sigma_obs ** 2)
        x = x + k * (y_pred[i] - x)
        p = (1 - k) * p
        smoothed[i] = x
    return smoothed


def apply_kalman_loso(loso_result: dict,
                      df: pd.DataFrame,
                      sigma_p: float = 15.0,
                      sigma_obs: float = 150.0) -> float:
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
    return mean_absolute_error(np.concatenate(trues_k),
                                np.concatenate(preds_k)) / 60.0


# ─── Зоопарк ──────────────────────────────────────────────────────────────────

def _run_one(cfg: dict, df: pd.DataFrame, feat_cols: list[str],
             target_col: str, feature_set: str, target_name: str,
             n_subj: int, n_feat: int, sigma_p: float, sigma_obs: float) -> dict:
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


def run_zoo(df: pd.DataFrame, feat_cols: list[str], target_col: str,
            feature_set: str, target_name: str, zoo: list[dict],
            sigma_p: float = 15.0, sigma_obs: float = 150.0,
            n_jobs: int = -1) -> list[dict]:
    n_subj = df["subject_id"].nunique()
    n_feat = len(feat_cols)
    print(f"\n  [{feature_set} / {target_name}]  n={n_subj}, {n_feat} признаков, "
          f"{len(zoo)} конфигов...")

    records = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_run_one)(cfg, df, feat_cols, target_col,
                          feature_set, target_name, n_subj, n_feat,
                          sigma_p, sigma_obs)
        for cfg in zoo
    )

    for r in sorted(records, key=lambda x: x.get("kalman_mae_min") or 99):
        if r.get("_error"):
            print(f"    {r['model']:<30s}  ОШИБКА: {r['_error']}")
        else:
            print(f"    {r['model']:<30s}  raw={r['raw_mae_min']:.3f}  "
                  f"kalman={r['kalman_mae_min']:.3f}  ({r['sec']:.1f}s)")

    return records


# ─── Per-subject Δ MAE ────────────────────────────────────────────────────────

def compute_per_subject_delta(loso_base: dict | None,
                               loso_bc: dict | None,
                               df: pd.DataFrame,
                               feature_set_base: str,
                               feature_set_bc: str,
                               target_name: str) -> pd.DataFrame | None:
    """Вычисляет per-subject MAE и Δ = bc − base (отрицательное = лучше)."""
    if loso_base is None or loso_bc is None:
        return None

    rows = []
    subjects = sorted(df["subject_id"].unique())
    for s in subjects:
        mask_b = loso_base["subjects"] == s
        mask_c = loso_bc["subjects"] == s
        if not mask_b.any() or not mask_c.any():
            continue
        mae_b = mean_absolute_error(loso_base["y_true"][mask_b],
                                     loso_base["y_pred"][mask_b]) / 60.0
        mae_c = mean_absolute_error(loso_bc["y_true"][mask_c],
                                     loso_bc["y_pred"][mask_c]) / 60.0
        rows.append({
            "subject_id":   s,
            "target":       target_name,
            "set_base":     feature_set_base,
            "set_bc":       feature_set_bc,
            "mae_base_min": round(mae_b, 4),
            "mae_bc_min":   round(mae_c, 4),
            "delta_min":    round(mae_c - mae_b, 4),
            "improved":     mae_c < mae_b,
        })
    return pd.DataFrame(rows)


# ─── Визуализация Δ MAE ──────────────────────────────────────────────────────

def plot_delta_bar(delta_df: pd.DataFrame, out_dir: Path) -> None:
    """Bar-chart per-subject Δ MAE для каждой пары (base → BC)."""
    if delta_df is None or delta_df.empty:
        return

    pairs = delta_df.groupby(["set_base", "set_bc", "target"])
    for (s_base, s_bc, tgt), grp in pairs:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["#d62728" if d > 0 else "#2ca02c" for d in grp["delta_min"]]
        ax.bar(grp["subject_id"], grp["delta_min"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Per-subject Δ MAE: {s_bc} − {s_base} / {tgt.upper()}", fontsize=12)
        ax.set_xlabel("Субъект")
        ax.set_ylabel("ΔMAE (мин)  [отриц. = лучше с BC]")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fname = f"delta_{s_bc.replace('+', '_')}_{tgt}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ─── Отчёт ────────────────────────────────────────────────────────────────────

def write_report(summary: list[dict], best: pd.DataFrame,
                 out_dir: Path, honest_md: str = "",
                 v0011_best: pd.DataFrame | None = None) -> None:
    lines = [
        "# v0012 — Body Composition: отчёт",
        "",
        "## Лучшие модели по (набор признаков, таргет)",
        "",
        best[["feature_set", "target", "model",
              "raw_mae_min", "kalman_mae_min", "r2", "rho"]].to_markdown(index=False),
        "",
    ]

    if v0011_best is not None:
        lines += [
            "## Сравнение с v0011",
            "",
            "| Набор | Таргет | MAE v0011 | MAE v0012 | Δ MAE |",
            "|---|---|---|---|---|",
        ]
        for _, row in best.iterrows():
            fs = row["feature_set"]
            tgt = row["target"]
            v11 = v0011_best.loc[
                (v0011_best["feature_set"] == fs) &
                (v0011_best["target"] == tgt), "kalman_mae_min"]
            if v11.empty:
                continue
            v11_val = float(v11.iloc[0])
            delta = row["kalman_mae_min"] - v11_val
            sign = "+" if delta > 0 else ""
            lines.append(f"| {fs} | {tgt} | {v11_val:.3f} | {row['kalman_mae_min']:.3f} "
                          f"| {sign}{delta:.3f} |")
        lines += [""]

    if honest_md:
        lines += ["## Честные baselines (MeanPredictor, FirstWindowPredictor)", "", honest_md]

    lines += [
        "## Защита от переобучения",
        "",
        "- Статические BC-признаки одинаковы для всех окон субъекта.",
        "- При N=14 (LT2) каждый субъект в тесте видит BC-вектор, не участвовавший в обучении,",
        "  но корреляция BC с LT может быть подобрана на 13 субъектах → переобучение.",
        "- Ориентир: gap (raw − FirstWindow) для BC-наборов не должен ухудшаться vs v0011.",
        "- Per-subject Δ MAE: если улучшение на 1–2 субъектах — не надёжно.",
        "",
    ]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Отчёт: {report_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target", nargs="+", default=["lt2", "lt1"],
                   choices=["lt2", "lt1"])
    p.add_argument("--feature-set", nargs="+", dest="feature_set",
                   default=["BodyComp", "EMG+BC", "EMG+NIRS+BC", "EMG+NIRS+HRV+BC",
                             "EMG", "EMG+NIRS", "EMG+NIRS+HRV"])
    p.add_argument("--sigma-p",   type=float, default=15.0)
    p.add_argument("--sigma-obs", type=float, default=150.0)
    p.add_argument("--no-delta",  action="store_true",
                   help="Пропустить вычисление per-subject Δ MAE")
    p.add_argument("--no-honest", action="store_true",
                   help="Пропустить честные baselines")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = OUT_DIR / "viz"
    viz_dir.mkdir(exist_ok=True)

    dataset_dir = DEFAULT_DATASET_DIR
    print(f"Датасет: {dataset_dir}")

    # ── Загрузка данных ──
    df_raw = pd.read_parquet(dataset_dir / "merged_features_ml.parquet")
    session_params_path = dataset_dir / "session_params.parquet"
    session_params = (pd.read_parquet(session_params_path)
                      if session_params_path.exists() else pd.DataFrame())

    bc = _load_body_comp(dataset_dir)
    bc_ok = bc[BODY_COMP_FEATURES].notna().all(axis=1).sum()
    print(f"  Субъекты с полными BC-данными: {bc_ok}/{len(bc)}")

    # ── v0011 лучшее (для сравнения Δ) ──
    v0011_best_path = _ROOT / "results" / "v0011" / "best_per_set.csv"
    v0011_best = pd.read_csv(v0011_best_path) if v0011_best_path.exists() else None

    zoo = _build_zoo()
    all_records: list[dict] = []
    honest_blocks: list[str] = []
    all_delta_dfs: list[pd.DataFrame] = []

    TARGET_COLS = {
        "lt2": "target_time_to_lt2_center_sec",
        "lt1": "target_time_to_lt1_sec",
    }

    for target in args.target:
        target_col = TARGET_COLS[target]
        print(f"\n{'='*60}")
        print(f"ТАРГЕТ: {target.upper()} / {target_col}")

        df_prep = prepare_data(df_raw, session_params, target, bc)
        df_prep = df_prep.dropna(subset=[target_col])

        # Словарь: feature_set → лучший _loso для per-subject Δ
        best_losos: dict[str, dict | None] = {}

        for fs in args.feature_set:
            try:
                feat_cols = get_feature_cols(df_prep, fs)
            except ValueError as e:
                print(f"  Пропуск {fs}: {e}")
                continue

            if not feat_cols:
                print(f"  Пропуск {fs}: нет признаков")
                continue

            records = run_zoo(df_prep, feat_cols, target_col,
                              fs, target, zoo,
                              args.sigma_p, args.sigma_obs)
            all_records.extend(records)

            # Сохраняем _loso лучшего для per-subject Δ
            valid = [r for r in records if r.get("_loso") is not None
                     and not np.isnan(r.get("kalman_mae_min", np.nan))]
            if valid:
                best_rec = min(valid, key=lambda r: r["kalman_mae_min"])
                best_losos[fs] = best_rec["_loso"]

                # Честные baselines
                if not args.no_honest:
                    best_factory_name = best_rec["model"]
                    best_cfg = next((c for c in zoo if c["name"] == best_factory_name), None)
                    if best_cfg is not None:
                        raw_ps = _loso_to_per_subject(
                            df_prep, feat_cols, target_col, best_cfg["factory"])
                        hb = run_honest_baselines(
                            df_prep, feat_cols, target_col,
                            raw_per_subject=raw_ps,
                            model_factory=best_cfg["factory"],
                            kalman_fn=kalman_smooth,
                            sigma_p=args.sigma_p,
                            sigma_obs_ref=150.0,
                        )
                        honest_blocks.append(format_honest_block(hb, fs, target))

                        # Сохраняем в CSV
                        hb_row = {
                            "feature_set": fs, "target": target,
                            "mean_pred_mae": hb["mean_pred"]["mae_min"],
                            "fw_mae": hb["first_win"]["mae_min"],
                            "raw_mae": hb["raw"]["mae_min"],
                            "kalman_ref_mae": hb["kalman_ref"]["mae_min"],
                            "gap_raw_vs_fw": hb["gap_raw_vs_fw"],
                            "verdict": hb["verdict"],
                            "stability_raw": hb["stability_raw"],
                            "stability_fw": hb["stability_fw"],
                        }
                        hb_path = OUT_DIR / "honest_baselines.csv"
                        hb_df = pd.DataFrame([hb_row])
                        if hb_path.exists():
                            old = pd.read_csv(hb_path)
                            hb_df = pd.concat([old, hb_df], ignore_index=True)
                        hb_df.to_csv(hb_path, index=False)
                        print(f"\n  [Честные baselines] gap={hb['gap_raw_vs_fw']:+.3f} мин  "
                              f"{hb['verdict']}")
            else:
                best_losos[fs] = None

        # Per-subject Δ MAE: базовые наборы vs BC-версии
        if not args.no_delta:
            pairs = [
                ("EMG",          "EMG+BC"),
                ("EMG+NIRS",     "EMG+NIRS+BC"),
                ("EMG+NIRS+HRV", "EMG+NIRS+HRV+BC"),
            ]
            for base_fs, bc_fs in pairs:
                if base_fs in best_losos and bc_fs in best_losos:
                    delta_df = compute_per_subject_delta(
                        best_losos[base_fs], best_losos[bc_fs],
                        df_prep, base_fs, bc_fs, target)
                    if delta_df is not None:
                        all_delta_dfs.append(delta_df)

    # ── Сводные таблицы ──
    if not all_records:
        print("Нет результатов. Завершение.")
        return

    df_sum = pd.DataFrame([{k: v for k, v in r.items() if k != "_loso"}
                            for r in all_records])
    df_sum.to_csv(OUT_DIR / "summary.csv", index=False)

    best_rows = []
    for (fs, tgt), grp in df_sum.groupby(["feature_set", "target"]):
        valid = grp.dropna(subset=["kalman_mae_min"])
        if valid.empty:
            continue
        best_rows.append(valid.loc[valid["kalman_mae_min"].idxmin()])
    best = pd.DataFrame(best_rows)
    best.to_csv(OUT_DIR / "best_per_set.csv", index=False)

    print("\n\n=== ЛУЧШИЕ РЕЗУЛЬТАТЫ ===")
    print(best[["feature_set", "target", "model",
                "raw_mae_min", "kalman_mae_min"]].to_string(index=False))

    # Per-subject Δ
    if all_delta_dfs:
        delta_all = pd.concat(all_delta_dfs, ignore_index=True)
        delta_all.to_csv(OUT_DIR / "body_comp_delta.csv", index=False)
        print("\n=== Per-subject Δ MAE (BC − base, отриц. = лучше) ===")
        print(delta_all[["subject_id", "target", "set_base", "set_bc",
                          "mae_base_min", "mae_bc_min", "delta_min", "improved"]].to_string(index=False))
        plot_delta_bar(delta_all, viz_dir)

    # Отчёт
    write_report(
        [r for r in all_records if not r.get("_error")],
        best,
        OUT_DIR,
        honest_md="\n\n".join(honest_blocks),
        v0011_best=v0011_best,
    )


if __name__ == "__main__":
    main()
