"""evaluate_all.py — Расширенная оценка всех версий моделей.

Не версионируется (не обучает, только оценивает), отслеживается через git.

Версии:
  v0001  ElasticNet(α=0.5, l1=0.9)           NIRS+HRV+inter+z_EMG / z_EMG+HRV
  v0002  ElasticNet + Kalman(σ_p=30,σ_o=120)  те же признаки + постобработка
  v0004  Ridge(α=100) / Huber(ε=1.35)         NIRS+HRV+inter+z_EMG / z_EMG+HRV+runNIRS
  v0006A Ridge/Huber + stage-0 norm            rr_pct_change вместо hrv_mean_rr_ms
  v0007  Ridge/Huber + CV-признаки             v0004 + CV амплитуды ЭМГ + CV ритмики

Артефакты:
  artefacts/v000X/lt2_acc_by_time.png    — Acc@δ(t_norm) с трендом и CI
  artefacts/v000X/lt2_tde.png            — Threshold Detection Error
  artefacts/comparison_lt2.png           — сравнение всех версий
  artefacts/comparison_lt1.png

Запуск:
  uv run python scripts/evaluate_all.py
  uv run python scripts/evaluate_all.py --target lt2
  uv run python scripts/evaluate_all.py --versions v0001 v0004
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from scripts.eval_utils import compute_all_metrics, plot_acc_by_time, plot_tde, plot_summary_comparison

ARTEFACTS_DIR = _ROOT / "artefacts"
DATASET_PATH = DEFAULT_DATASET_DIR / "merged_features_ml.parquet"


# ─── Общие наборы признаков ───────────────────────────────────────────────────

NIRS = [
    "trainred_smo2_mean", "trainred_smo2_drop", "trainred_smo2_slope",
    "trainred_smo2_std", "trainred_dsmo2_dt",
    "trainred_hhb_mean", "trainred_hhb_slope", "trainred_hhb_std", "trainred_dhhb_dt",
    "trainred_hbdiff_mean", "trainred_hbdiff_slope", "trainred_hbdiff_std",
    "trainred_thb_mean", "trainred_thb_slope", "trainred_thb_std",
]
HRV = [
    "hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms",
    "hrv_sd1_ms", "hrv_sd2_ms", "hrv_sd1_sd2_ratio", "hrv_dfa_alpha1",
]
HRV_SCALEFREE = ["hrv_sd1_sd2_ratio", "hrv_dfa_alpha1"]
Z_EMG = [
    "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
    "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
    "z_vl_prox_rest_rms", "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
]
INTER = ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"]
RUNNING_NIRS = ["smo2_from_running_max", "hhb_from_running_min", "smo2_rel_drop_pct"]
INTER_NORM = ["feat_smo2_x_rr_pct", "feat_rr_pct_per_watt", "feat_smo2_x_dfa"]

# Новые признаки вариабельности (v0007)
EMG_AMPL_CV = [
    "z_vl_dist_load_rms_cv", "z_vl_dist_rest_rms_cv",
    "z_vl_prox_load_rms_cv", "z_vl_prox_rest_rms_cv",
]
TIMING_CV = ["load_duration_cv", "rest_duration_cv"]


# ─── Feature engineering ─────────────────────────────────────────────────────

def _z_emg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [c for c in df.columns if c.startswith("vl_")]:
        m = df.groupby("subject_id")[col].transform("mean")
        s = df.groupby("subject_id")[col].transform("std")
        df[f"z_{col}"] = (df[col] - m) / (s + 1e-8)
    return df


def _interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    df["feat_smo2_x_rr"] = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
    df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    df["feat_rr_per_watt"] = df["hrv_mean_rr_ms"] / pw
    return df


def _running_nirs(df: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
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


_HRV_MS = ["hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_sd1_ms", "hrv_sd2_ms"]
_HRV_DELTA = ["hrv_dfa_alpha1"]
_NIRS_DELTA = ["trainred_smo2_mean", "trainred_hhb_mean", "trainred_hbdiff_mean", "trainred_thb_mean"]


def _stage0_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject baseline из первой ступени (stage_index==0) → delta и pct_change."""
    df = df.copy()
    s0 = df[df["stage_index"] == 0].groupby("subject_id")

    for col in _HRV_MS:
        if col not in df.columns:
            continue
        base = s0[col].mean().rename(f"_b_{col}")
        df = df.join(base, on="subject_id")
        df[f"{col}_pct_change"] = (df[col] - df[f"_b_{col}"]) / (df[f"_b_{col}"].abs() + 1e-8) * 100
        df[f"{col}_delta"] = df[col] - df[f"_b_{col}"]
        df = df.drop(columns=[f"_b_{col}"])

    for col in _HRV_DELTA:
        if col not in df.columns:
            continue
        base = s0[col].mean().rename(f"_b_{col}")
        df = df.join(base, on="subject_id")
        df[f"{col}_delta"] = df[col] - df[f"_b_{col}"]
        df = df.drop(columns=[f"_b_{col}"])

    for col in _NIRS_DELTA:
        if col not in df.columns:
            continue
        base = s0[col].mean().rename(f"_b_{col}")
        df = df.join(base, on="subject_id")
        df[f"{col}_delta"] = df[col] - df[f"_b_{col}"]
        df = df.drop(columns=[f"_b_{col}"])

    if "hrv_mean_rr_ms_pct_change" in df.columns:
        df["rr_slope_per_stage"] = df["hrv_mean_rr_ms_pct_change"] / (df["stage_index"] + 1)

    return df


def _interactions_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction на нормированных признаках (для v0006A)."""
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    if "hrv_mean_rr_ms_pct_change" in df.columns:
        df["feat_smo2_x_rr_pct"] = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms_pct_change"] / 1e2
        df["feat_rr_pct_per_watt"] = df["hrv_mean_rr_ms_pct_change"] / pw
    if "hrv_dfa_alpha1" in df.columns:
        df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    return df


# ─── Фабрики моделей ─────────────────────────────────────────────────────────

from sklearn.ensemble import GradientBoostingRegressor


def _elasticnet():
    return ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=5000, tol=1e-4,
                      fit_intercept=True, random_state=42)

def _ridge():
    return Ridge(alpha=100.0, fit_intercept=True, solver="auto")

def _huber():
    return HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500,
                          fit_intercept=True, warm_start=False)

def _gbm():
    return GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=3)


# ─── Kalman постобработка ─────────────────────────────────────────────────────

_KALMAN_SIGMA_PROCESS = 30.0   # сек — v0002: неопределённость перехода
_KALMAN_SIGMA_OBS = 120.0      # сек — v0002: доверие к модели


def _kalman_smooth(
    predictions_sec: np.ndarray,
    step_sec: float = 5.0,
    sigma_process: float = _KALMAN_SIGMA_PROCESS,
    sigma_obs: float = _KALMAN_SIGMA_OBS,
) -> np.ndarray:
    """1D Kalman-сглаживатель: τ_{t+1} = τ_t − step + w, z_t = τ_t + v."""
    n = len(predictions_sec)
    if n == 0:
        return predictions_sec.copy()
    Q = sigma_process ** 2
    R = sigma_obs ** 2
    tau = predictions_sec[0]
    P = R
    out = np.empty(n)
    out[0] = tau
    for t in range(1, n):
        tau_p = tau - step_sec
        P_p = P + Q
        K = P_p / (P_p + R)
        tau = tau_p + K * (predictions_sec[t] - tau_p)
        P = (1 - K) * P_p
        out[t] = tau
    return out


def apply_kalman_to_loso(
    loso_result: dict,
    sigma_process: float = _KALMAN_SIGMA_PROCESS,
    sigma_obs: float = _KALMAN_SIGMA_OBS,
) -> dict:
    """Применяет Kalman-сглаживание к результатам LOSO."""
    y_pred_raw = loso_result["y_pred"].copy()
    y_pred_smooth = y_pred_raw.copy()
    subj_all = loso_result["subjects"]
    elapsed_all = loso_result["elapsed"]

    for s in sorted(np.unique(subj_all)):
        idx = np.where(subj_all == s)[0]
        order = np.argsort(elapsed_all[idx])
        sorted_idx = idx[order]
        smoothed = _kalman_smooth(y_pred_raw[sorted_idx],
                                  sigma_process=sigma_process, sigma_obs=sigma_obs)
        y_pred_smooth[sorted_idx] = smoothed

    new = loso_result.copy()
    new["y_pred"] = y_pred_smooth
    new["mae_min"] = float(mean_absolute_error(loso_result["y_true"], y_pred_smooth)) / 60.0
    new["r2"] = float(r2_score(loso_result["y_true"], y_pred_smooth))
    new["rho"] = float(spearmanr(loso_result["y_true"], y_pred_smooth).statistic)
    return new


# ─── LOSO с elapsed_sec ───────────────────────────────────────────────────────

def loso_with_elapsed(df, features, target, model_factory) -> dict:
    subjects = sorted(df["subject_id"].unique())
    feat_cols = [f for f in features if f in df.columns]
    preds, trues, subjs, elapseds = [], [], [], []

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test = df[df["subject_id"] == test_s].sort_values("window_start_sec")
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        model = model_factory()
        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        model.fit(X_tr, train[target].values)
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        preds.append(model.predict(X_te))
        trues.append(test[target].values)
        subjs.append(np.full(len(test), test_s))
        elapseds.append(test["elapsed_sec"].values)

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    subjects_arr = np.concatenate(subjs)
    elapsed_arr = np.concatenate(elapseds)
    return {
        "y_true": y_true, "y_pred": y_pred,
        "subjects": subjects_arr, "elapsed": elapsed_arr,
        "mae_min": float(mean_absolute_error(y_true, y_pred)) / 60.0,
        "r2": float(r2_score(y_true, y_pred)),
        "rho": float(spearmanr(y_true, y_pred).statistic),
    }


# ─── Prepare functions ────────────────────────────────────────────────────────

def _base_lt2(df_full):
    df = df_full[df_full["window_valid_all_required"] == 1].copy()
    df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    return df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)


def _base_lt1(df_full):
    df = df_full[df_full["target_time_to_lt1_usable"] == 1].copy()
    df = df[df["nirs_valid"] == 1].copy()
    return df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)


def prep_lt2_v0001(df_full, sp):
    df = _base_lt2(df_full)
    df = _z_emg(df)
    df = _interactions(df)
    feats = NIRS + HRV + INTER + Z_EMG
    return df, feats, "target_time_to_lt2_center_sec"


def prep_lt1_v0001(df_full, sp):
    df = _base_lt1(df_full)
    df = _z_emg(df)
    feats = Z_EMG + HRV
    return df, feats, "target_time_to_lt1_sec"


def prep_lt2_v0004(df_full, sp):
    df = _base_lt2(df_full)
    df = _z_emg(df)
    df = _interactions(df)
    feats = NIRS + HRV + INTER + Z_EMG
    return df, feats, "target_time_to_lt2_center_sec"


def prep_lt1_v0004(df_full, sp):
    df = _base_lt1(df_full)
    df = _z_emg(df)
    if not sp.empty:
        df = _running_nirs(df, sp)
    feats = Z_EMG + HRV + RUNNING_NIRS
    return df, feats, "target_time_to_lt1_sec"


def prep_lt2_v0007(df_full, sp):
    """v0007: v0004 + лучший набор CV-признаков для LT2."""
    df = _base_lt2(df_full)
    df = _z_emg(df)
    df = _interactions(df)
    # Берём лучшую конфигурацию: базовый v0004 + все CV-признаки
    feats = NIRS + HRV + INTER + Z_EMG + EMG_AMPL_CV + TIMING_CV
    return df, feats, "target_time_to_lt2_center_sec"


def prep_lt1_v0007(df_full, sp):
    """v0007: лучший набор для LT1 — z_EMG + HRV + timing_CV (без running NIRS)."""
    df = _base_lt1(df_full)
    df = _z_emg(df)
    # Лучший по v0007_train: z_EMG+HRV+timing_CV (MAE=1.980, лучше running NIRS)
    feats = Z_EMG + HRV + TIMING_CV
    return df, feats, "target_time_to_lt1_sec"


def prep_lt2_v0006a(df_full, sp):
    """v0006 вариант A: hrv_mean_rr_ms → rr_pct_change + rr_delta."""
    df = _base_lt2(df_full)
    df = _stage0_baselines(df)
    df = _z_emg(df)
    df = _interactions_norm(df)
    feats = (NIRS + HRV_SCALEFREE
             + ["hrv_mean_rr_ms_pct_change", "hrv_mean_rr_ms_delta",
                "hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_sd1_ms", "hrv_sd2_ms"]
             + INTER_NORM + Z_EMG)
    return df, feats, "target_time_to_lt2_center_sec"


def prep_lt1_v0006a(df_full, sp):
    df = _base_lt1(df_full)
    df = _stage0_baselines(df)
    df = _z_emg(df)
    df = _interactions_norm(df)
    feats = (Z_EMG + HRV_SCALEFREE
             + ["hrv_mean_rr_ms_pct_change", "hrv_mean_rr_ms_delta",
                "hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_sd1_ms", "hrv_sd2_ms"])
    return df, feats, "target_time_to_lt1_sec"


SAMPEN_30S = ["load_sampen_30s", "rest_sampen_30s"]
SAMPEN_60S = ["load_sampen_60s", "rest_sampen_60s"]


def prep_lt2_v0008(df_full, sp):
    """v0008 LT2: лучшая конфигурация из v0008 — v0004-признаки + Ridge.
    SampEn не дал прироста для LT2 с текущим датасетом."""
    df = _base_lt2(df_full)
    df = _z_emg(df)
    df = _interactions(df)
    feats = NIRS + HRV + INTER + Z_EMG
    return df, feats, "target_time_to_lt2_center_sec"


def prep_lt1_v0008(df_full, sp):
    """v0008 LT1: SampEn 30s + z_EMG + HRV."""
    df = _base_lt1(df_full)
    df = _z_emg(df)
    feats = Z_EMG + HRV + SAMPEN_30S
    return df, feats, "target_time_to_lt1_sec"


def prep_lt2_v0009(df_full, sp):
    """v0009 LT2: те же признаки что v0008, Kalman σ_p=15с, σ_obs=150с."""
    return prep_lt2_v0008(df_full, sp)


def prep_lt1_v0009(df_full, sp):
    """v0009 LT1: те же признаки что v0008, Kalman σ_p=15с, σ_obs=150с."""
    return prep_lt1_v0008(df_full, sp)


def _kalman_v0009(loso_result: dict) -> dict:
    return apply_kalman_to_loso(loso_result, sigma_process=15.0, sigma_obs=150.0)


def _kalman_v0010_lt1(loso_result: dict) -> dict:
    """v0010 LT1: честный оптимум σ_p=15с, σ_obs=250с (Δ=-0.008 vs v0009)."""
    return apply_kalman_to_loso(loso_result, sigma_process=15.0, sigma_obs=250.0)


# ─── Реестр версий ────────────────────────────────────────────────────────────

VERSIONS: dict[str, dict] = {
    "v0001": {
        "lt2": (prep_lt2_v0001, _elasticnet, None),
        "lt1": (prep_lt1_v0001, _elasticnet, None),
        "label": "v0001\nElasticNet",
    },
    "v0002": {
        # Те же признаки что v0001, но с Kalman-постобработкой предсказаний
        "lt2": (prep_lt2_v0001, _elasticnet, "kalman"),
        "lt1": (prep_lt1_v0001, _elasticnet, "kalman"),
        "label": "v0002\nElasticNet\n+Kalman",
    },
    "v0004": {
        "lt2": (prep_lt2_v0004, _ridge, None),
        "lt1": (prep_lt1_v0004, _huber, None),
        "label": "v0004\nRidge/Huber",
    },
    "v0006A": {
        # Замена hrv_mean_rr_ms на stage-0 pct_change
        "lt2": (prep_lt2_v0006a, _ridge, None),
        "lt1": (prep_lt1_v0006a, _huber, None),
        "label": "v0006A\nstage-0 pct\n(без rr_ms)",
    },
    "v0007": {
        # v0004 + признаки вариабельности (CV амплитуды ЭМГ + CV ритмики)
        "lt2": (prep_lt2_v0007, _ridge, None),
        "lt1": (prep_lt1_v0007, _huber, None),
        "label": "v0007\nRidge/Huber\n+CV",
    },
    "v0008": {
        # LT2: v0004-признаки + Ridge (SampEn не дал прироста)
        # LT1: SampEn 30s + Huber
        "lt2": (prep_lt2_v0008, _ridge, None),
        "lt1": (prep_lt1_v0008, _huber, None),
        "label": "v0008\nSampEn\n+Ridge/Huber",
    },
    "v0009": {
        # Kalman σ_p=15с, σ_obs=150с поверх v0008-best
        "lt2": (prep_lt2_v0009, _ridge, _kalman_v0009),
        "lt1": (prep_lt1_v0009, _huber, _kalman_v0009),
        "label": "v0009\n+Kalman\nσ_p=15,σ_o=150",
    },
    "v0010": {
        # Честный оптимум расширенной сетки:
        # LT2: те же параметры что v0009 (σ_p=15, σ_obs=150)
        # LT1: σ_p=15, σ_obs=250 (Δ=-0.008 vs v0009, на уровне шума)
        "lt2": (prep_lt2_v0009, _ridge, _kalman_v0009),
        "lt1": (prep_lt1_v0009, _huber, _kalman_v0010_lt1),
        "label": "v0010\nKalman\next.grid",
    },
}


# ─── Оценка одной версии ─────────────────────────────────────────────────────

def evaluate_version(
    version: str,
    df_full: pd.DataFrame,
    sp: pd.DataFrame,
    task: str,
    no_plots: bool = False,
) -> dict | None:
    cfg = VERSIONS[version]
    prep_fn, factory_fn, postprocess = cfg[task]

    df, feats, target = prep_fn(df_full, sp)
    feat_cols = [f for f in feats if f in df.columns]
    if len(feat_cols) < 2:
        print(f"  {version}/{task.upper()}: пропуск (нет признаков)")
        return None

    print(f"  {version}/{task.upper()}: {df['subject_id'].nunique()} уч., "
          f"{len(df)} окон, {len(feat_cols)} признаков")

    loso_res = loso_with_elapsed(df, feat_cols, target, factory_fn)

    if postprocess == "kalman":
        loso_res = apply_kalman_to_loso(loso_res)
    elif callable(postprocess):
        loso_res = postprocess(loso_res)

    metrics = compute_all_metrics(
        loso_res["y_true"], loso_res["y_pred"],
        loso_res["elapsed"], loso_res["subjects"],
    )

    print(f"    MAE={metrics['mae_min']:.3f}  R²={loso_res['r2']:.3f}  ρ={loso_res['rho']:.3f}")
    print(f"    Acc@30s={metrics['acc_global'][30.0]:.1%}  "
          f"Acc@1min={metrics['acc_global'][60.0]:.1%}  "
          f"Acc@2min={metrics['acc_global'][120.0]:.1%}  "
          f"Acc@3min={metrics['acc_global'][180.0]:.1%}")
    print(f"    |TDE| mean={metrics['tde_mean_abs_min']:.2f} мин  "
          f"median={metrics['tde_median_abs_min']:.2f} мин")

    if not no_plots:
        out_dir = ARTEFACTS_DIR / version
        label_short = cfg["label"].replace("\n", ", ")
        plot_acc_by_time(
            metrics,
            title=f"Acc@δ по ходу теста — {version} {task.upper()} [{label_short}]",
            out_path=out_dir / f"{task}_acc_by_time.png",
        )
        plot_tde(
            metrics,
            title=f"Threshold Detection Error — {version} {task.upper()} [{label_short}]",
            out_path=out_dir / f"{task}_tde.png",
        )

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--versions", nargs="+", default=list(VERSIONS.keys()),
                   choices=list(VERSIONS.keys()))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("evaluate_all.py — Расширенные метрики по всем версиям")
    print("=" * 65)

    df_full = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    sp = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    tasks = []
    if args.target in ("lt2", "both"):
        tasks.append("lt2")
    if args.target in ("lt1", "both"):
        tasks.append("lt1")

    all_metrics: dict[str, dict[str, dict]] = {t: {} for t in tasks}

    for task in tasks:
        print(f"\n{'═' * 65}\n{task.upper()}\n{'═' * 65}")
        for version in args.versions:
            print(f"\n[{version}]")
            m = evaluate_version(version, df_full, sp, task, no_plots=args.no_plots)
            if m is not None:
                all_metrics[task][version] = m

    if not args.no_plots:
        for task in tasks:
            vm = all_metrics[task]
            if len(vm) >= 2:
                print(f"\nСравнительный график {task.upper()}...")
                renamed = {VERSIONS[v]["label"]: m for v, m in vm.items()}
                plot_summary_comparison(
                    renamed,
                    out_path=ARTEFACTS_DIR / f"comparison_{task}.png",
                    title=f"Сравнение версий — {task.upper()}",
                )

    print("\n" + "=" * 65)
    print("ИТОГ")
    print("=" * 65)
    for task in tasks:
        print(f"\n{task.upper()}:")
        print(f"  {'Версия':<10}  {'MAE':>7}  {'Acc@30s':>8}  "
              f"{'Acc@1min':>9}  {'Acc@2min':>9}  {'|TDE|':>7}")
        print("  " + "─" * 60)
        for v, m in all_metrics[task].items():
            print(f"  {v:<10}  {m['mae_min']:>7.3f}  "
                  f"{m['acc_global'][30.0]:>8.1%}  "
                  f"{m['acc_global'][60.0]:>9.1%}  "
                  f"{m['acc_global'][120.0]:>9.1%}  "
                  f"{m['tde_mean_abs_min']:>7.2f}")


if __name__ == "__main__":
    main()
