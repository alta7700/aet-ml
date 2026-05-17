"""Feature engineering для new_arch.

Самодостаточная копия логики из scripts/v0011_modality_ablation.py:
  • per-subject z-нормировка EMG/кинематики по baseline (stage_index=0);
  • running NIRS (накопленные от начала теста);
  • interaction-признаки (NIRS × HRV);
  • prepare_data(df_raw, session_params, target) — общая подготовка;
  • get_feature_cols(df, feature_set) — выбор набора признаков.

Списки модальностей (EMG, NIRS, HRV, EMG+NIRS, EMG+NIRS+HRV) и константа
EXCLUDE_ABS совместимы с v0011 (см. scripts/v0011_modality_ablation.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Абсолютные признаки — исключаются в варианте noabs (with_abs=False).
EXCLUDE_ABS: frozenset[str] = frozenset([
    "trainred_smo2_mean", "trainred_hhb_mean",
    "trainred_hbdiff_mean", "trainred_thb_mean",
    "hrv_mean_rr_ms", "feat_smo2_x_rr",
])

# Сырые ЭМГ-колонки (до z-norm) — 68 штук.
_EMG_RAW_PREFIX = "vl_"

# Кинематические признаки (тот же сенсорный блок, что ЭМГ).
KINEMATICS_FEATURES: list[str] = [
    "cadence_mean_rpm", "cadence_cv",
    "load_duration_ms", "rest_duration_ms", "load_rest_ratio",
    "load_duration_cv", "rest_duration_cv",
    "load_trend_cv_ratio_30s", "rest_trend_cv_ratio_30s",
    "load_sampen_30s", "rest_sampen_30s",
]

# NIRS — Train.Red, 15 признаков.
NIRS_FEATURES: list[str] = [
    "trainred_smo2_mean", "trainred_smo2_drop", "trainred_smo2_slope",
    "trainred_smo2_std", "trainred_dsmo2_dt",
    "trainred_hhb_mean", "trainred_hhb_slope", "trainred_hhb_std", "trainred_dhhb_dt",
    "trainred_hbdiff_mean", "trainred_hbdiff_slope", "trainred_hbdiff_std",
    "trainred_thb_mean", "trainred_thb_slope", "trainred_thb_std",
]

# Running NIRS — накопленные от начала теста (вычисляются динамически).
RUNNING_NIRS_FEATURES: list[str] = [
    "smo2_from_running_max",
    "hhb_from_running_min",
    "smo2_rel_drop_pct",
]

# HRV — 7 признаков из сырого RR (без QC-колонок).
HRV_FEATURES: list[str] = [
    "hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms",
    "hrv_sd1_ms", "hrv_sd2_ms", "hrv_sd1_sd2_ratio",
    "hrv_dfa_alpha1",
]

# Interaction (только когда есть и NIRS, и HRV).
INTERACTION_FEATURES: list[str] = ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"]


def _get_emg_raw_cols(df: pd.DataFrame) -> list[str]:
    """Возвращает все сырые vl_-колонки из датафрейма."""
    return [c for c in df.columns if c.startswith(_EMG_RAW_PREFIX)]


def _add_subject_z(df: pd.DataFrame, cols: list[str],
                   prefix: str = "z_") -> tuple[pd.DataFrame, list[str]]:
    """Per-subject z-нормировка по baseline (stage_index=0, 60 Вт).

    mean и std считаются только по окнам ступени покоя (stage_index=0),
    затем применяются ко всем окнам субъекта — без look-ahead.
    Если окон baseline нет, используется полная сессия (fallback).
    """
    df = df.copy()
    z_cols: list[str] = []

    baseline_mask = df["stage_index"] == 0
    baseline_df = df[baseline_mask]

    for col in cols:
        if col not in df.columns:
            continue

        if baseline_mask.any():
            stats = baseline_df.groupby("subject_id")[col].agg(["mean", "std"])
            m = df["subject_id"].map(stats["mean"])
            s = df["subject_id"].map(stats["std"])
        else:
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
        hhb = g["trainred_hhb_mean"].values
        rmax = np.maximum.accumulate(np.where(np.isfinite(smo2), smo2, -np.inf))
        rmin = np.minimum.accumulate(np.where(np.isfinite(hhb), hhb, np.inf))
        rmax = np.where(np.isinf(rmax), np.nan, rmax)
        rmin = np.where(np.isinf(rmin), np.nan, rmin)
        df.loc[idx, "smo2_from_running_max"] = rmax - smo2
        df.loc[idx, "hhb_from_running_min"] = hhb - rmin
        if not sp_idx.empty and subj in sp_idx.index:
            b = float(sp_idx.loc[subj, "nirs_smo2_baseline_mean"])
            if np.isfinite(b) and b > 0:
                df.loc[idx, "smo2_rel_drop_pct"] = (b - smo2) / b * 100.0
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет interaction-признаки (только для набора с HRV+NIRS)."""
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    df["feat_smo2_x_rr"] = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
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
    if target == "lt2":
        df = df_raw[df_raw["window_valid_all_required"] == 1].copy()
        df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    else:
        df = df_raw[df_raw["target_time_to_lt1_usable"] == 1].copy()

    df = df.sort_values(["subject_id", "window_start_sec"])

    emg_raw = _get_emg_raw_cols(df)
    df, _ = _add_subject_z(df, emg_raw, prefix="z_")

    kin_present = [c for c in KINEMATICS_FEATURES if c in df.columns]
    df, _ = _add_subject_z(df, kin_present, prefix="z_")

    df = _add_running_nirs(df, session_params)

    if "trainred_smo2_mean" in df.columns and "hrv_mean_rr_ms" in df.columns:
        df = _add_interactions(df)

    return df


def get_feature_cols(df: pd.DataFrame, feature_set: str,
                     with_abs: bool = True) -> list[str]:
    """Возвращает список колонок для заданного набора признаков.

    Поддерживаемые наборы: EMG, NIRS, HRV, EMG+NIRS, EMG+NIRS+HRV.
    Если with_abs=False, абсолютные признаки из EXCLUDE_ABS исключаются.
    """
    all_cols = set(df.columns)

    emg_cols = [c for c in df.columns if c.startswith("z_vl_")]
    kin_cols = [c for c in df.columns
                if c.startswith("z_") and not c.startswith("z_vl_")]

    nirs_cols = [c for c in NIRS_FEATURES if c in all_cols]
    run_nirs = [c for c in RUNNING_NIRS_FEATURES if c in all_cols]
    hrv_cols = [c for c in HRV_FEATURES if c in all_cols]
    inter_cols = [c for c in INTERACTION_FEATURES if c in all_cols]

    sets: dict[str, list[str]] = {
        "EMG": emg_cols + kin_cols,
        "NIRS": nirs_cols + run_nirs,
        "HRV": hrv_cols,
        "EMG+NIRS": emg_cols + kin_cols + nirs_cols + run_nirs,
        "EMG+NIRS+HRV": emg_cols + kin_cols + nirs_cols + run_nirs + hrv_cols + inter_cols,
    }

    if feature_set not in sets:
        raise ValueError(f"Неизвестный набор: {feature_set}. Доступны: {list(sets)}")

    cols = [c for c in sets[feature_set] if c in all_cols]
    if not with_abs:
        cols = [c for c in cols if c not in EXCLUDE_ABS]
    return cols
