"""Conformal prediction intervals для LT-time predictions.

Метод: leave-one-subject-out conformal (Jackknife+, Barber et al. 2021).
Для каждого test-subject S интервал на LT-time строится по абсолютным
ошибкам остальных 17 субъектов; благодаря тому что у нас LOSO и субъекты
exchangeable, это даёт marginal coverage guarantee.

Finite-sample correction
========================
Стандартный split conformal берёт ``q = quantile(errors, 1 - α)``. Для
маленьких N эту цифру корректируют:

    rank = ceil((n_cal + 1) · (1 - α)) / n_cal

При n_cal=17:
  α=0.20  →  rank = ceil(18·0.8)/17 = 15/17 ≈ 0.882
  α=0.30  →  rank = ceil(18·0.7)/17 = 13/17 ≈ 0.765
  α=0.10  →  rank = ceil(18·0.9)/17 = 17/17 = 1.0  →  q = max(errors)
  α=0.05  →  rank > 1 → invalid, нужно ещё больше калибровочной выборки

То есть 95%-интервал на N=18 субъектах = max-error, а 90% = max-error.
Поэтому ``cfg.conformal_alphas`` стартово хранит только {0.2, 0.3}.
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from analysis.schemas import (
    AnalysisConfig,
    CONFORMAL_INTERVAL_COLUMNS,
    CONFORMAL_SUMMARY_COLUMNS,
    LT_POLICIES,
)


# ─── Policy → колонки lt_point_metrics ──────────────────────────────────────

_POLICY_COLS = {
    "median": ("lt_hat_median_sec", "abs_lt_err_median_sec"),
    "crossing": ("lt_hat_crossing_sec", "abs_lt_err_crossing_sec"),
    "stable_median": ("lt_hat_stable_median_sec", "abs_lt_err_stable_median_sec"),
}


# ─── Core conformal helpers ─────────────────────────────────────────────────

def _qhat_jackknife(errors: np.ndarray, alpha: float) -> tuple[float, float]:
    """Jackknife+ q_hat по N калибровочным ошибкам.

    Возвращает (q_hat, фактический quantile_used). Если требуемый rank > 1
    (т.е. требуется больше точек чем есть) — возвращает max(errors) и
    quantile=1.0; вызывающая сторона должна это явно отрепортить.
    """
    n = len(errors)
    if n == 0:
        return float("nan"), float("nan")
    rank = math.ceil((n + 1) * (1.0 - alpha)) / n
    if rank > 1.0:
        return float(np.max(errors)), 1.0
    q = float(np.quantile(errors, rank, method="higher"))
    return q, float(rank)


def _bootstrap_coverage_ci(covered: np.ndarray, *, B: int,
                            seed: int, alpha_ci: float = 0.05
                            ) -> tuple[float, float]:
    """Bootstrap percentile CI для empirical coverage по субъектам."""
    n = len(covered)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = covered[idx].mean(axis=1)
    return (
        float(np.percentile(boots, 100 * alpha_ci / 2)),
        float(np.percentile(boots, 100 * (1 - alpha_ci / 2))),
    )


# ─── Layer-уровни ───────────────────────────────────────────────────────────

def build_conformal_intervals(lt_point_metrics: pd.DataFrame,
                              cfg: AnalysisConfig,
                              alphas: Iterable[float] | None = None
                              ) -> pd.DataFrame:
    """Строит таблицу интервалов (model_id × subject_id × policy × alpha).

    Для каждой модели независимо: внутри неё leave-one-subject-out по
    ошибкам, q_hat считается по оставшимся.
    """
    if alphas is None:
        alphas = cfg.conformal_alphas

    rows: list[dict] = []
    for model_id, g_m in lt_point_metrics.groupby("model_id", observed=True):
        if len(g_m) < 3:
            warnings.warn(
                f"{model_id}: только {len(g_m)} субъектов — conformal пропущен")
            continue
        arch = str(g_m["architecture_id"].iloc[0])
        target = str(g_m["target"].iloc[0])
        for policy in LT_POLICIES:
            hat_col, err_col = _POLICY_COLS[policy]
            if hat_col not in g_m.columns:
                continue
            errs = g_m[err_col].to_numpy(dtype=float)
            hats = g_m[hat_col].to_numpy(dtype=float)
            trues = g_m["lt_true_sec"].to_numpy(dtype=float)
            subjects = g_m["subject_id"].to_numpy()
            for alpha in alphas:
                nominal = 1.0 - alpha
                for i, sid in enumerate(subjects):
                    cal_mask = np.arange(len(errs)) != i
                    cal_errs = errs[cal_mask]
                    qhat, rank = _qhat_jackknife(cal_errs, alpha)
                    if not np.isfinite(qhat):
                        continue
                    lo = float(hats[i] - qhat)
                    hi = float(hats[i] + qhat)
                    covered = bool(lo <= trues[i] <= hi)
                    rows.append({
                        "model_id": model_id,
                        "architecture_id": arch,
                        "target": target,
                        "subject_id": str(sid),
                        "policy": policy,
                        "alpha": float(alpha),
                        "nominal_coverage": nominal,
                        "lt_true_sec": float(trues[i]),
                        "lt_hat_sec": float(hats[i]),
                        "abs_lt_err_sec": float(errs[i]),
                        "qhat_sec": qhat,
                        "qhat_quantile_used": rank,
                        "interval_low_sec": lo,
                        "interval_high_sec": hi,
                        "interval_width_sec": float(2.0 * qhat),
                        "covered": covered,
                        "calibration_n": int(cal_mask.sum()),
                        "qhat_method": "jackknife_plus",
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[[c for c in CONFORMAL_INTERVAL_COLUMNS if c in df.columns]]
    cfg.conformal_intervals_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.conformal_intervals_path, index=False)
    return df


def build_conformal_summary(intervals_df: pd.DataFrame,
                            subject_metrics: pd.DataFrame,
                            cfg: AnalysisConfig) -> pd.DataFrame:
    """Сводка: одна строка на (model_id, target, policy, alpha)."""
    if intervals_df.empty:
        empty = pd.DataFrame(columns=CONFORMAL_SUMMARY_COLUMNS)
        cfg.conformal_summary_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(cfg.conformal_summary_path, index=False)
        return empty

    # Сначала посчитаем mae_mean per (model_id, target) — для нормирования
    # interval_width / (2·MAE).
    mae_by_model = (
        subject_metrics.groupby(["model_id", "target"], observed=True)["mae"]
        .mean()
        .rename("mae_mean")
        .reset_index()
    )

    rows: list[dict] = []
    grp_cols = ["model_id", "architecture_id", "target", "policy", "alpha"]
    for keys, g in intervals_df.groupby(grp_cols, observed=True):
        model_id, arch, target, policy, alpha = keys
        nominal = 1.0 - float(alpha)
        covered = g["covered"].to_numpy(dtype=float)
        emp = float(covered.mean())
        lo, hi = _bootstrap_coverage_ci(
            covered, B=cfg.conformal_bootstrap_n, seed=cfg.conformal_bootstrap_seed)
        widths = g["interval_width_sec"].to_numpy(dtype=float)
        mae_mean = mae_by_model[
            (mae_by_model["model_id"] == model_id)
            & (mae_by_model["target"] == target)
        ]["mae_mean"]
        denom = float(mae_mean.iloc[0]) if not mae_mean.empty else float("nan")
        rows.append({
            "model_id": str(model_id),
            "architecture_id": str(arch),
            "target": str(target),
            "policy": str(policy),
            "alpha": float(alpha),
            "nominal_coverage": nominal,
            "empirical_coverage": emp,
            "coverage_gap": emp - nominal,
            "coverage_ci_low": lo,
            "coverage_ci_high": hi,
            "mean_interval_width_sec": float(np.mean(widths)),
            "median_interval_width_sec": float(np.median(widths)),
            "interval_width_over_2mae": (
                float(np.mean(widths) / (2.0 * denom))
                if denom and denom > 0 else float("nan")
            ),
            "calibration_n": int(g["calibration_n"].iloc[0]),
            "qhat_method": str(g["qhat_method"].iloc[0]),
        })

    df = pd.DataFrame(rows, columns=CONFORMAL_SUMMARY_COLUMNS)
    cfg.conformal_summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.conformal_summary_path, index=False)
    return df
