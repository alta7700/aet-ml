"""Парные стат-тесты для subject-aligned сравнения моделей.

Главный инвариант: сравнение моделей m_a и m_b делается на одинаковом
наборе subject_id — vec_a и vec_b строятся inner-join'ом по subject_id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class PairedTestResult:
    n_subjects: int
    mean_a: float
    mean_b: float
    delta_mean: float
    median_a: float
    median_b: float
    delta_median: float
    wilcoxon_stat: float
    wilcoxon_pvalue: float
    ttest_stat: float
    ttest_pvalue: float
    cohens_d: float
    cliffs_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float


# ─── Effect sizes ──────────────────────────────────────────────────────────

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d для парных образцов (среднее разностей / sd разностей)."""
    if len(a) < 2:
        return float("nan")
    diff = a - b
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.mean(diff) / sd)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's δ ∈ [-1, 1]. Без зависимости от распределения."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a = np.asarray(a)
    b = np.asarray(b)
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    n = a.size * b.size
    return float((gt - lt) / n)


def bootstrap_paired_diff_ci(a: np.ndarray, b: np.ndarray, *, B: int,
                              ci: float, seed: int) -> tuple[float, float]:
    """Bootstrap percentile CI для mean(a) - mean(b) на парных данных."""
    if len(a) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    idx = rng.integers(0, n, size=(B, n))
    boots = diffs[idx].mean(axis=1)
    alpha = 1.0 - ci
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


# ─── Парные тесты ──────────────────────────────────────────────────────────

def paired_test(a: np.ndarray, b: np.ndarray, *, bootstrap_n: int = 1000,
                ci_level: float = 0.95, seed: int = 42) -> PairedTestResult:
    """Полный набор парных тестов и эффект-сайзов."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n != len(b):
        raise ValueError("a и b должны быть одной длины")

    mean_a = float(np.mean(a)) if n else float("nan")
    mean_b = float(np.mean(b)) if n else float("nan")
    med_a = float(np.median(a)) if n else float("nan")
    med_b = float(np.median(b)) if n else float("nan")

    if n >= 2 and not np.all(a == b):
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", method="auto")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        try:
            t = stats.ttest_rel(a, b)
            t_stat, t_p = float(t.statistic), float(t.pvalue)
        except ValueError:
            t_stat, t_p = float("nan"), float("nan")
    else:
        w_stat = w_p = t_stat = t_p = float("nan")

    d = cohens_d_paired(a, b)
    cd = cliffs_delta(a, b)
    lo, hi = bootstrap_paired_diff_ci(
        a, b, B=bootstrap_n, ci=ci_level, seed=seed)

    return PairedTestResult(
        n_subjects=n,
        mean_a=mean_a, mean_b=mean_b, delta_mean=mean_a - mean_b,
        median_a=med_a, median_b=med_b, delta_median=med_a - med_b,
        wilcoxon_stat=w_stat, wilcoxon_pvalue=w_p,
        ttest_stat=t_stat, ttest_pvalue=t_p,
        cohens_d=d, cliffs_delta=cd,
        bootstrap_ci_low=lo, bootstrap_ci_high=hi,
    )


# ─── subject-alignment ─────────────────────────────────────────────────────

def subject_align(metric_df: pd.DataFrame, model_a: str, model_b: str,
                   metric: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Возвращает (vec_a, vec_b, subjects) по общим subject_id двух моделей."""
    a = metric_df[metric_df["model_id"] == model_a][["subject_id", metric]]
    b = metric_df[metric_df["model_id"] == model_b][["subject_id", metric]]
    merged = a.merge(b, on="subject_id", suffixes=("_a", "_b"))
    return (
        merged[f"{metric}_a"].to_numpy(),
        merged[f"{metric}_b"].to_numpy(),
        merged["subject_id"].tolist(),
    )
