"""Чистые функции метрик качества регрессии.

Все принимают вектора ``y_true`` и ``y_pred`` одинаковой длины,
возвращают скаляр. NaN в input — ответственность вызывающей стороны.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя ошибка со знаком (y_pred - y_true)."""
    return float(np.mean(y_pred - y_true))


def std_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Стандартное отклонение остатков."""
    return float(np.std(y_pred - y_true, ddof=1)) if len(y_true) > 1 else 0.0


def max_abs_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation. NaN при вырожденном случае."""
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(stats.pearsonr(y_true, y_pred).statistic)


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    res = stats.spearmanr(y_true, y_pred)
    return float(res.statistic)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Коэффициент детерминации R^2.

    Если var(y_true) = 0 — возвращает NaN (метрика не определена).
    """
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def catastrophic_rate(y_true: np.ndarray, y_pred: np.ndarray,
                      threshold_sec: float) -> float:
    """Доля точек с |err| > threshold_sec."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred) > threshold_sec))


def compute_all_subject_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                threshold_sec: float) -> dict:
    """Считает весь набор subject-level метрик дорожки A одним вызовом."""
    return {
        "n_samples": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "std_err": std_err(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_r": spearman_r(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "max_abs_err": max_abs_err(y_true, y_pred),
        "catastrophic_rate": catastrophic_rate(y_true, y_pred, threshold_sec),
    }
