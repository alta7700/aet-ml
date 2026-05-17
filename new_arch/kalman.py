"""Kalman постпроцессинг для монотонизации предсказаний.

Самодостаточная копия из scripts/v0011_modality_ablation.py.
"""

from __future__ import annotations

import numpy as np


def kalman_smooth(y_pred: np.ndarray,
                  sigma_p: float = 15.0,
                  sigma_obs: float = 150.0,
                  dt: float = 5.0) -> np.ndarray:
    """Одномерный фильтр Калмана для монотонизации предсказания.

    Модель: x[t] = x[t-1] - dt (время до порога убывает).
    dt = шаг окна, по умолчанию 5 с.
    """
    n = len(y_pred)
    x = y_pred[0]
    p = sigma_obs ** 2
    smoothed = np.empty(n)

    for i in range(n):
        x -= dt
        p += sigma_p ** 2
        k = p / (p + sigma_obs ** 2)
        x = x + k * (y_pred[i] - x)
        p = (1 - k) * p
        smoothed[i] = x

    return smoothed
