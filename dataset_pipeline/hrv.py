"""Признаки ВСР (вариабельность сердечного ритма) для датасета ML.

Реализует каузальный trailing-контекст 120 с: для каждого окна используются
RR-интервалы из диапазона [window_end - 120 с, window_end].

Переиспользует correct_rr_window() и dfa_alpha1() из methods.lt2 (без изменений).

Выходной файл: dataset/features_hrv.parquet (7 признаков + 4 QC-поля на окно).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from dataset_pipeline.common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_HRV_CONTEXT_SEC,
    load_subjects_table,
    load_windows_table,
    save_parquet,
)
from methods.lt2 import correct_rr_window, dfa_alpha1

# ─────────────────────── Константы ───────────────────────

# Канал RR в finaltest.h5
_RR_CHANNEL = "zephyr.rr"

# Анкорный канал для временны́х меток
_ANCHOR_CHANNEL = "channels/moxy.smo2/timestamps"

# Минимальное покрытие (доля от 120 с) для признания HRV-окна валидным
_MIN_HRV_COVERAGE = 0.8

# Максимально допустимая доля артефактных RR в окне
_MAX_ARTIFACT_FRACTION = 0.05

# RR-интервалы в ms: физиологические пределы
_RR_MIN_MS = 300.0
_RR_MAX_MS = 2000.0


# ─────────────────────── Загрузка RR ───────────────────────

def load_rr_signal(source_h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Загружает канал zephyr.rr из finaltest.h5.

    Параметры
    ----------
    source_h5_path : Path
        Путь к finaltest.h5.

    Возвращает
    ----------
    Кортеж (rr_times_sec, rr_values_sec):
    - rr_times_sec — метки времени в секундах от якоря
    - rr_values_sec — длительности RR-интервалов в секундах (channel хранит секунды)
    """
    with h5py.File(source_h5_path, "r") as handle:
        anchor_ms = float(handle[_ANCHOR_CHANNEL][0])
        raw_times = handle[f"channels/{_RR_CHANNEL}/timestamps"][:].astype(float)
        values_sec = handle[f"channels/{_RR_CHANNEL}/values"][:].astype(float)

    times_sec = (raw_times - anchor_ms) / 1000.0
    return times_sec, values_sec


# ─────────────────────── Извлечение признаков ───────────────────────

def extract_hrv_features(
    window_end_sec: float,
    rr_times_sec: np.ndarray,
    rr_values_sec: np.ndarray,
    hrv_context_sec: float = DEFAULT_HRV_CONTEXT_SEC,
) -> dict[str, object]:
    """Извлекает 7 ВСР-признаков + 4 QC-поля для одного окна.

    Использует trailing-контекст: [window_end - hrv_context_sec, window_end].
    Это гарантирует каузальность — признак не знает о будущих RR-интервалах.

    Параметры
    ----------
    window_end_sec : float
        Правая граница окна (конец каузального окна) в секундах.
    rr_times_sec : np.ndarray
        Метки времени RR-интервалов в секундах.
    rr_values_sec : np.ndarray
        RR-интервалы в секундах.
    hrv_context_sec : float
        Длина trailing-контекста в секундах (по умолчанию 120).
    """
    context_start = window_end_sec - hrv_context_sec
    context_end = window_end_sec

    mask = (rr_times_sec >= context_start) & (rr_times_sec <= context_end)
    rr_raw_ms = rr_values_sec[mask] * 1000.0  # секунды → миллисекунды

    # QC: покрытие (доля физиологически допустимых RR от ожидаемого числа за 120 с)
    # Ожидаемое число RR при среднем 60 уд/мин = 120, при 80 уд/мин ≈ 160
    # Используем простую оценку через длительность: сумма всех RR / context_sec
    valid_mask = np.isfinite(rr_raw_ms) & (rr_raw_ms >= _RR_MIN_MS) & (rr_raw_ms <= _RR_MAX_MS)
    total_rr_duration_sec = float(np.sum(rr_raw_ms[valid_mask])) / 1000.0
    hrv_coverage_fraction = float(min(1.0, total_rr_duration_sec / hrv_context_sec))

    nan_result: dict[str, object] = {
        **{k: float("nan") for k in _all_hrv_feature_names()},
        "hrv_coverage_fraction": hrv_coverage_fraction,
        "hrv_artifact_fraction": float("nan"),
        "hrv_valid": 0,
        "hrv_rr_count": int(np.sum(valid_mask)),
    }

    # Применяем коррекцию артефактов
    rr_ms_corrected, artifact_fraction = correct_rr_window(rr_raw_ms)

    hrv_valid = int(
        hrv_coverage_fraction >= _MIN_HRV_COVERAGE
        and artifact_fraction <= _MAX_ARTIFACT_FRACTION
    )

    if hrv_valid == 0:
        return {
            **{k: float("nan") for k in _all_hrv_feature_names()},
            "hrv_coverage_fraction": hrv_coverage_fraction,
            "hrv_artifact_fraction": float(artifact_fraction),
            "hrv_valid": 0,
            "hrv_rr_count": len(rr_ms_corrected),
        }

    # ─── Временны́е признаки ВСР ───
    mean_rr = float(np.mean(rr_ms_corrected))
    sdnn = float(np.std(rr_ms_corrected, ddof=1)) if len(rr_ms_corrected) > 1 else float("nan")
    rmssd = (
        float(np.sqrt(np.mean(np.diff(rr_ms_corrected) ** 2)))
        if len(rr_ms_corrected) > 1
        else float("nan")
    )

    # Признаки диаграммы Пуанкаре
    if np.isfinite(rmssd) and np.isfinite(sdnn):
        sd1 = rmssd / np.sqrt(2.0)
        sd2_sq = 2.0 * sdnn ** 2 - 0.5 * rmssd ** 2
        sd2 = float(np.sqrt(max(0.0, sd2_sq)))
        sd1_sd2_ratio = sd1 / sd2 if sd2 > 0 else float("nan")
    else:
        sd1 = float("nan")
        sd2 = float("nan")
        sd1_sd2_ratio = float("nan")

    # DFA α1 (каузальный, реиспользуем из methods.lt2)
    dfa_a1 = float(dfa_alpha1(rr_ms_corrected))

    return {
        "hrv_mean_rr_ms": mean_rr,
        "hrv_sdnn_ms": sdnn,
        "hrv_rmssd_ms": rmssd,
        "hrv_sd1_ms": float(sd1),
        "hrv_sd2_ms": sd2,
        "hrv_sd1_sd2_ratio": sd1_sd2_ratio,
        "hrv_dfa_alpha1": dfa_a1,
        "hrv_coverage_fraction": hrv_coverage_fraction,
        "hrv_artifact_fraction": float(artifact_fraction),
        "hrv_valid": hrv_valid,
        "hrv_rr_count": len(rr_ms_corrected),
    }


def _all_hrv_feature_names() -> list[str]:
    """Возвращает список всех 7 имён HRV-признаков."""
    return [
        "hrv_mean_rr_ms",
        "hrv_sdnn_ms",
        "hrv_rmssd_ms",
        "hrv_sd1_ms",
        "hrv_sd2_ms",
        "hrv_sd1_sd2_ratio",
        "hrv_dfa_alpha1",
    ]


# ─────────────────────── Сборка таблицы ───────────────────────

def build_hrv_table(
    subjects_path: Path,
    windows_path: Path,
    hrv_context_sec: float = DEFAULT_HRV_CONTEXT_SEC,
) -> pd.DataFrame:
    """Строит таблицу признаков HRV по всем участникам.

    Параметры
    ----------
    subjects_path : Path
        Путь к subjects.parquet.
    windows_path : Path
        Путь к windows.parquet.
    hrv_context_sec : float
        Длина trailing-контекста DFA/HRV в секундах (по умолчанию 120).
    """
    subjects = load_subjects_table(subjects_path)
    windows = load_windows_table(windows_path)

    rows: list[dict[str, object]] = []

    for subject_row in subjects.itertuples():
        subject_id = str(subject_row.subject_id)
        source_h5_path = Path(str(subject_row.source_h5_path))
        subject_windows = windows[windows["subject_id"] == subject_id]

        print(f"  Участник {subject_id}: загружаем RR...", flush=True)
        try:
            rr_times_sec, rr_values_sec = load_rr_signal(source_h5_path)
        except Exception as exc:
            # При ошибке загрузки RR сохраняем NaN-строки для всех окон участника
            print(f"  ПРЕДУПРЕЖДЕНИЕ: {subject_id} — ошибка загрузки RR: {exc}", flush=True)
            for window_row in subject_windows.itertuples():
                nan_row: dict[str, object] = {k: float("nan") for k in _all_hrv_feature_names()}
                nan_row.update({
                    "hrv_coverage_fraction": float("nan"),
                    "hrv_artifact_fraction": float("nan"),
                    "hrv_valid": 0,
                    "hrv_rr_count": 0,
                    "window_id": str(window_row.window_id),
                    "subject_id": subject_id,
                })
                rows.append(nan_row)
            continue

        for window_row in subject_windows.itertuples():
            row = extract_hrv_features(
                window_end_sec=float(window_row.window_end_sec),
                rr_times_sec=rr_times_sec,
                rr_values_sec=rr_values_sec,
                hrv_context_sec=hrv_context_sec,
            )
            row["window_id"] = str(window_row.window_id)
            row["subject_id"] = subject_id
            rows.append(row)

    data_frame = pd.DataFrame(rows)
    if not data_frame.empty:
        data_frame = data_frame.sort_values(["subject_id", "window_id"]).reset_index(drop=True)
        key_cols = ["window_id", "subject_id"]
        other_cols = [c for c in data_frame.columns if c not in key_cols]
        data_frame = data_frame[key_cols + other_cols]

    return data_frame
