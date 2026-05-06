"""Сборка таблицы QC-флагов по всем окнам датасета.

Объединяет QC-поля из трёх источников:
- features_emg_kinematics.parquet (emg_valid, kinematics_valid, покрытие, cycles_count)
- features_nirs.parquet (nirs_valid, покрытие)
- features_hrv.parquet (hrv_valid, артефакты, покрытие)

Выходной файл: dataset/qc_windows.parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ─────────────────────── Имена колонок ───────────────────────

# QC-поля из EMG-таблицы
_EMG_QC_COLS = [
    "emg_valid",
    "kinematics_valid",
    "emg_coverage_fraction",
    "kinematics_coverage_fraction",
    "cycles_count",
]

# QC-поля из NIRS-таблицы
_NIRS_QC_COLS = [
    "nirs_valid",
    "nirs_coverage_fraction",
]

# QC-поля из HRV-таблицы
_HRV_QC_COLS = [
    "hrv_valid",
    "hrv_coverage_fraction",
    "hrv_artifact_fraction",
    "hrv_rr_count",
]

# Ключи JOIN
_KEY_COLS = ["window_id", "subject_id"]


def build_qc_table(
    emg_path: Path,
    nirs_path: Path,
    hrv_path: Path,
) -> pd.DataFrame:
    """Строит таблицу QC-флагов по всем окнам.

    Выполняет JOIN трёх таблиц по (window_id, subject_id).
    Если у окна нет строки в одной из таблиц, соответствующие поля остаются NaN,
    а флаг valid = 0.

    Параметры
    ----------
    emg_path : Path
        Путь к features_emg_kinematics.parquet.
    nirs_path : Path
        Путь к features_nirs.parquet.
    hrv_path : Path
        Путь к features_hrv.parquet.
    """
    emg_df = pd.read_parquet(emg_path, columns=_KEY_COLS + _EMG_QC_COLS)
    nirs_df = pd.read_parquet(nirs_path, columns=_KEY_COLS + _NIRS_QC_COLS)
    hrv_df = pd.read_parquet(hrv_path, columns=_KEY_COLS + _HRV_QC_COLS)

    # Начинаем с EMG — она самая полная по числу окон
    merged = emg_df.copy()

    merged = merged.merge(nirs_df, on=_KEY_COLS, how="left")
    merged = merged.merge(hrv_df, on=_KEY_COLS, how="left")

    # Заполняем NaN в valid-колонках нулями (если модальность отсутствует — не валидна)
    for col in ("emg_valid", "kinematics_valid", "nirs_valid", "hrv_valid"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    # Итоговые агрегированные флаги
    merged["window_valid_any"] = (
        (merged["emg_valid"] | merged["nirs_valid"] | merged["hrv_valid"])
        .clip(0, 1)
        .astype(int)
    )
    merged["window_valid_all_required"] = (
        (merged["emg_valid"] & merged["nirs_valid"] & merged["hrv_valid"])
        .clip(0, 1)
        .astype(int)
    )

    merged = merged.sort_values(_KEY_COLS).reset_index(drop=True)
    return merged
