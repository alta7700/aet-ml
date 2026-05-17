"""Признаки NIRS (ближняя инфракрасная спектроскопия) для датасета ML.

Загружает 4 канала NIRS один раз на участника, извлекает 15 признаков
для каждого скользящего окна. Baseline SmO2 берётся из session_params.parquet.

Выходной файл: dataset/features_nirs.parquet (15 признаков + 3 QC-поля на окно).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from new_arch.dataset_pipeline.common import (
    DEFAULT_DATASET_DIR,
    load_subjects_table,
    load_windows_table,
    save_parquet,
)

# ─────────────────────── Константы ───────────────────────

# Каналы NIRS в finaltest.h5
_NIRS_CHANNELS = (
    "train.red.smo2",
    "train.red.hhb.unfiltered",
    "train.red.hbdiff",
    "train.red.thb.unfiltered",
)

# Анкорный канал для временны́х меток
_ANCHOR_CHANNEL = "channels/moxy.smo2/timestamps"

# Минимальное покрытие для признания NIRS-окна валидным
_MIN_NIRS_COVERAGE = 0.8

# Порог для определения «пропуска» в сигнале (множитель медианного шага)
_GAP_DT_MULTIPLIER = 5.0
_GAP_MIN_ABS_SEC = 1.0


# ─────────────────────── Загрузка сигналов ───────────────────────

def _load_channel_relative(
    handle: h5py.File,
    channel_name: str,
    anchor_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Загружает канал NIRS и переводит timestamps в секунды от якоря.

    Параметры
    ----------
    handle : h5py.File
        Открытый файл finaltest.h5.
    channel_name : str
        Имя канала внутри группы channels/.
    anchor_ms : float
        Якорное время в миллисекундах.
    """
    group = handle[f"channels/{channel_name}"]
    raw_times = group["timestamps"][:].astype(float)
    values = group["values"][:].astype(float)
    times_sec = (raw_times - anchor_ms) / 1000.0
    return times_sec, values


def load_nirs_signals(
    source_h5_path: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Загружает все 4 канала NIRS из finaltest.h5 в память.

    Параметры
    ----------
    source_h5_path : Path
        Путь к finaltest.h5.

    Возвращает
    ----------
    dict с ключами — именами каналов, значениями — (times_sec, values).
    """
    signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with h5py.File(source_h5_path, "r") as handle:
        anchor_ms = float(handle[_ANCHOR_CHANNEL][0])
        for channel_name in _NIRS_CHANNELS:
            times_sec, values = _load_channel_relative(handle, channel_name, anchor_ms)
            signals[channel_name] = (times_sec, values)
    return signals


# ─────────────────────── QC: покрытие ───────────────────────

def _compute_nirs_coverage(
    times_in_window: np.ndarray,
    window_duration_sec: float,
    median_dt_sec: float,
) -> float:
    """Вычисляет долю окна, не занятую пропусками в сигнале.

    Параметры
    ----------
    times_in_window : np.ndarray
        Метки времени точек SmO2 внутри окна.
    window_duration_sec : float
        Длина окна в секундах.
    median_dt_sec : float
        Медианный шаг дискретизации NIRS на всей сессии в секундах.
    """
    if times_in_window.size < 2:
        # Если 0 или 1 точек — считаем всё пропуском
        return 0.0 if times_in_window.size == 0 else (
            median_dt_sec / window_duration_sec if window_duration_sec > 0 else 0.0
        )

    gap_threshold_sec = max(_GAP_MIN_ABS_SEC, _GAP_DT_MULTIPLIER * median_dt_sec)
    diffs = np.diff(times_in_window)
    gaps_sec = float(np.sum(np.maximum(0.0, diffs[diffs > gap_threshold_sec] - median_dt_sec)))
    coverage = 1.0 - gaps_sec / window_duration_sec if window_duration_sec > 0 else 0.0
    return float(max(0.0, min(1.0, coverage)))


# ─────────────────────── Извлечение признаков ───────────────────────

def _extract_channel_slope_features(
    times_in_window: np.ndarray,
    values_in_window: np.ndarray,
    prefix: str,
    n_features: int,
) -> dict[str, float]:
    """Извлекает mean, slope, std (и для SmO2 дополнительно drop) для канала.

    Параметры
    ----------
    times_in_window, values_in_window : np.ndarray
        Точки NIRS внутри окна.
    prefix : str
        Префикс имён признаков.
    n_features : int
        3 для hhb/hbdiff/thb, 4 для smo2 (без drop — он добавляется снаружи).
    """
    if values_in_window.size < 2:
        names = [f"{prefix}_mean", f"{prefix}_slope", f"{prefix}_std"]
        return {k: float("nan") for k in names}

    mean_val = float(np.mean(values_in_window))
    std_val = float(np.std(values_in_window, ddof=1) if values_in_window.size > 1 else 0.0)

    # Наклон через polyfit (линейная регрессия по времени)
    t_centered = times_in_window - float(np.mean(times_in_window))
    try:
        slope = float(np.polyfit(t_centered, values_in_window, 1)[0])
    except Exception:
        slope = float("nan")

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_slope": slope,
        f"{prefix}_std": std_val,
    }


def extract_nirs_features(
    window_start_sec: float,
    window_end_sec: float,
    nirs_signals: dict[str, tuple[np.ndarray, np.ndarray]],
    smo2_baseline_mean: float,
    median_dt_sec: float,
) -> dict[str, object]:
    """Извлекает 15 признаков NIRS + 3 QC-поля для одного окна.

    Параметры
    ----------
    window_start_sec, window_end_sec : float
        Границы каузального окна.
    nirs_signals : dict
        Загруженные сигналы NIRS участника (результат load_nirs_signals).
    smo2_baseline_mean : float
        Среднее SmO2 на ступени baseline (из session_params.parquet).
    median_dt_sec : float
        Медианный шаг дискретизации SmO2 на сессии.
    """
    window_duration_sec = window_end_sec - window_start_sec

    # QC: покрытие по SmO2
    smo2_times, smo2_values = nirs_signals["train.red.smo2"]
    mask_smo2 = (smo2_times >= window_start_sec) & (smo2_times < window_end_sec)
    smo2_t_win = smo2_times[mask_smo2]
    smo2_v_win = smo2_values[mask_smo2]
    nirs_coverage_fraction = _compute_nirs_coverage(smo2_t_win, window_duration_sec, median_dt_sec)
    nirs_valid = int(nirs_coverage_fraction >= _MIN_NIRS_COVERAGE)

    if nirs_valid == 0:
        return {
            **{k: float("nan") for k in _all_nirs_feature_names()},
            "nirs_coverage_fraction": nirs_coverage_fraction,
            "nirs_valid": 0,
        }

    features: dict[str, object] = {}

    # ─── train.red.smo2 (5 признаков) ───
    smo2_feat = _extract_channel_slope_features(smo2_t_win, smo2_v_win, "trainred_smo2", 3)
    smo2_mean = smo2_feat.get("trainred_smo2_mean", float("nan"))
    if isinstance(smo2_mean, float) and np.isfinite(smo2_mean) and np.isfinite(smo2_baseline_mean):
        smo2_drop = smo2_baseline_mean - smo2_mean
        smo2_dsmo2_dt = smo2_feat.get("trainred_smo2_slope", float("nan"))
    else:
        smo2_drop = float("nan")
        smo2_dsmo2_dt = float("nan")
    features.update(smo2_feat)
    features["trainred_smo2_drop"] = smo2_drop
    features["trainred_dsmo2_dt"] = smo2_dsmo2_dt

    # ─── train.red.hhb.unfiltered (4 признака) ───
    hhb_times, hhb_values = nirs_signals["train.red.hhb.unfiltered"]
    mask_hhb = (hhb_times >= window_start_sec) & (hhb_times < window_end_sec)
    hhb_feat = _extract_channel_slope_features(
        hhb_times[mask_hhb], hhb_values[mask_hhb], "trainred_hhb", 3
    )
    features.update(hhb_feat)
    # trainred_dhhb_dt = то же что и slope (синоним)
    features["trainred_dhhb_dt"] = hhb_feat.get("trainred_hhb_slope", float("nan"))

    # ─── train.red.hbdiff (3 признака) ───
    hbdiff_times, hbdiff_values = nirs_signals["train.red.hbdiff"]
    mask_hbdiff = (hbdiff_times >= window_start_sec) & (hbdiff_times < window_end_sec)
    features.update(
        _extract_channel_slope_features(
            hbdiff_times[mask_hbdiff], hbdiff_values[mask_hbdiff], "trainred_hbdiff", 3
        )
    )

    # ─── train.red.thb.unfiltered (3 признака) ───
    thb_times, thb_values = nirs_signals["train.red.thb.unfiltered"]
    mask_thb = (thb_times >= window_start_sec) & (thb_times < window_end_sec)
    features.update(
        _extract_channel_slope_features(
            thb_times[mask_thb], thb_values[mask_thb], "trainred_thb", 3
        )
    )

    features["nirs_coverage_fraction"] = nirs_coverage_fraction
    features["nirs_valid"] = nirs_valid
    return features


def _all_nirs_feature_names() -> list[str]:
    """Возвращает список всех 15 имён NIRS-признаков."""
    return [
        # SmO2 (5)
        "trainred_smo2_mean",
        "trainred_smo2_slope",
        "trainred_smo2_std",
        "trainred_smo2_drop",
        "trainred_dsmo2_dt",
        # HHb (4)
        "trainred_hhb_mean",
        "trainred_hhb_slope",
        "trainred_hhb_std",
        "trainred_dhhb_dt",
        # HbDiff (3)
        "trainred_hbdiff_mean",
        "trainred_hbdiff_slope",
        "trainred_hbdiff_std",
        # tHb (3)
        "trainred_thb_mean",
        "trainred_thb_slope",
        "trainred_thb_std",
    ]


# ─────────────────────── Сборка таблицы ───────────────────────

def build_nirs_table(
    subjects_path: Path,
    windows_path: Path,
    session_params_path: Path,
) -> pd.DataFrame:
    """Строит таблицу признаков NIRS по всем участникам.

    Параметры
    ----------
    subjects_path : Path
        Путь к subjects.parquet.
    windows_path : Path
        Путь к windows.parquet.
    session_params_path : Path
        Путь к session_params.parquet (нужен для baseline SmO2).
    """
    subjects = load_subjects_table(subjects_path)
    windows = load_windows_table(windows_path)
    session_params = pd.read_parquet(session_params_path)

    rows: list[dict[str, object]] = []

    for subject_row in subjects.itertuples():
        subject_id = str(subject_row.subject_id)
        source_h5_path = Path(str(subject_row.source_h5_path))
        subject_windows = windows[windows["subject_id"] == subject_id]

        def _emit_nan_nirs_rows(reason: str) -> None:
            """Записывает NaN-строки для всех окон участника при недоступном NIRS."""
            print(f"  ПРЕДУПРЕЖДЕНИЕ: {subject_id} — {reason}", flush=True)
            for window_row in subject_windows.itertuples():
                nan_row: dict[str, object] = {k: float("nan") for k in _all_nirs_feature_names()}
                nan_row.update({
                    "nirs_coverage_fraction": float("nan"),
                    "nirs_valid": 0,
                    "window_id": str(window_row.window_id),
                    "subject_id": subject_id,
                })
                rows.append(nan_row)

        # Берём baseline SmO2 из session_params
        sp_mask = session_params["subject_id"] == subject_id
        if not sp_mask.any():
            _emit_nan_nirs_rows("нет строки в session_params")
            continue
        smo2_baseline_mean = float(session_params.loc[sp_mask, "nirs_smo2_baseline_mean"].iloc[0])

        print(f"  Участник {subject_id}: загружаем NIRS...", flush=True)
        try:
            nirs_signals = load_nirs_signals(source_h5_path)
        except Exception as exc:
            _emit_nan_nirs_rows(f"ошибка загрузки NIRS: {exc}")
            continue

        # Медианный шаг дискретизации SmO2 на сессии
        smo2_times = nirs_signals["train.red.smo2"][0]
        median_dt_sec = float(np.median(np.diff(smo2_times))) if smo2_times.size > 1 else 1.0

        for window_row in subject_windows.itertuples():
            row = extract_nirs_features(
                window_start_sec=float(window_row.window_start_sec),
                window_end_sec=float(window_row.window_end_sec),
                nirs_signals=nirs_signals,
                smo2_baseline_mean=smo2_baseline_mean,
                median_dt_sec=median_dt_sec,
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
