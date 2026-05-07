"""Признаки ЭМГ и кинематики педалирования для датасета ML.

Модуль реализует паттерн PhasedSession: один раз загружаем данные участника
(вызов detect_phases, кэш сырых сигналов), затем быстро извлекаем признаки
для каждого скользящего окна.

Выходные файлы:
- features_emg_kinematics.parquet  — 101 признак на окно (64 ЭМГ + 8 prox-dist + 7 кинематика + 4 EMG-CV + 18 variability)
- session_params.parquet            — калибровочные параметры на участника
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pywt

from dataset_pipeline.common import (
    DEFAULT_DATASET_DIR,
    load_subjects_table,
    load_windows_table,
    save_parquet,
)
from methods.pedal_cycles import (
    DEFAULT_CALIBRATION_MARGIN_SEC,
    DEFAULT_CALIBRATION_TAIL_SEC,
    DEFAULT_MAX_CADENCE_RPM,
    DEFAULT_MIN_CADENCE_RPM,
    DEFAULT_PEAK_PROMINENCE_STD,
    CyclePhase,
    detect_phases,
)

# ─────────────────────── Константы ───────────────────────

# Минимальное число сэмплов в сегменте для вычисления спектральных признаков
_MIN_SAMPLES_SPECTRAL = 32

# Минимальное число сэмплов в сегменте для вычисления вейвлетных признаков
_MIN_SAMPLES_WAVELET = 64

# Вейвлет и уровень декомпозиции (db4 — стандарт для ЭМГ)
_WAVELET = "db4"
_WAVELET_LEVEL = 5

# Частотные диапазоны (Гц) для полос спектра
_BAND_LOW_LOW_HZ = 20.0
_BAND_LOW_HIGH_HZ = 50.0
_BAND_MID_LOW_HZ = 50.0
_BAND_MID_HIGH_HZ = 150.0
_BAND_HIGH_LOW_HZ = 150.0
_BAND_HIGH_HIGH_HZ = 450.0

# Минимальное покрытие для признания EMG-сигнала валидным
_MIN_EMG_COVERAGE = 0.8

# Канал NIRS для baseline
_NIRS_SMO2_CHANNEL = "train.red.smo2"

# Анкорный канал для вычисления относительных временных меток
_ANCHOR_CHANNEL = "channels/moxy.smo2/timestamps"


# ─────────────────────── PhasedSession ───────────────────────

@dataclass
class PhasedSession:
    """Кэш данных одного участника для быстрого извлечения признаков по окнам.

    Создаётся один раз на участника, хранится в памяти на время сборки датасета.
    """

    subject_id: str
    source_h5_path: Path

    # Калибровочные параметры (сохраняются в session_params.parquet)
    emg_vl_dist_baseline_rms: float
    emg_vl_prox_baseline_rms: float
    pca_axis: np.ndarray          # shape (3,) — первая главная компонента гироскопа
    nirs_smo2_baseline_mean: float

    # Кэш сигналов (только в памяти)
    cycles: tuple[CyclePhase, ...]

    # ЭМГ VL-дист (trigno.vl.avanti) — отфильтрованные значения, нормированные на baseline RMS
    emg_vl_dist_times: np.ndarray
    emg_vl_dist_values: np.ndarray

    # ЭМГ VL-прокс (trigno.rf.avanti) — отфильтрованные значения, нормированные на baseline RMS
    emg_vl_prox_times: np.ndarray
    emg_vl_prox_values: np.ndarray

    # Частота дискретизации ЭМГ (одинакова для обоих каналов)
    emg_sample_rate_hz: float


def build_phased_session(source_h5_path: Path, subject_id: str) -> PhasedSession:
    """Строит PhasedSession для одного участника.

    Вызывает detect_phases с параметрами по умолчанию, извлекает сигналы
    из PhaseDetectionResult и загружает baseline SmO2 из HDF5.

    Параметры
    ----------
    source_h5_path : Path
        Путь к finaltest.h5 участника.
    subject_id : str
        Идентификатор участника (для логирования и ключей).

    Возвращает
    ----------
    PhasedSession
        Готовый кэш для извлечения признаков.
    """
    result = detect_phases(
        participant_name=subject_id,
        source_h5_path=source_h5_path,
        sensor_mode="auto",
        calibration_tail_sec=DEFAULT_CALIBRATION_TAIL_SEC,
        calibration_margin_sec=DEFAULT_CALIBRATION_MARGIN_SEC,
        min_cadence_rpm=DEFAULT_MIN_CADENCE_RPM,
        max_cadence_rpm=DEFAULT_MAX_CADENCE_RPM,
        peak_prominence_std=DEFAULT_PEAK_PROMINENCE_STD,
    )

    # Извлекаем каналы ЭМГ
    # emg_channels[0] = trigno.vl.avanti  → VL_dist
    # emg_channels[1] = trigno.rf.avanti  → VL_prox
    vl_dist_channel = result.emg_channels[0]
    vl_prox_channel = result.emg_channels[1]

    # Сигналы хранятся в filtered_values (полосовая фильтрация 20–500 Гц),
    # нормированных на baseline RMS. Для признаков используем именно их.
    vl_dist_values = (
        np.asarray(vl_dist_channel.filtered_values, dtype=float)
        if vl_dist_channel.filtered_values is not None
        else np.asarray(vl_dist_channel.values, dtype=float)
    )
    vl_prox_values = (
        np.asarray(vl_prox_channel.filtered_values, dtype=float)
        if vl_prox_channel.filtered_values is not None
        else np.asarray(vl_prox_channel.values, dtype=float)
    )

    # Загружаем baseline SmO2 из HDF5 напрямую
    nirs_smo2_baseline_mean = _load_nirs_baseline_mean(
        source_h5_path=source_h5_path,
        normalization_start_sec=result.normalization_stage.start_sec,
        normalization_end_sec=result.normalization_stage.end_sec,
    )

    return PhasedSession(
        subject_id=subject_id,
        source_h5_path=source_h5_path,
        emg_vl_dist_baseline_rms=float(vl_dist_channel.baseline_rms),
        emg_vl_prox_baseline_rms=float(vl_prox_channel.baseline_rms),
        pca_axis=np.asarray(result.selected_sensor.pca_vector, dtype=float),
        nirs_smo2_baseline_mean=nirs_smo2_baseline_mean,
        cycles=result.cycles,
        emg_vl_dist_times=np.asarray(vl_dist_channel.timestamps_sec, dtype=float),
        emg_vl_dist_values=vl_dist_values,
        emg_vl_prox_times=np.asarray(vl_prox_channel.timestamps_sec, dtype=float),
        emg_vl_prox_values=vl_prox_values,
        emg_sample_rate_hz=float(vl_dist_channel.sample_rate_hz),
    )


def _load_nirs_baseline_mean(
    source_h5_path: Path,
    normalization_start_sec: float,
    normalization_end_sec: float,
) -> float:
    """Загружает среднее значение SmO2 на ступени нормализации (baseline 30 Вт).

    Параметры
    ----------
    source_h5_path : Path
        Путь к finaltest.h5.
    normalization_start_sec, normalization_end_sec : float
        Временной интервал ступени нормализации в секундах от начала записи.
    """
    with h5py.File(source_h5_path, "r") as handle:
        anchor_ms = float(handle[_ANCHOR_CHANNEL][0])
        smo2_raw_times = handle[f"channels/{_NIRS_SMO2_CHANNEL}/timestamps"][:].astype(float)
        smo2_values = handle[f"channels/{_NIRS_SMO2_CHANNEL}/values"][:].astype(float)

    smo2_times = (smo2_raw_times - anchor_ms) / 1000.0
    baseline_mask = (
        (smo2_times >= normalization_start_sec) & (smo2_times <= normalization_end_sec)
    )
    baseline_segment = smo2_values[baseline_mask]
    if baseline_segment.size == 0:
        return float("nan")
    return float(np.mean(baseline_segment))


# ─────────────────────── Вспомогательные функции ───────────────────────

def _collect_phase_samples(
    times: np.ndarray,
    values: np.ndarray,
    cycles: list[CyclePhase],
    use_push: bool,
    window_start_sec: float,
    window_end_sec: float,
) -> np.ndarray:
    """Собирает сэмплы из фазы (push или pull), обрезая по границам окна.

    Каждая фаза обрезается до пересечения с [window_start_sec, window_end_sec],
    чтобы признаки строго каузальны и не содержали данных вне текущего окна.

    Параметры
    ----------
    times : np.ndarray
        Метки времени сигнала в секундах.
    values : np.ndarray
        Значения сигнала.
    cycles : list[CyclePhase]
        Список циклов, попавших в окно (хотя бы частично).
    use_push : bool
        True → push-фаза (нагрузка), False → pull-фаза (восстановление).
    window_start_sec, window_end_sec : float
        Границы текущего каузального окна — жёсткий ограничитель выборки.
    """
    # Собираем диапазоны индексов через бинарный поиск — O(log N) вместо O(N) на маску
    ranges: list[tuple[int, int]] = []
    total = 0
    for cycle in cycles:
        if use_push:
            phase_start, phase_end = cycle.push_start_sec, cycle.push_end_sec
        else:
            phase_start, phase_end = cycle.pull_start_sec, cycle.pull_end_sec
        t_start = max(phase_start, window_start_sec)
        t_end = min(phase_end, window_end_sec)
        if t_end <= t_start:
            continue
        i0 = int(np.searchsorted(times, t_start, side="left"))
        i1 = int(np.searchsorted(times, t_end, side="left"))
        if i1 > i0:
            ranges.append((i0, i1))
            total += i1 - i0
    if not ranges:
        return np.empty(0, dtype=float)
    # Одно выделение памяти вместо concatenate списка массивов
    result = np.empty(total, dtype=float)
    pos = 0
    for i0, i1 in ranges:
        n = i1 - i0
        result[pos:pos + n] = values[i0:i1]
        pos += n
    return result


def _collect_per_cycle_rms(
    times: np.ndarray,
    values: np.ndarray,
    cycles: list[CyclePhase],
    use_push: bool,
    window_start_sec: float,
    window_end_sec: float,
    min_samples: int = 4,
) -> np.ndarray:
    """Возвращает массив RMS по каждому полному (или частичному) циклу в окне.

    Используется для вычисления CV амплитуды от цикла к циклу.
    Циклы с менее чем min_samples точек пропускаются.
    """
    rms_values: list[float] = []
    for cycle in cycles:
        phase_start = cycle.push_start_sec if use_push else cycle.pull_start_sec
        phase_end = cycle.push_end_sec if use_push else cycle.pull_end_sec
        t_start = max(phase_start, window_start_sec)
        t_end = min(phase_end, window_end_sec)
        if t_end <= t_start:
            continue
        i0 = int(np.searchsorted(times, t_start, side="left"))
        i1 = int(np.searchsorted(times, t_end, side="left"))
        n = i1 - i0
        if n >= min_samples:
            seg = values[i0:i1]
            # np.dot избегает промежуточного массива seg**2
            rms_values.append(float(np.sqrt(np.dot(seg, seg) / n)))
    return np.array(rms_values, dtype=float)


def _timing_trend_features(durations: np.ndarray, suffix: str) -> dict[str, float]:
    """Признаки тренда CV: slope и ratio между первой и последней третями окна.

    Captures нарастание ритмической нестабильности, которую единственный CV не видит.
    Требует минимум 6 циклов (по 2 на треть).
    """
    nan = {"load" if "load" in suffix else suffix: float("nan")}  # заглушка не нужна
    n = len(durations)
    if n < 6:
        return {
            f"trend_cv_slope_{suffix}": float("nan"),
            f"trend_cv_ratio_{suffix}": float("nan"),
        }
    third = n // 3
    cv_early = _cv(durations[:third])
    cv_late = _cv(durations[n - third:])
    if not (math.isfinite(cv_early) and math.isfinite(cv_late)):
        return {
            f"trend_cv_slope_{suffix}": float("nan"),
            f"trend_cv_ratio_{suffix}": float("nan"),
        }
    slope = cv_late - cv_early
    ratio = cv_late / cv_early if abs(cv_early) > 1e-9 else float("nan")
    return {
        f"trend_cv_slope_{suffix}": slope,
        f"trend_cv_ratio_{suffix}": ratio,
    }


def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """SampleEntropy: мера непредсказуемости ряда длительностей циклов.

    Параметры: m=2 (длина шаблона), r=0.2*std (допуск совпадения).
    Рекомендуется N≥50 для стабильной оценки (60с-окно ≈ 75 циклов, 120с ≈ 150).
    """
    n = len(x)
    if n < m + 2:
        return float("nan")
    r = r_factor * float(np.std(x, ddof=1))
    if r < 1e-12:
        return float("nan")

    def _count_matches(length: int) -> int:
        # Матрица шаблонов (n-length, length) — numpy-векторизация вместо двойного цикла
        templates = np.array([x[i:i + length] for i in range(n - length)])
        diff = np.abs(templates[:, None, :] - templates[None, :, :])
        max_diff = diff.max(axis=2)
        matches = max_diff < r
        np.fill_diagonal(matches, False)
        return int(matches.sum())

    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b == 0 or a == 0:
        return float("nan")
    return float(-np.log(a / b))


def _cv(values: np.ndarray) -> float:
    """Коэффициент вариации (std/mean). NaN если менее 2 значений или mean≈0."""
    if values.size < 2:
        return float("nan")
    mean = float(np.mean(values))
    if not math.isfinite(mean) or abs(mean) < 1e-12:
        return float("nan")
    return float(np.std(values, ddof=1) / mean)


def _compute_time_domain_features(x: np.ndarray) -> dict[str, float]:
    """Вычисляет признаки временно́й области для одного сегмента.

    Параметры
    ----------
    x : np.ndarray
        Отсчёты сигнала.

    Возвращает
    ----------
    dict со скалярными значениями: rms, mav, wl, zcr.
    """
    if x.size < 2:
        return {"rms": float("nan"), "mav": float("nan"), "wl": float("nan"), "zcr": float("nan")}
    rms = float(np.sqrt(np.mean(x ** 2)))
    mav = float(np.mean(np.abs(x)))
    wl = float(np.sum(np.abs(np.diff(x))))
    sign_changes = float(np.sum(np.diff(np.sign(x)) != 0))
    zcr = sign_changes / float(len(x))
    return {"rms": rms, "mav": mav, "wl": wl, "zcr": zcr}


def _compute_spectral_features(x: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    """Вычисляет признаки частотной области через FFT.

    Параметры
    ----------
    x : np.ndarray
        Отсчёты сигнала.
    sample_rate_hz : float
        Частота дискретизации.

    Возвращает
    ----------
    dict: mdf, mnf, p_low, p_mid, p_high, ratio_mid_high.
    """
    nan_result: dict[str, float] = {
        "mdf": float("nan"),
        "mnf": float("nan"),
        "p_low": float("nan"),
        "p_mid": float("nan"),
        "p_high": float("nan"),
        "ratio_mid_high": float("nan"),
    }
    if x.size < _MIN_SAMPLES_SPECTRAL:
        return nan_result

    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    fft_vals = np.fft.rfft(x)
    psd = (np.abs(fft_vals) ** 2) / n

    total_power = float(np.sum(psd))
    if total_power <= 0:
        return nan_result

    # Медианная частота (50% кумулятивной мощности)
    cumsum_psd = np.cumsum(psd)
    idx_mdf = int(np.searchsorted(cumsum_psd, total_power * 0.5))
    idx_mdf = min(idx_mdf, len(freqs) - 1)
    mdf = float(freqs[idx_mdf])

    # Средняя частота (центроид спектра)
    mnf = float(np.sum(freqs * psd) / total_power)

    # Мощность в полосах
    p_low_abs = float(np.sum(psd[(freqs >= _BAND_LOW_LOW_HZ) & (freqs <= _BAND_LOW_HIGH_HZ)]))
    p_mid_abs = float(np.sum(psd[(freqs > _BAND_MID_LOW_HZ) & (freqs <= _BAND_MID_HIGH_HZ)]))
    p_high_abs = float(np.sum(psd[(freqs > _BAND_HIGH_LOW_HZ) & (freqs <= _BAND_HIGH_HIGH_HZ)]))
    band_total = p_low_abs + p_mid_abs + p_high_abs
    if band_total <= 0:
        return {**nan_result, "mdf": mdf, "mnf": mnf}

    p_low = p_low_abs / total_power
    p_mid = p_mid_abs / total_power
    p_high = p_high_abs / total_power
    ratio_mid_high = p_mid / p_high if p_high > 0 else float("nan")

    return {
        "mdf": mdf,
        "mnf": mnf,
        "p_low": p_low,
        "p_mid": p_mid,
        "p_high": p_high,
        "ratio_mid_high": ratio_mid_high,
    }


def _compute_wavelet_features(x: np.ndarray) -> dict[str, float]:
    """Вычисляет признаки вейвлетного разложения (db4, уровни d2–d5).

    Параметры
    ----------
    x : np.ndarray
        Отсчёты сигнала.

    Возвращает
    ----------
    dict: e_d2, e_d3, e_d4, e_d5, wavelet_entropy, ratio_low_high.
    """
    nan_result: dict[str, float] = {
        "e_d2": float("nan"),
        "e_d3": float("nan"),
        "e_d4": float("nan"),
        "e_d5": float("nan"),
        "wavelet_entropy": float("nan"),
        "ratio_low_high": float("nan"),
    }
    if x.size < _MIN_SAMPLES_WAVELET:
        return nan_result

    # pywt.wavedec возвращает [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # Для level=5: [a5, d5, d4, d3, d2, d1]
    try:
        coeffs = pywt.wavedec(x, _WAVELET, level=_WAVELET_LEVEL)
    except Exception:
        return nan_result

    # coeffs[5] = d1 (500–1000 Гц при fs=2000) — шум, не используем
    # coeffs[4] = d2 (~250–500 Гц)
    # coeffs[3] = d3 (~125–250 Гц)
    # coeffs[2] = d4 (~62–125 Гц)
    # coeffs[1] = d5 (~31–62 Гц)
    if len(coeffs) < 6:
        return nan_result

    e_d5 = float(np.sum(coeffs[1] ** 2))
    e_d4 = float(np.sum(coeffs[2] ** 2))
    e_d3 = float(np.sum(coeffs[3] ** 2))
    e_d2 = float(np.sum(coeffs[4] ** 2))

    energies = np.array([e_d2, e_d3, e_d4, e_d5], dtype=float)
    total_energy = float(np.sum(energies))

    if total_energy > 0 and all(math.isfinite(e) for e in energies):
        p = energies / total_energy
        p_safe = np.where(p > 1e-12, p, 1e-12)
        wavelet_entropy = float(-np.sum(p_safe * np.log(p_safe)))
        ratio_low_high = (e_d4 + e_d5) / (e_d2 + e_d3) if (e_d2 + e_d3) > 0 else float("nan")
    else:
        wavelet_entropy = float("nan")
        ratio_low_high = float("nan")

    return {
        "e_d2": e_d2,
        "e_d3": e_d3,
        "e_d4": e_d4,
        "e_d5": e_d5,
        "wavelet_entropy": wavelet_entropy,
        "ratio_low_high": ratio_low_high,
    }


def _compute_stream_features(
    x: np.ndarray,
    sample_rate_hz: float,
    prefix: str,
) -> dict[str, float]:
    """Вычисляет все 16 признаков для одного потока (stream).

    Параметры
    ----------
    x : np.ndarray
        Отсчёты сигнала для потока.
    sample_rate_hz : float
        Частота дискретизации.
    prefix : str
        Префикс имён признаков, например ``vl_dist_load``.
    """
    td = _compute_time_domain_features(x)
    sp = _compute_spectral_features(x, sample_rate_hz)
    wv = _compute_wavelet_features(x)
    merged = {**td, **sp, **wv}
    return {f"{prefix}_{k}": v for k, v in merged.items()}


# ─────────────────────── Главная функция извлечения признаков ───────────────────────

def extract_emg_kinematics_features(
    session: PhasedSession,
    window_start_sec: float,
    window_end_sec: float,
) -> dict[str, object]:
    """Извлекает 83 признака + QC-поля для одного скользящего окна.

    Параметры
    ----------
    session : PhasedSession
        Закешированные данные участника.
    window_start_sec, window_end_sec : float
        Границы каузального окна в секундах.

    Возвращает
    ----------
    dict с 83 признаками и 5 QC-полями (итого 88 полей).
    """
    window_duration_sec = window_end_sec - window_start_sec
    fs = session.emg_sample_rate_hz

    # Циклы, которые хотя бы частично попадают в окно
    cycles_in_window: list[CyclePhase] = [
        c for c in session.cycles
        if c.push_start_sec < window_end_sec and c.pull_end_sec > window_start_sec
    ]
    cycles_count = len(cycles_in_window)

    # QC: покрытие ЭМГ
    dist_mask = (
        (session.emg_vl_dist_times >= window_start_sec)
        & (session.emg_vl_dist_times < window_end_sec)
    )
    observed_samples = int(np.sum(dist_mask))
    expected_samples = window_duration_sec * fs
    emg_coverage_fraction = float(
        min(1.0, observed_samples / expected_samples) if expected_samples > 0 else 0.0
    )

    # QC: покрытие кинематики (доля времени окна, покрытая циклами)
    if cycles_count > 0:
        total_cycle_time = sum(
            min(c.pull_end_sec, window_end_sec) - max(c.push_start_sec, window_start_sec)
            for c in cycles_in_window
        )
        kinematics_coverage_fraction = float(min(1.0, total_cycle_time / window_duration_sec))
    else:
        kinematics_coverage_fraction = 0.0

    emg_valid = int(emg_coverage_fraction >= _MIN_EMG_COVERAGE and cycles_count >= 1)
    kinematics_valid = int(cycles_count >= 1)

    # Если цикломнет — все признаки NaN
    if cycles_count == 0:
        nan_features = {k: float("nan") for k in _all_feature_names()}
        return {
            **nan_features,
            "cycles_count": 0,
            "emg_coverage_fraction": emg_coverage_fraction,
            "kinematics_coverage_fraction": kinematics_coverage_fraction,
            "emg_valid": 0,
            "kinematics_valid": 0,
        }

    # Собираем сэмплы по фазам для обоих каналов, обрезая по границам окна
    dist_load = _collect_phase_samples(
        session.emg_vl_dist_times, session.emg_vl_dist_values,
        cycles_in_window, use_push=True,
        window_start_sec=window_start_sec, window_end_sec=window_end_sec,
    )
    dist_rest = _collect_phase_samples(
        session.emg_vl_dist_times, session.emg_vl_dist_values,
        cycles_in_window, use_push=False,
        window_start_sec=window_start_sec, window_end_sec=window_end_sec,
    )
    prox_load = _collect_phase_samples(
        session.emg_vl_prox_times, session.emg_vl_prox_values,
        cycles_in_window, use_push=True,
        window_start_sec=window_start_sec, window_end_sec=window_end_sec,
    )
    prox_rest = _collect_phase_samples(
        session.emg_vl_prox_times, session.emg_vl_prox_values,
        cycles_in_window, use_push=False,
        window_start_sec=window_start_sec, window_end_sec=window_end_sec,
    )

    # Признаки по 4 потокам (16 × 4 = 64)
    features: dict[str, object] = {}
    features.update(_compute_stream_features(dist_load, fs, "vl_dist_load"))
    features.update(_compute_stream_features(dist_rest, fs, "vl_dist_rest"))
    features.update(_compute_stream_features(prox_load, fs, "vl_prox_load"))
    features.update(_compute_stream_features(prox_rest, fs, "vl_prox_rest"))

    # Производные prox-dist признаки (8)
    features.update(_compute_prox_dist_derived(features))

    # Кинематические признаки (7: базовые + CV длительностей)
    features.update(_compute_kinematics_features(cycles_in_window))

    # CV амплитуды ЭМГ цикл-к-циклу (4): нестабильность паттерна как маркер усталости
    for ch_times, ch_vals, prefix, use_push in [
        (session.emg_vl_dist_times, session.emg_vl_dist_values, "vl_dist_load", True),
        (session.emg_vl_dist_times, session.emg_vl_dist_values, "vl_dist_rest", False),
        (session.emg_vl_prox_times, session.emg_vl_prox_values, "vl_prox_load", True),
        (session.emg_vl_prox_times, session.emg_vl_prox_values, "vl_prox_rest", False),
    ]:
        per_cycle_rms = _collect_per_cycle_rms(
            ch_times, ch_vals, cycles_in_window, use_push,
            window_start_sec, window_end_sec,
        )
        features[f"{prefix}_rms_cv"] = _cv(per_cycle_rms)

    # Trend CV и SampEn на 3 масштабах (30s / 60s / 120s) для timing-признаков
    # Каждый масштаб — отдельная группа признаков, в модель подаётся только один
    for lookback_sec, suffix in [(30.0, "30s"), (60.0, "60s"), (120.0, "120s")]:
        lb_start = window_end_sec - lookback_sec
        cycles_lb: list[CyclePhase] = [
            c for c in session.cycles
            if c.push_start_sec >= lb_start and c.pull_end_sec <= window_end_sec
        ]
        load_dur = np.array([c.push_duration_sec * 1000.0 for c in cycles_lb], dtype=float)
        rest_dur = np.array([c.pull_duration_sec * 1000.0 for c in cycles_lb], dtype=float)

        load_trend = _timing_trend_features(load_dur, suffix)
        rest_trend = _timing_trend_features(rest_dur, suffix)
        features.update({
            f"load_trend_cv_slope_{suffix}": load_trend[f"trend_cv_slope_{suffix}"],
            f"load_trend_cv_ratio_{suffix}": load_trend[f"trend_cv_ratio_{suffix}"],
            f"rest_trend_cv_slope_{suffix}": rest_trend[f"trend_cv_slope_{suffix}"],
            f"rest_trend_cv_ratio_{suffix}": rest_trend[f"trend_cv_ratio_{suffix}"],
            f"load_sampen_{suffix}": _sample_entropy(load_dur),
            f"rest_sampen_{suffix}": _sample_entropy(rest_dur),
        })

    # QC поля
    features["cycles_count"] = cycles_count
    features["emg_coverage_fraction"] = emg_coverage_fraction
    features["kinematics_coverage_fraction"] = kinematics_coverage_fraction
    features["emg_valid"] = emg_valid
    features["kinematics_valid"] = kinematics_valid

    return features


def _compute_prox_dist_derived(features: dict[str, object]) -> dict[str, float]:
    """Вычисляет 8 производных prox-dist признаков.

    Параметры
    ----------
    features : dict
        Уже вычисленные признаки потоков.
    """

    def safe_diff(a_key: str, b_key: str) -> float:
        a = features.get(a_key)
        b = features.get(b_key)
        if a is None or b is None or not math.isfinite(float(a)) or not math.isfinite(float(b)):
            return float("nan")
        return float(a) - float(b)

    def safe_ratio(a_key: str, b_key: str) -> float:
        a = features.get(a_key)
        b = features.get(b_key)
        if a is None or b is None or not math.isfinite(float(a)) or not math.isfinite(float(b)):
            return float("nan")
        denom = float(b)
        return float(a) / denom if denom != 0 else float("nan")

    return {
        "delta_rms_prox_dist_load": safe_diff("vl_prox_load_rms", "vl_dist_load_rms"),
        "ratio_rms_prox_dist_load": safe_ratio("vl_prox_load_rms", "vl_dist_load_rms"),
        "delta_mdf_prox_dist_load": safe_diff("vl_prox_load_mdf", "vl_dist_load_mdf"),
        "delta_we_prox_dist_load": safe_diff(
            "vl_prox_load_wavelet_entropy", "vl_dist_load_wavelet_entropy"
        ),
        "delta_rms_prox_dist_rest": safe_diff("vl_prox_rest_rms", "vl_dist_rest_rms"),
        "ratio_rms_prox_dist_rest": safe_ratio("vl_prox_rest_rms", "vl_dist_rest_rms"),
        "delta_mdf_prox_dist_rest": safe_diff("vl_prox_rest_mdf", "vl_dist_rest_mdf"),
        "delta_we_prox_dist_rest": safe_diff(
            "vl_prox_rest_wavelet_entropy", "vl_dist_rest_wavelet_entropy"
        ),
    }


def _compute_kinematics_features(cycles: list[CyclePhase]) -> dict[str, float]:
    """Вычисляет 7 кинематических признаков по набору циклов в окне.

    Параметры
    ----------
    cycles : list[CyclePhase]
        Циклы педалирования, попавшие в окно (len >= 1).
    """
    cadence_values = np.array([c.cadence_rpm for c in cycles], dtype=float)
    load_durations_ms = np.array([c.push_duration_sec * 1000.0 for c in cycles], dtype=float)
    rest_durations_ms = np.array([c.pull_duration_sec * 1000.0 for c in cycles], dtype=float)

    cadence_mean_rpm = float(np.mean(cadence_values))
    cadence_cv = _cv(cadence_values)
    load_duration_ms = float(np.mean(load_durations_ms))
    rest_duration_ms = float(np.mean(rest_durations_ms))
    load_rest_ratio = (
        float(np.mean(load_durations_ms) / np.mean(rest_durations_ms))
        if np.mean(rest_durations_ms) > 0
        else float("nan")
    )
    # CV длительностей: растёт при ритмической нестабильности (усталость, рекрутирование)
    load_duration_cv = _cv(load_durations_ms)
    rest_duration_cv = _cv(rest_durations_ms)

    return {
        "cadence_mean_rpm": cadence_mean_rpm,
        "cadence_cv": cadence_cv,
        "load_duration_ms": load_duration_ms,
        "rest_duration_ms": rest_duration_ms,
        "load_rest_ratio": load_rest_ratio,
        "load_duration_cv": load_duration_cv,
        "rest_duration_cv": rest_duration_cv,
    }


def _all_feature_names() -> list[str]:
    """Возвращает список всех 83 имён признаков (без QC-полей).

    Состав:
    - 64 stream-признака  (16 × 4 потока)
    - 8 prox-dist производных
    - 7 кинематических (включая CV длительностей)
    - 4 EMG CV амплитуды цикл-к-циклу
    - 18 timing Trend CV + SampEn на 3 масштабах (30s / 60s / 120s):
        load/rest × (trend_cv_slope, trend_cv_ratio, sampen) × 3 = 18
    Итого: 101
    """
    feature_suffixes = [
        "rms", "mav", "wl", "zcr",
        "mdf", "mnf", "p_low", "p_mid", "p_high", "ratio_mid_high",
        "e_d2", "e_d3", "e_d4", "e_d5", "wavelet_entropy", "ratio_low_high",
    ]
    streams = ["vl_dist_load", "vl_dist_rest", "vl_prox_load", "vl_prox_rest"]
    stream_features = [f"{s}_{f}" for s in streams for f in feature_suffixes]
    derived = [
        "delta_rms_prox_dist_load", "ratio_rms_prox_dist_load",
        "delta_mdf_prox_dist_load", "delta_we_prox_dist_load",
        "delta_rms_prox_dist_rest", "ratio_rms_prox_dist_rest",
        "delta_mdf_prox_dist_rest", "delta_we_prox_dist_rest",
    ]
    kinematic = [
        "cadence_mean_rpm", "cadence_cv",
        "load_duration_ms", "rest_duration_ms", "load_rest_ratio",
        "load_duration_cv", "rest_duration_cv",
    ]
    emg_cv = [
        "vl_dist_load_rms_cv", "vl_dist_rest_rms_cv",
        "vl_prox_load_rms_cv", "vl_prox_rest_rms_cv",
    ]
    variability = [
        feat
        for scale in ("30s", "60s", "120s")
        for feat in (
            f"load_trend_cv_slope_{scale}", f"load_trend_cv_ratio_{scale}",
            f"rest_trend_cv_slope_{scale}", f"rest_trend_cv_ratio_{scale}",
            f"load_sampen_{scale}", f"rest_sampen_{scale}",
        )
    ]
    return stream_features + derived + kinematic + emg_cv + variability


def _build_session_params_row(session: PhasedSession) -> dict[str, object]:
    """Строит строку session_params для одного участника."""
    return {
        "subject_id": session.subject_id,
        "emg_vl_dist_baseline_rms": session.emg_vl_dist_baseline_rms,
        "emg_vl_prox_baseline_rms": session.emg_vl_prox_baseline_rms,
        "pca_axis_x": float(session.pca_axis[0]),
        "pca_axis_y": float(session.pca_axis[1]),
        "pca_axis_z": float(session.pca_axis[2]),
        "nirs_smo2_baseline_mean": session.nirs_smo2_baseline_mean,
        "emg_sample_rate_hz": session.emg_sample_rate_hz,
    }


# ─────────────────────── Сборка таблицы ───────────────────────

def build_emg_kinematics_table(
    subjects_path: Path,
    windows_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит таблицы признаков EMG/кинематики и параметров сессий по всем участникам.

    Параметры
    ----------
    subjects_path : Path
        Путь к subjects.parquet.
    windows_path : Path
        Путь к windows.parquet.

    Возвращает
    ----------
    Кортеж (features_df, session_params_df).
    """
    subjects = load_subjects_table(subjects_path)
    windows = load_windows_table(windows_path)

    features_rows: list[dict[str, object]] = []
    session_params_rows: list[dict[str, object]] = []

    for subject_row in subjects.itertuples():
        subject_id = str(subject_row.subject_id)
        source_h5_path = Path(str(subject_row.source_h5_path))
        subject_windows = windows[windows["subject_id"] == subject_id]

        print(f"  Участник {subject_id}: строим PhasedSession...", flush=True)
        try:
            session = build_phased_session(source_h5_path, subject_id)
        except Exception as exc:
            # При ошибке детекции фаз сохраняем NaN-строки для всех окон участника,
            # чтобы итоговая таблица имела строго 1 строку на окно (контракт 3035 строк).
            print(f"  ПРЕДУПРЕЖДЕНИЕ: {subject_id} — ошибка detect_phases: {exc}", flush=True)
            for window_row in subject_windows.itertuples():
                nan_row: dict[str, object] = {k: float("nan") for k in _all_feature_names()}
                nan_row.update({
                    "cycles_count": 0,
                    "emg_coverage_fraction": float("nan"),
                    "kinematics_coverage_fraction": float("nan"),
                    "emg_valid": 0,
                    "kinematics_valid": 0,
                    "window_id": str(window_row.window_id),
                    "subject_id": subject_id,
                })
                features_rows.append(nan_row)
            continue

        session_params_rows.append(_build_session_params_row(session))

        for window_row in subject_windows.itertuples():
            row = extract_emg_kinematics_features(
                session=session,
                window_start_sec=float(window_row.window_start_sec),
                window_end_sec=float(window_row.window_end_sec),
            )
            row["window_id"] = str(window_row.window_id)
            row["subject_id"] = subject_id
            features_rows.append(row)

    features_df = pd.DataFrame(features_rows)
    session_params_df = pd.DataFrame(session_params_rows)

    if not features_df.empty:
        features_df = features_df.sort_values(["subject_id", "window_id"]).reset_index(drop=True)
        # Переставляем ключи в начало
        key_cols = ["window_id", "subject_id"]
        other_cols = [c for c in features_df.columns if c not in key_cols]
        features_df = features_df[key_cols + other_cols]

    return features_df, session_params_df
