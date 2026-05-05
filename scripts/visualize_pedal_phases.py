#!/usr/bin/env python3
"""Строит интервалы push/pull по гироскопам из finaltest.h5."""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.widgets import Slider


DATA_DIR = Path("/Users/tascan/Desktop/диссер/data")
DEFAULT_FILE_NAME = "finaltest.h5"
DEFAULT_OUTPUT_NAME = "pedal_phases.json"
DEFAULT_CALIBRATION_POWER_W = 60.0
DEFAULT_CALIBRATION_TAIL_SEC = 75.0
DEFAULT_CALIBRATION_MARGIN_SEC = 15.0
DEFAULT_MIN_CADENCE_RPM = 50.0
DEFAULT_MAX_CADENCE_RPM = 120.0
DEFAULT_BANDPASS_LOW_HZ = 0.4
DEFAULT_BANDPASS_HIGH_HZ = 4.0
DEFAULT_PEAK_PROMINENCE_STD = 0.5
DEFAULT_ZOOM_DURATION_SEC = 20.0
MIN_AUTOCORR_SCORE = 0.20


@dataclass(frozen=True)
class StageInterval:
    """Описание одной ступени мощности."""

    power_w: float
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class SensorProjection:
    """Результат PCA-проекции для одного гироскопа."""

    sensor_name: str
    sample_rate_hz: float
    pca_vector: np.ndarray
    centered_xyz: np.ndarray
    projected_raw: np.ndarray
    projected_filtered: np.ndarray
    calibration_mean: np.ndarray
    autocorr_score: float


@dataclass(frozen=True)
class SignalChannel:
    """Один временной ряд сигнала из HDF5."""

    channel_name: str
    sample_rate_hz: float
    timestamps_sec: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class CyclePhase:
    """Один цикл педалирования с интервалами push/pull."""

    cycle_index: int
    push_start_index: int
    push_end_index: int
    pull_start_index: int
    pull_end_index: int
    peak_index: int
    push_start_sec: float
    push_end_sec: float
    pull_start_sec: float
    pull_end_sec: float
    peak_time_sec: float
    cadence_rpm: float
    push_duration_sec: float
    pull_duration_sec: float
    cycle_duration_sec: float


@dataclass(frozen=True)
class PhaseDetectionResult:
    """Итог детекции фаз педалирования."""

    participant_name: str
    source_h5_path: Path
    selected_sensor: SensorProjection
    alternate_sensor: SensorProjection | None
    stage_60w: StageInterval
    calibration_start_sec: float
    calibration_end_sec: float
    cycles: tuple[CyclePhase, ...]
    timestamps_sec: np.ndarray
    emg_channels: tuple[SignalChannel, ...]


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description=(
            "Строит интервалы push/pull по гироскопам из finaltest.h5, "
            "сохраняет JSON и показывает график."
        )
    )
    parser.add_argument("participant", help="Название папки участника внутри data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Папка data с подпапками участников.",
    )
    parser.add_argument(
        "--file-name",
        default=DEFAULT_FILE_NAME,
        help="Имя HDF5-файла внутри папки участника.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Куда сохранить JSON с интервалами. По умолчанию рядом с finaltest.h5.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Куда сохранить PNG-график. Если не задано, график откроется в окне.",
    )
    parser.add_argument(
        "--sensor",
        choices=("auto", "vl", "rf"),
        default="auto",
        help="Какой гироскоп использовать: авто-выбор, VL или RF.",
    )
    parser.add_argument(
        "--calibration-tail-sec",
        type=float,
        default=DEFAULT_CALIBRATION_TAIL_SEC,
        help="Сколько секунд от конца ступени брать в хвостовой калибровочный блок.",
    )
    parser.add_argument(
        "--calibration-margin-sec",
        type=float,
        default=DEFAULT_CALIBRATION_MARGIN_SEC,
        help="Сколько секунд отступить от конца ступени перед калибровкой.",
    )
    parser.add_argument(
        "--min-cadence-rpm",
        type=float,
        default=DEFAULT_MIN_CADENCE_RPM,
        help="Минимально допустимый каданс для цикла.",
    )
    parser.add_argument(
        "--max-cadence-rpm",
        type=float,
        default=DEFAULT_MAX_CADENCE_RPM,
        help="Максимально допустимый каданс для цикла.",
    )
    parser.add_argument(
        "--peak-prominence-std",
        type=float,
        default=DEFAULT_PEAK_PROMINENCE_STD,
        help="Минимальная prominence как доля от STD фильтрованного сигнала в окне калибровки.",
    )
    parser.add_argument(
        "--zoom-start-sec",
        type=float,
        default=None,
        help="Начало увеличенного окна на графике.",
    )
    parser.add_argument(
        "--zoom-duration-sec",
        type=float,
        default=DEFAULT_ZOOM_DURATION_SEC,
        help="Длина увеличенного окна на графике.",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Не сохранять JSON с интервалами.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    """Нормализует имя папки для устойчивого сравнения Unicode."""
    return unicodedata.normalize("NFC", value).casefold().strip()


def resolve_participant_dir(data_dir: Path, participant_name: str) -> Path:
    """Ищет папку участника по имени с нормализацией Unicode."""
    expected = normalize_name(participant_name)
    for candidate in data_dir.iterdir():
        if candidate.is_dir() and normalize_name(candidate.name) == expected:
            return candidate
    raise FileNotFoundError(
        f"Не найдена папка участника '{participant_name}' в '{data_dir}'."
    )


def load_channel(handle: h5py.File, channel_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Загружает timestamps и values одного канала."""
    group = handle["channels"][channel_name]
    return group["timestamps"][:].astype(float), group["values"][:].astype(float)


def load_signal_channel(
    handle: h5py.File,
    channel_name: str,
    anchor_ms: float,
) -> SignalChannel:
    """Загружает канал сигнала для последующей визуализации."""
    raw_timestamps, values = load_channel(handle, channel_name)
    sample_rate_hz = 1.0 / float(np.median(np.diff(raw_timestamps)) / 1000.0)
    return SignalChannel(
        channel_name=channel_name,
        sample_rate_hz=sample_rate_hz,
        timestamps_sec=to_relative_seconds(raw_timestamps, anchor_ms),
        values=values.astype(np.float32),
    )


def to_relative_seconds(raw_timestamps_ms: np.ndarray, anchor_ms: float) -> np.ndarray:
    """Переводит временную ось HDF5 в секунды относительно начала записи."""
    return (np.asarray(raw_timestamps_ms, dtype=float) - float(anchor_ms)) / 1000.0


def load_power_stages(handle: h5py.File, anchor_ms: float) -> tuple[StageInterval, ...]:
    """Восстанавливает интервалы ступеней мощности из label-канала."""
    raw_ts, values = load_channel(handle, "power.label")
    times_sec = to_relative_seconds(raw_ts, anchor_ms)
    stages: list[StageInterval] = []
    for index, power in enumerate(values):
        start_sec = float(times_sec[index])
        if index + 1 < len(times_sec):
            end_sec = float(times_sec[index + 1])
        else:
            end_sec = float(handle.attrs["stop_time_sec"])
        if end_sec <= start_sec:
            continue
        stages.append(StageInterval(power_w=float(power), start_sec=start_sec, end_sec=end_sec))
    return tuple(stages)


def find_stage_by_power(stages: Iterable[StageInterval], power_w: float) -> StageInterval:
    """Ищет ступень по мощности."""
    for stage in stages:
        if math.isclose(stage.power_w, power_w, rel_tol=0.0, abs_tol=0.5):
            return stage
    raise ValueError(f"Не найдена ступень мощности {power_w:.1f} Вт.")


def build_calibration_window(
    stage: StageInterval,
    tail_sec: float,
    margin_sec: float,
) -> tuple[float, float]:
    """Строит окно калибровки PCA внутри выбранной ступени."""
    if stage.end_sec - stage.start_sec <= margin_sec + 10.0:
        raise ValueError(
            f"Ступень {stage.power_w:.1f} Вт слишком короткая для калибровки."
        )
    end_sec = stage.end_sec - margin_sec
    start_sec = max(stage.start_sec, end_sec - tail_sec)
    if end_sec - start_sec < 20.0:
        raise ValueError(
            "Окно калибровки получилось слишком коротким. "
            "Проверьте ступень мощности и параметры калибровки."
        )
    return float(start_sec), float(end_sec)


def build_pca_vector(calibration_xyz: np.ndarray) -> np.ndarray:
    """Строит главный вектор проекции через SVD."""
    centered = calibration_xyz - calibration_xyz.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0].astype(float)


def bandpass_pedaling(signal: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Оставляет диапазон частот, характерный для педалирования."""
    b, a = butter(
        3,
        [DEFAULT_BANDPASS_LOW_HZ, DEFAULT_BANDPASS_HIGH_HZ],
        btype="bandpass",
        fs=sample_rate_hz,
    )
    return filtfilt(b, a, signal)


def fix_projection_sign(
    filtered_signal: np.ndarray,
    timestamps_sec: np.ndarray,
    stage_60w: StageInterval,
) -> np.ndarray:
    """Делает первый значимый пик на ступени 60 Вт положительным."""
    inspection_mask = (timestamps_sec >= stage_60w.start_sec) & (
        timestamps_sec <= min(stage_60w.end_sec, stage_60w.start_sec + 20.0)
    )
    inspection = filtered_signal[inspection_mask]
    if inspection.size == 0:
        return filtered_signal
    peak_index = int(np.argmax(np.abs(inspection)))
    if inspection[peak_index] < 0:
        return -filtered_signal
    return filtered_signal


def autocorrelation_score(
    signal: np.ndarray,
    sample_rate_hz: float,
    min_cadence_rpm: float,
    max_cadence_rpm: float,
) -> float:
    """Оценивает периодичность сигнала через нормированную автокорреляцию."""
    centered = signal - float(np.mean(signal))
    autocorr = np.correlate(centered, centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    lag_min = max(1, int(sample_rate_hz * 60.0 / max_cadence_rpm))
    lag_max = max(lag_min + 1, int(sample_rate_hz * 60.0 / min_cadence_rpm))
    if lag_max >= len(autocorr):
        lag_max = len(autocorr) - 1
    peak_lag = lag_min + int(np.argmax(autocorr[lag_min:lag_max]))
    return float(autocorr[peak_lag] / (autocorr[0] + 1e-12))


def build_sensor_projection(
    handle: h5py.File,
    anchor_ms: float,
    sensor_name: str,
    calibration_start_sec: float,
    calibration_end_sec: float,
    stage_60w: StageInterval,
    min_cadence_rpm: float,
    max_cadence_rpm: float,
) -> tuple[SensorProjection, np.ndarray]:
    """Загружает гироскоп, строит PCA-проекцию и фильтрует её."""
    timestamps_ms, x = load_channel(handle, f"{sensor_name}.x")
    _, y = load_channel(handle, f"{sensor_name}.y")
    _, z = load_channel(handle, f"{sensor_name}.z")
    timestamps_sec = to_relative_seconds(timestamps_ms, anchor_ms)
    gyro_xyz = np.column_stack([x, y, z]).astype(float)

    calibration_mask = (timestamps_sec >= calibration_start_sec) & (
        timestamps_sec <= calibration_end_sec
    )
    if int(np.sum(calibration_mask)) < 100:
        raise ValueError(
            f"В окне калибровки слишком мало точек для гироскопа '{sensor_name}'."
        )

    calibration_xyz = gyro_xyz[calibration_mask]
    pca_vector = build_pca_vector(calibration_xyz)
    calibration_mean = calibration_xyz.mean(axis=0).astype(float)
    centered_xyz = gyro_xyz - calibration_mean[None, :]
    projected_raw = centered_xyz @ pca_vector

    sample_rate_hz = 1.0 / float(np.median(np.diff(timestamps_sec)))
    projected_filtered = bandpass_pedaling(projected_raw, sample_rate_hz)
    projected_filtered = fix_projection_sign(
        projected_filtered,
        timestamps_sec=timestamps_sec,
        stage_60w=stage_60w,
    )

    calibration_filtered = projected_filtered[calibration_mask]
    score = autocorrelation_score(
        calibration_filtered,
        sample_rate_hz=sample_rate_hz,
        min_cadence_rpm=min_cadence_rpm,
        max_cadence_rpm=max_cadence_rpm,
    )

    result = SensorProjection(
        sensor_name=sensor_name,
        sample_rate_hz=sample_rate_hz,
        pca_vector=pca_vector,
        centered_xyz=centered_xyz,
        projected_raw=projected_raw,
        projected_filtered=projected_filtered,
        calibration_mean=calibration_mean,
        autocorr_score=score,
    )
    return result, timestamps_sec


def choose_sensor(
    projections: dict[str, SensorProjection],
    sensor_mode: str,
) -> tuple[SensorProjection, SensorProjection | None]:
    """Выбирает рабочий гироскоп."""
    if sensor_mode == "vl":
        return projections["trigno.vl.avanti.gyro"], projections["trigno.rf.avanti.gyro"]
    if sensor_mode == "rf":
        return projections["trigno.rf.avanti.gyro"], projections["trigno.vl.avanti.gyro"]

    ordered = sorted(
        projections.values(),
        key=lambda item: item.autocorr_score,
        reverse=True,
    )
    selected = ordered[0]
    alternate = ordered[1] if len(ordered) > 1 else None
    return selected, alternate


def detect_cycles(
    timestamps_sec: np.ndarray,
    filtered_signal: np.ndarray,
    sample_rate_hz: float,
    calibration_mask: np.ndarray,
    min_cadence_rpm: float,
    max_cadence_rpm: float,
    peak_prominence_std: float,
) -> tuple[CyclePhase, ...]:
    """Строит интервалы фазы нагрузки и отдыха по минимумам и пикам."""
    calibration_segment = filtered_signal[calibration_mask]
    prominence = max(
        float(np.std(calibration_segment)) * peak_prominence_std,
        1e-8,
    )
    min_distance_samples = max(1, int(sample_rate_hz * 60.0 / max_cadence_rpm))
    peaks, _ = find_peaks(
        filtered_signal,
        distance=min_distance_samples,
        prominence=prominence,
    )
    troughs, _ = find_peaks(
        -filtered_signal,
        distance=min_distance_samples,
        prominence=prominence,
    )
    if len(peaks) < 3 or len(troughs) < 3:
        raise ValueError("Не удалось найти достаточно экстремумов педалирования.")

    min_cycle_sec = 60.0 / max_cadence_rpm
    max_cycle_sec = 60.0 / min_cadence_rpm
    cycles: list[CyclePhase] = []

    for peak_index in peaks:
        previous_troughs = troughs[troughs < peak_index]
        next_troughs = troughs[troughs > peak_index]
        if len(previous_troughs) == 0 or len(next_troughs) == 0:
            continue

        push_start_index = int(previous_troughs[-1])
        push_end_index = int(peak_index)
        next_push_start_index = int(next_troughs[0])

        push_start_sec = float(timestamps_sec[push_start_index])
        push_end_sec = float(timestamps_sec[push_end_index])
        next_push_start_sec = float(timestamps_sec[next_push_start_index])
        cycle_duration_sec = next_push_start_sec - push_start_sec
        if cycle_duration_sec < min_cycle_sec or cycle_duration_sec > max_cycle_sec:
            continue

        cadence_rpm = 60.0 / cycle_duration_sec
        push_duration_sec = push_end_sec - push_start_sec
        pull_duration_sec = next_push_start_sec - push_end_sec
        if push_duration_sec <= 0.0 or pull_duration_sec <= 0.0:
            continue

        cycles.append(
            CyclePhase(
                cycle_index=len(cycles),
                push_start_index=push_start_index,
                push_end_index=push_end_index,
                pull_start_index=push_end_index,
                pull_end_index=next_push_start_index,
                peak_index=int(peak_index),
                push_start_sec=push_start_sec,
                push_end_sec=push_end_sec,
                pull_start_sec=push_end_sec,
                pull_end_sec=next_push_start_sec,
                peak_time_sec=float(timestamps_sec[peak_index]),
                cadence_rpm=float(cadence_rpm),
                push_duration_sec=float(push_duration_sec),
                pull_duration_sec=float(pull_duration_sec),
                cycle_duration_sec=float(cycle_duration_sec),
            )
        )

    if len(cycles) < 10:
        raise ValueError(
            "После фильтрации осталось слишком мало циклов. "
            "Проверьте качество сигнала или параметры детекции."
        )
    return tuple(cycles)


def detect_phases(
    participant_name: str,
    source_h5_path: Path,
    sensor_mode: str,
    calibration_tail_sec: float,
    calibration_margin_sec: float,
    min_cadence_rpm: float,
    max_cadence_rpm: float,
    peak_prominence_std: float,
) -> PhaseDetectionResult:
    """Полный пайплайн детекции фаз педалирования."""
    with h5py.File(source_h5_path, "r") as handle:
        anchor_ms = float(handle["channels/moxy.smo2/timestamps"][0])
        stages = load_power_stages(handle, anchor_ms=anchor_ms)
        stage_60w = find_stage_by_power(stages, DEFAULT_CALIBRATION_POWER_W)
        calibration_start_sec, calibration_end_sec = build_calibration_window(
            stage=stage_60w,
            tail_sec=calibration_tail_sec,
            margin_sec=calibration_margin_sec,
        )
        emg_channels = (
            load_signal_channel(handle, "trigno.vl.avanti", anchor_ms=anchor_ms),
            load_signal_channel(handle, "trigno.rf.avanti", anchor_ms=anchor_ms),
        )

        projections: dict[str, SensorProjection] = {}
        timestamps_sec_ref: np.ndarray | None = None
        for sensor_name in ("trigno.vl.avanti.gyro", "trigno.rf.avanti.gyro"):
            projection, timestamps_sec = build_sensor_projection(
                handle=handle,
                anchor_ms=anchor_ms,
                sensor_name=sensor_name,
                calibration_start_sec=calibration_start_sec,
                calibration_end_sec=calibration_end_sec,
                stage_60w=stage_60w,
                min_cadence_rpm=min_cadence_rpm,
                max_cadence_rpm=max_cadence_rpm,
            )
            projections[sensor_name] = projection
            if timestamps_sec_ref is None:
                timestamps_sec_ref = timestamps_sec

    if timestamps_sec_ref is None:
        raise RuntimeError("Не удалось загрузить временную ось гироскопа.")

    selected_sensor, alternate_sensor = choose_sensor(projections, sensor_mode=sensor_mode)
    if sensor_mode == "auto" and selected_sensor.autocorr_score < MIN_AUTOCORR_SCORE:
        raise ValueError(
            "Автовыбор гироскопа дал слишком слабую периодичность. "
            "Проверьте запись или выберите датчик вручную."
        )

    calibration_mask = (timestamps_sec_ref >= calibration_start_sec) & (
        timestamps_sec_ref <= calibration_end_sec
    )
    cycles = detect_cycles(
        timestamps_sec=timestamps_sec_ref,
        filtered_signal=selected_sensor.projected_filtered,
        sample_rate_hz=selected_sensor.sample_rate_hz,
        calibration_mask=calibration_mask,
        min_cadence_rpm=min_cadence_rpm,
        max_cadence_rpm=max_cadence_rpm,
        peak_prominence_std=peak_prominence_std,
    )
    return PhaseDetectionResult(
        participant_name=participant_name,
        source_h5_path=source_h5_path,
        selected_sensor=selected_sensor,
        alternate_sensor=alternate_sensor,
        stage_60w=stage_60w,
        calibration_start_sec=calibration_start_sec,
        calibration_end_sec=calibration_end_sec,
        cycles=cycles,
        timestamps_sec=timestamps_sec_ref,
        emg_channels=emg_channels,
    )


def build_output_payload(result: PhaseDetectionResult) -> dict[str, object]:
    """Готовит JSON-представление результата."""
    alternate = None
    if result.alternate_sensor is not None:
        alternate = {
            "sensor_name": result.alternate_sensor.sensor_name,
            "autocorr_score": result.alternate_sensor.autocorr_score,
        }

    cycles_payload = []
    for cycle in result.cycles:
        cycles_payload.append(
            {
                "cycle_index": cycle.cycle_index,
                "push_start_sec": cycle.push_start_sec,
                "push_end_sec": cycle.push_end_sec,
                "pull_start_sec": cycle.pull_start_sec,
                "pull_end_sec": cycle.pull_end_sec,
                "load_start_sec": cycle.push_start_sec,
                "load_end_sec": cycle.push_end_sec,
                "rest_start_sec": cycle.pull_start_sec,
                "rest_end_sec": cycle.pull_end_sec,
                "peak_time_sec": cycle.peak_time_sec,
                "cadence_rpm": cycle.cadence_rpm,
                "push_duration_sec": cycle.push_duration_sec,
                "pull_duration_sec": cycle.pull_duration_sec,
                "load_duration_sec": cycle.push_duration_sec,
                "rest_duration_sec": cycle.pull_duration_sec,
                "cycle_duration_sec": cycle.cycle_duration_sec,
                "push_start_index": cycle.push_start_index,
                "push_end_index": cycle.push_end_index,
                "pull_start_index": cycle.pull_start_index,
                "pull_end_index": cycle.pull_end_index,
                "peak_index": cycle.peak_index,
            }
        )

    return {
        "participant_name": result.participant_name,
        "source_h5": str(result.source_h5_path),
        "phase_definition": (
            "Нагрузка = восходящая ветвь PCA-проекции от локального минимума до следующего пика; "
            "отдых = нисходящая ветвь от пика до следующего локального минимума."
        ),
        "selected_sensor": {
            "sensor_name": result.selected_sensor.sensor_name,
            "sample_rate_hz": result.selected_sensor.sample_rate_hz,
            "autocorr_score": result.selected_sensor.autocorr_score,
            "pca_vector": result.selected_sensor.pca_vector.tolist(),
            "calibration_mean": result.selected_sensor.calibration_mean.tolist(),
        },
        "alternate_sensor": alternate,
        "stage_60w": {
            "start_sec": result.stage_60w.start_sec,
            "end_sec": result.stage_60w.end_sec,
            "power_w": result.stage_60w.power_w,
        },
        "calibration_window": {
            "start_sec": result.calibration_start_sec,
            "end_sec": result.calibration_end_sec,
        },
        "cycle_count": len(result.cycles),
        "cadence_summary": {
            "median_rpm": float(np.median([cycle.cadence_rpm for cycle in result.cycles])),
            "p05_rpm": float(np.quantile([cycle.cadence_rpm for cycle in result.cycles], 0.05)),
            "p95_rpm": float(np.quantile([cycle.cadence_rpm for cycle in result.cycles], 0.95)),
        },
        "cycles": cycles_payload,
    }


def save_json(output_path: Path, payload: dict[str, object]) -> None:
    """Сохраняет JSON с интервалами."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_pca_overlay_for_window(
    emg_timestamps_sec: np.ndarray,
    emg_values: np.ndarray,
    pca_timestamps_sec: np.ndarray,
    pca_values: np.ndarray,
) -> np.ndarray:
    """Нормирует PCA-линию в амплитудный диапазон окна ЭМГ."""
    if emg_values.size == 0:
        return np.array([], dtype=float)
    interpolated_pca = np.interp(emg_timestamps_sec, pca_timestamps_sec, pca_values)
    emg_center = float(np.median(emg_values))
    pca_center = float(np.median(interpolated_pca))
    emg_scale = float(np.quantile(np.abs(emg_values - emg_center), 0.95))
    pca_scale = float(np.quantile(np.abs(interpolated_pca - pca_center), 0.95))
    if emg_scale <= 1e-12:
        emg_scale = float(np.std(emg_values) + 1e-12)
    if pca_scale <= 1e-12:
        pca_scale = float(np.std(interpolated_pca) + 1e-12)
    return emg_center + 0.9 * emg_scale * ((interpolated_pca - pca_center) / (pca_scale + 1e-12))


def render_emg_axis(
    ax: plt.Axes,
    emg_channel: SignalChannel,
    result: PhaseDetectionResult,
    window_start_sec: float,
    window_duration_sec: float,
) -> None:
    """Рисует один ЭМГ-канал с линией педалирования и интервалами фаз."""
    window_end_sec = window_start_sec + window_duration_sec
    emg_timestamps_sec = emg_channel.timestamps_sec
    emg_values = emg_channel.values
    emg_mask = (emg_timestamps_sec >= window_start_sec) & (emg_timestamps_sec <= window_end_sec)
    emg_window_timestamps = emg_timestamps_sec[emg_mask]
    emg_window_values = emg_values[emg_mask]

    ax.clear()
    if emg_window_timestamps.size == 0:
        ax.set_xlim(window_start_sec, window_end_sec)
        ax.set_title(f"{emg_channel.channel_name}: нет точек в окне")
        ax.set_xlabel("Время, с")
        ax.set_ylabel("ЭМГ, В")
        return

    overlay_values = build_pca_overlay_for_window(
        emg_timestamps_sec=emg_window_timestamps,
        emg_values=emg_window_values.astype(float),
        pca_timestamps_sec=result.timestamps_sec,
        pca_values=result.selected_sensor.projected_filtered,
    )

    channel_color = "#1f77b4" if "vl" in emg_channel.channel_name else "#2ca02c"
    ax.plot(
        emg_window_timestamps,
        emg_window_values,
        color=channel_color,
        linewidth=0.35,
        alpha=0.9,
        label=f"{emg_channel.channel_name} (ЭМГ)",
    )
    ax.plot(
        emg_window_timestamps,
        overlay_values,
        color="black",
        linewidth=1.0,
        alpha=0.95,
        label="PCA-линия педалирования (нормирована)",
    )
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)

    for cycle in result.cycles:
        if cycle.pull_end_sec < window_start_sec or cycle.push_start_sec > window_end_sec:
            continue
        ax.axvspan(cycle.push_start_sec, cycle.push_end_sec, color="#e67e22", alpha=0.18)
        ax.axvspan(cycle.pull_start_sec, cycle.pull_end_sec, color="#5dade2", alpha=0.18)
        ax.axvline(cycle.peak_time_sec, color="#c0392b", linewidth=0.6, alpha=0.45)

    cadence_in_window = np.array(
        [
            cycle.cadence_rpm
            for cycle in result.cycles
            if cycle.pull_end_sec >= window_start_sec and cycle.push_start_sec <= window_end_sec
        ],
        dtype=float,
    )
    push_values = np.array(
        [
            cycle.push_duration_sec
            for cycle in result.cycles
            if cycle.pull_end_sec >= window_start_sec and cycle.push_start_sec <= window_end_sec
        ],
        dtype=float,
    )
    pull_values = np.array(
        [
            cycle.pull_duration_sec
            for cycle in result.cycles
            if cycle.pull_end_sec >= window_start_sec and cycle.push_start_sec <= window_end_sec
        ],
        dtype=float,
    )
    cycle_count = len(cadence_in_window)
    if cycle_count == 0:
        cadence_in_window = np.array([np.nan], dtype=float)
        push_values = np.array([np.nan], dtype=float)
        pull_values = np.array([np.nan], dtype=float)

    flags: list[str] = []
    if result.stage_60w.start_sec < window_end_sec and result.stage_60w.end_sec > window_start_sec:
        flags.append("ступень 60 Вт")
    if result.calibration_start_sec < window_end_sec and result.calibration_end_sec > window_start_sec:
        flags.append("окно калибровки PCA")
    suffix = f" | {', '.join(flags)}" if flags else ""

    text = (
        f"Циклов в окне: {cycle_count}\n"
        f"Каданс, медиана: {np.nanmedian(cadence_in_window):.1f} rpm\n"
        f"Нагрузка, медиана: {np.nanmedian(push_values) * 1000:.0f} ms\n"
        f"Отдых, медиана: {np.nanmedian(pull_values) * 1000:.0f} ms"
    )
    ax.text(
        0.99,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d0d0d0"},
    )
    ax.set_xlim(window_start_sec, window_end_sec)
    ax.set_title(
        f"{emg_channel.channel_name} | окно {window_start_sec:.1f}–{window_end_sec:.1f} с{suffix}"
    )
    ax.set_xlabel("Время, с")
    ax.set_ylabel("ЭМГ, В")
    ax.legend(loc="upper left")


def render_window_axes(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    result: PhaseDetectionResult,
    window_start_sec: float,
    window_duration_sec: float,
) -> None:
    """Перерисовывает оба ЭМГ-графика для текущего окна."""
    render_emg_axis(
        ax=ax_top,
        emg_channel=result.emg_channels[0],
        result=result,
        window_start_sec=window_start_sec,
        window_duration_sec=window_duration_sec,
    )
    render_emg_axis(
        ax=ax_bottom,
        emg_channel=result.emg_channels[1],
        result=result,
        window_start_sec=window_start_sec,
        window_duration_sec=window_duration_sec,
    )


def build_interactive_slider(
    fig: plt.Figure,
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    result: PhaseDetectionResult,
    initial_zoom_start_sec: float,
    zoom_duration_sec: float,
) -> None:
    """Добавляет слайдер для просмотра 20-секундных окон по всей записи."""
    timestamps_sec = result.timestamps_sec
    min_start = float(timestamps_sec[0])
    max_start = max(min_start, float(timestamps_sec[-1] - zoom_duration_sec))
    initial_start = min(max(float(initial_zoom_start_sec), min_start), max_start)

    slider_ax = fig.add_axes([0.12, 0.06, 0.76, 0.035])
    slider = Slider(
        ax=slider_ax,
        label="Старт окна, с",
        valmin=min_start,
        valmax=max_start,
        valinit=initial_start,
        valstep=1.0,
    )

    def update(window_start_sec: float) -> None:
        window_start_sec = float(window_start_sec)
        render_window_axes(
            ax_top=ax_top,
            ax_bottom=ax_bottom,
            result=result,
            window_start_sec=window_start_sec,
            window_duration_sec=zoom_duration_sec,
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(initial_start)

    # Сохраняем ссылки на виджеты, иначе matplotlib может их собрать.
    fig._phase_slider = slider
    fig._phase_slider_ax = slider_ax
    fig._phase_slider_update = update


def plot_result(
    result: PhaseDetectionResult,
    zoom_start_sec: float,
    zoom_duration_sec: float,
    output_plot: Path | None,
) -> None:
    """Рисует два синхронных 20-секундных окна и слайдер перемещения."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    ax_top, ax_bottom = axes

    if output_plot is not None:
        render_window_axes(
            ax_top=ax_top,
            ax_bottom=ax_bottom,
            result=result,
            window_start_sec=zoom_start_sec,
            window_duration_sec=zoom_duration_sec,
        )
        fig.tight_layout()
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot, dpi=150)
        plt.close(fig)
    else:
        fig.subplots_adjust(bottom=0.14, hspace=0.28)
        build_interactive_slider(
            fig=fig,
            ax_top=ax_top,
            ax_bottom=ax_bottom,
            result=result,
            initial_zoom_start_sec=zoom_start_sec,
            zoom_duration_sec=zoom_duration_sec,
        )
        plt.show()


def describe_result(result: PhaseDetectionResult) -> None:
    """Печатает сводку по детекции фаз."""
    cadence_values = np.array([cycle.cadence_rpm for cycle in result.cycles], dtype=float)
    push_values = np.array([cycle.push_duration_sec for cycle in result.cycles], dtype=float)
    pull_values = np.array([cycle.pull_duration_sec for cycle in result.cycles], dtype=float)

    print(f"Папка: {result.participant_name}")
    print(f"Файл: {result.source_h5_path}")
    print(f"Выбранный гироскоп: {result.selected_sensor.sensor_name}")
    print(f"Оценка автокорреляции: {result.selected_sensor.autocorr_score:.3f}")
    if result.alternate_sensor is not None:
        print(
            f"Альтернативный гироскоп: {result.alternate_sensor.sensor_name} "
            f"(score={result.alternate_sensor.autocorr_score:.3f})"
        )
    print(
        f"Ступень калибровки: {result.stage_60w.power_w:.0f} Вт | "
        f"{result.stage_60w.start_sec:.2f}–{result.stage_60w.end_sec:.2f} с"
    )
    print(
        f"Окно калибровки PCA: {result.calibration_start_sec:.2f}–"
        f"{result.calibration_end_sec:.2f} с"
    )
    print(f"Число валидных циклов: {len(result.cycles)}")
    print(
        f"Каданс: median={np.median(cadence_values):.1f} rpm | "
        f"p05={np.quantile(cadence_values, 0.05):.1f} | "
        f"p95={np.quantile(cadence_values, 0.95):.1f}"
    )
    print(
        f"Нагрузка: median={np.median(push_values) * 1000:.0f} ms | "
        f"Отдых: median={np.median(pull_values) * 1000:.0f} ms"
    )


def main() -> None:
    """Точка входа CLI."""
    args = parse_args()
    participant_dir = resolve_participant_dir(args.data_dir, args.participant)
    participant_name = participant_dir.name
    source_h5_path = participant_dir / args.file_name
    if not source_h5_path.exists():
        raise FileNotFoundError(f"Не найден файл '{source_h5_path}'.")

    result = detect_phases(
        participant_name=participant_name,
        source_h5_path=source_h5_path,
        sensor_mode=args.sensor,
        calibration_tail_sec=args.calibration_tail_sec,
        calibration_margin_sec=args.calibration_margin_sec,
        min_cadence_rpm=args.min_cadence_rpm,
        max_cadence_rpm=args.max_cadence_rpm,
        peak_prominence_std=args.peak_prominence_std,
    )
    describe_result(result)

    if not args.no_save_json:
        output_json = args.output_json or (participant_dir / DEFAULT_OUTPUT_NAME)
        payload = build_output_payload(result)
        save_json(output_json, payload)
        print(f"JSON сохранён: {output_json}")

    zoom_start_sec = (
        args.zoom_start_sec
        if args.zoom_start_sec is not None
        else max(result.stage_60w.start_sec + 5.0, result.calibration_start_sec - 5.0)
    )
    plot_result(
        result=result,
        zoom_start_sec=float(zoom_start_sec),
        zoom_duration_sec=float(args.zoom_duration_sec),
        output_plot=args.output_plot,
    )
    if args.output_plot is not None:
        print(f"График сохранён: {args.output_plot}")


if __name__ == "__main__":
    main()
