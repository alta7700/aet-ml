#!/usr/bin/env python3
"""Ищет LT2 по fulltest.h5 и визуализирует лактат, DFA-a1 и HHb."""

from __future__ import annotations

import argparse
import difflib
from dataclasses import dataclass
from pathlib import Path
import sys
import unicodedata

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.lt2 import compute_lt2 as compute_lt2_method
from methods.lt2 import load_fulltest as load_fulltest_method


DEFAULT_FULLTEST_FILENAME = "fulltest.h5"
DFA_WINDOW_SEC = 120.0
DFA_STEP_SEC = 5.0
DFA_SCALES = tuple(range(4, 17))
DFA_THRESHOLD = 0.50
POST_STOP_LACTATE_EXTENSION_SEC = 60.0
MIN_UNIQUE_LACTATE_STAGES = 4
MAX_ACCEPTABLE_ARTIFACT_SHARE = 0.05


@dataclass(frozen=True)
class FulltestData:
    """Данные, извлечённые из fulltest.h5."""

    path: Path
    stop_time_sec: float
    power_times_sec: np.ndarray
    power_values_w: np.ndarray
    lactate_times_sec: np.ndarray
    lactate_values_mmol: np.ndarray
    rr_times_sec: np.ndarray
    rr_values_sec: np.ndarray
    hhb_times_sec: np.ndarray
    hhb_values: np.ndarray


@dataclass(frozen=True)
class LactatePoint:
    """Одна точка лактатной кривой с привязкой к рабочей мощности."""

    sample_time_sec: float
    effective_time_sec: float
    power_w: float
    lactate_mmol: float


@dataclass(frozen=True)
class ModDmaxResult:
    """Результат расчёта modified Dmax."""

    points: tuple[LactatePoint, ...]
    start_index: int
    lt2_power_w: float
    lt2_lactate_mmol: float
    lt2_time_sec: float
    interval_start_sec: float
    interval_end_sec: float
    interval_start_power_w: float
    interval_end_power_w: float
    interval_start_lactate_mmol: float
    interval_end_lactate_mmol: float
    fit_x: np.ndarray
    fit_y: np.ndarray
    pchip_fit_y: np.ndarray
    line_y: np.ndarray
    distance_x: np.ndarray
    distance_y: np.ndarray
    pchip_lt2_power_w: float
    pchip_lt2_lactate_mmol: float
    pchip_lt2_time_sec: float


@dataclass(frozen=True)
class DfaResult:
    """Результат расчёта DFA-a1."""

    times_sec: np.ndarray
    alpha1: np.ndarray
    artifact_share: np.ndarray
    crossing_time_sec: float | None


@dataclass(frozen=True)
class HhbResult:
    """Кандидаты LT2 по каналу HHb."""

    window_start_sec: float
    window_end_sec: float
    smooth_times_sec: np.ndarray
    smooth_values: np.ndarray
    peak_time_sec: float | None
    peak_value: float | None
    breakpoint_time_sec: float | None


@dataclass(frozen=True)
class Lt2Result:
    """Итог по LT2: базовая лактатная оценка и уточняющие кандидаты."""

    moddmax: ModDmaxResult
    dfa: DfaResult
    hhb: HhbResult
    refined_time_sec: float
    refined_sources: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Определяет LT2 по fulltest.h5: modified Dmax по лактату, "
            "DFA-a1 по RR и HHb-маркеры в окрестности порога."
        )
    )
    parser.add_argument(
        "subject",
        help="Точное имя папки участника внутри data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Каталог с папками участников.",
    )
    parser.add_argument(
        "--fulltest",
        type=Path,
        help=(
            "Путь к fulltest.h5. "
            "По умолчанию используется fulltest.h5 в папке участника."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Путь для сохранения PNG. Если не указан, график будет показан на экране.",
    )
    return parser.parse_args()


def resolve_subject_dir(data_dir: Path, subject_name: str) -> Path:
    """Находит папку участника и подсказывает близкие варианты при опечатке."""

    exact_path = data_dir / subject_name
    if exact_path.is_dir():
        return exact_path

    normalized_target = unicodedata.normalize("NFC", subject_name).casefold()
    candidates = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    for candidate in candidates:
        if unicodedata.normalize("NFC", candidate).casefold() == normalized_target:
            return data_dir / candidate

    suggestions = difflib.get_close_matches(subject_name, candidates, n=3, cutoff=0.4)
    suggestion_text = f" Ближайшие варианты: {', '.join(suggestions)}." if suggestions else ""
    raise FileNotFoundError(
        f"Папка '{subject_name}' не найдена в {data_dir}.{suggestion_text}"
    )


def load_channel(handle: h5py.File, channel_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Загружает timestamps и values одного канала."""

    group = handle[f"channels/{channel_name}"]
    return group["timestamps"][:].astype(float), group["values"][:].astype(float)


def to_relative_seconds(raw_timestamps: np.ndarray, anchor_ms: float) -> np.ndarray:
    """Приводит временную шкалу HDF5 к секундам от начала сессии."""

    return (np.asarray(raw_timestamps, dtype=float) - anchor_ms) / 1000.0


def load_fulltest(path: Path) -> FulltestData:
    """Загружает нужные для LT2 каналы из fulltest.h5."""

    with h5py.File(path, "r") as handle:
        anchor_ms = float(handle["channels/moxy.smo2/timestamps"][0])
        stop_time_sec = float(handle.attrs["stop_time_sec"])

        power_ts, power_vals = load_channel(handle, "power.label")
        lactate_ts, lactate_vals = load_channel(handle, "lactate")
        rr_ts, rr_vals = load_channel(handle, "zephyr.rr")
        hhb_ts, hhb_vals = load_channel(handle, "train.red.hhb.unfiltered")

    return FulltestData(
        path=path,
        stop_time_sec=stop_time_sec,
        power_times_sec=to_relative_seconds(power_ts, anchor_ms),
        power_values_w=power_vals,
        lactate_times_sec=to_relative_seconds(lactate_ts, anchor_ms),
        lactate_values_mmol=lactate_vals,
        rr_times_sec=to_relative_seconds(rr_ts, anchor_ms),
        rr_values_sec=rr_vals,
        hhb_times_sec=to_relative_seconds(hhb_ts, anchor_ms),
        hhb_values=hhb_vals,
    )


def map_time_to_power(
    power_times_sec: np.ndarray,
    power_values_w: np.ndarray,
    time_sec: float,
) -> float:
    """Возвращает последнюю известную рабочую мощность на заданный момент."""

    index = int(np.searchsorted(power_times_sec, time_sec, side="right") - 1)
    index = max(0, min(index, len(power_values_w) - 1))
    return float(power_values_w[index])


def collapse_duplicate_powers(points: list[LactatePoint]) -> tuple[LactatePoint, ...]:
    """Сводит повторные посттестовые точки на одной мощности к одной записи.

    Modified Dmax предполагает по сути одну лактатную точку на одну ступень.
    Если после stop_time кровь брали повторно, все эти значения относятся к
    последней рабочей ступени, а не к recovery-мощности. Поэтому для одной и
    той же мощности оставляем точку с максимальным лактатом.
    """

    by_power: dict[float, LactatePoint] = {}
    for point in points:
        existing = by_power.get(point.power_w)
        if existing is None or point.lactate_mmol > existing.lactate_mmol:
            by_power[point.power_w] = point
    collapsed = sorted(by_power.values(), key=lambda point: (point.power_w, point.effective_time_sec))
    return tuple(collapsed)


def build_lactate_points(data: FulltestData) -> tuple[LactatePoint, ...]:
    """Формирует точки лактатной кривой для расчёта modified Dmax."""

    allowed_until_sec = data.stop_time_sec + POST_STOP_LACTATE_EXTENSION_SEC + 1e-9
    selected: list[LactatePoint] = []
    for sample_time_sec, lactate_mmol in zip(data.lactate_times_sec, data.lactate_values_mmol):
        if sample_time_sec > allowed_until_sec:
            continue

        effective_time_sec = min(float(sample_time_sec), data.stop_time_sec)
        power_w = map_time_to_power(
            power_times_sec=data.power_times_sec,
            power_values_w=data.power_values_w,
            time_sec=effective_time_sec,
        )
        selected.append(
            LactatePoint(
                sample_time_sec=float(sample_time_sec),
                effective_time_sec=effective_time_sec,
                power_w=power_w,
                lactate_mmol=float(lactate_mmol),
            )
        )

    points = collapse_duplicate_powers(selected)
    if len(points) < MIN_UNIQUE_LACTATE_STAGES:
        raise ValueError(
            "Для modified Dmax нужно хотя бы 4 уникальные ступени лактата "
            f"до stop_time + {POST_STOP_LACTATE_EXTENSION_SEC:.0f} с."
        )
    return points


def perpendicular_distance(
    x: np.ndarray,
    y: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    """Считает перпендикулярное расстояние от точек до прямой."""

    numerator = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = np.hypot(y2 - y1, x2 - x1)
    if denominator == 0.0:
        return np.zeros_like(x, dtype=float)
    return numerator / denominator


def interpolate_time_for_power(points: tuple[LactatePoint, ...], target_power_w: float) -> float:
    """Оценивает время LT2 по рабочим ступеням и их эффективному времени."""

    powers = np.array([point.power_w for point in points], dtype=float)
    times = np.array([point.effective_time_sec for point in points], dtype=float)
    return float(np.interp(target_power_w, powers, times))


def find_max_distance_point(
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Находит точку кривой с максимальным расстоянием до линии Dmax."""

    curve_mask = curve_x >= x1 - 1e-9
    distance_x = curve_x[curve_mask]
    distance_y = curve_y[curve_mask]
    distances = perpendicular_distance(distance_x, distance_y, x1, y1, x2, y2)
    max_index = int(np.argmax(distances))
    return (
        distance_x,
        distance_y,
        float(distance_x[max_index]),
        float(distance_y[max_index]),
    )


def build_moddmax(data: FulltestData) -> ModDmaxResult:
    """Считает modified Dmax по лактатной кривой."""

    points = build_lactate_points(data)
    powers = np.array([point.power_w for point in points], dtype=float)
    lactates = np.array([point.lactate_mmol for point in points], dtype=float)

    diffs = np.diff(lactates)
    rise_indices = np.where(diffs > 0.4)[0]
    start_index = int(rise_indices[0]) if len(rise_indices) else 0

    if len(points) < MIN_UNIQUE_LACTATE_STAGES:
        raise ValueError("Для cubic polynomial modified Dmax нужно минимум 4 точки.")

    poly = np.poly1d(np.polyfit(powers, lactates, 3))
    fit_x = np.linspace(float(powers[0]), float(powers[-1]), 800)
    fit_y = poly(fit_x)
    pchip = PchipInterpolator(powers, lactates)
    pchip_fit_y = pchip(fit_x)

    x1 = float(powers[start_index])
    y1 = float(lactates[start_index])
    x2 = float(powers[-1])
    y2 = float(lactates[-1])

    distance_x, distance_y, lt2_power_w, lt2_lactate_mmol = find_max_distance_point(
        curve_x=fit_x,
        curve_y=fit_y,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    _, _, pchip_lt2_power_w, pchip_lt2_lactate_mmol = find_max_distance_point(
        curve_x=fit_x,
        curve_y=pchip_fit_y,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    lt2_time_sec = interpolate_time_for_power(points, lt2_power_w)
    pchip_lt2_time_sec = interpolate_time_for_power(points, pchip_lt2_power_w)

    insert_index = int(np.searchsorted(powers, lt2_power_w, side="right"))
    lower_index = max(0, insert_index - 1)
    upper_index = min(len(points) - 1, insert_index)
    interval_start_sec = float(points[lower_index].effective_time_sec)
    interval_end_sec = float(points[upper_index].effective_time_sec)
    interval_start_power_w = float(points[lower_index].power_w)
    interval_end_power_w = float(points[upper_index].power_w)
    interval_start_lactate_mmol = float(points[lower_index].lactate_mmol)
    interval_end_lactate_mmol = float(points[upper_index].lactate_mmol)

    line_y = y1 + (y2 - y1) * (distance_x - x1) / (x2 - x1)
    return ModDmaxResult(
        points=points,
        start_index=start_index,
        lt2_power_w=lt2_power_w,
        lt2_lactate_mmol=lt2_lactate_mmol,
        lt2_time_sec=lt2_time_sec,
        interval_start_sec=interval_start_sec,
        interval_end_sec=interval_end_sec,
        interval_start_power_w=interval_start_power_w,
        interval_end_power_w=interval_end_power_w,
        interval_start_lactate_mmol=interval_start_lactate_mmol,
        interval_end_lactate_mmol=interval_end_lactate_mmol,
        fit_x=fit_x,
        fit_y=fit_y,
        pchip_fit_y=pchip_fit_y,
        line_y=line_y,
        distance_x=distance_x,
        distance_y=distance_y,
        pchip_lt2_power_w=pchip_lt2_power_w,
        pchip_lt2_lactate_mmol=pchip_lt2_lactate_mmol,
        pchip_lt2_time_sec=pchip_lt2_time_sec,
    )


def correct_rr_window(rr_ms: np.ndarray) -> tuple[np.ndarray, float]:
    """Применяет мягкую коррекцию артефактов к окну RR.

    Полностью воспроизвести Kubios automatic correction без их реализации
    нельзя, поэтому здесь используется близкая к канонической мягкая схема:
    физиологические пределы RR + локальная медианная проверка. Доля замен
    сохраняется отдельно, чтобы видеть доверие к окну.
    """

    rr_ms = np.asarray(rr_ms, dtype=float)
    if len(rr_ms) < 20:
        return rr_ms.copy(), 1.0

    corrected = rr_ms.copy()
    invalid = ~np.isfinite(corrected)
    invalid |= corrected < 300.0
    invalid |= corrected > 2000.0

    for index in range(len(corrected)):
        left = max(0, index - 5)
        right = min(len(corrected), index + 6)
        neighborhood = corrected[left:right]
        neighborhood = neighborhood[np.isfinite(neighborhood)]
        if len(neighborhood) < 3:
            continue
        local_median = float(np.median(neighborhood))
        if local_median <= 0.0:
            continue
        if abs(corrected[index] - local_median) / local_median > 0.20:
            invalid[index] = True

    valid_indices = np.where(~invalid)[0]
    artifact_share = float(invalid.mean())
    if len(valid_indices) < 2:
        return corrected, 1.0

    corrected[invalid] = np.interp(
        np.where(invalid)[0].astype(float),
        valid_indices.astype(float),
        corrected[valid_indices],
    )
    return corrected, artifact_share


def dfa_alpha1(rr_ms: np.ndarray, scales: tuple[int, ...] = DFA_SCALES) -> float:
    """Считает short-term scaling exponent DFA-a1.

    Используется стандартный DFA первого порядка: интегрированный ряд RR,
    линейное вычитание тренда в каждом боксе и аппроксимация slope на
    логарифмических масштабах 4–16 ударов.
    """

    rr_ms = np.asarray(rr_ms, dtype=float)
    if len(rr_ms) < max(scales) * 4:
        return np.nan

    centered = rr_ms - float(np.mean(rr_ms))
    integrated = np.cumsum(centered)

    fluctuation_sizes: list[float] = []
    valid_scales: list[int] = []
    for scale in scales:
        if scale < 2:
            continue
        segment_count = len(integrated) // scale
        if segment_count < 2:
            continue

        rms_values: list[float] = []
        for direction in (0, 1):
            series = integrated if direction == 0 else integrated[::-1]
            usable = series[: segment_count * scale].reshape(segment_count, scale)
            x = np.arange(scale, dtype=float)
            for segment in usable:
                coeffs = np.polyfit(x, segment, 1)
                trend = coeffs[0] * x + coeffs[1]
                rms_values.append(float(np.sqrt(np.mean((segment - trend) ** 2))))

        rms_values = [value for value in rms_values if np.isfinite(value) and value > 0.0]
        if not rms_values:
            continue

        fluctuation_sizes.append(float(np.sqrt(np.mean(np.square(rms_values)))))
        valid_scales.append(scale)

    if len(valid_scales) < 4:
        return np.nan

    log_scales = np.log10(np.asarray(valid_scales, dtype=float))
    log_fluctuation = np.log10(np.asarray(fluctuation_sizes, dtype=float))
    slope, _ = np.polyfit(log_scales, log_fluctuation, 1)
    return float(slope)


def build_dfa_series(data: FulltestData) -> DfaResult:
    """Строит временной ряд DFA-a1 по RR-интервалам."""

    rr_times = np.asarray(data.rr_times_sec, dtype=float)
    rr_ms = np.asarray(data.rr_values_sec, dtype=float) * 1000.0
    if len(rr_times) < 50:
        raise ValueError("В zephyr.rr слишком мало точек для расчёта DFA-a1.")

    start_center = float(rr_times[0] + DFA_WINDOW_SEC / 2.0)
    end_center = float(rr_times[-1] - DFA_WINDOW_SEC / 2.0)
    centers = np.arange(start_center, end_center + 1e-9, DFA_STEP_SEC)

    alpha_values: list[float] = []
    artifact_share: list[float] = []
    for center_sec in centers:
        mask = (
            (rr_times >= center_sec - DFA_WINDOW_SEC / 2.0)
            & (rr_times <= center_sec + DFA_WINDOW_SEC / 2.0)
        )
        window_rr_ms = rr_ms[mask]
        corrected_rr_ms, share = correct_rr_window(window_rr_ms)
        artifact_share.append(share)
        alpha_values.append(dfa_alpha1(corrected_rr_ms))

    alpha = np.asarray(alpha_values, dtype=float)
    share = np.asarray(artifact_share, dtype=float)
    crossing_time = find_dfa_crossing(
        centers_sec=centers,
        alpha1=alpha,
    )
    return DfaResult(
        times_sec=centers,
        alpha1=alpha,
        artifact_share=share,
        crossing_time_sec=crossing_time,
    )


def find_dfa_crossing(centers_sec: np.ndarray, alpha1: np.ndarray) -> float | None:
    """Ищет первый нисходящий переход DFA-a1 через 0.50."""

    finite_mask = np.isfinite(alpha1)
    centers = np.asarray(centers_sec, dtype=float)[finite_mask]
    values = np.asarray(alpha1, dtype=float)[finite_mask]
    if len(values) < 2:
        return None

    for index in range(len(values) - 1):
        left = float(values[index])
        right = float(values[index + 1])
        if left >= DFA_THRESHOLD and right < DFA_THRESHOLD and right != left:
            fraction = (DFA_THRESHOLD - left) / (right - left)
            return float(centers[index] + fraction * (centers[index + 1] - centers[index]))
    return None


def smooth_to_grid(times_sec: np.ndarray, values: np.ndarray, step_sec: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Интерполирует сигнал на равномерную сетку и сглаживает окном."""

    times_sec = np.asarray(times_sec, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(times_sec) & np.isfinite(values)
    times_sec = times_sec[finite]
    values = values[finite]
    if len(times_sec) < 2:
        return np.array([]), np.array([])

    grid = np.arange(float(times_sec[0]), float(times_sec[-1]) + step_sec, step_sec)
    interpolated = np.interp(grid, times_sec, values)

    # HHb шумный, поэтому используем двухступенчатое сглаживание:
    # сначала короткую медиану, затем более мягкое среднее.
    median_radius = 2
    median_smoothed = interpolated.copy()
    for index in range(len(interpolated)):
        left = max(0, index - median_radius)
        right = min(len(interpolated), index + median_radius + 1)
        median_smoothed[index] = float(np.median(interpolated[left:right]))

    mean_radius = 7
    smoothed = median_smoothed.copy()
    for index in range(len(median_smoothed)):
        left = max(0, index - mean_radius)
        right = min(len(median_smoothed), index + mean_radius + 1)
        smoothed[index] = float(np.mean(median_smoothed[left:right]))

    return grid, smoothed


def piecewise_breakpoint(times_sec: np.ndarray, values: np.ndarray) -> float | None:
    """Ищет breakpoint по двум линейным регрессиям на сглаженном HHb."""

    if len(times_sec) < 20:
        return None

    best_index = None
    best_sse = None
    left_limit = max(5, int(len(times_sec) * 0.2))
    right_limit = min(len(times_sec) - 5, int(len(times_sec) * 0.8))
    if left_limit >= right_limit:
        return None

    for index in range(left_limit, right_limit):
        left_x = times_sec[: index + 1]
        left_y = values[: index + 1]
        right_x = times_sec[index:]
        right_y = values[index:]

        left_coeffs = np.polyfit(left_x, left_y, 1)
        right_coeffs = np.polyfit(right_x, right_y, 1)
        left_fit = np.polyval(left_coeffs, left_x)
        right_fit = np.polyval(right_coeffs, right_x)
        sse = float(np.sum((left_y - left_fit) ** 2) + np.sum((right_y - right_fit) ** 2))

        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_index = index

    return None if best_index is None else float(times_sec[best_index])


def build_hhb_markers(data: FulltestData, moddmax: ModDmaxResult) -> HhbResult:
    """Ищет HHb-кандидаты LT2 в окрестности лактатного интервала."""

    margin_sec = 45.0
    window_start_sec = max(0.0, moddmax.interval_start_sec - margin_sec)
    window_end_sec = moddmax.interval_end_sec + margin_sec

    mask = (
        (data.hhb_times_sec >= window_start_sec - 1e-9)
        & (data.hhb_times_sec <= window_end_sec + 1e-9)
    )
    times = data.hhb_times_sec[mask]
    values = data.hhb_values[mask]
    smooth_times, smooth_values = smooth_to_grid(times, values, step_sec=1.0)

    peak_time_sec = None
    peak_value = None
    breakpoint_time_sec = None
    if len(smooth_times) >= 5:
        peak_index = int(np.argmax(smooth_values))
        peak_time_sec = float(smooth_times[peak_index])
        peak_value = float(smooth_values[peak_index])
        breakpoint_time_sec = piecewise_breakpoint(smooth_times, smooth_values)

    return HhbResult(
        window_start_sec=window_start_sec,
        window_end_sec=window_end_sec,
        smooth_times_sec=smooth_times,
        smooth_values=smooth_values,
        peak_time_sec=peak_time_sec,
        peak_value=peak_value,
        breakpoint_time_sec=breakpoint_time_sec,
    )


def choose_refined_time(
    moddmax: ModDmaxResult,
    dfa: DfaResult,
    hhb: HhbResult,
) -> tuple[float, tuple[str, ...]]:
    """Выбирает итоговое уточнённое время LT2 по согласованным маркерам."""

    candidates: list[tuple[str, float]] = [("modDmax", moddmax.lt2_time_sec)]
    extended_start = moddmax.interval_start_sec - 60.0
    extended_end = moddmax.interval_end_sec + 60.0

    if dfa.crossing_time_sec is not None and extended_start <= dfa.crossing_time_sec <= extended_end:
        candidates.append(("DFA-a1=0.50", float(dfa.crossing_time_sec)))

    if hhb.breakpoint_time_sec is not None and extended_start <= hhb.breakpoint_time_sec <= extended_end:
        candidates.append(("HHb breakpoint", float(hhb.breakpoint_time_sec)))
    elif hhb.peak_time_sec is not None and extended_start <= hhb.peak_time_sec <= extended_end:
        candidates.append(("HHb peak", float(hhb.peak_time_sec)))

    candidate_times = np.array([time_sec for _, time_sec in candidates], dtype=float)
    if len(candidate_times) == 1:
        return float(candidate_times[0]), (candidates[0][0],)

    spread_sec = float(candidate_times.max() - candidate_times.min())
    if spread_sec <= 60.0:
        return float(np.median(candidate_times)), tuple(name for name, _ in candidates)

    # Если маркеры расходятся слишком сильно, не делаем вид, что знаем
    # «точную секунду», и оставляем лактатный центр как якорь.
    return float(moddmax.lt2_time_sec), ("modDmax",)


def seconds_to_mmss(seconds: float | None) -> str:
    """Форматирует секунды в ММ:СС."""

    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    total = int(round(float(seconds)))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def build_plot(subject_name: str, result: Lt2Result) -> plt.Figure:
    """Строит итоговую фигуру по LT2."""

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.32)

    # Панель 1: лактатная кривая и two-fit comparison для modDmax.
    ax = axes[0]
    powers = np.array([point.power_w for point in result.moddmax.points], dtype=float)
    lactates = np.array([point.lactate_mmol for point in result.moddmax.points], dtype=float)
    ax.axvspan(
        result.moddmax.interval_start_power_w,
        result.moddmax.interval_end_power_w,
        color="#cccccc",
        alpha=0.22,
        label="Сырой интервал LT2 по реальным точкам",
    )
    ax.scatter(powers, lactates, color="#1f77b4", s=55, label="Сырые точки лактата", zorder=4)
    ax.plot(
        result.moddmax.fit_x,
        result.moddmax.fit_y,
        color="#2ca02c",
        linewidth=1.8,
        label="Канонический cubic fit",
    )
    ax.plot(
        result.moddmax.fit_x,
        result.moddmax.pchip_fit_y,
        color="#ff7f0e",
        linewidth=1.5,
        linestyle="--",
        label="PCHIP fit (проверка устойчивости)",
    )
    ax.plot(
        result.moddmax.distance_x,
        result.moddmax.line_y,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.3,
        label="Линия modDmax",
    )
    ax.scatter(
        [result.moddmax.lt2_power_w],
        [result.moddmax.lt2_lactate_mmol],
        color="#d62728",
        s=80,
        label="LT2 modDmax (cubic)",
        zorder=5,
    )
    ax.scatter(
        [result.moddmax.pchip_lt2_power_w],
        [result.moddmax.pchip_lt2_lactate_mmol],
        color="#9467bd",
        s=70,
        label="LT2 modDmax (PCHIP)",
        zorder=5,
    )
    ax.axvline(result.moddmax.lt2_power_w, color="#d62728", linestyle=":", linewidth=1.2)
    ax.axvline(result.moddmax.pchip_lt2_power_w, color="#9467bd", linestyle=":", linewidth=1.2)
    for power_w, label in (
        (result.moddmax.interval_start_power_w, "левая граница"),
        (result.moddmax.interval_end_power_w, "правая граница"),
    ):
        ax.axvline(power_w, color="#999999", linestyle="-.", linewidth=0.9)
    info_text = (
        f"ось X = мощность, Вт\n"
        f"сырой интервал: {result.moddmax.interval_start_power_w:.0f}–"
        f"{result.moddmax.interval_end_power_w:.0f} Вт | "
        f"{seconds_to_mmss(result.moddmax.interval_start_sec)}–"
        f"{seconds_to_mmss(result.moddmax.interval_end_sec)}\n"
        f"cubic LT2: {result.moddmax.lt2_power_w:.1f} Вт | "
        f"PCHIP LT2: {result.moddmax.pchip_lt2_power_w:.1f} Вт"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    ax.set_title("LT2 по кривой лактат–мощность: канонический modDmax и проверка через PCHIP")
    ax.set_xlabel("Мощность ступени, Вт")
    ax.set_ylabel("Лактат, mmol/L")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    # Панель 2: DFA-a1.
    ax = axes[1]
    ax.plot(result.dfa.times_sec, result.dfa.alpha1, color="#ff7f0e", linewidth=1.3, label="DFA-a1")
    high_artifact = result.dfa.artifact_share > MAX_ACCEPTABLE_ARTIFACT_SHARE
    if high_artifact.any():
        ax.scatter(
            result.dfa.times_sec[high_artifact],
            result.dfa.alpha1[high_artifact],
            color="#d62728",
            s=14,
            label="Окна RR с артефактами >5%",
        )
    ax.axhline(DFA_THRESHOLD, color="#9467bd", linestyle="--", linewidth=1.2, label="DFA-a1 = 0.50")
    if result.dfa.crossing_time_sec is not None:
        ax.axvline(
            result.dfa.crossing_time_sec,
            color="#9467bd",
            linestyle=":",
            linewidth=1.4,
            label=f"HRVT2 {seconds_to_mmss(result.dfa.crossing_time_sec)}",
        )
    ax.axvspan(result.moddmax.interval_start_sec, result.moddmax.interval_end_sec, color="#cccccc", alpha=0.2)
    ax.axvline(result.moddmax.lt2_time_sec, color="#d62728", linestyle=":", linewidth=1.2, label="Центр modDmax")
    ax.set_title("Канонический DFA-a1 из RR (2 мин окно, шаг 5 с, масштабы 4–16 ударов)")
    ax.set_xlabel("Время от начала теста, с")
    ax.set_ylabel("DFA-a1")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    # Панель 3: HHb в окрестности LT2.
    ax = axes[2]
    ax.plot(
        result.hhb.smooth_times_sec,
        result.hhb.smooth_values,
        color="#8c564b",
        linewidth=1.6,
        label="HHb unfiltered (сглаженный)",
    )
    ax.axvspan(result.moddmax.interval_start_sec, result.moddmax.interval_end_sec, color="#cccccc", alpha=0.25)
    ax.axvline(result.moddmax.lt2_time_sec, color="#d62728", linestyle=":", linewidth=1.2, label="Центр modDmax")
    if result.hhb.breakpoint_time_sec is not None:
        ax.axvline(
            result.hhb.breakpoint_time_sec,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.3,
            label=f"HHb breakpoint {seconds_to_mmss(result.hhb.breakpoint_time_sec)}",
        )
    if result.hhb.peak_time_sec is not None:
        ax.axvline(
            result.hhb.peak_time_sec,
            color="#1f77b4",
            linestyle=":",
            linewidth=1.2,
            label=f"HHb peak {seconds_to_mmss(result.hhb.peak_time_sec)}",
        )
    if result.dfa.crossing_time_sec is not None:
        ax.axvline(result.dfa.crossing_time_sec, color="#9467bd", linestyle=":", linewidth=1.1, label="HRVT2")
    ax.axvline(result.refined_time_sec, color="#000000", linestyle="-", linewidth=1.6, label="Итоговый LT2")
    ax.set_title("HHb-кандидаты LT2 в окрестности лактатного интервала")
    ax.set_xlabel("Время от начала теста, с")
    ax.set_ylabel("HHb")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    fig.suptitle(
        (
            f"{subject_name} | LT2 modDmax={result.moddmax.lt2_power_w:.1f} Вт, "
            f"{seconds_to_mmss(result.moddmax.lt2_time_sec)} | "
            f"итог={seconds_to_mmss(result.refined_time_sec)} "
            f"({', '.join(result.refined_sources)})"
        ),
        fontsize=13,
    )
    return fig


def compute_lt2(data: FulltestData) -> Lt2Result:
    """Собирает все шаги определения LT2."""

    moddmax = build_moddmax(data)
    dfa = build_dfa_series(data)
    hhb = build_hhb_markers(data, moddmax)
    refined_time_sec, refined_sources = choose_refined_time(moddmax, dfa, hhb)
    return Lt2Result(
        moddmax=moddmax,
        dfa=dfa,
        hhb=hhb,
        refined_time_sec=refined_time_sec,
        refined_sources=refined_sources,
    )


def main() -> None:
    """Точка входа: ищет LT2 по fulltest.h5 и строит график."""

    args = parse_args()
    subject_dir = resolve_subject_dir(args.data_dir, args.subject)
    fulltest_path = args.fulltest or (subject_dir / DEFAULT_FULLTEST_FILENAME)
    if not fulltest_path.exists():
        raise FileNotFoundError(
            f"Файл fulltest.h5 не найден: {fulltest_path}. "
            "Сначала создайте его из тест+ред.h5."
        )

    data = load_fulltest_method(fulltest_path)
    result = compute_lt2_method(data)

    print(f"Папка: {subject_dir.name}")
    print(f"Файл: {fulltest_path}")
    print(f"stop_time: {seconds_to_mmss(data.stop_time_sec)} ({data.stop_time_sec:.1f} с)")
    print(
        "Точек лактата для modDmax: "
        f"{len(result.moddmax.points)}"
    )
    print(
        "LT2 modDmax: "
        f"{result.moddmax.lt2_power_w:.1f} Вт, "
        f"{result.moddmax.lt2_lactate_mmol:.2f} mmol/L, "
        f"{seconds_to_mmss(result.moddmax.lt2_time_sec)}"
    )
    print(
        "LT2 modDmax (PCHIP-проверка): "
        f"{result.moddmax.pchip_lt2_power_w:.1f} Вт, "
        f"{result.moddmax.pchip_lt2_lactate_mmol:.2f} mmol/L, "
        f"{seconds_to_mmss(result.moddmax.pchip_lt2_time_sec)}"
    )
    print(
        "Лактатный интервал: "
        f"{result.moddmax.interval_start_power_w:.0f}–{result.moddmax.interval_end_power_w:.0f} Вт | "
        f"{seconds_to_mmss(result.moddmax.interval_start_sec)}–"
        f"{seconds_to_mmss(result.moddmax.interval_end_sec)}"
    )
    print(
        "Разница cubic vs PCHIP: "
        f"{abs(result.moddmax.lt2_power_w - result.moddmax.pchip_lt2_power_w):.1f} Вт, "
        f"{abs(result.moddmax.lt2_time_sec - result.moddmax.pchip_lt2_time_sec):.1f} с"
    )
    print(
        "HRVT2 по DFA-a1=0.50: "
        f"{seconds_to_mmss(result.dfa.crossing_time_sec)}"
    )
    if np.isfinite(result.dfa.artifact_share).any():
        print(
            "RR окна с артефактами >5%: "
            f"{int(np.sum(result.dfa.artifact_share > MAX_ACCEPTABLE_ARTIFACT_SHARE))}"
        )
    print(
        "HHb breakpoint: "
        f"{seconds_to_mmss(result.hhb.breakpoint_time_sec)}"
    )
    print(
        "HHb peak: "
        f"{seconds_to_mmss(result.hhb.peak_time_sec)}"
    )
    print(
        "Итоговый LT2: "
        f"{seconds_to_mmss(result.refined_time_sec)} "
        f"по {', '.join(result.refined_sources)}"
    )

    fig = build_plot(subject_dir.name, result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
