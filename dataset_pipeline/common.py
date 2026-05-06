"""Общие функции для сборки датасета."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import unicodedata

import h5py
import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("data")
DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_WINDOW_DURATION_SEC = 30.0
DEFAULT_WINDOW_STEP_SEC = 5.0
DEFAULT_HRV_CONTEXT_SEC = 120.0
BASELINE_STAGE_INDEX = 0
WORK_STAGE_START_INDEX = 1

SUMMARY_FIELDS = (
    "sex",
    "age",
    "height",
    "weight",
    "body_fat_mass",
    "skeletal_muscle_mass",
    "dominant_leg_lean_mass",
    "dominant_leg_fat_mass",
    "phase_angle",
    "dominant_leg_circumference",
)

LT2_SUBJECT_FIELDS = (
    "lt2_method",
    "lt2_lactate_stage_count",
    "lt2_power_w",
    "lt2_lactate_mmol",
    "lt2_time_center_sec",
    "lt2_time_sec",
    "lt2_interval_start_sec",
    "lt2_interval_end_sec",
    "lt2_interval_start_power_w",
    "lt2_interval_end_power_w",
    "lt2_pchip_power_w",
    "lt2_pchip_lactate_mmol",
    "lt2_pchip_time_sec",
    "lt2_pchip_delta_power_w",
    "lt2_pchip_delta_time_sec",
    "lt2_hrvt2_time_sec",
    "lt2_hhb_breakpoint_time_sec",
    "lt2_hhb_peak_time_sec",
    "lt2_refined_time_sec",
    "lt2_refined_sources",
    "lt2_refined_valid",
    "lt2_refined_window_start_sec",
    "lt2_refined_window_end_sec",
    "lt2_refined_spread_sec",
    "lt2_refined_source_count",
    "lt2_power_label_quality",
    "lt2_power_label_quality_reasons",
    "lt2_time_label_quality",
    "lt2_time_label_quality_reasons",
)


@dataclass(frozen=True)
class SubjectFile:
    """Путь к `finaltest.h5` и связанная с ним папка участника."""

    subject_dir: Path
    finaltest_path: Path


@dataclass(frozen=True)
class StageInterval:
    """Один интервал ступени мощности в относительной временной шкале."""

    stage_index: int
    power_w: float
    start_sec: float
    end_sec: float


def normalize_name(value: str) -> str:
    """Нормализует имя для стабильной сортировки."""

    return unicodedata.normalize("NFC", value).casefold()


def subject_sort_key(path: Path) -> tuple[str, str]:
    """Возвращает ключ сортировки для папок участников."""

    normalized = unicodedata.normalize("NFC", path.name)
    return normalized.casefold(), normalized


def list_subject_files(data_dir: Path) -> tuple[list[SubjectFile], list[str]]:
    """Собирает все доступные `finaltest.h5` и возвращает также список пропусков."""

    subject_files: list[SubjectFile] = []
    skipped: list[str] = []

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Каталог с данными не найден: {data_dir}")

    for subject_dir in sorted(
        (path for path in data_dir.iterdir() if path.is_dir()),
        key=subject_sort_key,
    ):
        finaltest_path = subject_dir / "finaltest.h5"
        if not finaltest_path.exists():
            skipped.append(subject_dir.name)
            continue
        subject_files.append(SubjectFile(subject_dir=subject_dir, finaltest_path=finaltest_path))

    return subject_files, skipped


def ensure_parent_dir(path: Path) -> None:
    """Создаёт родительский каталог для выходного файла."""

    path.parent.mkdir(parents=True, exist_ok=True)


def save_parquet(data_frame: pd.DataFrame, output_path: Path, force: bool) -> None:
    """Сохраняет таблицу в parquet, контролируя перезапись."""

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Выходной файл уже существует: {output_path}. "
            "Используйте --force для перезаписи."
        )

    ensure_parent_dir(output_path)
    data_frame.to_parquet(output_path, index=False)


def decode_attr_value(value: Any) -> Any:
    """Приводит значение атрибута HDF5 к обычному Python-типу."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_optional_attr(handle: h5py.File, attr_name: str) -> Any:
    """Читает необязательный атрибут HDF5, возвращая `None` при отсутствии."""

    if attr_name not in handle.attrs:
        return None
    return decode_attr_value(handle.attrs[attr_name])


def read_required_attr(handle: h5py.File, attr_name: str) -> Any:
    """Читает обязательный атрибут HDF5."""

    value = read_optional_attr(handle, attr_name)
    if value is None:
        raise KeyError(f"В файле {handle.filename} отсутствует атрибут '{attr_name}'.")
    return value


def to_relative_seconds(raw_timestamps_ms: np.ndarray, anchor_ms: float) -> np.ndarray:
    """Переводит timestamps из миллисекунд в секунды от начала сессии."""

    return (np.asarray(raw_timestamps_ms, dtype=float) - float(anchor_ms)) / 1000.0


def get_anchor_ms(handle: h5py.File) -> float:
    """Возвращает общий якорь времени по первому timestamp канала `moxy.smo2`."""

    return float(handle["channels/moxy.smo2/timestamps"][0])


def load_channel_relative(
    handle: h5py.File,
    channel_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Загружает канал и переводит его timestamps в секунды от якоря."""

    anchor_ms = get_anchor_ms(handle)
    group = handle[f"channels/{channel_name}"]
    timestamps_sec = to_relative_seconds(group["timestamps"][:], anchor_ms)
    values = group["values"][:].astype(float)
    return timestamps_sec, values


def build_work_stage_intervals(
    power_times_sec: np.ndarray,
    power_values_w: np.ndarray,
    stop_time_sec: float,
) -> list[StageInterval]:
    """Строит интервалы рабочих ступеней.

    Первая ступень протокола (`30 Вт`) используется как baseline и не входит
    в рабочие интервалы датасета. Поэтому стадия с индексом 0 пропускается.
    """

    if len(power_times_sec) != len(power_values_w):
        raise ValueError("power.label содержит несовпадающие размеры timestamps и values.")
    if len(power_values_w) < 2:
        raise ValueError("Для построения рабочих окон нужен хотя бы baseline и одна рабочая ступень.")

    intervals: list[StageInterval] = []
    for raw_stage_index in range(WORK_STAGE_START_INDEX, len(power_values_w)):
        start_sec = float(power_times_sec[raw_stage_index])
        if start_sec >= stop_time_sec:
            break

        if raw_stage_index + 1 < len(power_values_w):
            nominal_end_sec = float(power_times_sec[raw_stage_index + 1])
        else:
            nominal_end_sec = float(stop_time_sec)
        end_sec = min(nominal_end_sec, float(stop_time_sec))
        if end_sec <= start_sec:
            continue

        intervals.append(
            StageInterval(
                stage_index=len(intervals),
                power_w=float(power_values_w[raw_stage_index]),
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    if not intervals:
        raise ValueError("Не удалось выделить ни одной рабочей ступени до stop_time.")
    return intervals


def find_stage_for_time(stage_intervals: list[StageInterval], time_sec: float) -> StageInterval:
    """Находит ступень, активную в заданный момент времени."""

    for interval in stage_intervals:
        if interval.start_sec <= time_sec < interval.end_sec:
            return interval

    # Последняя ступень включает правую границу окна при совпадении с концом теста.
    last_interval = stage_intervals[-1]
    if np.isclose(time_sec, last_interval.end_sec):
        return last_interval

    raise ValueError(
        f"Время {time_sec:.3f} с не попадает ни в одну рабочую ступень."
    )


def compute_mode_power_for_window(
    stage_intervals: list[StageInterval],
    window_start_sec: float,
    window_end_sec: float,
) -> float:
    """Возвращает мощность, покрывающую наибольшую часть окна."""

    best_power_w: float | None = None
    best_overlap_sec = -1.0
    current_stage = find_stage_for_time(stage_intervals, window_end_sec)

    for interval in stage_intervals:
        overlap_sec = min(window_end_sec, interval.end_sec) - max(window_start_sec, interval.start_sec)
        if overlap_sec <= 0:
            continue

        # При равном покрытии берём ступень правой границы окна.
        if overlap_sec > best_overlap_sec or (
            np.isclose(overlap_sec, best_overlap_sec)
            and interval.power_w == current_stage.power_w
        ):
            best_overlap_sec = float(overlap_sec)
            best_power_w = float(interval.power_w)

    if best_power_w is None:
        raise ValueError(
            f"Окно {window_start_sec:.3f}–{window_end_sec:.3f} с не пересекает рабочие ступени."
        )
    return best_power_w


def build_window_starts(
    work_start_sec: float,
    stop_time_sec: float,
    window_duration_sec: float,
    step_sec: float,
) -> np.ndarray:
    """Строит стартовые позиции окон так, чтобы окно целиком лежало в рабочей части."""

    if window_duration_sec <= 0 or step_sec <= 0:
        raise ValueError("Длина окна и шаг должны быть положительными.")

    last_start_sec = float(stop_time_sec) - float(window_duration_sec)
    if last_start_sec < work_start_sec:
        return np.array([], dtype=float)

    starts = np.arange(work_start_sec, last_start_sec + 1e-9, step_sec, dtype=float)
    return starts


def load_subjects_table(subjects_path: Path) -> pd.DataFrame:
    """Читает таблицу испытуемых из parquet."""

    if not subjects_path.exists():
        raise FileNotFoundError(
            f"Таблица испытуемых не найдена: {subjects_path}. "
            "Сначала постройте subjects.parquet."
        )
    return pd.read_parquet(subjects_path)


def load_windows_table(windows_path: Path) -> pd.DataFrame:
    """Читает таблицу окон из parquet."""

    if not windows_path.exists():
        raise FileNotFoundError(
            f"Таблица окон не найдена: {windows_path}. "
            "Сначала постройте windows.parquet."
        )
    return pd.read_parquet(windows_path)

