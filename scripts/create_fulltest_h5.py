#!/usr/bin/env python3
"""Создаёт fulltest.h5, добавляя к тест+ред.h5 метаданные summary и канал лактата."""

from __future__ import annotations

import argparse
import csv
import difflib
from dataclasses import dataclass
from pathlib import Path
import unicodedata

import h5py
import numpy as np

from create_test_plus_red_h5 import copy_h5_tree

DEFAULT_SOURCE_FILENAME = "тест+ред.h5"
DEFAULT_SUMMARY_FILENAME = "summary.csv"
DEFAULT_OUTPUT_FILENAME = "fulltest.h5"
LACTATE_CHANNEL_NAME = "lactate"
H5_TIME_SOURCE_PATH = "channels/moxy.smo2/timestamps"

SUMMARY_ATTR_MAP = {
    "ФИ": "subject_name",
    "Рост": "height",
    "Вес": "weight",
    "Возраст": "age",
    "Пол": "sex",
    "Масса жира в теле": "body_fat_mass",
    "Масса скелетной мускулатуры": "skeletal_muscle_mass",
    "Тощая масса ведущей ноги": "dominant_leg_lean_mass",
    "Жировая масса ведущей ноги": "dominant_leg_fat_mass",
    "Фазовый угол": "phase_angle",
    "Обхват ведущей ноги": "dominant_leg_circumference",
    "Время остановки": "stop_time",
}

SUMMARY_FLOAT_FIELDS = (
    "Рост",
    "Вес",
    "Масса жира в теле",
    "Масса скелетной мускулатуры",
    "Тощая масса ведущей ноги",
    "Жировая масса ведущей ноги",
    "Фазовый угол",
    "Обхват ведущей ноги",
)

SUMMARY_INT_FIELDS = (
    "Возраст",
)


@dataclass(frozen=True)
class LactatePoint:
    """Одна точка лактата из summary.csv."""

    time_text: str
    time_sec: float
    lactate: float


@dataclass(frozen=True)
class SummaryPayload:
    """Разобранное содержимое summary.csv."""

    metadata: dict[str, str]
    lactate_points: tuple[LactatePoint, ...]
    warnings: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Создаёт fulltest.h5: копирует тест+ред.h5 и добавляет "
            "метаданные из summary.csv и канал /channels/lactate."
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
        "--source",
        type=Path,
        help=(
            "Путь к исходному HDF5. "
            "По умолчанию используется тест+ред.h5 в папке участника."
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help=(
            "Путь к summary.csv. "
            "По умолчанию используется summary.csv в папке участника."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Путь к итоговому HDF5. "
            "По умолчанию создаётся fulltest.h5 в папке участника."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать выходной файл, если он уже существует.",
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


def parse_decimal(value: str) -> float:
    """Преобразует число с запятой в float."""

    normalized = value.strip().replace(",", ".")
    if not normalized:
        raise ValueError("Пустое числовое значение.")
    return float(normalized)


def parse_time_to_seconds(value: str) -> float:
    """Преобразует строку времени в секунды от старта теста."""

    text = value.strip()
    if not text:
        raise ValueError("Пустое значение времени.")

    parts = text.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        raise ValueError(f"Неподдерживаемый формат времени: {value}")

    return (
        int(hours) * 3600.0
        + int(minutes) * 60.0
        + float(seconds.replace(",", "."))
    )


def read_summary_rows(path: Path) -> list[list[str]]:
    """Читает summary.csv как список строк без потери пустых ячеек."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        return [[cell.strip() for cell in row] for row in reader]


def locate_lactate_header(rows: list[list[str]]) -> int:
    """Находит строку, где начинается таблица лактата."""

    for index, row in enumerate(rows):
        first = row[0].strip().casefold() if len(row) > 0 else ""
        second = row[1].strip().casefold() if len(row) > 1 else ""
        if first == "время" and second == "лактат":
            return index
    raise ValueError("В summary.csv не найдена строка заголовка таблицы лактата.")


def normalize_name(text: str) -> str:
    """Нормализует Unicode-строку для мягкого сравнения имён."""

    return unicodedata.normalize("NFC", text).casefold().strip()


def load_summary(path: Path, subject_folder_name: str) -> SummaryPayload:
    """Разбирает первые строки метаданных и таблицу лактата из summary.csv."""

    rows = read_summary_rows(path)
    if len(rows) < 2:
        raise ValueError(f"В {path} недостаточно строк для чтения метаданных.")

    metadata_header = rows[0]
    metadata_values = rows[1]
    metadata: dict[str, str] = {}
    for index, key in enumerate(metadata_header):
        key = key.strip()
        if not key:
            continue
        metadata[key] = metadata_values[index].strip() if index < len(metadata_values) else ""

    lactate_header_row = locate_lactate_header(rows)
    lactate_points: list[LactatePoint] = []
    for row in rows[lactate_header_row + 1 :]:
        time_text = row[0].strip() if len(row) > 0 else ""
        lactate_text = row[1].strip() if len(row) > 1 else ""
        if not time_text and not lactate_text:
            continue
        if not time_text or not lactate_text:
            continue

        lactate_points.append(
            LactatePoint(
                time_text=time_text,
                time_sec=parse_time_to_seconds(time_text),
                lactate=parse_decimal(lactate_text),
            )
        )

    if not lactate_points:
        raise ValueError(f"В {path} не найдено ни одной точки лактата.")

    warnings: list[str] = []
    summary_subject_name = metadata.get("ФИ", "")
    if summary_subject_name and normalize_name(summary_subject_name) != normalize_name(subject_folder_name):
        warnings.append(
            "Имя в summary.csv не совпадает с именем папки: "
            f"summary='{summary_subject_name}', папка='{subject_folder_name}'."
        )

    return SummaryPayload(
        metadata=metadata,
        lactate_points=tuple(lactate_points),
        warnings=tuple(warnings),
    )


def build_lactate_timestamps(source_h5_path: Path, lactate_points: tuple[LactatePoint, ...]) -> np.ndarray:
    """Переносит времена лактата в временную шкалу HDF5."""

    with h5py.File(source_h5_path, "r") as handle:
        if H5_TIME_SOURCE_PATH not in handle:
            raise ValueError(
                f"В исходном HDF5 нет канала {H5_TIME_SOURCE_PATH}."
            )
        anchor_ms = float(handle[H5_TIME_SOURCE_PATH][0])

    lactate_seconds = np.array([point.time_sec for point in lactate_points], dtype=float)
    return anchor_ms + lactate_seconds * 1000.0


def write_summary_metadata(
    dest_path: Path,
    payload: SummaryPayload,
) -> None:
    """Записывает метаданные summary в корневые атрибуты HDF5."""

    with h5py.File(dest_path, "a") as handle:
        for raw_key, attr_name in SUMMARY_ATTR_MAP.items():
            if raw_key not in payload.metadata:
                continue
            raw_value = payload.metadata[raw_key]
            if raw_key in SUMMARY_FLOAT_FIELDS and raw_value:
                try:
                    handle.attrs[attr_name] = parse_decimal(raw_value)
                    continue
                except ValueError:
                    pass
            if raw_key in SUMMARY_INT_FIELDS and raw_value:
                try:
                    handle.attrs[attr_name] = int(round(parse_decimal(raw_value)))
                    continue
                except ValueError:
                    pass
            handle.attrs[attr_name] = raw_value

        stop_time_text = payload.metadata.get("Время остановки", "")
        if stop_time_text:
            try:
                handle.attrs["stop_time_sec"] = parse_time_to_seconds(stop_time_text)
            except ValueError:
                pass


def clean_output_h5(dest_path: Path) -> None:
    """Удаляет из копии старые служебные атрибуты и ранее созданный канал лактата."""

    with h5py.File(dest_path, "a") as handle:
        legacy_attr_names = []
        known_metadata_names = set(SUMMARY_ATTR_MAP.values()) | {"stop_time_sec"}
        for name in handle.attrs.keys():
            text_name = str(name)
            if (
                text_name.startswith("trainred_")
                or text_name.startswith("summary_")
                or text_name.endswith("_value")
                or text_name in known_metadata_names
            ):
                legacy_attr_names.append(name)
        for attr_name in legacy_attr_names:
            del handle.attrs[attr_name]

        channels_group = handle.get("channels")
        if channels_group is not None and LACTATE_CHANNEL_NAME in channels_group:
            del channels_group[LACTATE_CHANNEL_NAME]


def add_lactate_channel(
    dest_path: Path,
    lactate_timestamps: np.ndarray,
    lactate_points: tuple[LactatePoint, ...],
) -> None:
    """Добавляет канал лактата в раздел /channels."""

    with h5py.File(dest_path, "a") as handle:
        channels_group = handle.require_group("channels")
        if LACTATE_CHANNEL_NAME in channels_group:
            raise ValueError(
                f"Канал {LACTATE_CHANNEL_NAME} уже существует в {dest_path}."
            )

        channel_group = channels_group.create_group(LACTATE_CHANNEL_NAME)
        channel_group.create_dataset(
            "timestamps",
            data=lactate_timestamps.astype(np.float64),
            dtype=np.float64,
        )
        channel_group.create_dataset(
            "values",
            data=np.array([point.lactate for point in lactate_points], dtype=np.float32),
            dtype=np.float32,
        )


def create_fulltest_h5(
    source_h5_path: Path,
    summary_path: Path,
    output_path: Path,
    subject_folder_name: str,
    force: bool,
) -> SummaryPayload:
    """Создаёт fulltest.h5 из тест+ред.h5 и summary.csv."""

    if not source_h5_path.exists():
        raise FileNotFoundError(
            f"Исходный HDF5 не найден: {source_h5_path}. "
            "Сначала создайте тест+ред.h5."
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"Файл summary.csv не найден: {summary_path}.")

    payload = load_summary(summary_path, subject_folder_name=subject_folder_name)
    lactate_timestamps = build_lactate_timestamps(source_h5_path, payload.lactate_points)

    copy_h5_tree(
        source_path=source_h5_path,
        dest_path=output_path,
        force=force,
    )
    clean_output_h5(dest_path=output_path)
    write_summary_metadata(
        dest_path=output_path,
        payload=payload,
    )
    add_lactate_channel(
        dest_path=output_path,
        lactate_timestamps=lactate_timestamps,
        lactate_points=payload.lactate_points,
    )

    return payload


def main() -> None:
    """Точка входа: создаёт fulltest.h5 для выбранной папки."""

    args = parse_args()
    subject_dir = resolve_subject_dir(args.data_dir, args.subject)

    source_h5_path = args.source or (subject_dir / DEFAULT_SOURCE_FILENAME)
    summary_path = args.summary or (subject_dir / DEFAULT_SUMMARY_FILENAME)
    output_path = args.output or (subject_dir / DEFAULT_OUTPUT_FILENAME)

    payload = create_fulltest_h5(
        source_h5_path=source_h5_path,
        summary_path=summary_path,
        output_path=output_path,
        subject_folder_name=subject_dir.name,
        force=args.force,
    )

    print(f"Папка: {subject_dir.name}")
    print(f"Источник HDF5: {source_h5_path}")
    print(f"Источник summary: {summary_path}")
    print(f"Точек лактата: {len(payload.lactate_points)}")
    print(
        "Времена лактата: "
        f"{payload.lactate_points[0].time_text} … {payload.lactate_points[-1].time_text}"
    )
    if payload.warnings:
        for warning in payload.warnings:
            print(f"Предупреждение: {warning}")
    print(f"Создан файл: {output_path}")


if __name__ == "__main__":
    main()
