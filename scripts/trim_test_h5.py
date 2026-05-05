#!/usr/bin/env python3
"""Создаёт новый тест_trimmed.h5, асимметрично обрезая исходный тест.h5."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
from pathlib import Path
import unicodedata

import h5py
import numpy as np

DEFAULT_SOURCE_FILENAME = "тест.h5"
DEFAULT_OUTPUT_FILENAME = "тест_trimmed.h5"
POWER_CHANNEL_NAME = "power.label"


@dataclass(frozen=True)
class ChannelTrimSummary:
    """Краткая сводка по обрезке одного канала."""

    channel_name: str
    original_rows: int
    trimmed_rows: int
    mode: str


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Создаёт тест_trimmed.h5: копирует исходный тест.h5 и "
            "асимметрично режет каналы по времени."
        )
    )
    parser.add_argument(
        "subject",
        help="Точное имя папки участника внутри data.",
    )
    parser.add_argument(
        "--minutes",
        "-m",
        type=float,
        required=True,
        help="Сколько минут удалить по заданным правилам.",
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
            "По умолчанию используется тест.h5 в папке участника."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Путь к итоговому HDF5. "
            "По умолчанию создаётся тест_trimmed.h5 в папке участника."
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


def validate_minutes(minutes: float) -> float:
    """Проверяет, что длительность обрезки положительна."""

    if not np.isfinite(minutes) or minutes <= 0.0:
        raise ValueError(
            f"Параметр --minutes должен быть положительным числом, получено: {minutes}"
        )
    return float(minutes)


def prepare_output_path(dest_path: Path, force: bool) -> None:
    """Готовит путь к выходному файлу и защищает от тихой перезаписи."""

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        if not force:
            raise FileExistsError(
                f"Выходной файл уже существует: {dest_path}. "
                "Используйте --force для перезаписи."
            )
        dest_path.unlink()


def copy_root_structure(source: h5py.File, dest: h5py.File) -> None:
    """Копирует root attrs и все верхнеуровневые объекты, кроме channels."""

    for attr_name, attr_value in source.attrs.items():
        dest.attrs[attr_name] = attr_value

    for item_name in source.keys():
        if item_name == "channels":
            continue
        source.copy(item_name, dest)


def copy_group_attrs(source_group: h5py.Group, dest_group: h5py.Group) -> None:
    """Копирует attrs одной HDF5-группы."""

    for attr_name, attr_value in source_group.attrs.items():
        dest_group.attrs[attr_name] = attr_value


def trim_channel_data(
    channel_name: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    trim_ms: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Обрезает канал по выбранному правилу и возвращает новый массив времени."""

    if timestamps.ndim != 1 or values.ndim != 1:
        raise ValueError(
            f"Канал {channel_name} должен содержать одномерные timestamps и values."
        )
    if len(timestamps) != len(values):
        raise ValueError(
            f"Канал {channel_name} повреждён: длины timestamps и values не совпадают."
        )
    if len(timestamps) == 0:
        raise ValueError(f"Канал {channel_name} пуст ещё до обрезки.")

    if channel_name == POWER_CHANNEL_NAME:
        cutoff_ms = float(timestamps[-1]) - trim_ms
        mask = timestamps <= cutoff_ms
        mode = "drop_tail"
        trimmed_timestamps = timestamps[mask]
    else:
        cutoff_ms = float(timestamps[0]) + trim_ms
        mask = timestamps >= cutoff_ms
        mode = "drop_head_and_shift_back"
        trimmed_timestamps = timestamps[mask] - trim_ms

    trimmed_values = values[mask]

    if trimmed_timestamps.size == 0:
        raise ValueError(
            f"После обрезки канал {channel_name} оказался пустым. "
            f"Попробуйте меньшее значение --minutes."
        )

    return trimmed_timestamps, trimmed_values, mode


def create_trimmed_channel(
    source_channel: h5py.Group,
    dest_channels_group: h5py.Group,
    channel_name: str,
    trim_ms: float,
) -> ChannelTrimSummary:
    """Создаёт в новом файле один уже обрезанный канал."""

    timestamps_ds = source_channel["timestamps"]
    values_ds = source_channel["values"]

    timestamps = timestamps_ds[:]
    values = values_ds[:]
    trimmed_timestamps, trimmed_values, mode = trim_channel_data(
        channel_name=channel_name,
        timestamps=timestamps,
        values=values,
        trim_ms=trim_ms,
    )

    channel_group = dest_channels_group.create_group(channel_name)
    copy_group_attrs(source_channel, channel_group)

    # Типы исходных датасетов сохраняем, чтобы новый файл не отличался
    # по схеме хранения, а только по длине и временной оси.
    timestamps_out = trimmed_timestamps.astype(timestamps_ds.dtype, copy=False)
    values_out = trimmed_values.astype(values_ds.dtype, copy=False)

    timestamps_new = channel_group.create_dataset(
        "timestamps",
        data=timestamps_out,
        dtype=timestamps_ds.dtype,
    )
    values_new = channel_group.create_dataset(
        "values",
        data=values_out,
        dtype=values_ds.dtype,
    )

    for attr_name, attr_value in timestamps_ds.attrs.items():
        timestamps_new.attrs[attr_name] = attr_value
    for attr_name, attr_value in values_ds.attrs.items():
        values_new.attrs[attr_name] = attr_value

    return ChannelTrimSummary(
        channel_name=channel_name,
        original_rows=len(timestamps),
        trimmed_rows=len(trimmed_timestamps),
        mode=mode,
    )


def create_trimmed_test_h5(
    source_path: Path,
    dest_path: Path,
    minutes: float,
    force: bool,
) -> tuple[ChannelTrimSummary, ...]:
    """Создаёт новый HDF5 с асимметрично обрезанными каналами."""

    if not source_path.exists():
        raise FileNotFoundError(f"Исходный HDF5 не найден: {source_path}")

    trim_ms = validate_minutes(minutes) * 60_000.0
    prepare_output_path(dest_path, force=force)

    summaries: list[ChannelTrimSummary] = []
    try:
        with h5py.File(source_path, "r") as source, h5py.File(dest_path, "w") as dest:
            if "channels" not in source:
                raise ValueError(f"В файле {source_path} нет группы /channels.")
            if POWER_CHANNEL_NAME not in source["channels"]:
                raise ValueError(
                    f"В файле {source_path} нет канала {POWER_CHANNEL_NAME}."
                )

            copy_root_structure(source, dest)

            source_channels = source["channels"]
            dest_channels = dest.create_group("channels")
            copy_group_attrs(source_channels, dest_channels)

            for channel_name in sorted(source_channels.keys()):
                summary = create_trimmed_channel(
                    source_channel=source_channels[channel_name],
                    dest_channels_group=dest_channels,
                    channel_name=channel_name,
                    trim_ms=trim_ms,
                )
                summaries.append(summary)
    except Exception:
        # Не оставляем после ошибки частично собранный файл, чтобы случайно
        # не принять его за корректный результат пайплайна.
        if dest_path.exists():
            dest_path.unlink()
        raise

    return tuple(summaries)


def print_summary(
    subject_dir: Path,
    source_path: Path,
    dest_path: Path,
    minutes: float,
    summaries: tuple[ChannelTrimSummary, ...],
) -> None:
    """Печатает краткую сводку по выполненной обрезке."""

    print(f"Папка: {subject_dir.name}")
    print(f"Источник HDF5: {source_path}")
    print(f"Выходной HDF5: {dest_path}")
    print(f"Обрезка: {minutes:.3f} мин")
    for summary in summaries:
        print(
            f"{summary.channel_name}: "
            f"{summary.original_rows} -> {summary.trimmed_rows} "
            f"({summary.mode})"
        )


def main() -> None:
    """Точка входа: создаёт trimmed-версию исходного тест.h5."""

    try:
        args = parse_args()
        subject_dir = resolve_subject_dir(args.data_dir, args.subject)
        source_path = args.source or (subject_dir / DEFAULT_SOURCE_FILENAME)
        dest_path = args.output or (subject_dir / DEFAULT_OUTPUT_FILENAME)

        summaries = create_trimmed_test_h5(
            source_path=source_path,
            dest_path=dest_path,
            minutes=args.minutes,
            force=args.force,
        )
        print_summary(
            subject_dir=subject_dir,
            source_path=source_path,
            dest_path=dest_path,
            minutes=args.minutes,
            summaries=summaries,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
