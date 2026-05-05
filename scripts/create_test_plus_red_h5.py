#!/usr/bin/env python3
"""Создаёт тест+ред.h5, добавляя в тест.h5 вырезанные каналы из train.red.csv."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import sys
import unicodedata

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.trainred_alignment import load_trainred

DEFAULT_POINTS_FILENAME = "trainred_alignment_points.json"
DEFAULT_OUTPUT_FILENAME = "тест+ред.h5"
TRAINRED_TIMESTAMP_COLUMN = "Timestamp (seconds passed)"

# Сохраняем все нужные каналы сразу, даже если часть из них пока
# не используется в анализе напрямую. Это расширение файла, а не
# минималистичный экспорт только «самого полезного».
TRAINRED_TO_H5_CHANNELS = {
    "train.red.smo2": "SmO2",
    "train.red.hbdiff": "HBDiff",
    "train.red.smo2.unfiltered": "SmO2 unfiltered",
    "train.red.o2hb.unfiltered": "O2HB unfiltered",
    "train.red.hhb.unfiltered": "HHb unfiltered",
    "train.red.thb.unfiltered": "THb unfiltered",
    "train.red.hbdiff.unfiltered": "HBDiff unfiltered",
}


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Создаёт файл тест+ред.h5: копирует тест.h5 и добавляет "
            "в него вырезанный фрагмент каналов из train.red.csv."
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
        "--points-file",
        type=Path,
        help=(
            "Путь к JSON с точками выравнивания. "
            "По умолчанию используется trainred_alignment_points.json в папке участника."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Путь к выходному h5. "
            "По умолчанию создаётся файл тест+ред.h5 в папке участника."
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


def load_points(path: Path) -> dict:
    """Читает JSON с точками выравнивания и валидирует базовую структуру."""

    if not path.exists():
        raise FileNotFoundError(
            f"Файл с точками выравнивания не найден: {path}. "
            "Сначала запустите visualize_trainred_alignment.py."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "match" not in payload:
        raise ValueError(f"В файле {path} нет секции 'match'.")

    required_keys = {
        "train_start_sec",
        "train_end_sec",
        "h5_start_sec",
        "h5_end_sec",
        "best_feature",
        "correlation",
        "status",
    }
    missing = sorted(required_keys - set(payload["match"]))
    if missing:
        raise ValueError(f"В файле {path} не хватает ключей: {', '.join(missing)}")

    return payload


def copy_h5_tree(source_path: Path, dest_path: Path, force: bool) -> None:
    """Копирует исходный тест.h5 целиком в новый файл."""

    if dest_path.exists():
        if not force:
            raise FileExistsError(
                f"Выходной файл уже существует: {dest_path}. "
                "Используйте --force для перезаписи."
            )
        dest_path.unlink()

    with h5py.File(source_path, "r") as source, h5py.File(dest_path, "w") as dest:
        for attr_name, attr_value in source.attrs.items():
            dest.attrs[attr_name] = attr_value

        for item_name in source.keys():
            source.copy(item_name, dest)


def build_trainred_window(
    train_frame: pd.DataFrame,
    train_start_sec: float,
    train_end_sec: float,
) -> pd.DataFrame:
    """Вырезает окно train.red по относительному времени от начала CSV."""

    if TRAINRED_TIMESTAMP_COLUMN not in train_frame.columns:
        raise ValueError(
            f"В train.red нет колонки '{TRAINRED_TIMESTAMP_COLUMN}'."
        )

    train_window = train_frame.copy()
    train_window["train_relative_sec"] = (
        train_window[TRAINRED_TIMESTAMP_COLUMN].to_numpy(dtype=float)
        - float(train_window[TRAINRED_TIMESTAMP_COLUMN].iloc[0])
    )

    tolerance_sec = 1e-9
    mask = (
        (train_window["train_relative_sec"] >= train_start_sec - tolerance_sec)
        & (train_window["train_relative_sec"] <= train_end_sec + tolerance_sec)
    )
    train_window = train_window.loc[mask].copy()

    if train_window.empty:
        raise ValueError(
            "После вырезания окно train.red оказалось пустым. "
            "Проверьте точки выравнивания."
        )

    return train_window


def build_target_timestamps(
    source_h5_path: Path,
    train_window: pd.DataFrame,
    train_start_sec: float,
    h5_start_sec: float,
) -> np.ndarray:
    """Строит временную шкалу для вставленных каналов в системе времени h5."""

    with h5py.File(source_h5_path, "r") as handle:
        h5_anchor_ms = float(handle["channels/moxy.smo2/timestamps"][0]) + h5_start_sec * 1000.0

    # Здесь принципиально сохраняется накопленная временная структура train.red:
    # мы не ресемплируем и не навязываем шаг h5, а лишь переносим исходные
    # приращения времени в систему координат тест.h5.
    offsets_ms = (
        train_window["train_relative_sec"].to_numpy(dtype=float) - train_start_sec
    ) * 1000.0
    return (h5_anchor_ms + offsets_ms).astype(np.float64)


def validate_trainred_columns(train_frame: pd.DataFrame) -> None:
    """Проверяет, что в train.red есть все колонки для расширенного h5."""

    missing = sorted(column for column in TRAINRED_TO_H5_CHANNELS.values() if column not in train_frame.columns)
    if missing:
        raise ValueError(
            "В train.red не найдены нужные колонки: "
            f"{', '.join(missing)}"
        )


def add_trainred_channels(
    dest_path: Path,
    target_timestamps: np.ndarray,
    train_window: pd.DataFrame,
) -> None:
    """Добавляет вырезанные train.red каналы в уже скопированный h5."""

    with h5py.File(dest_path, "a") as handle:
        channels_group = handle.require_group("channels")

        for h5_channel_name, trainred_column in TRAINRED_TO_H5_CHANNELS.items():
            if h5_channel_name in channels_group:
                raise ValueError(
                    f"Канал {h5_channel_name} уже существует в {dest_path}."
                )

            values = train_window[trainred_column].to_numpy(dtype=np.float32)
            channel_group = channels_group.create_group(h5_channel_name)
            channel_group.create_dataset(
                "timestamps",
                data=target_timestamps,
                dtype=np.float64,
            )
            channel_group.create_dataset(
                "values",
                data=values,
                dtype=np.float32,
            )


def write_selection_metadata(
    dest_path: Path,
    match: dict,
    selection_mode: str,
    selection_comment: str,
) -> None:
    """Сохранён для совместимости интерфейса, но больше ничего не записывает."""

    _ = dest_path
    _ = match
    _ = selection_mode
    _ = selection_comment


def create_test_plus_red_h5(
    source_h5_path: Path,
    source_trainred_path: Path,
    output_path: Path,
    match: dict,
    force: bool,
    selection_mode: str,
    selection_comment: str,
) -> int:
    """Создаёт итоговый тест+ред.h5 по уже известным точкам выравнивания."""

    train_frame = load_trainred(source_trainred_path)
    validate_trainred_columns(train_frame)

    train_window = build_trainred_window(
        train_frame=train_frame,
        train_start_sec=float(match["train_start_sec"]),
        train_end_sec=float(match["train_end_sec"]),
    )
    target_timestamps = build_target_timestamps(
        source_h5_path=source_h5_path,
        train_window=train_window,
        train_start_sec=float(match["train_start_sec"]),
        h5_start_sec=float(match["h5_start_sec"]),
    )

    copy_h5_tree(
        source_path=source_h5_path,
        dest_path=output_path,
        force=force,
    )
    add_trainred_channels(
        dest_path=output_path,
        target_timestamps=target_timestamps,
        train_window=train_window,
    )
    write_selection_metadata(
        dest_path=output_path,
        match=match,
        selection_mode=selection_mode,
        selection_comment=selection_comment,
    )

    return len(target_timestamps)


def main() -> None:
    """Точка входа: создаёт расширенный файл тест+ред.h5."""

    args = parse_args()
    subject_dir = resolve_subject_dir(args.data_dir, args.subject)

    source_h5_path = subject_dir / "тест.h5"
    source_trainred_path = subject_dir / "train.red.csv"
    points_path = args.points_file or (subject_dir / DEFAULT_POINTS_FILENAME)
    output_path = args.output or (subject_dir / DEFAULT_OUTPUT_FILENAME)

    points = load_points(points_path)
    match = points["match"]
    selection_mode = str(points.get("selection_mode", "points_file"))
    selection_comment = str(
        points.get(
            "selection_comment",
            "Собрано из сохранённых точек выравнивания.",
        )
    )
    inserted_points = create_test_plus_red_h5(
        source_h5_path=source_h5_path,
        source_trainred_path=source_trainred_path,
        output_path=output_path,
        match=match,
        force=args.force,
        selection_mode=selection_mode,
        selection_comment=selection_comment,
    )

    print(f"Папка: {subject_dir.name}")
    print(f"Файл с точками: {points_path}")
    print(f"Источник h5: {source_h5_path}")
    print(f"Источник train.red: {source_trainred_path}")
    print(
        "Окно train.red: "
        f"{float(match['train_start_sec']):.1f}–{float(match['train_end_sec']):.1f} с"
    )
    print(
        "Окно h5: "
        f"{float(match['h5_start_sec']):.1f}–{float(match['h5_end_sec']):.1f} с"
    )
    print(f"Режим подбора: {selection_mode}")
    print(f"Комментарий: {selection_comment}")
    print(f"Добавлено точек в каждый train.red канал: {inserted_points}")
    print(f"Создан файл: {output_path}")


if __name__ == "__main__":
    main()
