#!/usr/bin/env python3
"""Строит таблицу каузальных окон `windows.parquet`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from dataset_pipeline.common import DEFAULT_WINDOW_DURATION_SEC
from dataset_pipeline.common import DEFAULT_WINDOW_STEP_SEC
from dataset_pipeline.common import save_parquet
from dataset_pipeline.windows import build_windows_table


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Строит dataset/windows.parquet по таблице subjects.parquet."
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "subjects.parquet",
        help="Путь к subjects.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "windows.parquet",
        help="Путь к выходному parquet.",
    )
    parser.add_argument(
        "--window-duration-sec",
        type=float,
        default=DEFAULT_WINDOW_DURATION_SEC,
        help="Длина окна в секундах.",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=DEFAULT_WINDOW_STEP_SEC,
        help="Шаг окна в секундах.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать выходной файл, если он уже существует.",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа: строит windows.parquet."""

    args = parse_args()
    data_frame = build_windows_table(
        subjects_path=args.subjects_file,
        window_duration_sec=args.window_duration_sec,
        step_sec=args.step_sec,
    )
    save_parquet(data_frame, args.output, force=args.force)

    print(f"Построен файл: {args.output}")
    print(f"Окон: {len(data_frame)}")
    print(f"Участников: {data_frame['subject_id'].nunique() if not data_frame.empty else 0}")
    print(f"Окно: {args.window_duration_sec:.1f} с | шаг: {args.step_sec:.1f} с")


if __name__ == "__main__":
    main()

