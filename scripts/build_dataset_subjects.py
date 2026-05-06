#!/usr/bin/env python3
"""Строит таблицу испытуемых `subjects.parquet` из `finaltest.h5`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.common import DEFAULT_DATA_DIR
from dataset_pipeline.common import DEFAULT_DATASET_DIR
from dataset_pipeline.common import save_parquet
from dataset_pipeline.subjects import build_subjects_table


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Строит dataset/subjects.parquet по всем доступным finaltest.h5."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Каталог с папками участников.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "subjects.parquet",
        help="Путь к выходному parquet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать выходной файл, если он уже существует.",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа: строит subjects.parquet."""

    args = parse_args()
    data_frame, skipped = build_subjects_table(args.data_dir)
    save_parquet(data_frame, args.output, force=args.force)

    print(f"Построен файл: {args.output}")
    print(f"Строк: {len(data_frame)}")
    print(f"Участников: {data_frame['subject_id'].nunique() if not data_frame.empty else 0}")
    if skipped:
        print("Пропущены папки без finaltest.h5:")
        for name in skipped:
            print(f"- {name}")


if __name__ == "__main__":
    main()

