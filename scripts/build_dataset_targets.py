#!/usr/bin/env python3
"""Строит таблицу таргетов `targets.parquet` по окнам."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from dataset_pipeline.common import save_parquet
from dataset_pipeline.targets import build_targets_table


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Строит dataset/targets.parquet по subjects.parquet и windows.parquet."
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "subjects.parquet",
        help="Путь к subjects.parquet.",
    )
    parser.add_argument(
        "--windows-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "windows.parquet",
        help="Путь к windows.parquet.",
    )
    parser.add_argument(
        "--lt1-labels-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "lt1_labels.parquet",
        help="Путь к lt1_labels.parquet. Если файл существует — LT1-таргеты добавляются автоматически.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "targets.parquet",
        help="Путь к выходному parquet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать выходной файл, если он уже существует.",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа: строит targets.parquet."""

    args = parse_args()
    lt1_path = args.lt1_labels_file if args.lt1_labels_file.exists() else None
    data_frame = build_targets_table(
        subjects_path=args.subjects_file,
        windows_path=args.windows_file,
        lt1_labels_path=lt1_path,
    )
    save_parquet(data_frame, args.output, force=args.force)

    print(f"Построен файл: {args.output}")
    print(f"Строк: {len(data_frame)}")
    print(
        "Бинарно валидных окон: "
        f"{int(data_frame['target_binary_valid'].sum()) if not data_frame.empty else 0}"
    )
    print(
        "Refined-usable окон: "
        f"{int(data_frame['target_time_to_lt2_refined_usable'].sum()) if not data_frame.empty else 0}"
    )


if __name__ == "__main__":
    main()
