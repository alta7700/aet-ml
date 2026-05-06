"""CLI-скрипт: сборка признаков NIRS.

Выход: dataset/features_nirs.parquet (15 признаков + 3 QC-поля на окно).

Использование:
  python scripts/build_dataset_nirs.py
  python scripts/build_dataset_nirs.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR, save_parquet
from dataset_pipeline.nirs import build_nirs_table


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Собирает признаки NIRS для всех окон датасета."
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "subjects.parquet",
    )
    parser.add_argument(
        "--windows-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "windows.parquet",
    )
    parser.add_argument(
        "--session-params-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "session_params.parquet",
        help="Путь к session_params.parquet (нужен для baseline SmO2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "features_nirs.parquet",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Точка входа скрипта."""
    args = parse_args()

    if args.output.exists() and not args.force:
        print(f"Файл уже существует: {args.output}. Используйте --force для перезаписи.")
        return

    print(f"Читаем участников: {args.subjects_file}")
    print(f"Читаем окна: {args.windows_file}")
    print(f"Параметры сессий: {args.session_params_file}")
    print("Строим признаки NIRS...")

    t_start = time.perf_counter()
    data_frame = build_nirs_table(
        subjects_path=args.subjects_file,
        windows_path=args.windows_file,
        session_params_path=args.session_params_file,
    )
    elapsed = time.perf_counter() - t_start

    print(f"\nГотово за {elapsed:.1f} с:")
    print(f"  Окон: {len(data_frame)}")
    if not data_frame.empty and "nirs_valid" in data_frame.columns:
        n_valid = int(data_frame["nirs_valid"].sum())
        pct = 100.0 * n_valid / len(data_frame)
        print(f"  NIRS-валидных окон: {n_valid} ({pct:.1f}%)")

    save_parquet(data_frame, args.output, args.force)
    print(f"\nЗаписано: {args.output}")


if __name__ == "__main__":
    main()
