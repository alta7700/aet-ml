"""CLI-скрипт: сборка признаков ЭМГ и кинематики педалирования.

Выход:
  dataset/features_emg_kinematics.parquet  — 77 признаков + 5 QC полей на окно
  dataset/session_params.parquet           — калибровочные параметры на участника

Использование:
  python scripts/build_dataset_emg_kinematics.py
  python scripts/build_dataset_emg_kinematics.py --force
  python scripts/build_dataset_emg_kinematics.py \\
      --subjects-file dataset/subjects.parquet \\
      --windows-file dataset/windows.parquet \\
      --output dataset/features_emg_kinematics.parquet \\
      --session-params-output dataset/session_params.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импортировать dataset_pipeline и methods
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR, save_parquet
from dataset_pipeline.emg_kinematics import build_emg_kinematics_table


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Собирает признаки ЭМГ и кинематики для всех окон датасета."
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "subjects.parquet",
        help="Путь к subjects.parquet (по умолчанию: dataset/subjects.parquet).",
    )
    parser.add_argument(
        "--windows-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "windows.parquet",
        help="Путь к windows.parquet (по умолчанию: dataset/windows.parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "features_emg_kinematics.parquet",
        help="Путь для записи features_emg_kinematics.parquet.",
    )
    parser.add_argument(
        "--session-params-output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "session_params.parquet",
        help="Путь для записи session_params.parquet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать выходные файлы, если они уже существуют.",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа скрипта."""
    args = parse_args()

    # Проверяем, нужно ли пересчитывать
    if (
        args.output.exists()
        and args.session_params_output.exists()
        and not args.force
    ):
        print(
            f"Файлы уже существуют:\n"
            f"  {args.output}\n"
            f"  {args.session_params_output}\n"
            "Используйте --force для перезаписи."
        )
        return

    print(f"Читаем участников: {args.subjects_file}")
    print(f"Читаем окна: {args.windows_file}")
    print("Строим признаки ЭМГ и кинематики...")

    t_start = time.perf_counter()
    features_df, session_params_df = build_emg_kinematics_table(
        subjects_path=args.subjects_file,
        windows_path=args.windows_file,
    )
    elapsed = time.perf_counter() - t_start

    print(f"\nГотово за {elapsed:.1f} с:")
    print(f"  Окон: {len(features_df)}")
    print(f"  Участников: {len(session_params_df)}")

    if not features_df.empty:
        n_valid = int(features_df["emg_valid"].sum())
        pct = 100.0 * n_valid / len(features_df)
        print(f"  EMG-валидных окон: {n_valid} ({pct:.1f}%)")

    save_parquet(features_df, args.output, args.force)
    save_parquet(session_params_df, args.session_params_output, args.force)

    print(f"\nЗаписано: {args.output}")
    print(f"Записано: {args.session_params_output}")


if __name__ == "__main__":
    main()
