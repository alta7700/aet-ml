"""CLI-скрипт: сборка признаков HRV / ВСР.

Выход: dataset/features_hrv.parquet (7 признаков + 4 QC-поля на окно).
Использует trailing-контекст 120 с (каузальный DFA-α1).

Использование:
  python scripts/build_dataset_hrv.py
  python scripts/build_dataset_hrv.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR, DEFAULT_HRV_CONTEXT_SEC, save_parquet
from dataset_pipeline.hrv import build_hrv_table


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Собирает признаки HRV (включая DFA-α1) для всех окон датасета."
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
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "features_hrv.parquet",
    )
    parser.add_argument(
        "--hrv-context-sec",
        type=float,
        default=DEFAULT_HRV_CONTEXT_SEC,
        help=f"Длина trailing-контекста ВСР в секундах (по умолчанию {DEFAULT_HRV_CONTEXT_SEC}).",
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
    print(f"Trailing-контекст HRV: {args.hrv_context_sec} с")
    print("Строим признаки HRV...")

    t_start = time.perf_counter()
    data_frame = build_hrv_table(
        subjects_path=args.subjects_file,
        windows_path=args.windows_file,
        hrv_context_sec=args.hrv_context_sec,
    )
    elapsed = time.perf_counter() - t_start

    print(f"\nГотово за {elapsed:.1f} с:")
    print(f"  Окон: {len(data_frame)}")
    if not data_frame.empty and "hrv_valid" in data_frame.columns:
        n_valid = int(data_frame["hrv_valid"].sum())
        pct = 100.0 * n_valid / len(data_frame)
        print(f"  HRV-валидных окон: {n_valid} ({pct:.1f}%)")
        if "hrv_dfa_alpha1" in data_frame.columns:
            valid_dfa = data_frame.loc[data_frame["hrv_valid"] == 1, "hrv_dfa_alpha1"].dropna()
            if len(valid_dfa) > 0:
                print(
                    f"  DFA-α1 (валидные): "
                    f"min={valid_dfa.min():.3f}, "
                    f"median={valid_dfa.median():.3f}, "
                    f"max={valid_dfa.max():.3f}"
                )

    save_parquet(data_frame, args.output, args.force)
    print(f"\nЗаписано: {args.output}")


if __name__ == "__main__":
    main()
