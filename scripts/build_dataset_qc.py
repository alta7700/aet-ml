"""CLI-скрипт: сборка QC-таблицы датасета.

Читает три таблицы признаков, извлекает QC-поля и объединяет их в qc_windows.parquet.

Выход: dataset/qc_windows.parquet

Использование:
  python scripts/build_dataset_qc.py
  python scripts/build_dataset_qc.py --force
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
from dataset_pipeline.qc import build_qc_table


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Собирает QC-флаги по всем окнам из трёх таблиц признаков."
    )
    parser.add_argument(
        "--emg-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "features_emg_kinematics.parquet",
    )
    parser.add_argument(
        "--nirs-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "features_nirs.parquet",
    )
    parser.add_argument(
        "--hrv-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "features_hrv.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "qc_windows.parquet",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Точка входа скрипта."""
    args = parse_args()

    if args.output.exists() and not args.force:
        print(f"Файл уже существует: {args.output}. Используйте --force для перезаписи.")
        return

    print("Строим QC-таблицу...")
    t_start = time.perf_counter()
    qc_df = build_qc_table(
        emg_path=args.emg_file,
        nirs_path=args.nirs_file,
        hrv_path=args.hrv_file,
    )
    elapsed = time.perf_counter() - t_start

    print(f"\nГотово за {elapsed:.1f} с:")
    print(f"  Окон всего: {len(qc_df)}")
    if not qc_df.empty:
        n_any = int(qc_df["window_valid_any"].sum())
        n_all = int(qc_df["window_valid_all_required"].sum())
        total = len(qc_df)
        print(f"  Валидных (хотя бы 1 модальность): {n_any} ({100.0 * n_any / total:.1f}%)")
        print(f"  Валидных (все 3 модальности):      {n_all} ({100.0 * n_all / total:.1f}%)")
        for col in ("emg_valid", "nirs_valid", "hrv_valid"):
            if col in qc_df.columns:
                n = int(qc_df[col].sum())
                print(f"  {col}: {n} ({100.0 * n / total:.1f}%)")

    save_parquet(qc_df, args.output, args.force)
    print(f"\nЗаписано: {args.output}")


if __name__ == "__main__":
    main()
