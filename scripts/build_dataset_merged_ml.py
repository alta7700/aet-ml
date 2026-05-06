"""CLI-скрипт: финальная сборка объединённого датасета ML.

Выходные файлы:
  dataset/merged_features_ml.parquet  — полный датасет для классического ML
  dataset/sequence_index.parquet       — lookup-таблица для нейросети (DataLoader)
  dataset/qc_summary.md                — отчёт о качестве датасета

Использование:
  python scripts/build_dataset_merged_ml.py
  python scripts/build_dataset_merged_ml.py --force
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
from dataset_pipeline.merge import build_merged_table, generate_qc_summary


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Объединяет все таблицы признаков в merged_features_ml.parquet."
    )
    parser.add_argument(
        "--windows-file", type=Path, default=DEFAULT_DATASET_DIR / "windows.parquet"
    )
    parser.add_argument(
        "--targets-file", type=Path, default=DEFAULT_DATASET_DIR / "targets.parquet"
    )
    parser.add_argument(
        "--emg-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "features_emg_kinematics.parquet",
    )
    parser.add_argument(
        "--nirs-file", type=Path, default=DEFAULT_DATASET_DIR / "features_nirs.parquet"
    )
    parser.add_argument(
        "--hrv-file", type=Path, default=DEFAULT_DATASET_DIR / "features_hrv.parquet"
    )
    parser.add_argument(
        "--qc-file", type=Path, default=DEFAULT_DATASET_DIR / "qc_windows.parquet"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet",
    )
    parser.add_argument(
        "--sequence-index-output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "sequence_index.parquet",
    )
    parser.add_argument(
        "--qc-summary-output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "qc_summary.md",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Точка входа скрипта."""
    args = parse_args()

    if (
        args.output.exists()
        and args.sequence_index_output.exists()
        and not args.force
    ):
        print(
            f"Файлы уже существуют:\n"
            f"  {args.output}\n"
            f"  {args.sequence_index_output}\n"
            "Используйте --force для перезаписи."
        )
        return

    print("Объединяем таблицы признаков...")
    t_start = time.perf_counter()
    merged_df, sequence_index_df = build_merged_table(
        windows_path=args.windows_file,
        targets_path=args.targets_file,
        emg_path=args.emg_file,
        nirs_path=args.nirs_file,
        hrv_path=args.hrv_file,
        qc_path=args.qc_file,
    )
    elapsed = time.perf_counter() - t_start

    print(f"\nГотово за {elapsed:.1f} с:")
    print(f"  Окон: {len(merged_df)}")
    print(f"  Признаков: {len(merged_df.columns)}")
    print(f"  window_id уникальны: {merged_df['window_id'].is_unique}")

    if not merged_df.empty:
        n_any = int(merged_df.get("window_valid_any", 0).sum())
        n_all = int(merged_df.get("window_valid_all_required", 0).sum())
        total = len(merged_df)
        print(f"  Валидных (хотя бы 1): {n_any} ({100.0 * n_any / total:.1f}%)")
        print(f"  Валидных (все 3):     {n_all} ({100.0 * n_all / total:.1f}%)")

    save_parquet(merged_df, args.output, args.force)
    save_parquet(sequence_index_df, args.sequence_index_output, args.force)
    print(f"\nЗаписано: {args.output}")
    print(f"Записано: {args.sequence_index_output}")

    # Генерируем QC-отчёт (передаём subjects для LT2 quality breakdown)
    generate_qc_summary(merged_df, args.qc_summary_output, subjects_path=args.windows_file.parent / "subjects.parquet")


if __name__ == "__main__":
    main()
