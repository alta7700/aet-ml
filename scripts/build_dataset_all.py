"""CLI-скрипт: полный пайплайн сборки датасета.

Запускает все шаги последовательно:
  0. subjects.parquet      (уже должен существовать)
  1. windows.parquet       (уже должен существовать)
  2. targets.parquet       (уже должен существовать)
  3. session_params.parquet + features_emg_kinematics.parquet
  4. features_nirs.parquet
  5. features_hrv.parquet
  6. qc_windows.parquet
  7. merged_features_ml.parquet + sequence_index.parquet + qc_summary.md

Каждый шаг пропускается, если выходной файл уже существует и --force не передан.

Использование:
  python scripts/build_dataset_all.py
  python scripts/build_dataset_all.py --force
  python scripts/build_dataset_all.py --data-dir /path/to/data --dataset-dir /path/to/dataset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_pipeline.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_HRV_CONTEXT_SEC,
    save_parquet,
)
from dataset_pipeline.emg_kinematics import build_emg_kinematics_table
from dataset_pipeline.hrv import build_hrv_table
from dataset_pipeline.merge import build_merged_table, generate_qc_summary
from dataset_pipeline.nirs import build_nirs_table
from dataset_pipeline.qc import build_qc_table
from dataset_pipeline.subjects import build_subjects_table
from dataset_pipeline.targets import build_targets_table
from dataset_pipeline.windows import build_windows_table


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description=(
            "Полный пайплайн сборки датасета: от subjects.parquet до merged_features_ml.parquet."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Корневая папка с данными участников (по умолчанию: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Папка для выходных файлов (по умолчанию: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перестроить все шаги, даже если файлы уже существуют.",
    )
    return parser.parse_args()


def _should_skip(path: Path, force: bool) -> bool:
    """Возвращает True, если файл уже существует и --force не указан."""
    if path.exists() and not force:
        print(f"  ⏭  Пропускаем (уже есть): {path.name}")
        return True
    return False


def _step_header(step: int, name: str) -> None:
    """Печатает заголовок шага."""
    print(f"\n{'─' * 60}")
    print(f"Шаг {step}: {name}")
    print(f"{'─' * 60}")


def main() -> None:
    """Полный пайплайн сборки датасета."""
    args = parse_args()
    d = args.dataset_dir
    d.mkdir(parents=True, exist_ok=True)

    t_pipeline_start = time.perf_counter()
    print(f"Пайплайн датасета: data={args.data_dir}, dataset={d}, force={args.force}")

    # ── Шаг 0: subjects ──
    _step_header(0, "subjects.parquet")
    subjects_path = d / "subjects.parquet"
    if not _should_skip(subjects_path, args.force):
        t0 = time.perf_counter()
        subjects_df, skipped = build_subjects_table(args.data_dir)
        save_parquet(subjects_df, subjects_path, force=True)
        skip_msg = f", пропущено папок: {len(skipped)}" if skipped else ""
        print(f"  ✓  {len(subjects_df)} участников за {time.perf_counter() - t0:.1f} с{skip_msg}")

    # ── Шаг 1: windows ──
    _step_header(1, "windows.parquet")
    windows_path = d / "windows.parquet"
    if not _should_skip(windows_path, args.force):
        t0 = time.perf_counter()
        windows_df = build_windows_table(subjects_path=subjects_path)
        save_parquet(windows_df, windows_path, force=True)
        print(f"  ✓  {len(windows_df)} окон за {time.perf_counter() - t0:.1f} с")

    # ── Шаг 2: targets ──
    _step_header(2, "targets.parquet")
    targets_path = d / "targets.parquet"
    if not _should_skip(targets_path, args.force):
        t0 = time.perf_counter()
        targets_df = build_targets_table(
            subjects_path=subjects_path,
            windows_path=windows_path,
        )
        save_parquet(targets_df, targets_path, force=True)
        print(f"  ✓  {len(targets_df)} таргетов за {time.perf_counter() - t0:.1f} с")

    # ── Шаг 3: EMG + кинематика ──
    _step_header(3, "features_emg_kinematics.parquet + session_params.parquet")
    emg_path = d / "features_emg_kinematics.parquet"
    session_params_path = d / "session_params.parquet"
    if not (_should_skip(emg_path, args.force) and _should_skip(session_params_path, args.force)):
        t0 = time.perf_counter()
        emg_df, session_params_df = build_emg_kinematics_table(
            subjects_path=subjects_path,
            windows_path=windows_path,
        )
        save_parquet(emg_df, emg_path, force=True)
        save_parquet(session_params_df, session_params_path, force=True)
        n_valid = int(emg_df["emg_valid"].sum()) if not emg_df.empty else 0
        print(
            f"  ✓  {len(emg_df)} окон, {n_valid} EMG-валидных, "
            f"{len(session_params_df)} участников за {time.perf_counter() - t0:.1f} с"
        )

    # ── Шаг 4: NIRS ──
    _step_header(4, "features_nirs.parquet")
    nirs_path = d / "features_nirs.parquet"
    if not _should_skip(nirs_path, args.force):
        t0 = time.perf_counter()
        nirs_df = build_nirs_table(
            subjects_path=subjects_path,
            windows_path=windows_path,
            session_params_path=session_params_path,
        )
        save_parquet(nirs_df, nirs_path, force=True)
        n_valid = int(nirs_df["nirs_valid"].sum()) if not nirs_df.empty else 0
        print(
            f"  ✓  {len(nirs_df)} окон, {n_valid} NIRS-валидных "
            f"за {time.perf_counter() - t0:.1f} с"
        )

    # ── Шаг 5: HRV ──
    _step_header(5, "features_hrv.parquet")
    hrv_path = d / "features_hrv.parquet"
    if not _should_skip(hrv_path, args.force):
        t0 = time.perf_counter()
        hrv_df = build_hrv_table(
            subjects_path=subjects_path,
            windows_path=windows_path,
            hrv_context_sec=DEFAULT_HRV_CONTEXT_SEC,
        )
        save_parquet(hrv_df, hrv_path, force=True)
        n_valid = int(hrv_df["hrv_valid"].sum()) if not hrv_df.empty else 0
        print(
            f"  ✓  {len(hrv_df)} окон, {n_valid} HRV-валидных "
            f"за {time.perf_counter() - t0:.1f} с"
        )

    # ── Шаг 6: QC ──
    _step_header(6, "qc_windows.parquet")
    qc_path = d / "qc_windows.parquet"
    if not _should_skip(qc_path, args.force):
        t0 = time.perf_counter()
        qc_df = build_qc_table(
            emg_path=emg_path,
            nirs_path=nirs_path,
            hrv_path=hrv_path,
        )
        save_parquet(qc_df, qc_path, force=True)
        n_all = int(qc_df["window_valid_all_required"].sum()) if not qc_df.empty else 0
        print(
            f"  ✓  {len(qc_df)} окон, {n_all} валидных по всем 3 модальностям "
            f"за {time.perf_counter() - t0:.1f} с"
        )

    # ── Шаг 7: Merge + QC summary ──
    _step_header(7, "merged_features_ml.parquet + sequence_index.parquet + qc_summary.md")
    merged_path = d / "merged_features_ml.parquet"
    seq_index_path = d / "sequence_index.parquet"
    qc_summary_path = d / "qc_summary.md"
    if not (_should_skip(merged_path, args.force) and _should_skip(seq_index_path, args.force)):
        t0 = time.perf_counter()
        merged_df, sequence_index_df = build_merged_table(
            windows_path=windows_path,
            targets_path=targets_path,
            emg_path=emg_path,
            nirs_path=nirs_path,
            hrv_path=hrv_path,
            qc_path=qc_path,
        )
        save_parquet(merged_df, merged_path, force=True)
        save_parquet(sequence_index_df, seq_index_path, force=True)
        generate_qc_summary(merged_df, qc_summary_path, subjects_path=subjects_path)
        print(
            f"  ✓  {len(merged_df)} окон, {len(merged_df.columns)} признаков "
            f"за {time.perf_counter() - t0:.1f} с"
        )
    else:
        # Если merged уже есть, но qc_summary нет — перегенерируем отчёт
        if not qc_summary_path.exists() or args.force:
            import pandas as pd
            merged_df = pd.read_parquet(merged_path)
            generate_qc_summary(merged_df, qc_summary_path, subjects_path=subjects_path)

    total_elapsed = time.perf_counter() - t_pipeline_start
    print(f"\n{'═' * 60}")
    print(f"✅  Пайплайн завершён за {total_elapsed:.1f} с")
    print(f"    Датасет: {d.resolve()}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
