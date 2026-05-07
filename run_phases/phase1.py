"""Фаза 1 — сборка датасета.

Топология (зависимости между шагами):

  subjects.parquet
      │
      ├─── windows.parquet ──────────────────────────────────┐
      │                                                       │
      └─── lt1_labels.parquet (параллельно с windows)        │
                  │                                          │
                  └───────────► targets.parquet ◄────────────┘
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
              features_emg_kinematics.parquet    features_hrv.parquet
              + session_params.parquet           (параллельно с EMG)
                      │
                      ▼
              features_nirs.parquet
              (нужен session_params от EMG)
                      │               │               │
                      └───────────────┴───────────────┘
                                      │
                               qc_windows.parquet
                                      │
                        merged_features_ml.parquet
                        sequence_index.parquet
                        qc_summary.md

Запуск:
  uv run python run_phases/phase1.py
  uv run python run_phases/phase1.py --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_DATASET = _ROOT / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Фаза 1: сборка датасета.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_ROOT / "data",
        help="Папка с участниками (по умолчанию: data/).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET,
        help="Выходная папка датасета (по умолчанию: dataset/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересобрать все файлы, даже если они уже существуют.",
    )
    return parser.parse_args()


def _cmd(script: str, *extra: str) -> list[str]:
    """Собирает команду uv run python scripts/{script} {extra}."""
    return ["uv", "run", "python", str(_SCRIPTS / script), *extra]


def run_seq(cmd: list[str], label: str) -> None:
    """Запускает команду синхронно; прерывает фазу при ошибке."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─' * 60}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=_ROOT)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n[ОШИБКА] {label} завершилась с кодом {result.returncode}. Останавливаемся.")
        sys.exit(result.returncode)
    print(f"  ✓ {label} — {elapsed:.1f} с")


def run_parallel(jobs: list[tuple[list[str], str]]) -> None:
    """Запускает список (cmd, label) параллельно; прерывает при любой ошибке."""
    labels = [label for _, label in jobs]
    print(f"\n{'─' * 60}")
    print(f"  Параллельно: {', '.join(labels)}")
    print(f"{'─' * 60}")

    t0 = time.perf_counter()

    def _run(item: tuple[list[str], str]) -> tuple[str, int]:
        cmd, label = item
        result = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
        # Выводим captured вывод по завершении чтобы не перемешивалось
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return label, result.returncode

    failed = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(_run, job): job[1] for job in jobs}
        for future in as_completed(futures):
            label, code = future.result()
            elapsed = time.perf_counter() - t0
            if code != 0:
                print(f"  ✗ {label} — код {code}")
                failed.append(label)
            else:
                print(f"  ✓ {label} — {elapsed:.1f} с")

    if failed:
        print(f"\n[ОШИБКА] Следующие шаги завершились с ошибкой: {failed}. Останавливаемся.")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    data_dir = str(args.data_dir)
    dataset_dir = str(args.dataset_dir)
    force_flag = ["--force"] if args.force else []

    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  ФАЗА 1: сборка датасета")
    print(f"  data={data_dir}  dataset={dataset_dir}  force={args.force}")
    print(f"{'#' * 60}")

    t_start = time.perf_counter()

    # ── Шаг 1: subjects ──────────────────────────────────────────────────────
    run_seq(
        _cmd("build_dataset_subjects.py",
             "--data-dir", data_dir, "--output", f"{dataset_dir}/subjects.parquet",
             *force_flag),
        "subjects.parquet",
    )

    # ── Шаг 2: windows + lt1_labels (параллельно, оба зависят только от subjects) ──
    run_parallel([
        (_cmd("build_dataset_windows.py",
              "--subjects-file", f"{dataset_dir}/subjects.parquet",
              "--output", f"{dataset_dir}/windows.parquet",
              *force_flag),
         "windows.parquet"),
        (_cmd("extract_lt1_labels.py",
              "--subjects-file", f"{dataset_dir}/subjects.parquet",
              "--output", f"{dataset_dir}/lt1_labels.parquet",
              *force_flag),
         "lt1_labels.parquet"),
    ])

    # ── Шаг 3: targets (нужны subjects + windows + lt1_labels) ───────────────
    run_seq(
        _cmd("build_dataset_targets.py",
             "--subjects-file", f"{dataset_dir}/subjects.parquet",
             "--windows-file", f"{dataset_dir}/windows.parquet",
             "--lt1-labels-file", f"{dataset_dir}/lt1_labels.parquet",
             "--output", f"{dataset_dir}/targets.parquet",
             *force_flag),
        "targets.parquet",
    )

    # ── Шаг 4a: EMG + HRV параллельно (оба зависят только от subjects + windows) ──
    # NIRS нельзя запускать здесь — ей нужен session_params.parquet из EMG
    run_parallel([
        (_cmd("build_dataset_emg_kinematics.py",
              "--subjects-file", f"{dataset_dir}/subjects.parquet",
              "--windows-file", f"{dataset_dir}/windows.parquet",
              "--output", f"{dataset_dir}/features_emg_kinematics.parquet",
              "--session-params-output", f"{dataset_dir}/session_params.parquet",
              *force_flag),
         "features_emg_kinematics.parquet + session_params.parquet"),
        (_cmd("build_dataset_hrv.py",
              "--subjects-file", f"{dataset_dir}/subjects.parquet",
              "--windows-file", f"{dataset_dir}/windows.parquet",
              "--output", f"{dataset_dir}/features_hrv.parquet",
              *force_flag),
         "features_hrv.parquet"),
    ])

    # ── Шаг 4b: NIRS (нужен session_params от EMG — запускаем после шага 4a) ──
    run_seq(
        _cmd("build_dataset_nirs.py",
             "--subjects-file", f"{dataset_dir}/subjects.parquet",
             "--windows-file", f"{dataset_dir}/windows.parquet",
             "--session-params-file", f"{dataset_dir}/session_params.parquet",
             "--output", f"{dataset_dir}/features_nirs.parquet",
             *force_flag),
        "features_nirs.parquet",
    )

    # ── Шаг 5: QC (нужны все три features) ───────────────────────────────────
    run_seq(
        _cmd("build_dataset_qc.py",
             "--emg-file", f"{dataset_dir}/features_emg_kinematics.parquet",
             "--nirs-file", f"{dataset_dir}/features_nirs.parquet",
             "--hrv-file", f"{dataset_dir}/features_hrv.parquet",
             "--output", f"{dataset_dir}/qc_windows.parquet",
             *force_flag),
        "qc_windows.parquet",
    )

    # ── Шаг 6: merge → финальный датасет ─────────────────────────────────────
    run_seq(
        _cmd("build_dataset_merged_ml.py",
             "--windows-file", f"{dataset_dir}/windows.parquet",
             "--targets-file", f"{dataset_dir}/targets.parquet",
             "--emg-file", f"{dataset_dir}/features_emg_kinematics.parquet",
             "--nirs-file", f"{dataset_dir}/features_nirs.parquet",
             "--hrv-file", f"{dataset_dir}/features_hrv.parquet",
             "--qc-file", f"{dataset_dir}/qc_windows.parquet",
             "--output", f"{dataset_dir}/merged_features_ml.parquet",
             "--sequence-index-output", f"{dataset_dir}/sequence_index.parquet",
             "--qc-summary-output", f"{dataset_dir}/qc_summary.md",
             *force_flag),
        "merged_features_ml.parquet + sequence_index.parquet + qc_summary.md",
    )

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'#' * 60}")
    print(f"  ФАЗА 1 ЗАВЕРШЕНА — {elapsed_total:.1f} с")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
