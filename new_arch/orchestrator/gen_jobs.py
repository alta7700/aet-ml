"""Генератор jobs.csv для orchestrator'а.

Создаёт декартов набор:
  архитектура × target × feature_set × with_abs.

  Lin    — одна batch-задача (linear_runner --grid-all), считает все 580 моделей.
  LSTM1..LSTM16 × {lt1, lt2} × {EMG, EMG+NIRS, EMG+NIRS+HRV} × {abs, no_abs}
  TCN1..TCN4    × {lt1, lt2} × {EMG, EMG+NIRS, EMG+NIRS+HRV} × {abs, no_abs}

Для stateful LSTM (LSTM7..LSTM12) runner отличается — берётся lstm_stateful_runner.py.

Опции:
  --gpu-only   только NN (LSTM + TCN), без линейных моделей.

Выход: orchestrator/jobs.csv.
  job_id, runner, architecture_id, target, feature_set, with_abs, wavelet_mode, cmd
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from architectures import LINEAR_ARCHS, LSTM_ARCHS, TCN_ARCHS


OUT_PATH = Path(__file__).resolve().parent / "jobs.csv"

TARGETS = ["lt1", "lt2"]
FEATURE_SETS_LIN = ["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"]
FEATURE_SETS_NN = ["EMG", "EMG+NIRS", "EMG+NIRS+HRV"]
WITH_ABS_VARIANTS = [True, False]


def _runner_for(arch) -> str:
    """Какой runner запускать для архитектуры."""
    if arch.family == "Lin":
        return "linear_runner.py"
    if arch.family == "TCN":
        return "tcn_runner.py"
    if arch.family == "LSTM":
        if arch.model_class_name == "LSTMStatefulRegressor":
            return "lstm_stateful_runner.py"
        return "lstm_runner.py"
    raise ValueError(f"Неизвестная family={arch.family!r}")


def _build_cmd(runner: str, arch, target: str, fset: str, with_abs: bool) -> str:
    flag = "--with-abs" if with_abs else "--no-abs"
    return (
        f"PYTHONPATH=. uv run python {runner} "
        f"--architecture {arch.architecture_id} "
        f"--target {target} "
        f"--feature-set {fset} {flag}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="генератор jobs.csv для orchestrator'а")
    p.add_argument("--gpu-only", action="store_true",
                   help="только NN (LSTM+TCN), пропустить linear batch job")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    n = 0

    def add(arch, fset_list):
        nonlocal n
        runner = _runner_for(arch)
        for target, fset, with_abs in itertools.product(TARGETS, fset_list, WITH_ABS_VARIANTS):
            n += 1
            rows.append({
                "job_id": f"J{n:04d}",
                "runner": Path(runner).name,
                "architecture_id": arch.architecture_id,
                "target": target,
                "feature_set": fset,
                "with_abs": str(with_abs).lower(),
                "wavelet_mode": arch.forced_wavelet_mode or "none",
                "cmd": _build_cmd(runner, arch, target, fset, with_abs),
            })

    if not args.gpu_only:
        # Лин — батч-режим: одна job на весь декартов набор Lin × target × fset × abs.
        # Работает на CPU (joblib parallel), не нагружает GPU.
        n += 1
        rows.append({
            "job_id": f"J{n:04d}",
            "runner": "linear_runner.py",
            "architecture_id": "ALL_LIN",
            "target": "both",
            "feature_set": "all",
            "with_abs": "both",
            "wavelet_mode": "none",
            "cmd": "PYTHONPATH=. uv run python linear_runner.py --grid-all",
        })

    for arch in LSTM_ARCHS:
        add(arch, FEATURE_SETS_NN)
    for arch in TCN_ARCHS:
        add(arch, FEATURE_SETS_NN)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Сводка по семействам.
    from collections import Counter
    c = Counter()
    for r in rows:
        family = r["architecture_id"][:3]  # Lin, LST, TCN
        c[family] += 1
    print(f"Сгенерировано {len(rows)} задач:")
    for fam, count in sorted(c.items()):
        print(f"  {fam}*  {count}")
    print(f"\n→ {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
