"""Фаза 3 — агрегированная оценка всех версий моделей.

Запускает evaluate_all.py, который:
  - прогоняет LOSO для каждой зарегистрированной версии
  - строит графики Acc@δ(t_norm) и TDE по участникам
  - сохраняет сравнительные графики в artefacts/

Выходные артефакты:
  artefacts/v{NNNN}/lt2_acc_by_time.png
  artefacts/v{NNNN}/lt2_tde.png
  artefacts/comparison_lt2.png
  artefacts/comparison_lt1.png

Запуск:
  uv run python run_phases/phase3.py
  uv run python run_phases/phase3.py --target lt2
  uv run python run_phases/phase3.py --versions v0004 v0009
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Фаза 3: агрегированная оценка версий.")
    parser.add_argument(
        "--target",
        choices=["lt1", "lt2", "both"],
        default="both",
        help="Какой таргет оценивать (по умолчанию: both).",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        metavar="vNNNN",
        help="Оценить только указанные версии.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не сохранять графики (только вывод метрик в stdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cmd = ["uv", "run", "python", str(_SCRIPTS / "evaluate_all.py")]

    if args.target != "both":
        cmd += ["--target", args.target]
    if args.versions:
        cmd += ["--versions"] + args.versions
    if args.no_plots:
        cmd.append("--no-plots")

    print(f"\n{'#' * 60}")
    print(f"  ФАЗА 3: агрегированная оценка")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'#' * 60}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=_ROOT)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n[ОШИБКА] evaluate_all.py завершился с кодом {result.returncode}.")
        sys.exit(result.returncode)

    print(f"\n{'#' * 60}")
    print(f"  ФАЗА 3 ЗАВЕРШЕНА — {elapsed:.1f} с")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
