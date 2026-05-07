"""Главный оркестратор — запускает все фазы последовательно.

Фазы:
  0  phase0.py  — подготовка сырых данных (ID-проверка, выравнивание, fulltest, finaltest)
  1  phase1.py  — сборка датасета (subjects → windows → targets → features → merged)  [TODO]
  2  phase2.py  — обучение всех версий моделей                                         [TODO]
  3  phase3.py  — агрегированная оценка (evaluate_all)                                 [TODO]

Запуск всего пайплайна:
  uv run python run_phases/run_all.py

Запуск отдельной фазы:
  uv run python run_phases/run_all.py --from-phase 1
  uv run python run_phases/run_all.py --only-phase 0
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Реестр фаз: (номер, модуль, описание)
PHASES: list[tuple[int, str, str]] = [
    (0, "run_phases.phase0", "Подготовка сырых данных"),
    (1, "run_phases.phase1", "Сборка датасета"),
    (2, "run_phases.phase2", "Обучение моделей"),
    (3, "run_phases.phase3", "Агрегированная оценка"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Главный оркестратор пайплайна.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from-phase",
        type=int,
        metavar="N",
        help="Начать с фазы N (включительно), пропустить предыдущие.",
    )
    group.add_argument(
        "--only-phase",
        type=int,
        metavar="N",
        help="Запустить только фазу N.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_ROOT / "data",
        help="Папка с участниками (передаётся в phase0).",
    )
    return parser.parse_args()


def run_phase(module_path: str, phase_num: int, description: str) -> None:
    print(f"\n{'#' * 70}")
    print(f"#  ФАЗА {phase_num}: {description}")
    print(f"{'#' * 70}\n")
    mod = importlib.import_module(module_path)
    mod.main()


def main() -> None:
    args = parse_args()

    # Пробрасываем --data-dir в sys.argv для phase0, которая сама парсит аргументы
    # Простой способ: перезаписываем sys.argv перед импортом фазы
    # TODO: рефакторинг — передавать конфиг объектом, а не через sys.argv

    phases_to_run = [
        (num, mod, desc)
        for num, mod, desc in PHASES
        if (args.only_phase is None or num == args.only_phase)
        and (args.from_phase is None or num >= args.from_phase)
    ]

    if not phases_to_run:
        print("Нет фаз для запуска. Проверьте --from-phase / --only-phase.")
        sys.exit(1)

    for num, mod, desc in phases_to_run:
        run_phase(mod, num, desc)

    print(f"\n{'#' * 70}")
    print(f"#  ВСЕ ФАЗЫ ЗАВЕРШЕНЫ")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
