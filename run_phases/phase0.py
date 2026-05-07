"""Фаза 0 — подготовка сырых данных.

Обрабатывает каждого участника последовательно:
  1. Проверка наличия файла ID у всех участников (стоп если нет).
  2. Выравнивание тест.h5 + train.red.csv:
       - если нет trainred_alignment_points.json → запускает визуализацию,
         пользователь вручную сохраняет точки и жмёт кнопку создания тест+ред.h5
       - если json есть, но тест+ред.h5 нет → вызывает create_test_plus_red_h5 напрямую
  3. Создание fulltest.h5 (--force).
  4. Создание finaltest.h5 (--force).

Запуск:
  uv run python run_phases/phase0.py
  uv run python run_phases/phase0.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"

ALIGNMENT_ARTIFACT = "trainred_alignment_points.json"
TESTRED_FILENAME = "тест+ред.h5"
ID_FILENAME = "ID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Фаза 0: подготовка сырых данных.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_ROOT / "data",
        help="Папка с участниками (по умолчанию: data/).",
    )
    return parser.parse_args()


def run(cmd: list[str], desc: str) -> None:
    """Запускает команду, прокидывает вывод в терминал, прерывает при ошибке."""
    print(f"\n{'─' * 60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, cwd=_ROOT)
    if result.returncode != 0:
        print(f"\n[ОШИБКА] Команда завершилась с кодом {result.returncode}. Останавливаемся.")
        sys.exit(result.returncode)


def subject_folders(data_dir: Path) -> list[Path]:
    """Возвращает список папок участников (все подпапки data_dir)."""
    return sorted(p for p in data_dir.iterdir() if p.is_dir())


# ─── Шаг 1: проверка ID ──────────────────────────────────────────────────────

def check_ids(subjects: list[Path]) -> None:
    """Проверяет, что у каждого участника есть файл ID. Останавливается если нет."""
    missing = [s for s in subjects if not (s / ID_FILENAME).exists()]
    if not missing:
        print(f"[OK] Файл ID найден у всех {len(subjects)} участников.")
        return
    print("\n[СТОП] Файл ID отсутствует у следующих участников:")
    for s in missing:
        print(f"  • {s.name}")
    print("\nЗапустите assign_subject_ids.py, затем повторите фазу 0.")
    sys.exit(1)


# ─── Шаг 2: выравнивание тест.h5 + train.red.csv ─────────────────────────────

def ensure_alignment(subject_dir: Path, data_dir: Path) -> None:
    """Проверяет артефакт выравнивания; при необходимости запускает GUI."""
    name = subject_dir.name
    alignment_file = subject_dir / ALIGNMENT_ARTIFACT
    testred_file = subject_dir / TESTRED_FILENAME

    if not alignment_file.exists():
        # Пользователь должен выполнить выравнивание вручную через GUI
        print(f"\n[ВЫРАВНИВАНИЕ] {name}: нет {ALIGNMENT_ARTIFACT} → запускаем визуализацию.")
        print("  Настройте ползунок, затем нажмите кнопку «Сохранить и создать тест+ред.h5».")
        run(
            ["uv", "run", "python", str(_SCRIPTS / "visualize_trainred_alignment.py"), name,
             "--data-dir", str(data_dir)],
            f"{name}: visualize_trainred_alignment (ручное сопоставление)",
        )
        # После закрытия GUI оба файла должны существовать
        if not alignment_file.exists():
            print(f"[ОШИБКА] {ALIGNMENT_ARTIFACT} так и не создан. Завершаем.")
            sys.exit(1)
        if not testred_file.exists():
            print(f"[ОШИБКА] {TESTRED_FILENAME} так и не создан. Завершаем.")
            sys.exit(1)
        print(f"[OK] {name}: выравнивание выполнено.")
        return

    if not testred_file.exists():
        # JSON уже есть, но h5 пересоздали вручную или удалили — строим напрямую
        print(f"\n[ВЫРАВНИВАНИЕ] {name}: json есть, но {TESTRED_FILENAME} отсутствует → пересоздаём.")
        run(
            ["uv", "run", "python", str(_SCRIPTS / "create_test_plus_red_h5.py"), name,
             "--data-dir", str(data_dir)],
            f"{name}: create_test_plus_red_h5",
        )
        return

    print(f"[SKIP] {name}: выравнивание уже выполнено ({ALIGNMENT_ARTIFACT} + {TESTRED_FILENAME}).")


# ─── Шаг 3: создание fulltest.h5 ─────────────────────────────────────────────

def build_fulltest(subject_dir: Path, data_dir: Path) -> None:
    name = subject_dir.name
    run(
        ["uv", "run", "python", str(_SCRIPTS / "create_fulltest_h5.py"), name,
         "--data-dir", str(data_dir), "--force"],
        f"{name}: create_fulltest_h5 --force",
    )


# ─── Шаг 4: создание finaltest.h5 ────────────────────────────────────────────

def build_finaltest(subject_dir: Path, data_dir: Path) -> None:
    name = subject_dir.name
    run(
        ["uv", "run", "python", str(_SCRIPTS / "create_finaltest_h5.py"), name,
         "--data-dir", str(data_dir), "--force"],
        f"{name}: create_finaltest_h5 --force",
    )


# ─── Точка входа ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir

    if not data_dir.exists():
        print(f"[ОШИБКА] Папка с данными не найдена: {data_dir}")
        sys.exit(1)

    subjects = subject_folders(data_dir)
    if not subjects:
        print(f"[ОШИБКА] В {data_dir} нет папок участников.")
        sys.exit(1)

    print(f"\n=== ФАЗА 0: подготовка сырых данных ===")
    print(f"Найдено участников: {len(subjects)}")

    # Шаг 1 — общая проверка ID перед любой обработкой
    print("\n--- Шаг 1: проверка файлов ID ---")
    check_ids(subjects)

    # Шаги 2–4 — последовательно по каждому участнику
    for subject_dir in subjects:
        name = subject_dir.name
        print(f"\n{'═' * 60}")
        print(f"  Участник: {name}")
        print(f"{'═' * 60}")

        # Шаг 2: выравнивание
        ensure_alignment(subject_dir, data_dir)

        # Шаг 3: fulltest
        build_fulltest(subject_dir, data_dir)

        # Шаг 4: finaltest
        build_finaltest(subject_dir, data_dir)

    print(f"\n{'═' * 60}")
    print(f"  ФАЗА 0 ЗАВЕРШЕНА — {len(subjects)} участников обработано.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
