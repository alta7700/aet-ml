#!/usr/bin/env python3
"""Проставляет технические ID участникам, создавая файл ID в каждой папке."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import unicodedata

ID_FILENAME = "ID"
DEFAULT_PREFIX = "S"
DEFAULT_DIGITS = 3


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Создаёт в папках участников файл ID с техническим "
            "идентификатором вида S001."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Каталог с папками участников.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Префикс идентификатора.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=DEFAULT_DIGITS,
        help="Количество цифр в числовой части идентификатора.",
    )
    return parser.parse_args()


def sort_key(path: Path) -> tuple[str, str]:
    """Возвращает стабильный ключ сортировки для имён папок."""

    normalized = unicodedata.normalize("NFC", path.name)
    return normalized.casefold(), normalized


def list_subject_dirs(data_dir: Path) -> list[Path]:
    """Собирает и сортирует папки участников."""

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Каталог с данными не найден: {data_dir}")
    return sorted(
        (path for path in data_dir.iterdir() if path.is_dir()),
        key=sort_key,
    )


def build_id_pattern(prefix: str, digits: int) -> re.Pattern[str]:
    """Строит регулярное выражение для проверки ID."""

    return re.compile(rf"^{re.escape(prefix)}(\d{{{digits}}})$")


def read_subject_id(id_path: Path) -> str:
    """Читает ID из файла и валидирует базовый формат."""

    value = id_path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Файл ID пустой: {id_path}")
    if not value.isascii():
        raise ValueError(f"ID должен быть ASCII-строкой: {id_path}")
    if any(symbol.isspace() for symbol in value):
        raise ValueError(f"ID не должен содержать пробелы: {id_path}")
    return value


def format_subject_id(prefix: str, digits: int, number: int) -> str:
    """Формирует строковый идентификатор участника."""

    return f"{prefix}{number:0{digits}d}"


def find_first_free_number(used_numbers: set[int]) -> int:
    """Находит первый свободный номер ID без разрывов."""

    candidate = 1
    while candidate in used_numbers:
        candidate += 1
    return candidate


def assign_subject_ids(
    data_dir: Path,
    prefix: str,
    digits: int,
) -> list[tuple[Path, str, bool]]:
    """Назначает ID отсутствующим папкам и возвращает сводку."""

    if digits <= 0:
        raise ValueError("Параметр --digits должен быть положительным.")
    if not prefix or not prefix.isascii() or any(symbol.isspace() for symbol in prefix):
        raise ValueError("Параметр --prefix должен быть непустым ASCII-токеном без пробелов.")

    subject_dirs = list_subject_dirs(data_dir)
    id_pattern = build_id_pattern(prefix=prefix, digits=digits)
    used_ids: dict[str, Path] = {}
    used_numbers: set[int] = set()
    results: list[tuple[Path, str, bool]] = []

    # Сначала валидируем уже существующие ID, чтобы не создавать конфликтов.
    for subject_dir in subject_dirs:
        id_path = subject_dir / ID_FILENAME
        if not id_path.exists():
            continue

        subject_id = read_subject_id(id_path)
        match = id_pattern.fullmatch(subject_id)
        if match is None:
            raise ValueError(
                f"Файл {id_path} содержит ID '{subject_id}', "
                f"который не соответствует шаблону {prefix}{'0' * digits}."
            )
        if subject_id in used_ids:
            raise ValueError(
                f"Дублирующийся ID '{subject_id}' в папках "
                f"'{used_ids[subject_id].name}' и '{subject_dir.name}'."
            )

        used_ids[subject_id] = subject_dir
        used_numbers.add(int(match.group(1)))
        results.append((subject_dir, subject_id, False))

    # Новым папкам выдаём первый свободный номер, чтобы случайная потеря
    # одного файла ID не сдвигала идентичность участника в хвост диапазона.
    for subject_dir in subject_dirs:
        id_path = subject_dir / ID_FILENAME
        if id_path.exists():
            continue

        next_number = find_first_free_number(used_numbers)
        subject_id = format_subject_id(prefix=prefix, digits=digits, number=next_number)
        id_path.write_text(f"{subject_id}\n", encoding="utf-8")
        used_ids[subject_id] = subject_dir
        used_numbers.add(next_number)
        results.append((subject_dir, subject_id, True))

    return sorted(results, key=lambda item: sort_key(item[0]))


def main() -> None:
    """Точка входа: создаёт отсутствующие ID и печатает сводку."""

    args = parse_args()
    results = assign_subject_ids(
        data_dir=args.data_dir,
        prefix=args.prefix,
        digits=args.digits,
    )

    created_count = sum(1 for _, _, created in results if created)
    print(f"Каталог: {args.data_dir}")
    print(f"Создано новых ID: {created_count}")
    for subject_dir, subject_id, created in results:
        status = "создан" if created else "уже существует"
        print(f"{subject_id} | {status} | {subject_dir.name}")


if __name__ == "__main__":
    main()
