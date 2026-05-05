#!/usr/bin/env python3
"""Генерирует summary.csv для каждой папки в каталоге data."""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubjectData:
    """Нормализованные данные по участнику для выгрузки в итоговый CSV."""

    full_name: str
    height: str
    age: str
    sex: str
    body_fat_percent: str
    skeletal_muscle_mass: str
    dominant_leg_lean_mass: str
    dominant_leg_fat_mass: str
    phase_angle: str
    time_lactate: tuple[tuple[str, str], ...]


# Значения с InBody и протоколов внесены после ручной проверки сканов.
# Для рукописных строк слепой OCR давал слишком много тихих ошибок.
SUBJECTS: dict[str, SubjectData] = {
    "Анна Агаронова": SubjectData(
        full_name="Анна Агаронова",
        height="170,5",
        age="42",
        sex="Мужской",
        body_fat_percent="10,2",
        skeletal_muscle_mass="43,0",
        dominant_leg_lean_mass="10,90",
        dominant_leg_fat_mass="1,3",
        phase_angle="8,0",
        time_lactate=(
            ("3:30", "1,41"),
            ("6:30", "1,47"),
            ("9:30", "2,77"),
            ("12:30", "3,71"),
            ("15:30", "6,77"),
            ("16:40", "8,32"),
            ("18:00", "7,68"),
        ),
    ),
    "Васильева Ольга": SubjectData(
        full_name="Васильева Ольга",
        height="170",
        age="32",
        sex="Женский",
        body_fat_percent="29,7",
        skeletal_muscle_mass="26,7",
        dominant_leg_lean_mass="7,77",
        dominant_leg_fat_mass="3,3",
        phase_angle="5,8",
        time_lactate=(
            ("3:50", "1,06"),
            ("7:00", "1,25"),
            ("9:45", "2,19"),
            ("13:00", "3,79"),
            ("15:50", "6,48"),
            ("19:45", "1,88"),
            ("21:30", "6,92"),
        ),
    ),
    "Дмитрий Онищенко": SubjectData(
        full_name="Дмитрий Онищенко",
        height="183,2",
        age="40",
        sex="Мужской",
        body_fat_percent="6,9",
        skeletal_muscle_mass="40,8",
        dominant_leg_lean_mass="10,49",
        dominant_leg_fat_mass="0,9",
        phase_angle="7,2",
        time_lactate=(
            ("3:30", "0,89"),
            ("6:30", "1,32"),
            ("9:30", "2,00"),
            ("12:30", "2,23"),
            ("15:30", "3,92"),
            ("18:30", "6,41"),
        ),
    ),
    "Иван Афаневич": SubjectData(
        full_name="Иван Афаневич",
        height="177",
        age="38",
        sex="Мужской",
        body_fat_percent="12,2",
        skeletal_muscle_mass="38,4",
        dominant_leg_lean_mass="11,07",
        dominant_leg_fat_mass="1,6",
        phase_angle="6,2",
        time_lactate=(
            ("4:01", "0,93"),
            ("6:30", "1,08"),
            ("9:30", "1,44"),
            ("13:40", "1,42"),
            ("15:30", "1,48"),
            ("18:39", "1,88"),
            ("21:30", "2,30"),
            ("24:30", "2,85"),
            ("27:30", "3,79"),
            ("30:30", "5,23"),
        ),
    ),
    "Мартынова Елена": SubjectData(
        full_name="Мартынова Елена",
        height="161",
        age="29",
        sex="Женский",
        body_fat_percent="20,9",
        skeletal_muscle_mass="23,3",
        dominant_leg_lean_mass="6,27",
        dominant_leg_fat_mass="1,8",
        phase_angle="5,2",
        time_lactate=(
            ("3:30", "1,4"),
            ("6:30", "1,8"),
            ("10:05", "3,89"),
            ("12:40", "6,44"),
            ("14:10", "6,55"),
            ("15:50", "8,98"),
        ),
    ),
    "Петр Макеев": SubjectData(
        full_name="Петр Макеев",
        height="170,5",
        age="42",
        sex="Мужской",
        body_fat_percent="10,2",
        skeletal_muscle_mass="43,0",
        dominant_leg_lean_mass="10,90",
        dominant_leg_fat_mass="1,3",
        phase_angle="8,0",
        time_lactate=(
            ("4:20", "1,29"),
            ("6:30", "1,19"),
            ("9:30", "1,74"),
            ("12:30", "2,25"),
            ("15:30", "2,85"),
            ("18:30", "4,08"),
            ("21:30", "5,12"),
            ("24:30", "8,1"),
            ("26:40", "10,28"),
        ),
    ),
    "Сергей Дубровин": SubjectData(
        full_name="Сергей Дубровин",
        height="181,4",
        age="23",
        sex="Мужской",
        body_fat_percent="20,7",
        skeletal_muscle_mass="35,8",
        dominant_leg_lean_mass="10,33",
        dominant_leg_fat_mass="2,5",
        phase_angle="6,0",
        time_lactate=(
            ("3:30", "2,19"),
            ("6:30", "2,11"),
            ("9:30", "2,84"),
            ("12:30", "4,07"),
            ("15:30", "5,97"),
            ("18:30", "9,04"),
            ("22:00", "11,77"),
            ("23:30", "11,80"),
        ),
    ),
    "Хренова Александра": SubjectData(
        full_name="Хренова Александра",
        height="165,6",
        age="25",
        sex="Женский",
        body_fat_percent="23,2",
        skeletal_muscle_mass="22,4",
        dominant_leg_lean_mass="6,62",
        dominant_leg_fat_mass="2,1",
        phase_angle="5,2",
        time_lactate=(
            ("3:30", "1,91"),
            ("6:30", "2,21"),
            ("9:40", "3,67"),
            ("12:15", "5,54"),
            ("14:40", "6,34"),
            ("17:00", "9,56"),
            ("19:40", "9,25"),
        ),
    ),
}

SUMMARY_HEADER = [
    "ФИ",
    "Рост",
    "Возраст",
    "Пол",
    "% содержания жира в теле",
    "Масса скелетной мускулатуры",
    "Тощая масса ведущей ноги",
    "Жировая масса ведущей ноги",
    "Фазовый угол",
    "Обхват ведущей ноги",
    "Время остановки",
]


def normalize_name(name: str) -> str:
    """Приводит имя папки к единой Unicode-нормализации."""

    return unicodedata.normalize("NFC", name)


def build_rows(subject: SubjectData) -> list[list[str]]:
    """Собирает строки итогового CSV в согласованном формате."""

    rows = [
        SUMMARY_HEADER,
        [
            subject.full_name,
            subject.height,
            subject.age,
            subject.sex,
            subject.body_fat_percent,
            subject.skeletal_muscle_mass,
            subject.dominant_leg_lean_mass,
            subject.dominant_leg_fat_mass,
            subject.phase_angle,
            "",
            "",
        ],
        [""] * len(SUMMARY_HEADER),
        [""] * len(SUMMARY_HEADER),
        ["Время", "Лактат", *[""] * (len(SUMMARY_HEADER) - 2)],
    ]
    for time_value, lactate in subject.time_lactate:
        rows.append([time_value, lactate, *[""] * (len(SUMMARY_HEADER) - 2)])
    return rows


def write_subject_csv(folder: Path, subject: SubjectData) -> None:
    """Записывает summary.csv в папку участника."""

    output_path = folder / "summary.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerows(build_rows(subject))


def main() -> None:
    """Проверяет состав папок и создает итоговые CSV-файлы."""

    data_dir = Path(__file__).resolve().parent.parent / "data"
    actual_folders = sorted(path for path in data_dir.iterdir() if path.is_dir())
    actual_keys = {normalize_name(folder.name) for folder in actual_folders}
    expected_keys = set(SUBJECTS)

    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise SystemExit(
            "Несовпадение набора папок и словаря данных: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for folder in actual_folders:
        subject = SUBJECTS[normalize_name(folder.name)]
        write_subject_csv(folder, subject)
        print(folder / "summary.csv")


if __name__ == "__main__":
    main()
