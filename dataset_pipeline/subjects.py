"""Сборка таблицы испытуемых из `finaltest.h5`."""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

from dataset_pipeline.common import LT2_SUBJECT_FIELDS
from dataset_pipeline.common import SUMMARY_FIELDS
from dataset_pipeline.common import SubjectFile
from dataset_pipeline.common import list_subject_files
from dataset_pipeline.common import read_optional_attr
from dataset_pipeline.common import read_required_attr

SUMMARY_NUMERIC_FIELDS = {
    "age",
    "height",
    "weight",
    "body_fat_mass",
    "skeletal_muscle_mass",
    "dominant_leg_lean_mass",
    "dominant_leg_fat_mass",
    "phase_angle",
    "dominant_leg_circumference",
}

SUMMARY_STRING_FIELDS = {
    "sex",
}


def normalize_missing_marker(value: object) -> object | None:
    """Сводит типовые текстовые маркеры пропуска к `None`."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped in {"-", "—", "nan", "NaN", "None"}:
            return None
        return stripped
    return value


def parse_summary_value(field_name: str, raw_value: object) -> object | None:
    """Приводит summary-значение к целевому типу."""

    value = normalize_missing_marker(raw_value)
    if value is None:
        return None

    if field_name in SUMMARY_STRING_FIELDS:
        return str(value)

    if field_name in SUMMARY_NUMERIC_FIELDS:
        if isinstance(value, (int, float)):
            if field_name == "age":
                return int(value)
            return float(value)

        text = str(value).replace(",", ".")
        number = float(text)
        if field_name == "age":
            return int(round(number))
        return number

    return value


def build_subject_row(subject_file: SubjectFile) -> dict[str, object]:
    """Собирает одну строку таблицы испытуемых."""

    row: dict[str, object] = {
        "subject_dir_name": subject_file.subject_dir.name,
        "source_h5_path": str(subject_file.finaltest_path.resolve()),
    }

    with h5py.File(subject_file.finaltest_path, "r") as handle:
        row["subject_id"] = read_required_attr(handle, "subject_id")
        row["subject_name"] = read_optional_attr(handle, "subject_name") or subject_file.subject_dir.name
        row["stop_time"] = read_optional_attr(handle, "stop_time")
        row["stop_time_sec"] = read_optional_attr(handle, "stop_time_sec")

        for field_name in SUMMARY_FIELDS:
            value = parse_summary_value(field_name, read_optional_attr(handle, field_name))
            row[field_name] = value
            row[f"{field_name}_is_missing"] = int(value is None)

        for field_name in LT2_SUBJECT_FIELDS:
            row[field_name] = read_optional_attr(handle, field_name)

    return row


def build_subjects_table(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Строит таблицу испытуемых и возвращает также список пропущенных папок."""

    subject_files, skipped = list_subject_files(data_dir)
    rows = [build_subject_row(subject_file) for subject_file in subject_files]
    data_frame = pd.DataFrame(rows)
    if not data_frame.empty:
        data_frame = data_frame.sort_values(["subject_id", "subject_dir_name"]).reset_index(drop=True)
    return data_frame, skipped
