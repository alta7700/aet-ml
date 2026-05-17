"""Сборка таблицы испытуемых из `finaltest.h5`."""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

from new_arch.dataset_pipeline.common import LT2_SUBJECT_FIELDS
from new_arch.dataset_pipeline.common import SUMMARY_FIELDS
from new_arch.dataset_pipeline.common import SubjectFile
from new_arch.dataset_pipeline.common import list_subject_files
from new_arch.dataset_pipeline.common import read_optional_attr
from new_arch.dataset_pipeline.common import read_required_attr

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
        "source_h5_path": str(subject_file.finaltest_path.resolve()),
    }

    with h5py.File(subject_file.finaltest_path, "r") as handle:
        row["subject_id"] = read_required_attr(handle, "subject_id")
        row["stop_time"] = read_optional_attr(handle, "stop_time")
        row["stop_time_sec"] = read_optional_attr(handle, "stop_time_sec")

        for field_name in SUMMARY_FIELDS:
            value = parse_summary_value(field_name, read_optional_attr(handle, field_name))
            row[field_name] = value
            row[f"{field_name}_is_missing"] = int(value is None)

        for field_name in LT2_SUBJECT_FIELDS:
            row[field_name] = read_optional_attr(handle, field_name)

    return row


# Коэффициенты импутации: вычислены по 12/14 субъектов с полными данными.
# leg_lean / skeletal_muscle_mass = 0.271 (CV 6%)
# leg_fat  / body_fat_mass        = 0.157 (CV 10%)
_LEG_LEAN_FRAC = 0.271
_LEG_FAT_FRAC  = 0.157


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные антропометрические индексы.

    Импутация leg_lean / leg_fat: если значение отсутствует,
    восстанавливается из общих параметров тела по стабильным пропорциям
    (_LEG_LEAN_FRAC, _LEG_FAT_FRAC), вычисленным по имеющимся данным.
    """
    df = df.copy()

    # ── Импутация ноги ────────────────────────────────────────────────────────
    for col, total_col, frac in [
        ("dominant_leg_lean_mass", "skeletal_muscle_mass", _LEG_LEAN_FRAC),
        ("dominant_leg_fat_mass",  "body_fat_mass",        _LEG_FAT_FRAC),
    ]:
        imputed_col = f"{col}_imputed"
        df[imputed_col] = False
        missing = df[col].isna() & df[total_col].notna()
        df.loc[missing, col] = df.loc[missing, total_col] * frac
        df.loc[missing, imputed_col] = True

    # ── Производные индексы ───────────────────────────────────────────────────
    h = df["height"].div(100)                          # см → м
    w = df["weight"]
    bf = df["body_fat_mass"]
    sm = df["skeletal_muscle_mass"]
    ll = df["dominant_leg_lean_mass"]
    lf = df["dominant_leg_fat_mass"]

    df["bmi"]                = (w / h**2).round(1)
    df["body_fat_pct"]       = (bf / w * 100).round(1)
    df["muscle_to_fat_total"] = (sm / bf).round(2)
    df["leg_fat_pct"]        = (lf / (ll + lf) * 100).round(1)   # ковариат NIRS-аттенюации
    df["muscle_to_fat_leg"]  = (ll / lf).round(2)

    return df


def build_subjects_table(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Строит таблицу испытуемых и возвращает также список пропущенных папок."""

    subject_files, skipped = list_subject_files(data_dir)
    rows = [build_subject_row(subject_file) for subject_file in subject_files]
    data_frame = pd.DataFrame(rows)
    if not data_frame.empty:
        data_frame = data_frame.sort_values("subject_id").reset_index(drop=True)
        data_frame = _add_derived_columns(data_frame)
    return data_frame, skipped
