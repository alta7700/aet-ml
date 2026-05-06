"""Сборка таблицы таргетов по окнам."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataset_pipeline.common import load_subjects_table
from dataset_pipeline.common import load_windows_table


TIME_TARGET_QUALITIES = {"high", "medium"}


def overlaps_interval(
    window_start_sec: float,
    window_end_sec: float,
    interval_start_sec: float,
    interval_end_sec: float,
) -> bool:
    """Проверяет, пересекается ли окно с интервалом."""

    return window_start_sec < interval_end_sec and window_end_sec > interval_start_sec


def build_targets_table(subjects_path: Path, windows_path: Path) -> pd.DataFrame:
    """Строит таблицу таргетов для окон."""

    subjects = load_subjects_table(subjects_path)
    windows = load_windows_table(windows_path)

    subject_fields = [
        "subject_id",
        "lt2_time_center_sec",
        "lt2_refined_time_sec",
        "lt2_interval_start_sec",
        "lt2_interval_end_sec",
        "lt2_refined_valid",
        "lt2_refined_window_start_sec",
        "lt2_refined_window_end_sec",
        "lt2_time_label_quality",
    ]
    merged = windows.merge(subjects[subject_fields], on="subject_id", how="left", validate="many_to_one")

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        window_start_sec = float(row["window_start_sec"])
        window_end_sec = float(row["window_end_sec"])
        coarse_start_sec = float(row["lt2_interval_start_sec"])
        coarse_end_sec = float(row["lt2_interval_end_sec"])

        refined_time_sec = float(row["lt2_refined_time_sec"])
        refined_valid = bool(int(row["lt2_refined_valid"]))
        refined_quality = str(row["lt2_time_label_quality"])
        refined_usable = refined_quality in TIME_TARGET_QUALITIES

        target_binary_label: int | None
        if window_end_sec < coarse_start_sec:
            target_binary_label = 0
        elif window_start_sec >= coarse_end_sec:
            target_binary_label = 1
        else:
            target_binary_label = None

        coarse_overlap = overlaps_interval(
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            interval_start_sec=coarse_start_sec,
            interval_end_sec=coarse_end_sec,
        )

        if refined_valid and pd.notna(row["lt2_refined_window_start_sec"]) and pd.notna(row["lt2_refined_window_end_sec"]):
            refined_start_sec = float(row["lt2_refined_window_start_sec"])
            refined_end_sec = float(row["lt2_refined_window_end_sec"])
            refined_overlap: bool | None = overlaps_interval(
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
                interval_start_sec=refined_start_sec,
                interval_end_sec=refined_end_sec,
            )
        else:
            refined_start_sec = np.nan
            refined_end_sec = np.nan
            refined_overlap = None

        rows.append(
            {
                "window_id": row["window_id"],
                "subject_id": row["subject_id"],
                "target_time_to_lt2_center_sec": float(row["lt2_time_center_sec"]) - window_end_sec,
                "target_time_to_lt2_refined_sec": refined_time_sec - window_end_sec,
                "target_time_to_lt2_refined_usable": int(refined_usable),
                "target_binary_label": target_binary_label,
                "target_binary_valid": int(target_binary_label is not None),
                "target_in_coarse_lt2_interval": int(coarse_overlap),
                "target_refined_window_valid": int(refined_valid),
                "target_in_refined_lt2_window": refined_overlap,
                "target_refined_window_start_sec": refined_start_sec,
                "target_refined_window_end_sec": refined_end_sec,
            }
        )

    data_frame = pd.DataFrame(rows)
    if not data_frame.empty:
        data_frame["target_binary_label"] = data_frame["target_binary_label"].astype("Int8")
        data_frame["target_in_refined_lt2_window"] = data_frame["target_in_refined_lt2_window"].astype("boolean")
        data_frame = data_frame.sort_values(["subject_id", "window_id"]).reset_index(drop=True)
    return data_frame

