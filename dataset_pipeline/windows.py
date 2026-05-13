"""Сборка таблицы каузальных окон из `finaltest.h5`."""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

from dataset_pipeline.common import DEFAULT_WINDOW_DURATION_SEC
from dataset_pipeline.common import DEFAULT_WINDOW_STEP_SEC
from dataset_pipeline.common import build_window_starts
from dataset_pipeline.common import build_work_stage_intervals
from dataset_pipeline.common import compute_mode_power_for_window
from dataset_pipeline.common import find_stage_for_time
from dataset_pipeline.common import load_channel_relative
from dataset_pipeline.common import load_subjects_table


def build_windows_for_subject(
    subject_row: pd.Series,
    window_duration_sec: float,
    step_sec: float,
) -> list[dict[str, object]]:
    """Строит окна для одного испытуемого."""

    finaltest_path = Path(str(subject_row["source_h5_path"]))
    rows: list[dict[str, object]] = []

    with h5py.File(finaltest_path, "r") as handle:
        power_times_sec, power_values_w = load_channel_relative(handle, "power.label")
        stop_time_sec = float(subject_row["stop_time_sec"])
        work_stages = build_work_stage_intervals(
            power_times_sec=power_times_sec,
            power_values_w=power_values_w,
            stop_time_sec=stop_time_sec,
        )

    work_start_sec = float(work_stages[0].start_sec)
    window_starts = build_window_starts(
        work_start_sec=work_start_sec,
        stop_time_sec=stop_time_sec,
        window_duration_sec=window_duration_sec,
        step_sec=step_sec,
    )

    for local_index, window_start_sec in enumerate(window_starts):
        window_end_sec = float(window_start_sec + window_duration_sec)
        window_center_sec = float(window_start_sec + window_duration_sec / 2.0)
        current_stage = find_stage_for_time(work_stages, window_end_sec)
        window_power_mode_w = compute_mode_power_for_window(
            stage_intervals=work_stages,
            window_start_sec=float(window_start_sec),
            window_end_sec=window_end_sec,
        )

        window_id = f"{subject_row['subject_id']}_w{local_index:04d}"
        rows.append(
            {
                "window_id": window_id,
                "subject_id": subject_row["subject_id"],
                "source_h5_path": str(finaltest_path.resolve()),
                "window_start_sec": float(window_start_sec),
                "window_end_sec": window_end_sec,
                "window_center_sec": window_center_sec,
                "window_duration_sec": float(window_duration_sec),
                "current_power_w": float(current_stage.power_w),
                "window_power_mode_w": float(window_power_mode_w),
                "stage_index": int(current_stage.stage_index),
                # elapsed_sec = время от начала рабочей части теста до правой
                # границы окна. Не зависит от длины baseline (30 Вт).
                "elapsed_sec": window_end_sec - work_start_sec,
                "is_work_phase": 1,
            }
        )

    return rows


def build_windows_table(
    subjects_path: Path,
    window_duration_sec: float = DEFAULT_WINDOW_DURATION_SEC,
    step_sec: float = DEFAULT_WINDOW_STEP_SEC,
) -> pd.DataFrame:
    """Строит таблицу каузальных окон по всем испытуемым."""

    subjects = load_subjects_table(subjects_path)
    rows: list[dict[str, object]] = []
    for _, subject_row in subjects.iterrows():
        rows.extend(
            build_windows_for_subject(
                subject_row=subject_row,
                window_duration_sec=window_duration_sec,
                step_sec=step_sec,
            )
        )

    data_frame = pd.DataFrame(rows)
    if not data_frame.empty:
        data_frame = data_frame.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    return data_frame

