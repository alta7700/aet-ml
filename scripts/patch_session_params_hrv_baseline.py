"""Одноразовый патч: добавляет hrv_hr_baseline_bpm и hrv_rmssd_baseline_ms
в существующий dataset/session_params.parquet.

Не пересобирает features_emg_kinematics — только вызывает detect_phases
для получения границ стадии 30 Вт и compute_hrv_baseline для каждого субъекта.

Запуск:
    uv run python scripts/patch_session_params_hrv_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset_pipeline.common import DEFAULT_DATASET_DIR, load_subjects_table, save_parquet
from dataset_pipeline.hrv import compute_hrv_baseline
from methods.pedal_cycles import (
    DEFAULT_CALIBRATION_MARGIN_SEC,
    DEFAULT_CALIBRATION_TAIL_SEC,
    DEFAULT_MAX_CADENCE_RPM,
    DEFAULT_MIN_CADENCE_RPM,
    DEFAULT_PEAK_PROMINENCE_STD,
    detect_phases,
)


def main() -> None:
    dataset_dir = DEFAULT_DATASET_DIR
    subjects_path = dataset_dir / "subjects.parquet"
    session_params_path = dataset_dir / "session_params.parquet"

    subjects = load_subjects_table(subjects_path)
    session_params = pd.read_parquet(session_params_path)
    print(f"Загружено session_params: {session_params.shape}")
    print(f"Колонки: {list(session_params.columns)}")

    hr_values: dict[str, float] = {}
    rmssd_values: dict[str, float] = {}

    for subject_row in subjects.itertuples():
        subject_id = str(subject_row.subject_id)
        source_h5_path = Path(str(subject_row.source_h5_path))

        print(f"  {subject_id}: detect_phases...", flush=True)
        try:
            result = detect_phases(
                participant_name=subject_id,
                source_h5_path=source_h5_path,
                sensor_mode="auto",
                calibration_tail_sec=DEFAULT_CALIBRATION_TAIL_SEC,
                calibration_margin_sec=DEFAULT_CALIBRATION_MARGIN_SEC,
                min_cadence_rpm=DEFAULT_MIN_CADENCE_RPM,
                max_cadence_rpm=DEFAULT_MAX_CADENCE_RPM,
                peak_prominence_std=DEFAULT_PEAK_PROMINENCE_STD,
            )
            hr_bpm, rmssd_ms = compute_hrv_baseline(
                source_h5_path=source_h5_path,
                baseline_start_sec=result.normalization_stage.start_sec,
                baseline_end_sec=result.normalization_stage.end_sec,
            )
            print(f"    HR={hr_bpm:.1f} bpm, RMSSD={rmssd_ms:.1f} ms")
        except Exception as exc:
            print(f"    ОШИБКА: {exc}")
            hr_bpm = float("nan")
            rmssd_ms = float("nan")

        hr_values[subject_id] = hr_bpm
        rmssd_values[subject_id] = rmssd_ms

    session_params["hrv_hr_baseline_bpm"] = session_params["subject_id"].map(hr_values)
    session_params["hrv_rmssd_baseline_ms"] = session_params["subject_id"].map(rmssd_values)

    # Сохраняем с сохранением порядка колонок: добавляем перед emg_sample_rate_hz
    cols = list(session_params.columns)
    new_cols = ["hrv_hr_baseline_bpm", "hrv_rmssd_baseline_ms"]
    base = [c for c in cols if c not in new_cols and c != "emg_sample_rate_hz"]
    ordered = base + new_cols + (["emg_sample_rate_hz"] if "emg_sample_rate_hz" in cols else [])
    session_params = session_params[ordered]

    save_parquet(session_params, session_params_path, force=True)
    print(f"\nСохранено: {session_params.shape}")
    print(session_params[["subject_id", "hrv_hr_baseline_bpm", "hrv_rmssd_baseline_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
