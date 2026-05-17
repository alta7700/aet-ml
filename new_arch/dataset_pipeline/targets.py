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


def build_targets_table(
    subjects_path: Path,
    windows_path: Path,
    lt1_labels_path: Path | None = None,
) -> pd.DataFrame:
    """Строит таблицу таргетов для окон.

    Если lt1_labels_path указан и существует — добавляет LT1-таргеты.
    """

    subjects = load_subjects_table(subjects_path)
    windows = load_windows_table(windows_path)

    # ── LT2-таргеты ──────────────────────────────────────────────────────────
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

    # ── LT1-таргеты: загружаем если файл есть ────────────────────────────────
    lt1_available = False
    lt1_df: pd.DataFrame | None = None
    if lt1_labels_path is not None and lt1_labels_path.exists():
        lt1_df = pd.read_parquet(lt1_labels_path)
        lt1_available = True
        merged = merged.merge(
            lt1_df[[
                "subject_id",
                "lt1_time_sec",
                "lt1_pchip_time_sec",
                "lt1_interval_start_sec",
                "lt1_interval_end_sec",
                "lt1_time_label_quality",
                "lt1_power_w",
                "lt1_available",
                "lt1_equals_lt2",
            ]],
            on="subject_id",
            how="left",
            validate="many_to_one",
        )

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        window_start_sec = float(row["window_start_sec"])
        window_end_sec = float(row["window_end_sec"])

        # ── LT2 ──
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

        entry: dict[str, object] = {
            "window_id": row["window_id"],
            "subject_id": row["subject_id"],
            # LT2-таргеты
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

        # ── LT1-таргеты ──────────────────────────────────────────────────────
        if lt1_available:
            subj_lt1_available = int(row.get("lt1_available", 0) or 0)
            lt1_time_sec = float(row["lt1_time_sec"]) if subj_lt1_available else np.nan
            lt1_quality = str(row.get("lt1_time_label_quality", "unavailable"))
            lt1_usable = lt1_quality in TIME_TARGET_QUALITIES and subj_lt1_available
            lt1_eq = int(row.get("lt1_equals_lt2", 0) or 0)

            lt1_pchip_time_sec = float(row["lt1_pchip_time_sec"]) if subj_lt1_available else np.nan

            if subj_lt1_available and np.isfinite(lt1_time_sec):
                time_to_lt1_sec: float | None = lt1_time_sec - window_end_sec
                time_to_lt1_pchip_sec: float | None = (
                    lt1_pchip_time_sec - window_end_sec
                    if np.isfinite(lt1_pchip_time_sec) else np.nan
                )
                lt1_interval_start = float(row["lt1_interval_start_sec"])
                lt1_interval_end = float(row["lt1_interval_end_sec"])
                lt1_binary: int | None
                if window_end_sec < lt1_interval_start:
                    lt1_binary = 0
                elif window_start_sec >= lt1_interval_end:
                    lt1_binary = 1
                else:
                    lt1_binary = None
                lt1_overlap = overlaps_interval(
                    window_start_sec=window_start_sec,
                    window_end_sec=window_end_sec,
                    interval_start_sec=lt1_interval_start,
                    interval_end_sec=lt1_interval_end,
                )
            else:
                time_to_lt1_sec = np.nan
                time_to_lt1_pchip_sec = np.nan
                lt1_binary = None
                lt1_overlap = None

            entry["target_time_to_lt1_sec"] = time_to_lt1_sec
            entry["target_time_to_lt1_pchip_sec"] = time_to_lt1_pchip_sec
            entry["target_time_to_lt1_usable"] = int(lt1_usable)
            entry["target_lt1_binary_label"] = lt1_binary
            entry["target_lt1_binary_valid"] = int(lt1_binary is not None)
            entry["target_in_lt1_interval"] = lt1_overlap
            entry["lt1_time_label_quality"] = lt1_quality
            entry["lt1_equals_lt2"] = lt1_eq
            entry["lt1_power_w"] = float(row["lt1_power_w"]) if subj_lt1_available else np.nan

        rows.append(entry)

    data_frame = pd.DataFrame(rows)
    if not data_frame.empty:
        data_frame["target_binary_label"] = data_frame["target_binary_label"].astype("Int8")
        data_frame["target_in_refined_lt2_window"] = data_frame["target_in_refined_lt2_window"].astype("boolean")
        if lt1_available:
            data_frame["target_lt1_binary_label"] = data_frame["target_lt1_binary_label"].astype("Int8")
            data_frame["target_in_lt1_interval"] = data_frame["target_in_lt1_interval"].astype("boolean")
        data_frame = data_frame.sort_values(["subject_id", "window_id"]).reset_index(drop=True)
    return data_frame
