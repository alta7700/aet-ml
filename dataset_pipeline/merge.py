"""Финальная сборка датасета ML: объединение всех таблиц признаков.

Выполняет LEFT JOIN цепочку:
  windows ← targets ← features_emg_kinematics ← features_nirs ← features_hrv ← qc_windows

Выходные файлы:
- merged_features_ml.parquet  — полный датасет для классического ML
- sequence_index.parquet       — lookup-таблица для нейросетей (DataLoader)

Генерирует также dataset/qc_summary.md — текстовый отчёт о качестве датасета.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_pipeline.common import DEFAULT_DATASET_DIR, save_parquet

# ─────────────────────── Выбор колонок ───────────────────────

# Колонки из windows, которые войдут в merged (без дублирующих метаданных)
_WINDOWS_COLS = [
    "window_id",
    "subject_id",
    "subject_name",
    "source_h5_path",
    "window_start_sec",
    "window_end_sec",
    "window_center_sec",
    "window_duration_sec",
    "current_power_w",
    "window_power_mode_w",
    "stage_index",
    "elapsed_sec",
    "is_work_phase",
]

# Все колонки targets
_TARGET_PREFIX = "target_"

# Служебные QC-колонки из qc_windows
_QC_AGG_COLS = ["window_valid_any", "window_valid_all_required"]
_QC_PER_MODALITY_COLS = [
    "emg_valid",
    "kinematics_valid",
    "nirs_valid",
    "hrv_valid",
    "emg_coverage_fraction",
    "kinematics_coverage_fraction",
    "nirs_coverage_fraction",
    "hrv_coverage_fraction",
    "hrv_artifact_fraction",
    "cycles_count",
    "hrv_rr_count",
]

# Ключи JOIN
_KEY_COLS = ["window_id", "subject_id"]


# ─────────────────────── Слияние ───────────────────────

def build_merged_table(
    windows_path: Path,
    targets_path: Path,
    emg_path: Path,
    nirs_path: Path,
    hrv_path: Path,
    qc_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Объединяет все таблицы в единый датасет ML.

    Параметры
    ----------
    windows_path, targets_path, emg_path, nirs_path, hrv_path, qc_path : Path
        Пути к промежуточным parquet-файлам.

    Возвращает
    ----------
    Кортеж (merged_df, sequence_index_df).
    """
    # Загружаем таблицы
    windows_df = pd.read_parquet(windows_path)
    targets_df = pd.read_parquet(targets_path)
    emg_df = pd.read_parquet(emg_path)
    nirs_df = pd.read_parquet(nirs_path)
    hrv_df = pd.read_parquet(hrv_path)
    qc_df = pd.read_parquet(qc_path)

    # ── Базовая таблица: windows ──
    available_windows_cols = [c for c in _WINDOWS_COLS if c in windows_df.columns]
    merged = windows_df[available_windows_cols].copy()

    # ── Таргеты ──
    # Берём колонки с префиксом target_, а также lt1_* (метаданные LT1-метки)
    target_cols = [
        c for c in targets_df.columns
        if c in _KEY_COLS
        or c.startswith(_TARGET_PREFIX)
        or c.startswith("lt1_")
    ]
    merged = merged.merge(
        targets_df[target_cols],
        on=_KEY_COLS,
        how="left",
        validate="many_to_one",
    )

    # ── Признаки EMG/кинематики ──
    emg_feature_cols = [c for c in emg_df.columns if c not in _KEY_COLS]
    # Исключаем QC-поля из EMG (они придут из qc_windows)
    emg_qc_set = {
        "emg_valid", "kinematics_valid",
        "emg_coverage_fraction", "kinematics_coverage_fraction", "cycles_count",
    }
    emg_pure_cols = [c for c in emg_feature_cols if c not in emg_qc_set]
    merged = merged.merge(
        emg_df[_KEY_COLS + emg_pure_cols],
        on=_KEY_COLS,
        how="left",
        validate="many_to_one",
    )

    # ── Признаки NIRS ──
    nirs_feature_cols = [c for c in nirs_df.columns if c not in _KEY_COLS]
    nirs_qc_set = {"nirs_valid", "nirs_coverage_fraction"}
    nirs_pure_cols = [c for c in nirs_feature_cols if c not in nirs_qc_set]
    merged = merged.merge(
        nirs_df[_KEY_COLS + nirs_pure_cols],
        on=_KEY_COLS,
        how="left",
        validate="many_to_one",
    )

    # ── Признаки HRV ──
    hrv_feature_cols = [c for c in hrv_df.columns if c not in _KEY_COLS]
    hrv_qc_set = {"hrv_valid", "hrv_coverage_fraction", "hrv_artifact_fraction", "hrv_rr_count"}
    hrv_pure_cols = [c for c in hrv_feature_cols if c not in hrv_qc_set]
    merged = merged.merge(
        hrv_df[_KEY_COLS + hrv_pure_cols],
        on=_KEY_COLS,
        how="left",
        validate="many_to_one",
    )

    # ── QC-флаги (включая агрегированные) ──
    available_qc_cols = [
        c for c in _QC_AGG_COLS + _QC_PER_MODALITY_COLS if c in qc_df.columns
    ]
    merged = merged.merge(
        qc_df[_KEY_COLS + available_qc_cols],
        on=_KEY_COLS,
        how="left",
        validate="many_to_one",
    )

    # Заполняем NaN в valid-флагах нулями
    for col in ("emg_valid", "kinematics_valid", "nirs_valid", "hrv_valid",
                "window_valid_any", "window_valid_all_required"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    merged = merged.sort_values(_KEY_COLS).reset_index(drop=True)

    # ── sequence_index: lookup-таблица для DataLoader нейросети ──
    seq_cols = [
        "window_id", "subject_id", "source_h5_path",
        "window_start_sec", "window_end_sec",
        "emg_valid", "nirs_valid", "hrv_valid", "kinematics_valid",
    ]
    available_seq_cols = [c for c in seq_cols if c in merged.columns]
    sequence_index = merged[available_seq_cols].copy()

    return merged, sequence_index


# ─────────────────────── QC-отчёт ───────────────────────

def generate_qc_summary(
    merged: pd.DataFrame,
    output_path: Path,
    subjects_path: Path | None = None,
) -> None:
    """Генерирует Markdown-отчёт о качестве датасета.

    Параметры
    ----------
    merged : pd.DataFrame
        Объединённый датасет (merged_features_ml).
    output_path : Path
        Куда записать qc_summary.md.
    subjects_path : Path | None
        Путь к subjects.parquet для получения breakdown по LT2 quality.
        Если None — секция LT2 quality будет пропущена.
    """
    lines: list[str] = []
    today = date.today().isoformat()

    lines.append(f"# Dataset QC Summary — {today}")
    lines.append("")

    # ── Объём ──
    lines.append("## Объём")
    n_windows = len(merged)
    n_subjects = merged["subject_id"].nunique() if "subject_id" in merged.columns else "?"
    per_subject = merged.groupby("subject_id").size() if "subject_id" in merged.columns else pd.Series(dtype=int)
    if len(per_subject) > 0:
        lines.append(f"- Участников: {n_subjects}")
        lines.append(f"- Окон всего: {n_windows}")
        lines.append(
            f"- Окон на участника: "
            f"min={int(per_subject.min())} / "
            f"median={int(per_subject.median())} / "
            f"max={int(per_subject.max())}"
        )
    else:
        lines.append(f"- Участников: {n_subjects}")
        lines.append(f"- Окон всего: {n_windows}")
    lines.append("")

    # ── Валидность по модальностям ──
    lines.append("## Валидность по модальностям")
    lines.append("")
    lines.append("| Модальность | Валидных окон | % от всех |")
    lines.append("|---|---|---|")
    for col, label in [
        ("emg_valid", "EMG"),
        ("nirs_valid", "NIRS"),
        ("hrv_valid", "HRV"),
        ("window_valid_all_required", "Все три"),
    ]:
        if col in merged.columns:
            n = int(merged[col].sum())
            pct = 100.0 * n / n_windows if n_windows > 0 else 0.0
            lines.append(f"| {label} | {n} | {pct:.1f}% |")
    lines.append("")

    # ── Таргеты ──
    lines.append("## Таргеты")
    if "target_binary_valid" in merged.columns:
        n_bin_valid = int(merged["target_binary_valid"].sum())
        pct_bin = 100.0 * n_bin_valid / n_windows if n_windows > 0 else 0.0
        lines.append(f"- Бинарно валидных окон: {n_bin_valid} ({pct_bin:.1f}%)")
    if "target_refined_window_valid" in merged.columns:
        n_ref = int(merged["target_refined_window_valid"].sum())
        pct_ref = 100.0 * n_ref / n_windows if n_windows > 0 else 0.0
        lines.append(f"- Refined-usable окон: {n_ref} ({pct_ref:.1f}%)")

    # LT2 power quality breakdown из subjects.parquet (на уровне участников)
    if subjects_path is not None and subjects_path.exists():
        try:
            subjects_df = pd.read_parquet(subjects_path)
            if "lt2_power_label_quality" in subjects_df.columns:
                quality_counts = subjects_df["lt2_power_label_quality"].value_counts()
                high = int(quality_counts.get("high", 0))
                medium = int(quality_counts.get("medium", 0))
                low = int(quality_counts.get("low", 0))
                lines.append(
                    f"- LT2 power quality (участников): "
                    f"high={high}, medium={medium}, low={low}"
                )
            if "lt2_time_label_quality" in subjects_df.columns:
                tq = subjects_df["lt2_time_label_quality"].value_counts()
                high_t = int(tq.get("high", 0))
                medium_t = int(tq.get("medium", 0))
                low_t = int(tq.get("low", 0))
                lines.append(
                    f"- LT2 time quality (участников): "
                    f"high={high_t}, medium={medium_t}, low={low_t}"
                )
        except Exception:
            pass
    lines.append("")

    # ── Диапазоны ключевых признаков ──
    lines.append("## Диапазоны ключевых признаков (валидные окна)")
    lines.append("")
    key_feature_info = [
        ("vl_dist_load_rms", "EMG", "emg_valid"),
        ("trainred_smo2_mean", "NIRS", "nirs_valid"),
        ("hrv_dfa_alpha1", "HRV", "hrv_valid"),
        ("cadence_mean_rpm", "Кинематика", "kinematics_valid"),
        ("hrv_mean_rr_ms", "HRV", "hrv_valid"),
    ]
    lines.append("| Признак | n | min | p25 | median | p75 | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for feat_col, _modality, valid_col in key_feature_info:
        if feat_col not in merged.columns:
            continue
        valid_flag = merged.get(valid_col, pd.Series(1, index=merged.index))
        subset = merged.loc[valid_flag == 1, feat_col].dropna()
        if len(subset) < 2:
            continue
        lines.append(
            f"| {feat_col} | {len(subset)} "
            f"| {subset.min():.3f} "
            f"| {subset.quantile(0.25):.3f} "
            f"| {subset.median():.3f} "
            f"| {subset.quantile(0.75):.3f} "
            f"| {subset.max():.3f} |"
        )
    lines.append("")

    # ── Предупреждения ──
    lines.append("## Предупреждения")
    warnings: list[str] = []

    # NaN > 50% в любом признаке
    feature_cols = [
        c for c in merged.columns
        if c.startswith(("vl_", "trainred_", "hrv_", "cadence_", "load_", "rest_"))
        and not c.endswith(("_valid", "_fraction", "_count"))
    ]
    for col in feature_cols:
        nan_frac = float(merged[col].isna().mean())
        if nan_frac > 0.5:
            warnings.append(f"- NaN > 50% в колонке `{col}` ({100 * nan_frac:.0f}%)")

    # Нулевое покрытие какой-либо модальности
    for col in ("emg_valid", "nirs_valid", "hrv_valid"):
        if col in merged.columns and merged[col].sum() == 0:
            warnings.append(f"- Все окна невалидны по модальности `{col}`!")

    # DFA-α1 вне физиологического диапазона
    if "hrv_dfa_alpha1" in merged.columns:
        dfa_valid = merged.loc[merged.get("hrv_valid", pd.Series(1, index=merged.index)) == 1, "hrv_dfa_alpha1"].dropna()
        if len(dfa_valid) > 0:
            out_of_range = ((dfa_valid < -0.5) | (dfa_valid > 2.0)).sum()
            if out_of_range > 0:
                warnings.append(
                    f"- DFA-α1 вне [-0.5, 2.0] у {out_of_range} валидных окон!"
                )

    if warnings:
        lines.extend(warnings)
    else:
        lines.append("_Предупреждений нет._")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Записан QC-отчёт: {output_path}")
