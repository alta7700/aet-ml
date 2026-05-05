#!/usr/bin/env python3
"""Строит графики h5 и train.red с найденным окном наилучшего совпадения."""

from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider

from create_test_plus_red_h5 import DEFAULT_OUTPUT_FILENAME, create_test_plus_red_h5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.trainred_alignment import (
    ALIGNMENT_FEATURES,
    classify_match,
    load_h5_smo2,
    load_trainred,
    resample_signal,
    zscore,
)

DEFAULT_POINTS_FILENAME = "trainred_alignment_points.json"
MANUAL_SELECTION_COMMENT = "Сохранено вручную через slider."


@dataclass(frozen=True)
class MatchWindow:
    """Описывает лучшее окно совпадения между h5 и train.red."""

    best_feature: str
    correlation: float
    sign: int
    status: str
    train_start_sec: float
    train_end_sec: float
    train_match_start_sec: float
    train_match_end_sec: float
    h5_start_sec: float
    h5_end_sec: float
    train_is_reference: bool
    gap_count_in_window: int
    max_gap_in_window_sec: float
    note: str


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Ищет лучшее соответствие между тест.h5 и train.red.csv "
            "для указанной папки и строит два графика."
        )
    )
    parser.add_argument(
        "subject",
        help="Точное имя папки внутри data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Каталог с папками участников.",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=1.0,
        help="Шаг ресемплинга в секундах для подбора окна.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Путь для сохранения PNG. Если не указан, график будет показан на экране.",
    )
    parser.add_argument(
        "--points-output",
        type=Path,
        help=(
            "Куда сохранить JSON с точками выравнивания. "
            "По умолчанию файл пишется в папку участника."
        ),
    )
    parser.add_argument(
        "--no-save-points",
        action="store_true",
        help="Не сохранять JSON с точками выравнивания.",
    )
    parser.add_argument(
        "--gap-threshold-sec",
        type=float,
        default=1.0,
        help=(
            "Порог длинной паузы в train.red. "
            "Паузы длиннее этого значения сжимаются при поиске окна."
        ),
    )
    return parser.parse_args()


def resolve_subject_dir(data_dir: Path, subject_name: str) -> Path:
    """Находит папку участника и подсказывает ближайшие совпадения при ошибке."""

    exact_path = data_dir / subject_name
    if exact_path.is_dir():
        return exact_path

    normalized_target = unicodedata.normalize("NFC", subject_name).casefold()
    candidates = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    for candidate in candidates:
        if unicodedata.normalize("NFC", candidate).casefold() == normalized_target:
            return data_dir / candidate

    suggestions = difflib.get_close_matches(subject_name, candidates, n=3, cutoff=0.4)
    suggestion_text = f" Ближайшие варианты: {', '.join(suggestions)}." if suggestions else ""
    raise FileNotFoundError(
        f"Папка '{subject_name}' не найдена в {data_dir}.{suggestion_text}"
    )


def choose_note(
    status: str,
    train_duration_sec: float,
    h5_duration_sec: float,
    sign: int,
    gap_count_in_window: int,
    max_gap_in_window_sec: float,
) -> str:
    """Формирует краткое пояснение по качеству совпадения."""

    notes: list[str] = []

    if train_duration_sec + 1e-6 < h5_duration_sec:
        notes.append("train.red короче h5, поэтому в окно попадает весь train.red")

    if sign < 0:
        notes.append("лучшее совпадение найдено по инвертированному признаку, нужна ручная проверка")

    if gap_count_in_window > 0:
        notes.append(
            "внутри окна есть разрывы train.red: "
            f"{gap_count_in_window} шт., максимум {max_gap_in_window_sec:.2f} с"
        )

    if status == "reliable":
        notes.append("форма сигнала совпадает хорошо")
    elif status == "usable":
        notes.append("совпадение рабочее, но границы лучше проверить вручную")
    elif status == "manual_review":
        notes.append("сходство умеренное, без визуальной проверки доверять нельзя")
    else:
        notes.append("устойчивого совпадения по форме сигнала не найдено")

    return "; ".join(notes)


def build_candidate(
    feature: str,
    correlation: float,
    sign: int,
    train_start_sec: float,
    train_end_sec: float,
    train_match_start_sec: float,
    train_match_end_sec: float,
    h5_start_sec: float,
    h5_end_sec: float,
    train_is_reference: bool,
    train_duration_sec: float,
    h5_duration_sec: float,
    gap_count_in_window: int,
    max_gap_in_window_sec: float,
) -> MatchWindow:
    """Собирает структуру результата из параметров найденного окна."""

    if gap_count_in_window > 0:
        # Если внутри окна есть реальные провалы, уверенность в «идеальном»
        # совпадении должна быть чуть ниже даже при высокой корреляции.
        correlation = correlation - min(0.08, 0.02 * gap_count_in_window)

    status = classify_match(correlation)
    note = choose_note(
        status,
        train_duration_sec,
        h5_duration_sec,
        sign,
        gap_count_in_window,
        max_gap_in_window_sec,
    )
    return MatchWindow(
        best_feature=feature,
        correlation=correlation,
        sign=sign,
        status=status,
        train_start_sec=train_start_sec,
        train_end_sec=train_end_sec,
        train_match_start_sec=train_match_start_sec,
        train_match_end_sec=train_match_end_sec,
        h5_start_sec=h5_start_sec,
        h5_end_sec=h5_end_sec,
        train_is_reference=train_is_reference,
        gap_count_in_window=gap_count_in_window,
        max_gap_in_window_sec=max_gap_in_window_sec,
        note=note,
    )


def build_train_time_axes(
    train_times_original: np.ndarray,
    gap_threshold_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Строит две временные оси train.red: исходную и ось для поиска окна."""

    train_times_original = np.asarray(train_times_original, dtype=float)
    if len(train_times_original) < 2:
        return train_times_original, train_times_original.copy(), np.zeros_like(train_times_original, dtype=bool)

    dt = np.diff(train_times_original, prepend=train_times_original[0])
    positive_dt = dt[1:]
    nominal_candidates = positive_dt[(positive_dt > 0.0) & (positive_dt <= gap_threshold_sec)]
    if len(nominal_candidates) == 0:
        nominal_candidates = positive_dt[positive_dt > 0.0]

    nominal_step_sec = float(np.median(nominal_candidates)) if len(nominal_candidates) else 0.0
    gap_mask = dt > gap_threshold_sec

    # Для поиска окна длинные паузы сжимаем до типичного шага записи,
    # чтобы выпадение train.red не растягивало физиологический фрагмент.
    dt_for_match = dt.copy()
    dt_for_match[0] = 0.0
    if nominal_step_sec > 0.0:
        dt_for_match[gap_mask] = nominal_step_sec

    train_times_matching = np.cumsum(dt_for_match)
    return train_times_original, train_times_matching, gap_mask


def project_matching_window_to_original_axis(
    train_times_original: np.ndarray,
    train_times_matching: np.ndarray,
    gap_mask: np.ndarray,
    train_match_start_sec: float,
    train_match_end_sec: float,
) -> tuple[float, float, int, float]:
    """Переносит найденное окно из сжатой оси обратно в исходную шкалу train.red."""

    tolerance_sec = 1e-9
    point_mask = (
        (train_times_matching >= train_match_start_sec - tolerance_sec)
        & (train_times_matching <= train_match_end_sec + tolerance_sec)
    )

    if not point_mask.any():
        start_index = int(np.searchsorted(train_times_matching, train_match_start_sec, side="left"))
        end_index = int(np.searchsorted(train_times_matching, train_match_end_sec, side="right")) - 1
        start_index = max(0, min(start_index, len(train_times_original) - 1))
        end_index = max(start_index, min(end_index, len(train_times_original) - 1))
        point_mask[start_index : end_index + 1] = True

    gap_mask_in_window = gap_mask & point_mask
    gap_sizes = np.diff(train_times_original, prepend=train_times_original[0])
    gap_count = int(gap_mask_in_window.sum())
    max_gap = float(gap_sizes[gap_mask_in_window].max()) if gap_count else 0.0

    selected_original = train_times_original[point_mask]
    return (
        float(selected_original[0]),
        float(selected_original[-1]),
        gap_count,
        max_gap,
    )


def find_best_overlap(
    train_times_original: np.ndarray,
    train_frame: pd.DataFrame,
    h5_times: np.ndarray,
    h5_values: np.ndarray,
    step_sec: float,
    gap_threshold_sec: float,
) -> MatchWindow:
    """Ищет максимальное сходство формы, даже если один сигнал короче другого."""

    h5_grid, h5_resampled = resample_signal(h5_times, h5_values, step_sec)
    if len(h5_grid) < 2:
        raise ValueError("В h5 недостаточно точек для выравнивания.")

    train_times_original, train_times_matching, gap_mask = build_train_time_axes(
        train_times_original=train_times_original,
        gap_threshold_sec=gap_threshold_sec,
    )

    train_duration_sec = (
        float(train_times_original[-1] - train_times_original[0])
        if len(train_times_original)
        else 0.0
    )
    h5_duration_sec = float(h5_times[-1] - h5_times[0]) if len(h5_times) else 0.0

    best_match: MatchWindow | None = None

    for feature in ALIGNMENT_FEATURES:
        if feature not in train_frame.columns:
            continue

        train_grid, train_resampled = resample_signal(
            train_times_matching,
            train_frame[feature].to_numpy(dtype=float),
            step_sec,
        )
        if len(train_grid) < 2:
            continue

        # Если train.red длиннее, двигаем окно h5 по train.red.
        if len(train_resampled) >= len(h5_resampled):
            reference = train_resampled
            target_z = zscore(h5_resampled)
            window_len = len(h5_resampled)

            for start_index in range(0, len(reference) - window_len + 1):
                segment_z = zscore(reference[start_index : start_index + window_len])
                if np.isnan(segment_z).any():
                    continue

                corr = float(np.corrcoef(segment_z, target_z)[0, 1])
                corr_inv = float(np.corrcoef(-segment_z, target_z)[0, 1])
                if not np.isfinite(corr) or not np.isfinite(corr_inv):
                    continue

                if corr_inv > corr:
                    sign = -1
                    best_corr = corr_inv
                else:
                    sign = 1
                    best_corr = corr

                train_match_start_sec = float(train_grid[start_index])
                train_match_end_sec = float(train_grid[start_index] + (h5_grid[-1] - h5_grid[0]))
                train_start_sec, train_end_sec, gap_count, max_gap = project_matching_window_to_original_axis(
                    train_times_original=train_times_original,
                    train_times_matching=train_times_matching,
                    gap_mask=gap_mask,
                    train_match_start_sec=train_match_start_sec,
                    train_match_end_sec=train_match_end_sec,
                )

                candidate = build_candidate(
                    feature=feature,
                    correlation=best_corr,
                    sign=sign,
                    train_start_sec=train_start_sec,
                    train_end_sec=train_end_sec,
                    train_match_start_sec=train_match_start_sec,
                    train_match_end_sec=train_match_end_sec,
                    h5_start_sec=float(h5_grid[0]),
                    h5_end_sec=float(h5_grid[-1]),
                    train_is_reference=True,
                    train_duration_sec=train_duration_sec,
                    h5_duration_sec=h5_duration_sec,
                    gap_count_in_window=gap_count,
                    max_gap_in_window_sec=max_gap,
                )

                if best_match is None or candidate.correlation > best_match.correlation:
                    best_match = candidate

        # Если h5 длиннее, считаем, что весь train.red — это лучшее доступное окно на train.red.
        else:
            reference = h5_resampled
            target_z = zscore(train_resampled)
            window_len = len(train_resampled)

            for start_index in range(0, len(reference) - window_len + 1):
                segment_z = zscore(reference[start_index : start_index + window_len])
                if np.isnan(segment_z).any():
                    continue

                corr = float(np.corrcoef(segment_z, target_z)[0, 1])
                corr_inv = float(np.corrcoef(-segment_z, target_z)[0, 1])
                if not np.isfinite(corr) or not np.isfinite(corr_inv):
                    continue

                if corr_inv > corr:
                    sign = -1
                    best_corr = corr_inv
                else:
                    sign = 1
                    best_corr = corr

                train_start_sec = float(train_times_original[0])
                train_end_sec = float(train_times_original[-1])
                train_match_start_sec = float(train_times_matching[0])
                train_match_end_sec = float(train_times_matching[-1])
                gap_count = int(gap_mask.sum())
                gap_sizes = np.diff(train_times_original, prepend=train_times_original[0])
                max_gap = float(gap_sizes[gap_mask].max()) if gap_count else 0.0

                candidate = build_candidate(
                    feature=feature,
                    correlation=best_corr,
                    sign=sign,
                    train_start_sec=train_start_sec,
                    train_end_sec=train_end_sec,
                    train_match_start_sec=train_match_start_sec,
                    train_match_end_sec=train_match_end_sec,
                    h5_start_sec=float(h5_grid[start_index]),
                    h5_end_sec=float(h5_grid[start_index] + (train_grid[-1] - train_grid[0])),
                    train_is_reference=False,
                    train_duration_sec=train_duration_sec,
                    h5_duration_sec=h5_duration_sec,
                    gap_count_in_window=gap_count,
                    max_gap_in_window_sec=max_gap,
                )

                if best_match is None or candidate.correlation > best_match.correlation:
                    best_match = candidate

    if best_match is None:
        raise ValueError("Не удалось найти совпадение: недостаточно валидных точек в сигналах.")

    return best_match


def build_plot(
    subject_name: str,
    train_frame: pd.DataFrame,
    h5_times: np.ndarray,
    h5_values: np.ndarray,
    match: MatchWindow,
    interactive: bool,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Строит один общий график с наложением h5 SmO2 на train.red SmO2."""

    timestamp_col = "Timestamp (seconds passed)"

    train_plot = train_frame.copy()
    train_plot[timestamp_col] = train_plot[timestamp_col] - float(train_plot[timestamp_col].iloc[0])

    fig, ax_train = plt.subplots(figsize=(15, 9))
    if interactive:
        fig.subplots_adjust(bottom=0.23)
    else:
        fig.subplots_adjust(bottom=0.12)

    ax_hbdiff = ax_train.twinx()

    ax_train.plot(
        train_plot[timestamp_col],
        train_plot["SmO2"],
        color="#2ca02c",
        linewidth=1.0,
        label="train.red SmO2",
        zorder=2,
    )
    ax_hbdiff.plot(
        train_plot[timestamp_col],
        train_plot["HBDiff"],
        color="#d62728",
        linewidth=1.0,
        alpha=0.8,
        label="train.red HbDiff",
        zorder=1,
    )

    overlay_x, overlay_y = prepare_overlay_segment(
        train_start_sec=match.train_start_sec,
        train_end_sec=match.train_end_sec,
        h5_times=h5_times,
        h5_values=h5_values,
        h5_start_sec=match.h5_start_sec,
    )
    ax_train.plot(
        overlay_x,
        overlay_y,
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.95,
        label="h5 SmO2 (наложение)",
        zorder=3,
    )

    ax_train.axvspan(
        match.train_start_sec,
        match.train_end_sec,
        color="#9467bd",
        alpha=0.12,
    )
    for boundary, label in (
        (match.train_start_sec, "Старт окна"),
        (match.train_end_sec, "Конец окна"),
    ):
        ax_train.axvline(
            boundary,
            color="#9467bd",
            linestyle="--",
            linewidth=1.5,
            label=label,
        )

    ax_train.set_title(
        (
            "Совмещение h5 SmO2 и train.red SmO2 | "
            f"лучший признак: {match.best_feature} | "
            f"corr={match.correlation:.3f} | "
            f"статус={match.status}"
        ),
        fontsize=12,
    )
    ax_train.set_xlabel("Время от начала train.red, с")
    ax_train.set_ylabel("SmO2, %")
    ax_hbdiff.set_ylabel("HbDiff")
    ax_train.grid(alpha=0.25)

    lines_left, labels_left = ax_train.get_legend_handles_labels()
    lines_right, labels_right = ax_hbdiff.get_legend_handles_labels()
    unique_labels: dict[str, object] = {}
    for line, label in zip(lines_left + lines_right, labels_left + labels_right):
        unique_labels.setdefault(label, line)
    ax_train.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")

    fig.suptitle(
        (
            f"{subject_name} | "
            f"окно train.red: {match.train_start_sec:.1f}–{match.train_end_sec:.1f} с | "
            f"{match.note}"
        ),
        fontsize=13,
    )

    return fig, ax_train, ax_hbdiff


def summarize_window_gaps(
    train_times: np.ndarray,
    start_sec: float,
    end_sec: float,
    gap_threshold_sec: float,
) -> tuple[int, float]:
    """Считает количество и максимальный размер разрывов в выбранном окне."""

    train_times = np.asarray(train_times, dtype=float)
    dt = np.diff(train_times, prepend=train_times[0]) if len(train_times) else np.array([])
    point_mask = (train_times >= start_sec - 1e-9) & (train_times <= end_sec + 1e-9)
    gap_mask = dt > gap_threshold_sec
    gap_mask_in_window = gap_mask & point_mask
    gap_count = int(gap_mask_in_window.sum())
    max_gap = float(dt[gap_mask_in_window].max()) if gap_count else 0.0
    return gap_count, max_gap


def compute_manual_correlation(
    train_times: np.ndarray,
    train_values: np.ndarray,
    h5_times: np.ndarray,
    h5_values: np.ndarray,
    start_sec: float,
    end_sec: float,
    step_sec: float,
) -> float:
    """Оценивает корреляцию формы SmO2 для текущего ручного окна."""

    mask = (train_times >= start_sec - 1e-9) & (train_times <= end_sec + 1e-9)
    if mask.sum() < 2:
        return 0.0

    train_rel = train_times[mask] - start_sec
    train_grid, train_resampled = resample_signal(train_rel, train_values[mask], step_sec)
    h5_grid, h5_resampled = resample_signal(h5_times - h5_times[0], h5_values, step_sec)

    compare_len = min(len(train_resampled), len(h5_resampled))
    if compare_len < 2:
        return 0.0

    train_z = zscore(train_resampled[:compare_len])
    h5_z = zscore(h5_resampled[:compare_len])
    if np.isnan(train_z).any() or np.isnan(h5_z).any():
        return 0.0

    correlation = float(np.corrcoef(train_z, h5_z)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def prepare_overlay_segment(
    train_start_sec: float,
    train_end_sec: float,
    h5_times: np.ndarray,
    h5_values: np.ndarray,
    h5_start_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает только видимую часть h5, которая помещается в текущее окно train.red."""

    if len(h5_times) == 0:
        return np.array([]), np.array([])

    visible_duration_sec = max(train_end_sec - train_start_sec, 0.0)
    h5_visible_end_sec = h5_start_sec + visible_duration_sec
    mask = (h5_times >= h5_start_sec - 1e-9) & (h5_times <= h5_visible_end_sec + 1e-9)
    visible_h5_times = h5_times[mask]
    visible_h5_values = h5_values[mask]

    if len(visible_h5_times) == 0:
        return np.array([]), np.array([])

    overlay_x = train_start_sec + (visible_h5_times - h5_start_sec)
    return overlay_x, visible_h5_values


def build_manual_match(
    automatic_match: MatchWindow,
    train_times: np.ndarray,
    h5_times: np.ndarray,
    h5_values: np.ndarray,
    train_smo2: np.ndarray,
    start_sec: float,
    gap_threshold_sec: float,
    step_sec: float,
) -> MatchWindow:
    """Собирает MatchWindow для вручную выставленного старта окна."""

    h5_duration_sec = float(h5_times[-1] - h5_times[0]) if len(h5_times) else 0.0
    end_sec = min(start_sec + h5_duration_sec, float(train_times[-1])) if len(train_times) else start_sec
    visible_duration_sec = max(end_sec - start_sec, 0.0)
    h5_end_sec = (float(h5_times[0]) + visible_duration_sec) if len(h5_times) else automatic_match.h5_end_sec
    gap_count, max_gap = summarize_window_gaps(
        train_times=train_times,
        start_sec=start_sec,
        end_sec=end_sec,
        gap_threshold_sec=gap_threshold_sec,
    )
    correlation = compute_manual_correlation(
        train_times=train_times,
        train_values=train_smo2,
        h5_times=h5_times,
        h5_values=h5_values,
        start_sec=start_sec,
        end_sec=end_sec,
        step_sec=step_sec,
    )
    manual_status = "manual_selection"
    manual_note = (
        f"{MANUAL_SELECTION_COMMENT} "
        f"Окно train.red: {start_sec:.1f}–{end_sec:.1f} с."
    )
    if visible_duration_sec + 1e-9 < h5_duration_sec:
        manual_note += " Окно короче длительности h5, поэтому используется только доступная часть train.red."
    if gap_count > 0:
        manual_note += (
            f" Внутри окна есть разрывы train.red: {gap_count} шт., "
            f"максимум {max_gap:.2f} с."
        )

    return MatchWindow(
        best_feature="manual_slider_smO2",
        correlation=correlation,
        sign=1,
        status=manual_status,
        train_start_sec=float(start_sec),
        train_end_sec=float(end_sec),
        train_match_start_sec=float(start_sec),
        train_match_end_sec=float(end_sec),
        h5_start_sec=float(h5_times[0]) if len(h5_times) else automatic_match.h5_start_sec,
        h5_end_sec=h5_end_sec,
        train_is_reference=True,
        gap_count_in_window=gap_count,
        max_gap_in_window_sec=max_gap,
        note=manual_note,
    )


def write_points_file(
    path: Path,
    subject_name: str,
    step_sec: float,
    match: MatchWindow,
    selection_mode: str,
    selection_comment: str,
) -> None:
    """Сохраняет точки выравнивания в JSON для последующей сборки h5."""

    payload = {
        "subject": subject_name,
        "step_sec": step_sec,
        "selection_mode": selection_mode,
        "selection_comment": selection_comment,
        "match": {
            "best_feature": match.best_feature,
            "correlation": match.correlation,
            "sign": match.sign,
            "status": match.status,
            "train_start_sec": match.train_start_sec,
            "train_end_sec": match.train_end_sec,
            "train_match_start_sec": match.train_match_start_sec,
            "train_match_end_sec": match.train_match_end_sec,
            "h5_start_sec": match.h5_start_sec,
            "h5_end_sec": match.h5_end_sec,
            "train_is_reference": match.train_is_reference,
            "gap_count_in_window": match.gap_count_in_window,
            "max_gap_in_window_sec": match.max_gap_in_window_sec,
            "note": match.note,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_interactive_overlay(
    fig: plt.Figure,
    ax_train: plt.Axes,
    subject_dir: Path,
    subject_name: str,
    train_times: np.ndarray,
    train_frame: pd.DataFrame,
    source_h5_path: Path,
    source_trainred_path: Path,
    h5_times: np.ndarray,
    h5_values: np.ndarray,
    automatic_match: MatchWindow,
    step_sec: float,
    gap_threshold_sec: float,
    output_h5_path: Path,
) -> None:
    """Добавляет слайдер и кнопку сохранения для ручной подгонки окна."""

    train_smo2 = train_frame["SmO2"].to_numpy(dtype=float)
    h5_duration_sec = float(h5_times[-1] - h5_times[0]) if len(h5_times) else 0.0
    max_start_sec = max(float(train_times[-1]), 0.0)
    slider_step_sec = max(step_sec / 2.0, 0.1)

    overlay_line = next(
        line for line in ax_train.get_lines()
        if line.get_label() == "h5 SmO2 (наложение)"
    )
    start_line = next(
        line for line in ax_train.get_lines()
        if line.get_label() == "Старт окна"
    )
    end_line = next(
        line for line in ax_train.get_lines()
        if line.get_label() == "Конец окна"
    )
    selection_patch = ax_train.patches[0]

    info_text = fig.text(
        0.10,
        0.06,
        (
            "Ручной режим: двигайте старт окна по train.red. "
            "Ширина окна фиксирована по длительности h5."
        ),
        fontsize=10,
    )

    slider_ax = fig.add_axes([0.10, 0.11, 0.62, 0.035])
    slider = Slider(
        ax=slider_ax,
        label="Старт окна train.red, с",
        valmin=0.0,
        valmax=max_start_sec,
        valinit=min(automatic_match.train_start_sec, max_start_sec),
        valstep=slider_step_sec,
    )

    button_ax = fig.add_axes([0.77, 0.095, 0.15, 0.06])
    save_button = Button(button_ax, "Создать тест+ред.h5", hovercolor="#c8f7c5")

    current_match = {"value": automatic_match}

    def update_overlay(start_sec: float) -> None:
        """Обновляет окно и наложенную кривую при движении слайдера."""

        end_sec = min(start_sec + h5_duration_sec, float(train_times[-1]))
        start_sec = max(start_sec, 0.0)

        overlay_x, overlay_y = prepare_overlay_segment(
            train_start_sec=start_sec,
            train_end_sec=end_sec,
            h5_times=h5_times,
            h5_values=h5_values,
            h5_start_sec=float(h5_times[0]) if len(h5_times) else 0.0,
        )
        overlay_line.set_xdata(overlay_x)
        overlay_line.set_ydata(overlay_y)
        start_line.set_xdata([start_sec, start_sec])
        end_line.set_xdata([end_sec, end_sec])
        selection_patch.set_x(start_sec)
        selection_patch.set_width(end_sec - start_sec)

        manual_match = build_manual_match(
            automatic_match=automatic_match,
            train_times=train_times,
            h5_times=h5_times,
            h5_values=h5_values,
            train_smo2=train_smo2,
            start_sec=start_sec,
            gap_threshold_sec=gap_threshold_sec,
            step_sec=step_sec,
        )
        current_match["value"] = manual_match
        fig.suptitle(
            (
                f"{subject_name} | "
                f"ручное окно train.red: {manual_match.train_start_sec:.1f}–{manual_match.train_end_sec:.1f} с | "
                f"{manual_match.note}"
            ),
            fontsize=13,
        )
        info_text.set_text(
            (
                f"Ручное окно: {manual_match.train_start_sec:.1f}–{manual_match.train_end_sec:.1f} с | "
                f"corr={manual_match.correlation:.3f} | "
                f"разрывы: {manual_match.gap_count_in_window}, "
                f"макс. {manual_match.max_gap_in_window_sec:.2f} с"
            )
        )
        fig.canvas.draw_idle()

    def save_manual_points(_: object) -> None:
        """Создаёт итоговый тест+ред.h5 по текущему ручному окну."""

        manual_match = current_match["value"]
        match_dict = {
            "best_feature": manual_match.best_feature,
            "correlation": manual_match.correlation,
            "sign": manual_match.sign,
            "status": manual_match.status,
            "train_start_sec": manual_match.train_start_sec,
            "train_end_sec": manual_match.train_end_sec,
            "train_match_start_sec": manual_match.train_match_start_sec,
            "train_match_end_sec": manual_match.train_match_end_sec,
            "h5_start_sec": manual_match.h5_start_sec,
            "h5_end_sec": manual_match.h5_end_sec,
            "train_is_reference": manual_match.train_is_reference,
            "gap_count_in_window": manual_match.gap_count_in_window,
            "max_gap_in_window_sec": manual_match.max_gap_in_window_sec,
            "note": manual_match.note,
        }

        try:
            inserted_points = create_test_plus_red_h5(
                source_h5_path=source_h5_path,
                source_trainred_path=source_trainred_path,
                output_path=output_h5_path,
                match=match_dict,
                force=True,
                selection_mode="manual_slider",
                selection_comment=MANUAL_SELECTION_COMMENT,
            )
        except Exception as exc:
            info_text.set_text(f"Ошибка создания {output_h5_path.name}: {exc}")
            fig.canvas.draw_idle()
            return

        info_text.set_text(
            (
                f"{MANUAL_SELECTION_COMMENT} Создан файл: {output_h5_path}. "
                f"Точек на канал: {inserted_points}. "
                f"Окно {manual_match.train_start_sec:.1f}–"
                f"{manual_match.train_end_sec:.1f} с"
            )
        )
        fig.canvas.draw_idle()

    slider.on_changed(update_overlay)
    save_button.on_clicked(save_manual_points)
    # Matplotlib-виджеты должны иметь живые ссылки, иначе после выхода из
    # функции они могут быть собраны GC и перестать реагировать на ввод.
    fig._trainred_interactive_controls = {
        "slider_ax": slider_ax,
        "slider": slider,
        "button_ax": button_ax,
        "save_button": save_button,
        "info_text": info_text,
        "current_match": current_match,
        "update_overlay": update_overlay,
        "save_manual_points": save_manual_points,
        "output_h5_path": output_h5_path,
        "source_h5_path": source_h5_path,
        "source_trainred_path": source_trainred_path,
        "subject_dir": subject_dir,
    }
    update_overlay(float(slider.val))


def main() -> None:
    """Точка входа: ищет совпадение и строит графики."""

    args = parse_args()
    subject_dir = resolve_subject_dir(args.data_dir, args.subject)

    train_frame = load_trainred(subject_dir / "train.red.csv")
    h5_times, h5_values = load_h5_smo2(subject_dir / "тест.h5")

    timestamp_col = "Timestamp (seconds passed)"
    train_times = train_frame[timestamp_col].to_numpy(dtype=float)
    train_times = train_times - float(train_times[0])

    match = find_best_overlap(
        train_times_original=train_times,
        train_frame=train_frame,
        h5_times=h5_times,
        h5_values=h5_values,
        step_sec=args.step_sec,
        gap_threshold_sec=args.gap_threshold_sec,
    )

    print(f"Папка: {subject_dir.name}")
    print(f"Лучший признак: {match.best_feature}")
    print(f"Корреляция: {match.correlation:.3f}")
    print(f"Статус: {match.status}")
    print(
        "Окно train.red: "
        f"{match.train_start_sec:.1f}–{match.train_end_sec:.1f} с"
    )
    print(
        "Окно train.red (ось подбора): "
        f"{match.train_match_start_sec:.1f}–{match.train_match_end_sec:.1f} с"
    )
    print(
        "Окно h5: "
        f"{match.h5_start_sec:.1f}–{match.h5_end_sec:.1f} с"
    )
    print(
        "Разрывы train.red внутри окна: "
        f"{match.gap_count_in_window}, максимум {match.max_gap_in_window_sec:.2f} с"
    )
    print(f"Примечание: {match.note}")

    points_output = None if args.no_save_points else (args.points_output or (subject_dir / DEFAULT_POINTS_FILENAME))
    if points_output is not None:
        write_points_file(
            path=points_output,
            subject_name=subject_dir.name,
            step_sec=args.step_sec,
            match=match,
            selection_mode="automatic",
            selection_comment="Сохранено автоматически по результату выравнивания.",
        )
        print(f"Точки сохранены: {points_output}")

    output_h5_path = subject_dir / DEFAULT_OUTPUT_FILENAME

    fig, ax_train, _ = build_plot(
        subject_name=subject_dir.name,
        train_frame=train_frame,
        h5_times=h5_times,
        h5_values=h5_values,
        match=match,
        interactive=args.output is None,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {args.output}")
    else:
        build_interactive_overlay(
            fig=fig,
            ax_train=ax_train,
            subject_dir=subject_dir,
            subject_name=subject_dir.name,
            train_times=train_times,
            train_frame=train_frame,
            source_h5_path=subject_dir / "тест.h5",
            source_trainred_path=subject_dir / "train.red.csv",
            h5_times=h5_times,
            h5_values=h5_values,
            automatic_match=match,
            step_sec=args.step_sec,
            gap_threshold_sec=args.gap_threshold_sec,
            output_h5_path=output_h5_path,
        )
        plt.show()


if __name__ == "__main__":
    main()
