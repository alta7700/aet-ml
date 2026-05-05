#!/usr/bin/env python3
"""Ищет окно из полной выгрузки train.red, соответствующее записи из тест.h5."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# Для сопоставления сначала пробуем штатный сглаженный SmO2.
# Остальные признаки нужны как резерв для случаев, где форма сигнала гуляет.
ALIGNMENT_FEATURES = (
    "SmO2",
    "SmO2 unfiltered",
    "HBDiff",
    "HHb unfiltered",
    "O2HB unfiltered",
)

# Эти колонки нужны либо для выравнивания, либо для последующей записи
# в расширенный файл тест+ред.h5.
TRAINRED_NUMERIC_COLUMNS = (
    "SmO2",
    "HBDiff",
    "SmO2 unfiltered",
    "O2HB unfiltered",
    "HHb unfiltered",
    "THb unfiltered",
    "HBDiff unfiltered",
)


@dataclass(frozen=True)
class AlignmentResult:
    """Результат выравнивания одной записи."""

    subject: str
    status: str
    best_feature: str
    correlation: float
    sign: int
    train_start_sec: float
    train_end_sec: float
    train_duration_sec: float
    h5_duration_sec: float
    train_rows: int
    h5_rows: int
    train_sampling_hz: float
    h5_sampling_hz: float
    note: str


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Находит внутри полного train.red.csv фрагмент, который лучше всего "
            "совпадает с каналом moxy.smo2 из тест.h5."
        )
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
        "--output-csv",
        type=Path,
        help="Куда сохранить итоговую таблицу с метриками совпадения.",
    )
    parser.add_argument(
        "--export-window",
        action="store_true",
        help=(
            "Экспортировать найденное окно train.red в файл "
            "aligned_trainred_window.csv внутри папки участника."
        ),
    )
    return parser.parse_args()


def find_trainred_header_row(path: Path) -> int:
    """Находит строку, с которой начинается табличная часть Train.Red."""

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            if line.startswith("Timestamp (seconds passed)"):
                return index
    raise ValueError(f"Не удалось найти заголовок данных в {path}")


def load_trainred(path: Path) -> pd.DataFrame:
    """Загружает табличную часть экспорта Train.Red и приводит числовые поля."""

    header_row = find_trainred_header_row(path)
    frame = pd.read_csv(path, skiprows=header_row)
    frame = frame.rename(columns={frame.columns[0]: "Timestamp (seconds passed)"})

    timestamp_col = "Timestamp (seconds passed)"
    frame = frame[pd.to_numeric(frame[timestamp_col], errors="coerce").notna()].copy()
    frame[timestamp_col] = pd.to_numeric(frame[timestamp_col], errors="coerce")

    for column in TRAINRED_NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def normalize_h5_time(raw_timestamps: np.ndarray) -> np.ndarray:
    """Приводит временную ось HDF5 к секундам от начала записи."""

    timestamps = raw_timestamps.astype(float)
    diffs = np.diff(timestamps)
    median_step = float(np.nanmedian(diffs)) if len(diffs) else 0.0

    # В текущих файлах шаг около 500, то есть время записано в миллисекундах.
    # Оставляем автоматическое определение, чтобы скрипт не был привязан к
    # одному формату выгрузки.
    divisor = 1000.0 if median_step > 10.0 else 1.0
    return (timestamps - timestamps[0]) / divisor


def load_h5_smo2(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Загружает временную ось и значения канала moxy.smo2."""

    with h5py.File(path, "r") as handle:
        raw_timestamps = handle["channels/moxy.smo2/timestamps"][:]
        values = handle["channels/moxy.smo2/values"][:].astype(float)
    return normalize_h5_time(raw_timestamps), values


def resample_signal(
    times: np.ndarray,
    values: np.ndarray,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ресемплирует сигнал на равномерную временную сетку."""

    finite_mask = np.isfinite(times) & np.isfinite(values)
    clean_times = np.asarray(times)[finite_mask]
    clean_values = np.asarray(values)[finite_mask]

    if len(clean_times) < 2:
        return np.array([]), np.array([])

    order = np.argsort(clean_times)
    clean_times = clean_times[order]
    clean_values = clean_values[order]

    unique_times, unique_indices = np.unique(clean_times, return_index=True)
    unique_values = clean_values[unique_indices]

    if len(unique_times) < 2:
        return np.array([]), np.array([])

    grid = np.arange(unique_times[0], unique_times[-1], step_sec)
    interpolated = np.interp(grid, unique_times, unique_values)
    return grid, interpolated


def zscore(values: np.ndarray) -> np.ndarray:
    """Нормализует сигнал, чтобы сравнивать форму, а не абсолютный уровень."""

    std = float(np.nanstd(values))
    if std == 0.0 or not np.isfinite(std):
        return np.full_like(values, np.nan, dtype=float)
    return (values - float(np.nanmean(values))) / std


def nominal_hz(times: np.ndarray) -> float:
    """Оценивает типичную частоту дискретизации по медианному шагу."""

    if len(times) < 2:
        return 0.0
    median_step = float(np.nanmedian(np.diff(times)))
    if median_step <= 0.0 or not np.isfinite(median_step):
        return 0.0
    return 1.0 / median_step


def classify_match(correlation: float) -> str:
    """Классифицирует качество совпадения по корреляции формы сигнала."""

    if correlation >= 0.9:
        return "reliable"
    if correlation >= 0.75:
        return "usable"
    if correlation >= 0.6:
        return "manual_review"
    return "no_match"


def choose_note(status: str, train_duration_sec: float, h5_duration_sec: float) -> str:
    """Добавляет краткое пояснение к результату."""

    if train_duration_sec + 1e-6 < h5_duration_sec:
        return "train.red короче теста из h5, автоматическое вложение невозможно"
    if status == "reliable":
        return "форма сигнала совпадает хорошо"
    if status == "usable":
        return "совпадение рабочее, но лучше проверить границы окна вручную"
    if status == "manual_review":
        return "есть только умеренное сходство формы, нужен визуальный контроль"
    return "устойчивого совпадения по форме SmO2 не найдено"


def find_best_alignment(
    train_times: np.ndarray,
    train_frame: pd.DataFrame,
    h5_times: np.ndarray,
    h5_values: np.ndarray,
    step_sec: float,
) -> tuple[str, float, int, float, float] | None:
    """Ищет лучшее окно train.red по набору признаков."""

    h5_grid, h5_resampled = resample_signal(h5_times, h5_values, step_sec)
    if len(h5_grid) < 2:
        return None

    h5_z = zscore(h5_resampled)
    best_match: tuple[str, float, int, float, float] | None = None

    for feature in ALIGNMENT_FEATURES:
        if feature not in train_frame.columns:
            continue

        train_grid, train_resampled = resample_signal(
            train_times,
            train_frame[feature].to_numpy(),
            step_sec,
        )
        if len(train_grid) < len(h5_grid):
            continue

        for start_index in range(0, len(train_resampled) - len(h5_resampled) + 1):
            segment = train_resampled[start_index : start_index + len(h5_resampled)]
            segment_z = zscore(segment)
            if np.isnan(segment_z).any():
                continue

            corr = float(np.corrcoef(segment_z, h5_z)[0, 1])
            corr_inv = float(np.corrcoef(-segment_z, h5_z)[0, 1])

            if corr_inv > corr:
                sign = -1
                best_corr = corr_inv
            else:
                sign = 1
                best_corr = corr

            if not np.isfinite(best_corr):
                continue

            start_sec = float(train_grid[start_index])
            end_sec = start_sec + float(h5_grid[-1] - h5_grid[0])
            candidate = (feature, best_corr, sign, start_sec, end_sec)

            if best_match is None or candidate[1] > best_match[1]:
                best_match = candidate

    return best_match


def export_window(
    subject_dir: Path,
    train_frame: pd.DataFrame,
    start_sec: float,
    end_sec: float,
) -> None:
    """Сохраняет найденное окно Train.Red в отдельный CSV."""

    timestamp_col = "Timestamp (seconds passed)"
    window = train_frame[
        (train_frame[timestamp_col] >= start_sec) & (train_frame[timestamp_col] <= end_sec)
    ].copy()
    window["aligned_time_sec"] = window[timestamp_col] - start_sec
    output_path = subject_dir / "aligned_trainred_window.csv"
    window.to_csv(output_path, index=False)


def analyze_subject(subject_dir: Path, step_sec: float, export_matched_window: bool) -> AlignmentResult:
    """Выполняет анализ одной папки участника."""

    train_path = subject_dir / "train.red.csv"
    h5_path = subject_dir / "тест.h5"

    train_frame = load_trainred(train_path)
    h5_times, h5_values = load_h5_smo2(h5_path)

    timestamp_col = "Timestamp (seconds passed)"
    train_times = train_frame[timestamp_col].to_numpy(dtype=float)
    train_times = train_times - float(train_times[0])

    train_duration_sec = float(train_times[-1] - train_times[0]) if len(train_times) else 0.0
    h5_duration_sec = float(h5_times[-1] - h5_times[0]) if len(h5_times) else 0.0

    best_match = find_best_alignment(
        train_times=train_times,
        train_frame=train_frame,
        h5_times=h5_times,
        h5_values=h5_values,
        step_sec=step_sec,
    )

    if best_match is None:
        status = "no_match"
        note = choose_note(status, train_duration_sec, h5_duration_sec)
        return AlignmentResult(
            subject=subject_dir.name,
            status=status,
            best_feature="",
            correlation=float("nan"),
            sign=1,
            train_start_sec=float("nan"),
            train_end_sec=float("nan"),
            train_duration_sec=train_duration_sec,
            h5_duration_sec=h5_duration_sec,
            train_rows=len(train_frame),
            h5_rows=len(h5_values),
            train_sampling_hz=nominal_hz(train_times),
            h5_sampling_hz=nominal_hz(h5_times),
            note=note,
        )

    feature, correlation, sign, start_sec, end_sec = best_match
    status = classify_match(correlation)
    note = choose_note(status, train_duration_sec, h5_duration_sec)

    # Сомнительные совпадения лучше сначала проверить вручную,
    # а уже потом экспортировать как окончательный результат.
    if export_matched_window and status in {"reliable", "usable"}:
        export_window(subject_dir, train_frame, start_sec, end_sec)

    return AlignmentResult(
        subject=subject_dir.name,
        status=status,
        best_feature=feature,
        correlation=correlation,
        sign=sign,
        train_start_sec=start_sec,
        train_end_sec=end_sec,
        train_duration_sec=train_duration_sec,
        h5_duration_sec=h5_duration_sec,
        train_rows=len(train_frame),
        h5_rows=len(h5_values),
        train_sampling_hz=nominal_hz(train_times),
        h5_sampling_hz=nominal_hz(h5_times),
        note=note,
    )


def print_results(results: list[AlignmentResult]) -> None:
    """Печатает сводку в читаемом табличном виде."""

    header = (
        "Участник",
        "Статус",
        "Признак",
        "corr",
        "Старт, с",
        "Финиш, с",
        "Train, c",
        "H5, c",
        "Train Hz",
        "H5 Hz",
        "Примечание",
    )
    print(" | ".join(header))
    print("-" * 160)

    for result in results:
        print(
            " | ".join(
                [
                    result.subject,
                    result.status,
                    result.best_feature or "-",
                    f"{result.correlation:.3f}" if np.isfinite(result.correlation) else "-",
                    f"{result.train_start_sec:.1f}" if np.isfinite(result.train_start_sec) else "-",
                    f"{result.train_end_sec:.1f}" if np.isfinite(result.train_end_sec) else "-",
                    f"{result.train_duration_sec:.1f}",
                    f"{result.h5_duration_sec:.1f}",
                    f"{result.train_sampling_hz:.2f}",
                    f"{result.h5_sampling_hz:.2f}",
                    result.note,
                ]
            )
        )


def write_csv(path: Path, results: list[AlignmentResult]) -> None:
    """Сохраняет результаты анализа в CSV."""

    fieldnames = [
        "subject",
        "status",
        "best_feature",
        "correlation",
        "sign",
        "train_start_sec",
        "train_end_sec",
        "train_duration_sec",
        "h5_duration_sec",
        "train_rows",
        "h5_rows",
        "train_sampling_hz",
        "h5_sampling_hz",
        "note",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> None:
    """Точка входа для пакетного анализа всех участников."""

    args = parse_args()
    subject_dirs = sorted(path for path in args.data_dir.iterdir() if path.is_dir())

    results = [
        analyze_subject(
            subject_dir=subject_dir,
            step_sec=args.step_sec,
            export_matched_window=args.export_window,
        )
        for subject_dir in subject_dirs
    ]

    print_results(results)

    if args.output_csv:
        write_csv(args.output_csv, results)


if __name__ == "__main__":
    main()
