#!/usr/bin/env python3
"""Создаёт finaltest.h5, добавляя к fulltest.h5 метрики LT2."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import unicodedata

import h5py

from create_test_plus_red_h5 import copy_h5_tree
from visualize_lt2_from_fulltest import compute_lt2, load_fulltest

DEFAULT_SOURCE_FILENAME = "fulltest.h5"
DEFAULT_OUTPUT_FILENAME = "finaltest.h5"


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Создаёт finaltest.h5: копирует fulltest.h5 и добавляет "
            "в корень файла основные метрики LT2."
        )
    )
    parser.add_argument(
        "subject",
        help="Точное имя папки участника внутри data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Каталог с папками участников.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help=(
            "Путь к исходному HDF5. "
            "По умолчанию используется fulltest.h5 в папке участника."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Путь к итоговому HDF5. "
            "По умолчанию создаётся finaltest.h5 в папке участника."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать выходной файл, если он уже существует.",
    )
    return parser.parse_args()


def resolve_subject_dir(data_dir: Path, subject_name: str) -> Path:
    """Находит папку участника и подсказывает близкие варианты при опечатке."""

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


def write_lt2_metadata(dest_path: Path) -> dict[str, float | str | int | None]:
    """Считает LT2 по fulltest.h5 и записывает основные метрики в attrs."""

    data = load_fulltest(dest_path)
    result = compute_lt2(data)

    refined_sources = ",".join(result.refined_sources)

    payload: dict[str, float | str | int | None] = {
        "lt2_method": "moddmax_cubic",
        "lt2_power_w": float(result.moddmax.lt2_power_w),
        "lt2_lactate_mmol": float(result.moddmax.lt2_lactate_mmol),
        "lt2_time_sec": float(result.moddmax.lt2_time_sec),
        "lt2_interval_start_sec": float(result.moddmax.interval_start_sec),
        "lt2_interval_end_sec": float(result.moddmax.interval_end_sec),
        "lt2_interval_start_power_w": float(result.moddmax.interval_start_power_w),
        "lt2_interval_end_power_w": float(result.moddmax.interval_end_power_w),
        "lt2_pchip_power_w": float(result.moddmax.pchip_lt2_power_w),
        "lt2_pchip_lactate_mmol": float(result.moddmax.pchip_lt2_lactate_mmol),
        "lt2_pchip_time_sec": float(result.moddmax.pchip_lt2_time_sec),
        "lt2_pchip_delta_power_w": float(
            abs(result.moddmax.lt2_power_w - result.moddmax.pchip_lt2_power_w)
        ),
        "lt2_pchip_delta_time_sec": float(
            abs(result.moddmax.lt2_time_sec - result.moddmax.pchip_lt2_time_sec)
        ),
        "lt2_hrvt2_time_sec": (
            None if result.dfa.crossing_time_sec is None else float(result.dfa.crossing_time_sec)
        ),
        "lt2_hhb_breakpoint_time_sec": (
            None if result.hhb.breakpoint_time_sec is None else float(result.hhb.breakpoint_time_sec)
        ),
        "lt2_hhb_peak_time_sec": (
            None if result.hhb.peak_time_sec is None else float(result.hhb.peak_time_sec)
        ),
        "lt2_refined_time_sec": float(result.refined_time_sec),
        "lt2_refined_sources": refined_sources,
    }

    with h5py.File(dest_path, "a") as handle:
        for attr_name in list(handle.attrs.keys()):
            if str(attr_name).startswith("lt2_"):
                del handle.attrs[attr_name]

        for attr_name, attr_value in payload.items():
            if attr_value is None:
                continue
            handle.attrs[attr_name] = attr_value

    return payload


def create_finaltest_h5(
    source_h5_path: Path,
    output_path: Path,
    force: bool,
) -> dict[str, float | str | int | None]:
    """Копирует fulltest.h5 и дописывает LT2-метрики."""

    if not source_h5_path.exists():
        raise FileNotFoundError(
            f"Исходный HDF5 не найден: {source_h5_path}. "
            "Сначала создайте fulltest.h5."
        )

    copy_h5_tree(
        source_path=source_h5_path,
        dest_path=output_path,
        force=force,
    )
    return write_lt2_metadata(dest_path=output_path)


def main() -> None:
    """Точка входа: создаёт finaltest.h5 для выбранной папки."""

    args = parse_args()
    subject_dir = resolve_subject_dir(args.data_dir, args.subject)

    source_h5_path = args.source or (subject_dir / DEFAULT_SOURCE_FILENAME)
    output_path = args.output or (subject_dir / DEFAULT_OUTPUT_FILENAME)

    payload = create_finaltest_h5(
        source_h5_path=source_h5_path,
        output_path=output_path,
        force=args.force,
    )

    print(f"Папка: {subject_dir.name}")
    print(f"Источник HDF5: {source_h5_path}")
    print(f"LT2 method: {payload['lt2_method']}")
    print(
        "LT2 cubic: "
        f"{float(payload['lt2_power_w']):.1f} Вт, "
        f"{float(payload['lt2_lactate_mmol']):.2f} mmol/L, "
        f"{float(payload['lt2_time_sec']):.1f} с"
    )
    print(
        "LT2 интервал: "
        f"{float(payload['lt2_interval_start_power_w']):.0f}–"
        f"{float(payload['lt2_interval_end_power_w']):.0f} Вт | "
        f"{float(payload['lt2_interval_start_sec']):.1f}–"
        f"{float(payload['lt2_interval_end_sec']):.1f} с"
    )
    print(
        "PCHIP-проверка: "
        f"{float(payload['lt2_pchip_power_w']):.1f} Вт, "
        f"расхождение {float(payload['lt2_pchip_delta_power_w']):.1f} Вт"
    )
    print(
        "Итоговое время LT2: "
        f"{float(payload['lt2_refined_time_sec']):.1f} с "
        f"({payload['lt2_refined_sources']})"
    )
    print(f"Создан файл: {output_path}")


if __name__ == "__main__":
    main()
