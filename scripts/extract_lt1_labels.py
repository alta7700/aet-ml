"""CLI-скрипт: расчёт LT1 методом D-max по лактатным кривым.

Тот же алгоритм, что и modified D-max для LT2 (cubic fit + PCHIP-проверка),
но линия проводится от первой точки до последней (start_index = 0).
Физиологически это нахождение первого «плеча» лактатной кривой — LT1.

Выходной файл: dataset/lt1_labels.parquet
  subject_id, lt1_power_w, lt1_lactate_mmol, lt1_time_sec,
  lt1_pchip_power_w, lt1_pchip_time_sec,
  lt1_interval_start_sec, lt1_interval_end_sec,
  lt1_power_label_quality, lt1_time_label_quality,
  lt1_available

Использование:
  python scripts/extract_lt1_labels.py
  python scripts/extract_lt1_labels.py --dataset-dir /path/to/dataset --force
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_pipeline.common import (
    DEFAULT_DATASET_DIR,
    save_parquet,
)
from methods.lt2 import (
    LactatePoint,
    build_lactate_points,
    find_max_distance_point,
    interpolate_time_for_power,
    load_fulltest,
    perpendicular_distance,
)

# Минимальное число уникальных ступеней лактата для D-max LT1
MIN_STAGES_LT1 = 4
# Порог расхождения cubic vs PCHIP (Вт) для оценки качества
PCHIP_DELTA_HIGH_W = 10.0
PCHIP_DELTA_MEDIUM_W = 25.0
MIN_STAGES_HIGH = 5


@dataclass(frozen=True)
class DmaxLt1Result:
    """Результат D-max LT1."""

    lt1_power_w: float
    lt1_lactate_mmol: float
    lt1_time_sec: float
    lt1_pchip_power_w: float
    lt1_pchip_lactate_mmol: float
    lt1_pchip_time_sec: float
    interval_start_sec: float
    interval_end_sec: float
    interval_start_power_w: float
    interval_end_power_w: float
    power_label_quality: str
    time_label_quality: str
    n_stages: int
    shoulder_dist_mmol: float  # максимальное перп. расстояние (мера выраженности «плеча»)


def build_dmax_lt1(points: tuple[LactatePoint, ...]) -> DmaxLt1Result:
    """Применяет D-max от первой до последней точки — LT1.

    В отличие от modified D-max (LT2), start_index всегда равен 0:
    мы ищем первый перегиб кривой, а не второй.
    """
    powers = np.array([p.power_w for p in points], dtype=float)
    lactates = np.array([p.lactate_mmol for p in points], dtype=float)

    # Кубический полином + PCHIP по всей кривой
    poly = np.poly1d(np.polyfit(powers, lactates, 3))
    fit_x = np.linspace(float(powers[0]), float(powers[-1]), 800)
    fit_y = poly(fit_x)
    pchip = PchipInterpolator(powers, lactates)
    pchip_fit_y = pchip(fit_x)

    # Линия D-max: от первой до последней точки (start_index = 0)
    x1, y1 = float(powers[0]), float(lactates[0])
    x2, y2 = float(powers[-1]), float(lactates[-1])

    # Cubic fit — основная оценка
    _, _, lt1_power_w, lt1_lactate_mmol = find_max_distance_point(
        curve_x=fit_x, curve_y=fit_y, x1=x1, y1=y1, x2=x2, y2=y2
    )
    # PCHIP — проверка устойчивости
    _, _, pchip_lt1_power_w, pchip_lt1_lactate_mmol = find_max_distance_point(
        curve_x=fit_x, curve_y=pchip_fit_y, x1=x1, y1=y1, x2=x2, y2=y2
    )

    lt1_time_sec = interpolate_time_for_power(points, lt1_power_w)
    pchip_lt1_time_sec = interpolate_time_for_power(points, pchip_lt1_power_w)

    # Окрестность LT1 по реальным точкам
    insert_idx = int(np.searchsorted(powers, lt1_power_w, side="right"))
    lower_idx = max(0, insert_idx - 1)
    upper_idx = min(len(points) - 1, insert_idx)
    interval_start_sec = float(points[lower_idx].effective_time_sec)
    interval_end_sec = float(points[upper_idx].effective_time_sec)
    interval_start_power_w = float(points[lower_idx].power_w)
    interval_end_power_w = float(points[upper_idx].power_w)

    # Качество метки мощности
    pchip_delta_w = abs(lt1_power_w - pchip_lt1_power_w)
    n_stages = len(points)
    if pchip_delta_w <= PCHIP_DELTA_HIGH_W and n_stages >= MIN_STAGES_HIGH:
        power_quality = "high"
    elif pchip_delta_w <= PCHIP_DELTA_MEDIUM_W and n_stages >= MIN_STAGES_LT1:
        power_quality = "medium"
    else:
        power_quality = "low"

    # Качество метки времени: только по мощности (нет secondary markers для LT1)
    # high = high power quality + ≥5 ступеней
    if power_quality == "high":
        time_quality = "high"
    elif power_quality == "medium":
        time_quality = "medium"
    else:
        time_quality = "low"

    # Проверка наличия «плеча»: вычисляем максимальное перпендикулярное расстояние.
    # Если оно мало — кривая почти линейна, LT1 неотличим от LT2.
    all_distances = perpendicular_distance(fit_x, fit_y, x1, y1, x2, y2)
    max_dist = float(np.max(all_distances))
    # Порог: расстояние < 0.5 ммоль/л → явного плеча нет
    if max_dist < 0.5:
        power_quality = "low"
        time_quality = "low"

    return DmaxLt1Result(
        lt1_power_w=float(lt1_power_w),
        lt1_lactate_mmol=float(lt1_lactate_mmol),
        lt1_time_sec=float(lt1_time_sec),
        lt1_pchip_power_w=float(pchip_lt1_power_w),
        lt1_pchip_lactate_mmol=float(pchip_lt1_lactate_mmol),
        lt1_pchip_time_sec=float(pchip_lt1_time_sec),
        interval_start_sec=float(interval_start_sec),
        interval_end_sec=float(interval_end_sec),
        interval_start_power_w=float(interval_start_power_w),
        interval_end_power_w=float(interval_end_power_w),
        power_label_quality=power_quality,
        time_label_quality=time_quality,
        n_stages=n_stages,
        shoulder_dist_mmol=float(max_dist),
    )


def compute_lt1_for_subject(h5_path: Path) -> dict[str, object]:
    """Вычисляет D-max LT1 для одного участника."""
    data = load_fulltest(h5_path)

    try:
        points = build_lactate_points(data)
    except ValueError as exc:
        # Недостаточно уникальных ступеней → LT1 недоступен
        return {
            "lt1_available": 0,
            "lt1_power_w": np.nan,
            "lt1_lactate_mmol": np.nan,
            "lt1_time_sec": np.nan,
            "lt1_pchip_power_w": np.nan,
            "lt1_pchip_time_sec": np.nan,
            "lt1_interval_start_sec": np.nan,
            "lt1_interval_end_sec": np.nan,
            "lt1_interval_start_power_w": np.nan,
            "lt1_interval_end_power_w": np.nan,
            "lt1_power_label_quality": "unavailable",
            "lt1_time_label_quality": "unavailable",
            "lt1_n_stages": 0,
            "lt1_error": str(exc),
        }

    result = build_dmax_lt1(points)
    return {
        "lt1_available": 1,
        "lt1_power_w": result.lt1_power_w,
        "lt1_lactate_mmol": result.lt1_lactate_mmol,
        "lt1_time_sec": result.lt1_time_sec,
        "lt1_pchip_power_w": result.lt1_pchip_power_w,
        "lt1_pchip_time_sec": result.lt1_pchip_time_sec,
        "lt1_interval_start_sec": result.interval_start_sec,
        "lt1_interval_end_sec": result.interval_end_sec,
        "lt1_interval_start_power_w": result.interval_start_power_w,
        "lt1_interval_end_power_w": result.interval_end_power_w,
        "lt1_power_label_quality": result.power_label_quality,
        "lt1_time_label_quality": result.time_label_quality,
        "lt1_n_stages": result.n_stages,
        "lt1_shoulder_dist_mmol": result.shoulder_dist_mmol,
        "lt1_error": None,
    }


def build_lt1_labels(subjects_path: Path) -> pd.DataFrame:
    """Строит таблицу LT1-меток для всех участников.

    После вычисления D-max сравниваем LT1 с LT2 из subjects.parquet.
    Если D-max находит ту же точку, что и для LT2 (кривая стартует в зоне
    накопления), помечаем lt1_equals_lt2=1 и снижаем качество до 'low'.
    """
    subjects = pd.read_parquet(subjects_path)
    rows: list[dict[str, object]] = []

    for _, subj in subjects.iterrows():
        h5_path = Path(str(subj["source_h5_path"]))
        row = compute_lt1_for_subject(h5_path)
        row["subject_id"] = subj["subject_id"]
        # Сравниваем с LT2-мощностью из subjects.parquet
        lt2_power_w = float(subj.get("lt2_power_w") or np.nan)
        lt1_power_w = float(row["lt1_power_w"]) if row["lt1_available"] else np.nan
        lt1_eq_lt2 = int(
            np.isfinite(lt1_power_w)
            and np.isfinite(lt2_power_w)
            and abs(lt1_power_w - lt2_power_w) < 1.0
        )
        row["lt1_equals_lt2"] = lt1_eq_lt2
        # Если LT1 = LT2 → нет различимого аэробного порога
        if lt1_eq_lt2:
            row["lt1_time_label_quality"] = "low"
            row["lt1_power_label_quality"] = "low"
        rows.append(row)

        avail = row["lt1_available"]
        quality = row["lt1_time_label_quality"]
        lt1_t = row["lt1_time_sec"]
        lt1_p = row["lt1_power_w"]
        eq_tag = " [=LT2!]" if lt1_eq_lt2 else ""
        if avail:
            print(
                f"  {subj['subject_id']}: LT1={lt1_p:.0f} Вт, "
                f"t={lt1_t:.0f} с, quality={quality}{eq_tag}"
            )
        else:
            print(f"  {subj['subject_id']}: недоступно ({row.get('lt1_error', '')})")

    df = pd.DataFrame(rows)
    df = df.sort_values("subject_id").reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Вычисляет LT1 методом D-max для всех участников."
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "subjects.parquet",
        help="Путь к subjects.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "lt1_labels.parquet",
        help="Выходной файл lt1_labels.parquet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать выходной файл, если уже существует.",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа."""
    args = parse_args()

    if args.output.exists() and not args.force:
        print(f"⏭  Файл уже существует: {args.output}. Используйте --force для перезаписи.")
        return

    print(f"Вычисляем LT1 D-max для всех участников...")
    print(f"  subjects: {args.subjects_file}")
    print(f"  output:   {args.output}")

    df = build_lt1_labels(args.subjects_file)
    save_parquet(df, args.output, force=True)

    n_available = int(df["lt1_available"].sum())
    n_high = int((df["lt1_time_label_quality"] == "high").sum())
    n_medium = int((df["lt1_time_label_quality"] == "medium").sum())
    print(f"\n✓ Готово: {n_available}/{len(df)} участников с LT1")
    print(f"  Качество: high={n_high}, medium={n_medium}, "
          f"low/unavail={len(df) - n_high - n_medium}")
    print(f"  Сохранено: {args.output}")


if __name__ == "__main__":
    main()
