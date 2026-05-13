"""CLI-скрипт: расчёт LT1 методом baseline+0.4 ммоль/л по лактатным кривым.

Метод: строим PCHIP по дискретным точкам лактата (мощность→лактат),
вычисляем базовый уровень как среднее первых двух точек, ищем первое
пересечение кривой с уровнем baseline + BASELINE_DELTA_MMOL.

Это стандартный физиологический критерий LT1 (first lactate threshold):
первое устойчивое превышение лактата покоя на 0.4 ммоль/л.
ICC с log-log методом ~0.98 (Newell et al., 2007).

lt1_time_sec     — дискретный момент замера (первая точка выше порога)
lt1_pchip_time_sec — интерполированный PCHIP-переход через baseline+0.4

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
    interpolate_time_for_power,
    load_fulltest,
)

# Минимальное число уникальных ступеней лактата для LT1
MIN_STAGES_LT1 = 4
MIN_STAGES_HIGH = 5
# Прирост лактата над базовым уровнем для определения LT1
BASELINE_DELTA_MMOL = 0.4
# Только первая точка (60 Вт, разминка) как baseline.
# Анализ показал: у 4/18 субъектов p2 (90 Вт) уже выше baseline+0.4 →
# усреднение двух точек искусственно завышало порог и смещало LT1 вправо.
BASELINE_N_POINTS = 1


@dataclass(frozen=True)
class BaselineLt1Result:
    """Результат LT1 методом baseline + BASELINE_DELTA_MMOL."""

    lt1_power_w: float           # дискретная мощность (первая точка выше порога)
    lt1_lactate_mmol: float      # лактат в этой точке
    lt1_time_sec: float          # дискретное время
    lt1_pchip_power_w: float     # мощность пересечения PCHIP с порогом
    lt1_pchip_lactate_mmol: float  # лактат в точке пересечения (≈ baseline + 0.4)
    lt1_pchip_time_sec: float    # интерполированное время пересечения
    interval_start_sec: float    # начало окрестности (точка до пересечения)
    interval_end_sec: float      # конец окрестности (точка после пересечения)
    interval_start_power_w: float
    interval_end_power_w: float
    power_label_quality: str
    time_label_quality: str
    n_stages: int
    shoulder_dist_mmol: float    # превышение над baseline (мера выраженности LT1)


def build_baseline_lt1(points: tuple[LactatePoint, ...]) -> BaselineLt1Result:
    """Определяет LT1 как первое пересечение PCHIP-кривой с уровнем baseline+0.4.

    baseline = среднее первых BASELINE_N_POINTS точек лактата.
    Порог = baseline + BASELINE_DELTA_MMOL (0.4 ммоль/л).

    Дискретная метка (lt1_time_sec) — момент замера первой точки выше порога.
    PCHIP-метка (lt1_pchip_time_sec) — интерполированное время пересечения
    между предыдущей (ниже) и текущей (выше) точками.
    """
    powers   = np.array([p.power_w      for p in points], dtype=float)
    lactates = np.array([p.lactate_mmol for p in points], dtype=float)
    n_stages = len(points)

    # ── Baseline и порог ─────────────────────────────────────────────────────
    n_base   = min(BASELINE_N_POINTS, n_stages)
    baseline = float(np.mean(lactates[:n_base]))
    threshold = baseline + BASELINE_DELTA_MMOL

    # ── Поиск первой дискретной точки выше порога ────────────────────────────
    above = np.where(lactates > threshold)[0]
    if len(above) == 0:
        # Порог не достигнут — возвращаем последнюю точку с пометкой low
        disc_idx = n_stages - 1
        power_quality = "low"
    else:
        disc_idx = int(above[0])
        power_quality = "high" if n_stages >= MIN_STAGES_HIGH else (
            "medium" if n_stages >= MIN_STAGES_LT1 else "low"
        )

    lt1_power_w    = float(powers[disc_idx])
    lt1_lactate    = float(lactates[disc_idx])
    lt1_time_sec   = float(points[disc_idx].effective_time_sec)

    # ── PCHIP-интерполяция точки пересечения ─────────────────────────────────
    pchip = PchipInterpolator(powers, lactates)

    if disc_idx == 0 or len(above) == 0:
        # Порог уже превышен с первой точки — интерп невозможна
        pchip_power_w   = lt1_power_w
        pchip_lactate   = lt1_lactate
        pchip_time_sec  = lt1_time_sec
        lower_idx       = 0
        upper_idx       = 0
    else:
        lower_idx = disc_idx - 1
        upper_idx = disc_idx
        # Бисекция на плотной сетке между двумя соседними точками
        search_x = np.linspace(powers[lower_idx], powers[upper_idx], 2000)
        search_y = pchip(search_x)
        cross = np.where(search_y >= threshold)[0]
        if len(cross) == 0:
            pchip_power_w  = lt1_power_w
            pchip_lactate  = lt1_lactate
            pchip_time_sec = lt1_time_sec
        else:
            pchip_power_w  = float(search_x[cross[0]])
            pchip_lactate  = float(threshold)
            pchip_time_sec = float(interpolate_time_for_power(points, pchip_power_w))

    interval_start_sec      = float(points[lower_idx].effective_time_sec)
    interval_end_sec        = float(points[upper_idx].effective_time_sec)
    interval_start_power_w  = float(points[lower_idx].power_w)
    interval_end_power_w    = float(points[upper_idx].power_w)

    time_quality = power_quality

    # shoulder_dist — превышение максимального лактата над baseline
    # (мера выраженности ответа, аналог прежнего perpendicular distance)
    shoulder_dist = float(np.max(lactates) - baseline)

    return BaselineLt1Result(
        lt1_power_w=lt1_power_w,
        lt1_lactate_mmol=lt1_lactate,
        lt1_time_sec=lt1_time_sec,
        lt1_pchip_power_w=pchip_power_w,
        lt1_pchip_lactate_mmol=pchip_lactate,
        lt1_pchip_time_sec=pchip_time_sec,
        interval_start_sec=interval_start_sec,
        interval_end_sec=interval_end_sec,
        interval_start_power_w=interval_start_power_w,
        interval_end_power_w=interval_end_power_w,
        power_label_quality=power_quality,
        time_label_quality=time_quality,
        n_stages=n_stages,
        shoulder_dist_mmol=shoulder_dist,
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

    result = build_baseline_lt1(points)
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
