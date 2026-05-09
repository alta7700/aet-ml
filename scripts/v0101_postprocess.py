"""Постпроцессинг v0101: сборка честных baselines и итогового отчета.

Запускается после того как основное обучение завершилось.
Собирает результаты из results/v0101/, запускает честные baselines,
генерирует финальный report.md с сравнением с v0011.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from scripts.v0011_modality_ablation import (
    prepare_data,
    get_feature_cols,
)

OUT_DIR = _ROOT / "results" / "v0101"


def main() -> None:
    """Собирает результаты обучения и выводит финальный отчет."""
    print("=" * 70)
    print("v0101 ПОСТПРОЦЕССИНГ: сборка результатов")
    print("=" * 70)

    # Проверяем наличие результатов
    summary_file = OUT_DIR / "summary.csv"
    best_file = OUT_DIR / "best_per_set.csv"
    honest_file = OUT_DIR / "honest_baselines.csv"
    report_file = OUT_DIR / "report.md"

    if not summary_file.exists():
        print("\n❌ summary.csv не найден. Дождитесь завершения обучения.")
        return

    summary = pd.read_csv(summary_file)
    best = pd.read_csv(best_file) if best_file.exists() else summary.groupby(
        ["feature_set", "target"]).apply(
        lambda x: x.loc[x["kalman_mae_min"].idxmin()]).reset_index(drop=True)

    print(f"\n✅ Загружено {len(summary)} записей из summary.csv")

    # Сравнение с v0011
    print("\n" + "═" * 70)
    print("СРАВНЕНИЕ С v0011 (ElasticNet/GBM)")
    print("═" * 70)

    v0011_results = {
        ("lt2", "EMG"): 3.198,
        ("lt2", "NIRS"): 3.938,
        ("lt2", "EMG+NIRS"): 3.117,
        ("lt2", "EMG+NIRS+HRV"): 1.859,
        ("lt1", "EMG"): 2.896,
        ("lt1", "NIRS"): 3.842,
        ("lt1", "EMG+NIRS"): 2.750,
        ("lt1", "EMG+NIRS+HRV"): 2.277,
    }

    improvements = []
    for _, row in best.iterrows():
        key = (row["target"], row["feature_set"])
        if key in v0011_results:
            v0011_mae = v0011_results[key]
            v0101_mae = row["kalman_mae_min"]
            delta = v0101_mae - v0011_mae
            delta_pct = (delta / v0011_mae) * 100
            improvements.append({
                "target": row["target"],
                "feature_set": row["feature_set"],
                "model": row["model"],
                "v0011_mae": v0011_mae,
                "v0101_mae": v0101_mae,
                "delta": delta,
                "delta_pct": delta_pct,
            })
            status = "✓" if delta < 0 else "⚠" if delta > 0.2 else "~"
            print(f"  {status} {row['target'].upper()} / {row['feature_set']:<16s}  "
                  f"v0011={v0011_mae:.3f}  v0101={v0101_mae:.3f}  "
                  f"Δ={delta:+.3f} ({delta_pct:+.1f}%)")

    # Статистика по честным baselines
    if honest_file.exists():
        honest = pd.read_csv(honest_file)
        print("\n" + "─" * 70)
        print("ЧЕСТНЫЕ BASELINES (gap = raw_MAE - FirstWindow_MAE):")
        print("─" * 70)
        for _, row in honest.iterrows():
            gap = row["gap_raw_vs_fw"]
            verdict = row["verdict"]
            print(f"  {row['target'].upper()} / {row['feature_set']:<16s}  "
                  f"gap={gap:+.3f}  {verdict}")

    # Итоги
    print("\n" + "═" * 70)
    print("ИТОГИ:")
    if improvements:
        improvements_df = pd.DataFrame(improvements)
        better = (improvements_df["delta"] < 0).sum()
        worse = (improvements_df["delta"] > 0.2).sum()
        print(f"\n  Лучше чем v0011: {better}/{len(improvements)}")
        print(f"  Хуже чем v0011: {worse}/{len(improvements)}")
        if better > 0:
            print(f"\n  Лучшие улучшения:")
            top = improvements_df.nsmallest(3, "delta")
            for _, row in top.iterrows():
                print(f"    {row['target'].upper()} / {row['feature_set']:<16s}: "
                      f"{row['delta']:+.3f} мин ({row['delta_pct']:+.1f}%)")

    if report_file.exists():
        print(f"\n✅ Итоговый отчет: {report_file.name}")
        print(f"Результаты: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
