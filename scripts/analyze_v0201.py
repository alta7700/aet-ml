#!/usr/bin/env python3
"""v0201 — Анализ результатов: сравнение с v0011"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_results(results_dir="results/v0201"):
    """Загружает результаты v0201"""
    res_dir = Path(results_dir)

    results = {}

    # Основные результаты
    summary_csv = res_dir / "summary.csv"
    if summary_csv.exists():
        results['summary'] = pd.read_csv(summary_csv)
        results['per_subject'] = pd.read_csv(res_dir / "per_subject.csv") if (res_dir / "per_subject.csv").exists() else None

    # NPY файлы
    results['npy_files'] = list(res_dir.glob("**/*.npy"))

    return results

def analyze_v0201(results):
    """Анализирует результаты v0201 и сравнивает с v0011"""

    if 'summary' not in results:
        print("❌ summary.csv не найден")
        return

    df = results['summary']

    # v0011 референсные значения
    v0011_ref = {
        ("lt2", "EMG+NIRS+HRV"): 1.859,
        ("lt1", "EMG+NIRS+HRV"): 2.277,
    }

    print("\n" + "="*80)
    print("v0201 — АНАЛИЗ РЕЗУЛЬТАТОВ (Wavelet-TCN с нормализацией таргета)")
    print("="*80)

    print("\n📊 СРАВНЕНИЕ С v0011 (Базовая модель):\n")

    for (target, fset), ref_mae in v0011_ref.items():
        v0201_data = df[(df['target'] == target) & (df['feature_set'] == fset)]

        if v0201_data.empty:
            print(f"  {target.upper()} / {fset:<16s} — НЕТ РЕЗУЛЬТАТОВ")
            continue

        best = v0201_data[v0201_data['kalman_mae_min'] == v0201_data['kalman_mae_min'].min()].iloc[0]
        mae_v0201 = best['kalman_mae_min']
        delta = mae_v0201 - ref_mae
        pct = (delta / ref_mae * 100) if ref_mae != 0 else 0

        symbol = "🟢" if delta < 0 else "🔴"
        print(f"  {symbol} {target.upper()} / {fset:<16s}")
        print(f"     v0011:  {ref_mae:.3f} мин (ref)")
        print(f"     v0201:  {mae_v0201:.3f} мин")
        print(f"     Δ:      {delta:+.3f} мин ({pct:+.1f}%)")
        print(f"     σ_obs:  {best['best_sigma_obs']:.1f}")
        print(f"     R²:     {best['r2']:.3f}\n")

    print("\n📈 ВСЕ РЕЗУЛЬТАТЫ v0201:\n")

    for target in df['target'].unique():
        print(f"  {target.upper()}:")
        sub = df[df['target'] == target].sort_values('kalman_mae_min')
        for _, row in sub.iterrows():
            print(f"    {row['feature_set']:<16s} (variant={row['variant']:<7s}) "
                  f"MAE={row['kalman_mae_min']:.3f}  R²={row['r2']:.3f}")
        print()

    print("\n" + "="*80)

if __name__ == "__main__":
    import sys

    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/v0201"
    results = load_results(results_dir)
    analyze_v0201(results)
