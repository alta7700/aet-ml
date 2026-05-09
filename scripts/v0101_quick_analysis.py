"""Быстрый анализ v0101 vs v0011 после завершения обучения."""

from pathlib import Path
import pandas as pd

OUT_DIR = Path("results/v0101")

def main():
    print("\n" + "="*70)
    print("v0101 БЫСТРЫЙ АНАЛИЗ")
    print("="*70)

    # Загрузить результаты
    summary_file = OUT_DIR / "summary.csv"
    best_file = OUT_DIR / "best_per_set.csv"
    honest_file = OUT_DIR / "honest_baselines.csv"

    if not summary_file.exists():
        print("\n❌ summary.csv не найден")
        return

    summary = pd.read_csv(summary_file)
    print(f"\n✅ Загружено {len(summary)} записей")

    # Лучший per набор
    best = summary.sort_values("kalman_mae_min").groupby(
        ["feature_set", "target"]).first().reset_index()

    print("\n" + "-"*70)
    print("ЛУЧШИЕ МОДЕЛИ V0101")
    print("-"*70)
    for _, row in best.iterrows():
        print(f"  {row['target'].upper()} / {row['feature_set']:<16s}  "
              f"{row['model']:<30s}  MAE={row['kalman_mae_min']:.3f} мин")

    # Сравнение с v0011
    v0011_ref = {
        ("lt2", "EMG"): 3.198,
        ("lt2", "NIRS"): 3.938,
        ("lt2", "EMG+NIRS"): 3.117,
        ("lt2", "EMG+NIRS+HRV"): 1.859,
        ("lt1", "EMG"): 2.896,
        ("lt1", "NIRS"): 3.842,
        ("lt1", "EMG+NIRS"): 2.750,
        ("lt1", "EMG+NIRS+HRV"): 2.277,
    }

    print("\n" + "-"*70)
    print("СРАВНЕНИЕ С V0011")
    print("-"*70)
    improvements = []
    for _, row in best.iterrows():
        key = (row["target"], row["feature_set"])
        if key in v0011_ref:
            v0011_mae = v0011_ref[key]
            v0101_mae = row["kalman_mae_min"]
            delta = v0101_mae - v0011_mae
            delta_pct = (delta / v0011_mae) * 100

            if delta < -0.2:
                emoji = "✓"
            elif delta > 0.2:
                emoji = "⚠"
            else:
                emoji = "~"

            print(f"  {emoji} {row['target'].upper()} / {row['feature_set']:<16s}  "
                  f"v0011={v0011_mae:.3f}  v0101={v0101_mae:.3f}  "
                  f"Δ={delta:+.3f} ({delta_pct:+.1f}%)")
            improvements.append(delta)

    if improvements:
        better = sum(1 for d in improvements if d < -0.2)
        worse = sum(1 for d in improvements if d > 0.2)
        neutral = len(improvements) - better - worse
        print(f"\n  Сводка: {better} лучше, {neutral} нейтрально, {worse} хуже чем v0011")

    # Честные baselines
    if honest_file.exists():
        honest = pd.read_csv(honest_file)
        print("\n" + "-"*70)
        print("ЧЕСТНЫЕ BASELINES (gap = raw_MAE - FirstWindow_MAE)")
        print("-"*70)
        for _, row in honest.iterrows():
            print(f"  {row['target'].upper()} / {row['feature_set']:<16s}  "
                  f"gap={row['gap_raw_vs_fw']:+.3f}  {row['verdict']}")

    print("\n✅ Результаты в: " + str(OUT_DIR.resolve()))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
