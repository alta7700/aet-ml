"""Боксплот kalman_gain по семействам архитектур, отдельно LT1 и LT2.

Назначение: иллюстрация к разделу 3.1 — фильтр Калмана не даёт
систематического выигрыша; знак эффекта зависит от семейства и порога.

Источник: results/final/01_ranking_wide.csv
Выход:    results/final/plots/kalman_gain_by_family.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "final" / "01_ranking_wide.csv"
OUT_LT1 = ROOT / "results" / "final" / "plots" / "kalman_gain_by_family_lt1.png"
OUT_LT2 = ROOT / "results" / "final" / "plots" / "kalman_gain_by_family_lt2.png"

# Порядок семейств — от классики к ансамблю.
FAMILY_ORDER = ["linear", "lstm", "tcn", "wavelet", "ensemble"]
FAMILY_LABEL = {
    "linear":   "Классика\n(Ridge/EN/SVR/GBM)",
    "lstm":     "LSTM",
    "tcn":      "TCN",
    "wavelet":  "Wavelet",
    "ensemble": "Ансамбль",
}


def plot_one(df: pd.DataFrame, target: str, out: Path) -> None:
    sub = df[df["target"] == target]
    data = [sub.loc[sub["family"] == f, "kalman_gain"].values
            for f in FAMILY_ORDER]
    positions = list(range(1, len(FAMILY_ORDER) + 1))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.4),
        boxprops=dict(facecolor="#cfe2f3", edgecolor="#444"),
        whiskerprops=dict(color="#444"),
        capprops=dict(color="#444"),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#888",
                        markeredgecolor="none", alpha=0.6),
    )
    # Точки конфигураций поверх — для прозрачности при малом n.
    for x, d in zip(positions, data):
        if len(d) == 0:
            continue
        jitter = 0.08
        xs = [x + (i / max(len(d) - 1, 1) - 0.5) * jitter * 2 for i in range(len(d))]
        ax.scatter(xs, d, s=14, color="#1f4e79", alpha=0.7, zorder=3)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILY_ORDER], fontsize=9)
    ax.set_ylabel("MAE_raw − MAE_kalman, мин")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved: {out}")


def main() -> None:
    df = pd.read_csv(SRC).dropna(subset=["kalman_gain", "family"])
    plot_one(df, "lt1", OUT_LT1)
    plot_one(df, "lt2", OUT_LT2)


if __name__ == "__main__":
    main()
