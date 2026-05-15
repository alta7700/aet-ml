"""Per-subject scatter MAE: with_abs vs noabs для v0011 по каждой модальности.

5 модальностей × 2 таргета = 10 панелей. По каждой — 18 точек (испытуемые);
ось X — MAE с абсолютными признаками, ось Y — без них. Диагональ y=x —
индикатор индивидуального изменения. Точка выше диагонали = ухудшение при
удалении абсолютных признаков, ниже — улучшение.

Источник: results/v0011/per_subject_full.csv (создан Блоком 1.1).

Запуск:
    PYTHONPATH=. uv run python scripts/final_plot_abs_noabs_per_subject.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / "results" / "v0011" / "per_subject_full.csv"
OUT = ROOT / "results" / "final" / "plots" / "abs_vs_noabs_per_subject.png"

FSETS = ["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"]
TARGETS = [("lt1", "LT1"), ("lt2", "LT2")]


def main() -> None:
    df = pd.read_csv(PS)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=False, sharey=False)
    for r, (target_key, target_lbl) in enumerate(TARGETS):
        for c, fset in enumerate(FSETS):
            ax = axes[r, c]
            sub = df[(df["target"] == target_key) & (df["feature_set"] == fset)]
            wa = sub[sub["variant"] == "with_abs"].set_index("subject_id")["mae_min"]
            na = sub[sub["variant"] == "noabs"].set_index("subject_id")["mae_min"]
            common = wa.index.intersection(na.index)
            wa, na = wa.loc[common], na.loc[common]

            ax.scatter(wa, na, s=50, alpha=0.7, color="C0")
            lo = min(wa.min(), na.min())
            hi = max(wa.max(), na.max())
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    "k--", lw=1, alpha=0.7)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("MAE with_abs, мин")
            ax.set_ylabel("MAE noabs, мин")
            ax.set_title(f"{target_lbl} / {fset}", fontsize=10)
            # Подсветка: сколько испытуемых ухудшились/улучшились
            n_worse = int((na > wa).sum())
            n_better = int((na < wa).sum())
            ax.text(0.03, 0.97, f"noabs хуже: {n_worse}\nnoabs лучше: {n_better}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8, edgecolor="grey"))
            ax.grid(alpha=0.3)

    fig.suptitle("Per-subject MAE: with_abs vs noabs (v0011, n=18)\n"
                 "Точка выше диагонали = удаление абсолютных признаков ухудшило предсказание",
                 fontsize=12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
