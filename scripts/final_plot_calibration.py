"""Calibration scatter ypred × ytrue для четырёх финалистов в обоих вариантах.

Для каждой комбинации (target, feature_set, variant) загружает сохранённые
y_pred/y_true.npy из results/v0011/{noabs/,} и строит scatter ypred vs ytrue
с диагональю y=x. Цвет точек — плотность (через hexbin / KDE). 2×4 панели.

Источник: results/v0011/ypred_{lt1,lt2}_{HRV,EMG_NIRS,EMG_NIRS_HRV}.npy и аналог в noabs/.

Запуск:
    PYTHONPATH=. uv run python scripts/final_plot_calibration.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V0011 = ROOT / "results" / "v0011"
OUT = ROOT / "results" / "final" / "plots" / "calibration_finalists.png"

# (target, fset_tag, label_for_panel)
FINALISTS = [
    ("lt2", "HRV", "LT2 / HRV"),
    ("lt2", "EMG_NIRS_HRV", "LT2 / EMG+NIRS+HRV"),
    ("lt1", "EMG_NIRS", "LT1 / EMG+NIRS"),
    ("lt1", "EMG_NIRS_HRV", "LT1 / EMG+NIRS+HRV"),
]


def load_pair(variant: str, target: str, fset_tag: str):
    base = V0011 if variant == "with_abs" else V0011 / "noabs"
    yp = np.load(base / f"ypred_{target}_{fset_tag}.npy") / 60.0
    yt = np.load(base / f"ytrue_{target}_{fset_tag}.npy") / 60.0
    return yp, yt


def draw_panel(ax, yp: np.ndarray, yt: np.ndarray, title: str) -> None:
    lo = min(yp.min(), yt.min())
    hi = max(yp.max(), yt.max())
    # Hexbin с плотностью
    hb = ax.hexbin(yt, yp, gridsize=40, cmap="viridis",
                   mincnt=1, extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Истинное время до порога, мин")
    ax.set_ylabel("Предсказанное, мин")
    mae = float(np.abs(yp - yt).mean())
    ax.set_title(f"{title}\nMAE = {mae:.2f} мин", fontsize=10)
    ax.grid(alpha=0.3)
    return hb


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    last_hb = None

    for col, (target, fset_tag, label) in enumerate(FINALISTS):
        for row, variant in enumerate(["with_abs", "noabs"]):
            ax = axes[row, col]
            try:
                yp, yt = load_pair(variant, target, fset_tag)
            except FileNotFoundError as e:
                ax.text(0.5, 0.5, f"missing\n{e}", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            title = f"{label}\n({variant})"
            last_hb = draw_panel(ax, yp, yt, title)

    if last_hb is not None:
        cb = fig.colorbar(last_hb, ax=axes.ravel().tolist(),
                          shrink=0.7, pad=0.02)
        cb.set_label("плотность окон")

    fig.suptitle("Калибровка предсказаний (window-level): четыре финалиста × {with_abs, noabs}",
                 fontsize=12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
