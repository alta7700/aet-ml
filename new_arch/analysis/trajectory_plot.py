"""Слайд 4: траектории предсказаний лучшей модели по всем участникам.

Для выбранной модели строит сетку 3×6 (или 4×5 — по числу субъектов) панелей:
- ось X: время теста в минутах (sample_end_sec / 60);
- ось Y: предсказанное и истинное время до порога (time_to_lt) в секундах;
- горизонтальная линия `y=0` — момент достижения порога;
- вертикальная линия — момент истинного LT (sample_end_sec, на котором y_true=0).

Подписи на русском, кроме слова "LT".
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds


_BLUE = "#1f77b4"
_RED = "#d62728"
_GREY = "#666666"


def _load_predictions(cache_dir: Path, model_id: str,
                      architecture_id: str) -> pd.DataFrame:
    d = ds.dataset(str(cache_dir / "predictions_selected"),
                   format="parquet", partitioning="hive")
    t = d.to_table(filter=(ds.field("architecture_id") == architecture_id)).to_pandas()
    t = t[t["model_id"] == model_id].copy()
    t["time_min"] = t["sample_end_sec"] / 60.0
    return t.sort_values(["subject_id", "sample_end_sec"], kind="stable")


def plot_trajectories(df: pd.DataFrame, *,
                      title: str,
                      out_path: Path,
                      ncols: int = 6) -> Path:
    subjects = sorted(df["subject_id"].unique())
    n = len(subjects)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows),
                             sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    for idx, subj in enumerate(subjects):
        ax = axes[idx // ncols][idx % ncols]
        g = df[df["subject_id"] == subj]
        t = g["time_min"].to_numpy()
        # обе оси в минутах
        y_true = g["y_true"].to_numpy() / 60.0
        y_pred = g["y_pred"].to_numpy() / 60.0

        ax.plot(t, y_true, color=_BLUE, lw=1.6, label="истинное")
        ax.plot(t, y_pred, color=_RED, lw=1.0, alpha=0.85, label="предсказание")
        ax.axhline(0.0, color=_GREY, ls=":", lw=0.8)

        # момент истинного LT — где y_true пересекает 0 (в минутах)
        zero_idx = int(np.argmin(np.abs(y_true)))
        if abs(y_true[zero_idx]) < 5.0 / 60.0:
            ax.axvline(t[zero_idx], color=_BLUE, ls="--", lw=0.7, alpha=0.6)

        # шкала Y динамическая по данным субъекта (без обрезки хвостов)

        ax.set_title(subj, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, lw=0.4)

    # выключаем пустые
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # общая подпись осей
    fig.supxlabel("время теста [мин]", fontsize=10)
    fig.supylabel("время до LT [мин]", fontsize=10)
    # fig.suptitle(title, fontsize=11)

    # одна общая легенда сверху справа
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=9, frameon=False)

    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    cache = Path("analysis_out/cache")
    out_dir = Path("analysis_out/figures/trajectories")

    # LT1 winner — LSTM3
    df1 = _load_predictions(cache, "LSTM3_31f2b090", "LSTM3")
    plot_trajectories(
        df1,
        title="Траектории предсказаний LSTM3 (ЭМГ+NIRS+ВСР) — LT1, все участники",
        out_path=out_dir / "trajectories_lt1_LSTM3.png",
    )
    print("→", out_dir / "trajectories_lt1_LSTM3.png")

    # LT2 winner — LSTM5 (EMG+NIRS+HRV+|EMG|)
    df2 = _load_predictions(cache, "LSTM5_0470318e", "LSTM5")
    plot_trajectories(
        df2,
        title="Траектории предсказаний LSTM5 (ЭМГ+NIRS+ВСР+|ЭМГ|) — LT2, все участники",
        out_path=out_dir / "trajectories_lt2_LSTM5.png",
    )
    print("→", out_dir / "trajectories_lt2_LSTM5.png")


if __name__ == "__main__":
    main()
