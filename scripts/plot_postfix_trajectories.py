"""Траектории предсказаний после ynorm+MSE-фикса.

Сравнивает на одном субъекте:
  Ridge (v0011, классика)  vs  LSTM (v0101, лучший NN после фикса)
для LT1 ENH и LT2 ENH.

Дополнительно — TCN (v0102), который НЕ восстановился после фикса
(остался коллапсированным), как контрпример.

Сетка 2×3:
  (LT1, медианный субъект v0101) → Ridge / LSTM / TCN
  (LT2, медианный субъект v0101) → Ridge / LSTM / TCN

Выход: results/final/plots/postfix_trajectories.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.common import DEFAULT_DATASET_DIR  # noqa: E402

OUT = ROOT / "results" / "final" / "plots" / "postfix_trajectories.png"


def idx_for(target: str) -> pd.DataFrame:
    df = pd.read_parquet(DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    if target == "lt2":
        d = df[df["window_valid_all_required"] == 1] \
              .dropna(subset=["target_time_to_lt2_center_sec"])
    else:
        d = df[df["target_time_to_lt1_usable"] == 1]
    return (d.sort_values(["subject_id", "window_start_sec"])
              [["subject_id", "window_start_sec"]].reset_index(drop=True))


def load_per_subject(version: str, target: str, fset: str = "EMG_NIRS_HRV"):
    yp = np.load(ROOT / "results" / version / f"ypred_{target}_{fset}.npy")
    yt = np.load(ROOT / "results" / version / f"ytrue_{target}_{fset}.npy")
    idx = idx_for(target)
    if len(idx) != len(yp):
        # Эвристика: дропаем самых коротких субъектов до совпадения размеров.
        sizes = idx.groupby("subject_id").size().sort_values()
        kept = set(sizes.index)
        for s in sizes.index:
            kept -= {s}
            if sizes.loc[list(kept)].sum() == len(yp):
                break
        idx = idx[idx.subject_id.isin(kept)].reset_index(drop=True)
    out = {}
    for s, sub in idx.groupby("subject_id"):
        m = idx.subject_id.values == s
        out[s] = {"yp": yp[m], "yt": yt[m], "ws": sub["window_start_sec"].values}
    return out


def pick_median_subject(per: dict) -> str:
    maes = {s: float(np.mean(np.abs(per[s]["yp"] - per[s]["yt"]))) / 60
            for s in per}
    med = float(np.median(list(maes.values())))
    return min(maes, key=lambda s: abs(maes[s] - med))


def plot_panel(ax, data: dict, subj: str, title: str) -> None:
    if subj not in data:
        ax.text(0.5, 0.5, f"нет данных по {subj}", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return
    skip = 3
    d = data[subj]
    t = d["ws"][skip:] / 60
    yp = d["yp"][skip:] / 60
    yt = d["yt"][skip:] / 60
    ax.plot(t, yt, color="black", lw=1.8, label="истина", zorder=3)
    ax.plot(t, yp, color="#d62728", lw=1.4, label="ŷ")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    mae = float(np.mean(np.abs(d["yp"] - d["yt"]))) / 60
    std_ratio = np.std(d["yp"]) / max(np.std(d["yt"]), 1e-6)
    ax.set_title(f"{title} · {subj}\nMAE={mae:.2f} мин · std(ŷ)/std(y)={std_ratio:.2f}",
                 fontsize=10)
    ax.grid(alpha=0.3)
    all_v = np.concatenate([yp, yt])
    pad = (all_v.max() - all_v.min()) * 0.08
    ax.set_ylim(all_v.min() - pad, all_v.max() + pad)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for row, tgt in enumerate(["lt1", "lt2"]):
        ridge = load_per_subject("v0011", tgt)
        lstm  = load_per_subject("v0101", tgt)
        tcn   = load_per_subject("v0102", tgt)
        # Берём медианный по MAE субъект из v0101 (лучшая NN); ему же показываем
        # Ridge и TCN — для прямого сравнения на одном субъекте.
        subj = pick_median_subject(lstm)
        plot_panel(axes[row, 0], ridge, subj,
                   f"{tgt.upper()} · Ridge (классика)")
        plot_panel(axes[row, 1], lstm,  subj,
                   f"{tgt.upper()} · LSTM (после фикса)")
        plot_panel(axes[row, 2], tcn,   subj,
                   f"{tgt.upper()} · TCN (после фикса, остался слабым)")

    for ax in axes[1]:
        ax.set_xlabel("Время записи, мин")
    for ax in axes[:, 0]:
        ax.set_ylabel("Время до порога, мин")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
