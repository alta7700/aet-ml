"""Траектория лучшей NN-модели (v0101 LSTM, ENH) — raw vs Kalman vs истина.

Две панели: LT1 и LT2. На каждой — медианный по MAE субъект, чтобы картина
отражала «типичное», а не «звёздное» поведение модели.

Выход: results/final/plots/best_nn_kalman_trajectories.png
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

OUT = ROOT / "results" / "final" / "plots" / "best_nn_kalman_trajectories.png"

VERSION = "v0101"
FSET = "EMG_NIRS_HRV"
SIGMA_OBS = 30.0  # типичный best_sigma_obs для семейства
SIGMA_P = 5.0
DT = 5.0  # шаг прогноза в секундах


def kalman_smooth(y_pred: np.ndarray, sigma_p: float, sigma_obs: float,
                  dt: float = DT) -> np.ndarray:
    """Одномерный фильтр Калмана: модель x[t] = x[t−1] − dt."""
    x = float(y_pred[0]); p = sigma_obs ** 2
    out = np.empty(len(y_pred))
    for i in range(len(y_pred)):
        x -= dt; p += sigma_p ** 2
        k = p / (p + sigma_obs ** 2)
        x = x + k * (y_pred[i] - x)
        p = (1 - k) * p
        out[i] = x
    return out


def idx_for(target: str) -> pd.DataFrame:
    df = pd.read_parquet(DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    if target == "lt2":
        d = df[df["window_valid_all_required"] == 1] \
              .dropna(subset=["target_time_to_lt2_center_sec"])
    else:
        d = df[df["target_time_to_lt1_usable"] == 1]
    return (d.sort_values(["subject_id", "window_start_sec"])
              [["subject_id", "window_start_sec"]].reset_index(drop=True))


def load_per_subject(target: str):
    yp = np.load(ROOT / "results" / VERSION / f"ypred_{target}_{FSET}.npy")
    yt = np.load(ROOT / "results" / VERSION / f"ytrue_{target}_{FSET}.npy")
    idx = idx_for(target)
    if len(idx) != len(yp):
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
        raw  = yp[m]
        kal  = kalman_smooth(raw, SIGMA_P, SIGMA_OBS)
        out[s] = {"raw": raw, "kal": kal, "yt": yt[m],
                  "ws": sub["window_start_sec"].values}
    return out


def pick_median_subject(per: dict) -> str:
    maes = {s: float(np.mean(np.abs(per[s]["raw"] - per[s]["yt"]))) / 60
            for s in per}
    med = float(np.median(list(maes.values())))
    return min(maes, key=lambda s: abs(maes[s] - med))


def plot_panel(ax, d: dict, subj: str, target: str) -> None:
    skip = 3
    t  = d["ws"][skip:] / 60
    yt = d["yt"][skip:]  / 60
    yr = d["raw"][skip:] / 60
    yk = d["kal"][skip:] / 60
    ax.plot(t, yt, color="black",   lw=1.8, label="истина", zorder=3)
    ax.plot(t, yr, color="#1f77b4", lw=1.0, alpha=0.7, label="raw")
    ax.plot(t, yk, color="#d62728", lw=1.6, label=f"Kalman (σ_obs={SIGMA_OBS:g})")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    mae_raw = float(np.mean(np.abs(d["raw"] - d["yt"]))) / 60
    mae_kal = float(np.mean(np.abs(d["kal"] - d["yt"]))) / 60
    ax.set_title(
        f"{target.upper()} ENH · v0101 LSTM · субъект {subj}\n"
        f"MAE raw = {mae_raw:.2f} мин · MAE Kalman = {mae_kal:.2f} мин",
        fontsize=11)
    ax.grid(alpha=0.3)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tgt in zip(axes, ["lt1", "lt2"]):
        per = load_per_subject(tgt)
        subj = pick_median_subject(per)
        plot_panel(ax, per[subj], subj, tgt)
        ax.set_xlabel("Время записи, мин")
    axes[0].set_ylabel("Время до порога, мин")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
