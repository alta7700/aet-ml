"""Сравнение траекторий v0106b vs v0201 на одной фигуре.

Показывает, действительно ли нормировка таргета изменила поведение
предсказаний Wavelet-TCN качественно, или сдвиг MAE 4,57 → 4,14 мин
остаётся в пределах того же «почти-плоского» режима.

Сетка 2×2: (LT1 ENH, LT2 ENH) × (медианный субъект, лучший субъект v0201).
Слева/справа: v0106b vs v0201, наложены raw-предсказания + истина.

Выход: results/final/plots/v0201_vs_v0106b_trajectories.png
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

OUT = ROOT / "results" / "final" / "plots" / "v0201_vs_v0106b_trajectories.png"


def idx_for(target: str) -> pd.DataFrame:
    df = pd.read_parquet(DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    if target == "lt2":
        d = df[df["window_valid_all_required"] == 1] \
              .dropna(subset=["target_time_to_lt2_center_sec"])
    else:
        d = df[df["target_time_to_lt1_usable"] == 1]
    return (d.sort_values(["subject_id", "window_start_sec"])
              [["subject_id", "window_start_sec"]].reset_index(drop=True))


def load_per_subject(version: str, target: str, fset: str = "EMG_NIRS_HRV"
                     ) -> dict[str, dict[str, np.ndarray]]:
    yp = np.load(ROOT / "results" / version / f"ypred_{target}_{fset}.npy")
    yt = np.load(ROOT / "results" / version / f"ytrue_{target}_{fset}.npy")
    idx = idx_for(target)
    if len(idx) != len(yp):
        ps_path = ROOT / "results" / version / "per_subject.csv"
        ps = pd.read_csv(ps_path)
        ps = ps[(ps.target == target) & (ps.feature_set == fset.replace("_", "+"))]
        idx = idx[idx.subject_id.isin(ps.subject_id.unique())].reset_index(drop=True)
    out = {}
    for s, sub in idx.groupby("subject_id"):
        m = idx.subject_id.values == s
        out[s] = {"yp": yp[m], "yt": yt[m], "ws": sub["window_start_sec"].values}
    return out


def pick_subjects(p_new: dict, p_ref: dict) -> tuple[str, str]:
    """Возвращает (медианный по MAE v0201, лучший по MAE v0201)."""
    maes = {s: float(np.mean(np.abs(p_new[s]["yp"] - p_new[s]["yt"]))) / 60
            for s in p_new}
    med_v = float(np.median(list(maes.values())))
    median_subj = min(maes, key=lambda s: abs(maes[s] - med_v))
    best_subj = min(maes, key=maes.get)
    return median_subj, best_subj


def plot_panel(ax, p_new: dict, p_ref: dict, subj: str, title: str) -> None:
    skip = 3
    for src, color, label in [("ref", "#1f77b4", "v0106b"),
                              ("new", "#d62728", "v0201")]:
        d = (p_new if src == "new" else p_ref)[subj]
        t = d["ws"][skip:] / 60
        ax.plot(t, d["yp"][skip:] / 60, color=color, lw=1.4, alpha=0.9, label=label)
    d = p_new[subj]
    ax.plot(d["ws"][skip:] / 60, d["yt"][skip:] / 60,
            color="black", lw=1.8, label="истина", zorder=3)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    mae_new = float(np.mean(np.abs(p_new[subj]["yp"] - p_new[subj]["yt"]))) / 60
    mae_ref = float(np.mean(np.abs(p_ref[subj]["yp"] - p_ref[subj]["yt"]))) / 60
    std_new = np.std(p_new[subj]["yp"]) / max(np.std(p_new[subj]["yt"]), 1e-6)
    std_ref = np.std(p_ref[subj]["yp"]) / max(np.std(p_ref[subj]["yt"]), 1e-6)
    ax.set_title(
        f"{title}, субъект {subj}\n"
        f"MAE: v0106b={mae_ref:.2f}, v0201={mae_new:.2f} мин · "
        f"std(ŷ)/std(y): v0106b={std_ref:.2f}, v0201={std_new:.2f}",
        fontsize=10,
    )
    ax.grid(alpha=0.3)


def main() -> None:
    cells = []
    for tgt in ["lt1", "lt2"]:
        new = load_per_subject("v0201", tgt)
        ref = load_per_subject("v0106b", tgt)
        common = sorted(set(new) & set(ref))
        new = {s: new[s] for s in common}
        ref = {s: ref[s] for s in common}
        med, best = pick_subjects(new, ref)
        cells.append((tgt, "медианный по MAE (v0201)", med, new, ref))
        cells.append((tgt, "лучший по MAE (v0201)", best, new, ref))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    for ax, (tgt, label, subj, new, ref) in zip(axes, cells):
        plot_panel(ax, new, ref, subj, f"{tgt.upper()} ENH · {label}")

    for ax in axes[2:]:
        ax.set_xlabel("Время записи, мин")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Время до порога, мин")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
