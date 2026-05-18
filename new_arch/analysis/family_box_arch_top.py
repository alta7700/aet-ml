"""Боксплоты по семействам на основе per-architecture champions.

Для каждого (target, family, architecture_id) берём модель с минимальным
lt_mae_median_policy_mean. Затем для каждого (target, family, subject_id)
берём минимальную abs_lt_err_median_sec по архитектурам семейства —
"лучшее, на что способно семейство для этого субъекта".

Сравнение строится на этих per-subject векторах: парный Wilcoxon + FDR-BH.
Боксплоты визуализируют именно эти 18-точечные распределения.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


_FAMILIES = ["Lin", "LSTM", "TCN"]
_COLORS = {"Lin": "#4c72b0", "LSTM": "#dd8452", "TCN": "#55a868"}


def _per_subject_min(target: str) -> pd.DataFrame:
    summary = pd.read_parquet("analysis_out/cache/model_summary.parquet")
    lt = pd.read_parquet("analysis_out/cache/lt_point_metrics.parquet")

    summary = summary[summary["target"] == target].copy()
    summary["rank"] = summary.groupby(["family", "architecture_id"])\
        ["lt_mae_median_policy_mean"].rank(method="min")
    champs = summary[summary["rank"] == 1]

    lt_c = lt[(lt["target"] == target)
              & lt["model_id"].isin(champs["model_id"])].copy()
    fam_map = champs.set_index("model_id")["family"]
    lt_c["family"] = lt_c["model_id"].map(fam_map)

    agg = lt_c.groupby(["family", "subject_id"])\
        ["abs_lt_err_median_sec"].min().reset_index(name="lt_err_min_sec")
    return agg


def _paired_tests(piv: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float]]:
    out = {}
    raw = []
    keys = []
    for a, b in combinations(_FAMILIES, 2):
        x = piv[a].to_numpy()
        y = piv[b].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        _, p = wilcoxon(x[mask], y[mask])
        med_d = float(np.median(x[mask] - y[mask]))
        out[(a, b)] = (p, med_d, float(mask.sum()))
        raw.append(p)
        keys.append((a, b))
    _, padj, _, _ = multipletests(raw, alpha=0.05, method="fdr_bh")
    return {k: (out[k][0], padj[i], out[k][1], out[k][2]) for i, k in enumerate(keys)}


def _plot(agg: pd.DataFrame, target: str, out_path: Path) -> Path:
    piv = agg.pivot(index="subject_id", columns="family", values="lt_err_min_sec")
    piv = piv[_FAMILIES]
    tests = _paired_tests(piv)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    data = [piv[f].dropna().to_numpy() for f in _FAMILIES]
    bp = ax.boxplot(data, positions=range(len(_FAMILIES)), widths=0.55,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    for patch, fam in zip(bp["boxes"], _FAMILIES):
        patch.set_facecolor(_COLORS[fam])
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.4)

    # точки субъектов поверх
    rng = np.random.default_rng(0)
    for i, vec in enumerate(data):
        x = rng.normal(loc=i, scale=0.05, size=len(vec))
        ax.scatter(x, vec, s=14, color="black", alpha=0.55, zorder=3)

    ax.set_xticks(range(len(_FAMILIES)))
    ax.set_xticklabels(_FAMILIES)
    ax.set_ylabel("|ошибка предсказания LT| [сек]\n(min по архитектурам семейства)")
    ax.set_title(f"{target.upper()}: семейства по архитектурным чемпионам", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)

    # подписи p-value сверху
    y_max = float(np.nanmax(piv.to_numpy()))
    span = y_max * 0.08
    base_y = y_max + span
    bracket_order = [("Lin", "LSTM"), ("LSTM", "TCN"), ("Lin", "TCN")]
    for k, pair in enumerate(bracket_order):
        a, b = pair
        i, j = _FAMILIES.index(a), _FAMILIES.index(b)
        y = base_y + k * span * 1.1
        ax.plot([i, i, j, j], [y - span * 0.15, y, y, y - span * 0.15],
                color="black", lw=0.8)
        p_raw, p_adj, d, n = tests[pair]
        star = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        ax.text((i + j) / 2, y + span * 0.12,
                f"p_adj={p_adj:.3f} {star}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(top=base_y + span * len(bracket_order) * 1.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    out_dir = Path("analysis_out/figures/box")
    for t in ("lt1", "lt2"):
        agg = _per_subject_min(t)
        out = out_dir / f"box_family_arch_top_{t}.png"
        _plot(agg, t, out)
        print("→", out)


if __name__ == "__main__":
    main()
