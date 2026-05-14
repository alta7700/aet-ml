"""Шаг 03: ablation `with_abs vs noabs` — paired Wilcoxon на per-subject MAE.

Гипотеза: модель, обученная только на трендах (без абсолютных значений
HRV/SmO₂/EMG-уровней), не хуже модели с абсолютными признаками. Если
подтвердится — это самостоятельный результат: «модель обобщается без
индивидуальных норм».

Источник: results/<version>/per_subject_full.csv (NN-версии содержат обе
variants; v0011 содержит только with_abs — для него ablation вырожден,
пропускаем).

Артефакты — в results/final/.

Запуск:
    PYTHONPATH=. uv run python scripts/final_abs_vs_noabs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FINAL = RESULTS / "final"

VERSIONS_NN = ["v0101", "v0102", "v0103", "v0104", "v0105",
               "v0106a", "v0106b", "v0106c", "v0107"]

EQUIVALENCE_MARGIN_MIN = 0.3  # |Δmedian| ниже которого считаем «эквивалентным» при p>0.05
ALPHA = 0.05


def load_all_per_subject() -> pd.DataFrame:
    rows = []
    for v in VERSIONS_NN:
        path = RESULTS / v / "per_subject_full.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["version"] = v
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def verdict(median_delta: float, p_value: float) -> str:
    """median_delta = MAE_noabs − MAE_with_abs. Положительное → noabs хуже."""
    if not np.isfinite(p_value):
        return "underpowered"
    if p_value >= ALPHA and abs(median_delta) < EQUIVALENCE_MARGIN_MIN:
        return "эквивалентно"
    if p_value < ALPHA:
        return "лучше with_abs" if median_delta > 0 else "лучше noabs"
    return "неопределённо"


def compute_paired(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (version, target, fset), g in df.groupby(["version", "target", "feature_set"]):
        wa = g[g["variant"] == "with_abs"][["subject_id", "mae_min"]].rename(columns={"mae_min": "mae_wa"})
        na = g[g["variant"] == "noabs"][["subject_id", "mae_min"]].rename(columns={"mae_min": "mae_na"})
        pair = wa.merge(na, on="subject_id").dropna()
        n = len(pair)
        if n < 6:
            rows.append({"version": version, "target": target, "feature_set": fset,
                         "n_paired": n, "verdict": "underpowered"})
            continue
        delta = pair["mae_na"] - pair["mae_wa"]
        try:
            stat = wilcoxon(pair["mae_wa"], pair["mae_na"], zero_method="wilcox",
                            alternative="two-sided")
            p = float(stat.pvalue)
            # Эффект-сайз r = Z/sqrt(N). scipy wilcoxon не возвращает Z напрямую;
            # аппроксимируем через mean rank — пропустим формальный r, оставим p.
        except Exception:
            p = float("nan")
        med_delta = float(delta.median())
        rows.append({
            "version": version,
            "target": target,
            "feature_set": fset,
            "n_paired": n,
            "median_mae_with_abs": round(float(pair["mae_wa"].median()), 4),
            "median_mae_noabs":    round(float(pair["mae_na"].median()), 4),
            "median_delta_mae":    round(med_delta, 4),
            "mean_delta_mae":      round(float(delta.mean()), 4),
            "wilcoxon_p":          round(p, 4) if np.isfinite(p) else float("nan"),
            "verdict":             verdict(med_delta, p),
        })
    return pd.DataFrame(rows)


def plot_summary(res: pd.DataFrame, path: Path) -> None:
    """Сводный график: scatter Δmedian × p, маркеры по таргетам, цвет по версии."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, tgt in zip(axes, ["lt1", "lt2"]):
        sub = res[(res["target"] == tgt) & res["wilcoxon_p"].notna()]
        for v, g in sub.groupby("version"):
            ax.scatter(g["median_delta_mae"], -np.log10(g["wilcoxon_p"]),
                       s=70, alpha=0.7, label=v)
            for _, r in g.iterrows():
                ax.annotate(r["feature_set"],
                            (r["median_delta_mae"], -np.log10(r["wilcoxon_p"])),
                            fontsize=7, xytext=(3, 3), textcoords="offset points",
                            alpha=0.6)
        ax.axhline(-np.log10(ALPHA), color="red", ls="--", lw=0.8,
                   label=f"p={ALPHA}")
        ax.axvline(0, color="black", lw=0.5)
        ax.axvspan(-EQUIVALENCE_MARGIN_MIN, EQUIVALENCE_MARGIN_MIN,
                   color="green", alpha=0.08, label=f"|Δ|<{EQUIVALENCE_MARGIN_MIN}")
        ax.set_xlabel("Δmedian MAE = MAE(noabs) − MAE(with_abs), мин")
        ax.set_ylabel("−log10(p) Wilcoxon")
        ax.set_title(f"{tgt.upper()}: ablation with_abs vs noabs")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    print("Загрузка per_subject_full…")
    df = load_all_per_subject()
    print(f"  {len(df)} строк, {df['version'].nunique()} версий")

    print("Paired Wilcoxon по (version × target × feature_set)…")
    res = compute_paired(df)
    res = res.sort_values(["target", "feature_set", "version"])
    out_path = FINAL / "11_abs_vs_noabs.csv"
    res.to_csv(out_path, index=False)
    print(f"  → {out_path.name} ({len(res)} строк)")

    # Сводка вердиктов
    print("\nСводка вердиктов:")
    print(res.groupby(["target", "verdict"]).size().unstack(fill_value=0).to_string())

    print("\nПо feature_set:")
    print(res.groupby(["feature_set", "verdict"]).size().unstack(fill_value=0).to_string())

    plot_path = FINAL / "plots" / "abs_vs_noabs_volcano.png"
    plot_summary(res, plot_path)
    print(f"\n→ {plot_path.name}")


if __name__ == "__main__":
    main()
