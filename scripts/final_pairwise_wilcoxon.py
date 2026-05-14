"""Шаг 04: парные сравнения моделей — Wilcoxon + Holm.

Внутри каждой группы `(target, feature_set, variant)` сравнивает все версии
попарно по per-subject MAE (paired Wilcoxon на пересечении субъектов),
применяет поправку Holm на множественные сравнения.

Дополнительно: кросс-модальное сравнение чемпионов
(v0011 HRV vs v0011 EMG+NIRS+HRV для lt2; v0011 EMG+NIRS vs EMG+NIRS+HRV для lt1).

Артефакты — в results/final/.

Запуск:
    PYTHONPATH=. uv run python scripts/final_pairwise_wilcoxon.py
"""

from __future__ import annotations

from itertools import combinations
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

ALL_VERSIONS = ["v0011", "v0101", "v0102", "v0103", "v0104", "v0105",
                "v0106a", "v0106b", "v0106c", "v0107"]
MIN_PAIRED = 12
ALPHA = 0.05


def load_all() -> pd.DataFrame:
    rows = []
    for v in ALL_VERSIONS:
        path = RESULTS / v / "per_subject_full.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["version"] = v
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def holm_correction(pvals: list[float]) -> list[float]:
    """Поправка Холма. Возвращает скорректированные p в исходном порядке."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def pairwise_in_group(g: pd.DataFrame, group_key: tuple) -> list[dict]:
    """Все пары версий внутри группы (target, feature_set, variant)."""
    target, fset, variant = group_key
    versions = sorted(g["version"].unique())
    # subject → mae по версии
    pivots = {v: g[g["version"] == v].set_index("subject_id")["mae_min"]
              for v in versions}

    raw = []
    for va, vb in combinations(versions, 2):
        sa, sb = pivots[va], pivots[vb]
        common = sa.index.intersection(sb.index)
        a = sa.loc[common].dropna()
        b = sb.loc[common].reindex(a.index).dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        n = len(a)
        if n < MIN_PAIRED:
            raw.append({"target": target, "feature_set": fset, "variant": variant,
                        "model_a": va, "model_b": vb, "n_paired": n,
                        "median_diff": float("nan"), "p": float("nan"),
                        "underpowered": True})
            continue
        try:
            p = float(wilcoxon(a.values, b.values, alternative="two-sided").pvalue)
        except Exception:
            p = float("nan")
        raw.append({"target": target, "feature_set": fset, "variant": variant,
                    "model_a": va, "model_b": vb, "n_paired": n,
                    "median_a": round(float(a.median()), 4),
                    "median_b": round(float(b.median()), 4),
                    "median_diff": round(float((a - b).median()), 4),
                    "p": p, "underpowered": False})

    # Holm внутри группы (только для не-underpowered)
    valid = [r for r in raw if not r["underpowered"] and np.isfinite(r["p"])]
    if valid:
        adj = holm_correction([r["p"] for r in valid])
        for r, pa in zip(valid, adj):
            r["p_holm"] = round(pa, 4)
            r["p"] = round(r["p"], 4)
            if pa < ALPHA:
                r["winner"] = r["model_a"] if r["median_diff"] < 0 else r["model_b"]
            else:
                r["winner"] = "n.s."
    for r in raw:
        r.setdefault("p_holm", float("nan"))
        r.setdefault("winner", "underpowered" if r["underpowered"] else "n.s.")
    return raw


def win_counts(res: pd.DataFrame) -> pd.DataFrame:
    """Сколько раз каждая версия значимо победила/проиграла в каждой группе."""
    rows = []
    for (target, fset, variant), g in res.groupby(["target", "feature_set", "variant"]):
        sig = g[(g["winner"] != "n.s.") & (g["winner"] != "underpowered")]
        all_versions = sorted(set(g["model_a"]) | set(g["model_b"]))
        for v in all_versions:
            wins = int((sig["winner"] == v).sum())
            losses = int(((sig["model_a"] == v) | (sig["model_b"] == v)).sum()) - wins
            rows.append({"target": target, "feature_set": fset, "variant": variant,
                         "version": v, "sig_wins": wins, "sig_losses": losses,
                         "net": wins - losses})
    return pd.DataFrame(rows).sort_values(["target", "feature_set", "variant", "net"],
                                          ascending=[True, True, True, False])


def plot_heatmaps(res: pd.DataFrame, out_dir: Path) -> None:
    """Для каждой группы — heatmap p_holm."""
    for (target, fset, variant), g in res.groupby(["target", "feature_set", "variant"]):
        versions = sorted(set(g["model_a"]) | set(g["model_b"]))
        n = len(versions)
        mat = np.full((n, n), np.nan)
        idx = {v: i for i, v in enumerate(versions)}
        for _, r in g.iterrows():
            if np.isfinite(r.get("p_holm", np.nan)):
                i, j = idx[r["model_a"]], idx[r["model_b"]]
                mat[i, j] = r["p_holm"]
                mat[j, i] = r["p_holm"]
        fig, ax = plt.subplots(figsize=(1 + n * 0.7, 1 + n * 0.6))
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=0.1)
        ax.set_xticks(range(n)); ax.set_xticklabels(versions, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(versions, fontsize=7)
        for i in range(n):
            for j in range(n):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if mat[i, j] < 0.03 else "black")
        ax.set_title(f"{target} / {fset} / {variant}\np_holm (зелёное = значимо)", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        tag = f"{target}_{fset.replace('+', '_')}_{variant}"
        fig.savefig(out_dir / f"pairwise_{tag}.png", dpi=110)
        plt.close(fig)


def champion_cross_modality(df: pd.DataFrame) -> pd.DataFrame:
    """Сравнение чемпиона с альтернативной модальностью внутри v0011."""
    comparisons = [
        ("lt2", "with_abs", "HRV", "EMG+NIRS+HRV", "v0011"),
        ("lt1", "with_abs", "EMG+NIRS", "EMG+NIRS+HRV", "v0011"),
        ("lt2", "with_abs", "HRV", "EMG+NIRS", "v0011"),
    ]
    rows = []
    for target, variant, fa, fb, version in comparisons:
        sub = df[(df["version"] == version) & (df["target"] == target)
                 & (df["variant"] == variant)]
        a = sub[sub["feature_set"] == fa].set_index("subject_id")["mae_min"]
        b = sub[sub["feature_set"] == fb].set_index("subject_id")["mae_min"]
        common = a.index.intersection(b.index)
        a, b = a.loc[common].dropna(), b.loc[common].dropna()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        if len(a) < MIN_PAIRED:
            continue
        p = float(wilcoxon(a.values, b.values, alternative="two-sided").pvalue)
        rows.append({
            "version": version, "target": target, "variant": variant,
            "fset_a": fa, "fset_b": fb, "n_paired": len(a),
            "median_a": round(float(a.median()), 4),
            "median_b": round(float(b.median()), 4),
            "median_diff": round(float((a - b).median()), 4),
            "p": round(p, 4),
            "verdict": (fa if (a - b).median() < 0 else fb) if p < ALPHA else "n.s.",
        })
    return pd.DataFrame(rows)


def main() -> None:
    plots_dir = FINAL / "plots" / "pairwise"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Загрузка per_subject_full…")
    df = load_all()
    print(f"  {len(df)} строк, {df['version'].nunique()} версий")

    print("Парные сравнения по группам (target × feature_set × variant)…")
    all_raw = []
    for key, g in df.groupby(["target", "feature_set", "variant"]):
        if g["version"].nunique() < 2:
            continue
        all_raw.extend(pairwise_in_group(g, key))
    res = pd.DataFrame(all_raw)
    res = res.sort_values(["target", "feature_set", "variant", "p_holm"])
    out_path = FINAL / "12_pairwise_wilcoxon.csv"
    res.to_csv(out_path, index=False)
    print(f"  → {out_path.name} ({len(res)} пар)")

    sig = res[(res["winner"] != "n.s.") & (res["winner"] != "underpowered")]
    print(f"  Значимых различий (p_holm < {ALPHA}): {len(sig)}/{len(res)}")

    print("\nWin-counts по группам…")
    wc = win_counts(res)
    wc_path = FINAL / "13_pairwise_win_counts.csv"
    wc.to_csv(wc_path, index=False)
    print(f"  → {wc_path.name}")

    print("Кросс-модальное сравнение чемпионов…")
    champ = champion_cross_modality(df)
    champ_path = FINAL / "14_champion_cross_modality.csv"
    champ.to_csv(champ_path, index=False)
    print(f"  → {champ_path.name}")
    print(champ.to_string(index=False))

    print("\nHeatmaps…")
    plot_heatmaps(res, plots_dir)
    print(f"  → {plots_dir}")


if __name__ == "__main__":
    main()
