"""Шаг 02a: распределение per-subject ошибок у топ-моделей.

Берёт топ-3 модели на target из шага 02 и для них:
- собирает MAE per-subject + ковариаты (HR_baseline, time_to_lt, age, bmi);
- строит гистограммы, scatter с ковариатами;
- сравнивает trained/untrained (порог time_to_lt2 > 13 мин);
- считает кросс-target корреляцию.

Артефакты — в results/final/eda/per_subject_dist/.

Запуск:
    PYTHONPATH=. uv run python scripts/final_per_subject_dist.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EDA = RESULTS / "final" / "eda" / "per_subject_dist"
PLOTS = EDA / "plots"

# Топ-модели для анализа: (version, variant, feature_set, target, short_id).
TOP_MODELS = [
    # lt2
    ("v0011", "with_abs", "HRV",          "lt2", "v0011_HRV"),
    ("v0011", "with_abs", "EMG+NIRS+HRV", "lt2", "v0011_ENH"),
    ("v0107", "with_abs", "EMG+NIRS+HRV", "lt2", "v0107_ENH"),
    # lt1
    ("v0011", "with_abs", "EMG+NIRS",     "lt1", "v0011_EN_lt1"),
    ("v0011", "with_abs", "EMG+NIRS+HRV", "lt1", "v0011_ENH_lt1"),
    ("v0107", "noabs",    "EMG+NIRS+HRV", "lt1", "v0107_ENH_lt1"),
]

TRAINED_THRESHOLD_MIN = 13.0


def load_subject_covariates() -> pd.DataFrame:
    """Собирает: subject_id, hr_baseline, age, bmi, body_fat_pct, time_to_lt1, time_to_lt2."""
    sp = pd.read_parquet(ROOT / "dataset" / "session_params.parquet")
    subs = pd.read_parquet(ROOT / "dataset" / "subjects.parquet")
    wn = pd.read_parquet(ROOT / "dataset" / "windows.parquet")
    tg = pd.read_parquet(ROOT / "dataset" / "targets.parquet")

    m = wn[["window_id", "subject_id", "window_end_sec", "elapsed_sec"]].merge(
        tg[["window_id", "target_time_to_lt1_pchip_sec",
            "target_time_to_lt2_center_sec"]],
        on="window_id",
    )
    first = m.sort_values(["subject_id", "window_end_sec"]).groupby("subject_id").first().reset_index()
    first["time_to_lt1_min"] = (first["target_time_to_lt1_pchip_sec"]
                                + first["elapsed_sec"]) / 60.0
    first["time_to_lt2_min"] = (first["target_time_to_lt2_center_sec"]
                                + first["elapsed_sec"]) / 60.0
    times = first[["subject_id", "time_to_lt1_min", "time_to_lt2_min"]]

    cov = (subs[["subject_id", "age", "sex", "bmi", "body_fat_pct",
                 "phase_angle", "weight", "height"]]
           .merge(sp[["subject_id", "hrv_hr_baseline_bpm",
                      "hrv_rmssd_baseline_ms"]], on="subject_id")
           .merge(times, on="subject_id"))
    cov["trained"] = (cov["time_to_lt2_min"] > TRAINED_THRESHOLD_MIN).astype(int)
    return cov


def load_model_per_subject(version: str, variant: str,
                            feature_set: str, target: str) -> pd.DataFrame:
    """Возвращает per-subject MAE/R² для одной модели."""
    path = RESULTS / version / "per_subject_full.csv"
    df = pd.read_csv(path)
    mask = ((df["variant"] == variant)
            & (df["feature_set"] == feature_set)
            & (df["target"] == target))
    sub = df[mask][["subject_id", "mae_min", "r2"]].copy()
    return sub


def build_long_table(cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version, variant, fset, target, short in TOP_MODELS:
        ps = load_model_per_subject(version, variant, fset, target)
        if ps.empty:
            continue
        ps = ps.merge(cov, on="subject_id", how="left")
        ps["model_id"] = short
        ps["version"] = version
        ps["variant"] = variant
        ps["feature_set"] = fset
        ps["target"] = target
        rows.append(ps)
    return pd.concat(rows, ignore_index=True)


def plot_hist_mae(long: pd.DataFrame, path: Path) -> None:
    targets = ["lt1", "lt2"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, t in zip(axes, targets):
        sub = long[long["target"] == t]
        for mid, g in sub.groupby("model_id"):
            ax.hist(g["mae_min"].dropna(), bins=12, alpha=0.5, label=mid)
        ax.set_title(f"Per-subject MAE — {t.upper()}")
        ax.set_xlabel("MAE, мин")
        ax.set_ylabel("Число субъектов")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_scatter(long: pd.DataFrame, x_col: str, xlabel: str, path: Path) -> None:
    targets = ["lt1", "lt2"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, t in zip(axes, targets):
        sub = long[long["target"] == t].dropna(subset=["mae_min", x_col])
        for mid, g in sub.groupby("model_id"):
            ax.scatter(g[x_col], g["mae_min"], s=40, alpha=0.7, label=mid)
            # Spearman per-model
            if len(g) > 3:
                rho, p = spearmanr(g[x_col], g["mae_min"])
                ax.text(0.02, 0.98 - 0.05 * list(sub["model_id"].unique()).index(mid),
                        f"{mid}: ρ={rho:+.2f}, p={p:.3f}",
                        transform=ax.transAxes, fontsize=8, va="top")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MAE, мин")
        ax.set_title(f"{t.upper()}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def trained_vs_untrained_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, g in long.groupby("model_id"):
        t = g[g["trained"] == 1]["mae_min"].dropna()
        u = g[g["trained"] == 0]["mae_min"].dropna()
        # Wilcoxon — independent (Mann-Whitney), не paired (разные субъекты).
        from scipy.stats import mannwhitneyu
        if len(t) >= 2 and len(u) >= 2:
            try:
                stat, p = mannwhitneyu(t, u, alternative="two-sided")
            except Exception:
                stat, p = float("nan"), float("nan")
        else:
            stat, p = float("nan"), float("nan")
        rows.append({
            "model_id": mid,
            "target": g["target"].iloc[0],
            "n_trained": int(len(t)),
            "n_untrained": int(len(u)),
            "mae_trained_median": round(float(t.median()), 3) if len(t) else float("nan"),
            "mae_untrained_median": round(float(u.median()), 3) if len(u) else float("nan"),
            "mae_trained_mean": round(float(t.mean()), 3) if len(t) else float("nan"),
            "mae_untrained_mean": round(float(u.mean()), 3) if len(u) else float("nan"),
            "mannwhitney_U": round(float(stat), 2) if np.isfinite(stat) else float("nan"),
            "p_value": round(float(p), 4) if np.isfinite(p) else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_trained_groups(long: pd.DataFrame, path: Path) -> None:
    targets = ["lt1", "lt2"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, t in zip(axes, targets):
        sub = long[long["target"] == t].dropna(subset=["mae_min", "trained"])
        models = sub["model_id"].unique()
        positions = np.arange(len(models))
        for i, mid in enumerate(models):
            g = sub[sub["model_id"] == mid]
            t_mae = g[g["trained"] == 1]["mae_min"].dropna()
            u_mae = g[g["trained"] == 0]["mae_min"].dropna()
            ax.boxplot([u_mae.values, t_mae.values],
                       positions=[i - 0.18, i + 0.18], widths=0.3,
                       patch_artist=True,
                       boxprops=dict(facecolor="#cce5ff"),
                       medianprops=dict(color="black"))
        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=15, fontsize=8)
        ax.set_title(f"{t.upper()} — untrained (левая) vs trained (правая)")
        ax.set_ylabel("MAE, мин")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_lt1_vs_lt2_per_subject(long: pd.DataFrame, path: Path) -> None:
    """Скаттер MAE_lt1 × MAE_lt2 для каждого субъекта по совпадающим парам моделей."""
    # Пара: v0011 EMG+NIRS+HRV with_abs для lt1 и lt2.
    lt1 = long[(long["model_id"] == "v0011_ENH_lt1")][["subject_id", "mae_min"]].rename(columns={"mae_min": "mae_lt1"})
    lt2 = long[(long["model_id"] == "v0011_ENH")][["subject_id", "mae_min"]].rename(columns={"mae_min": "mae_lt2"})
    merged = lt1.merge(lt2, on="subject_id")
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(merged["mae_lt1"], merged["mae_lt2"], s=50, alpha=0.7)
    for _, r in merged.iterrows():
        ax.annotate(r["subject_id"], (r["mae_lt1"], r["mae_lt2"]),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")
    rho, p = spearmanr(merged["mae_lt1"], merged["mae_lt2"])
    ax.set_xlabel("MAE на LT1, мин")
    ax.set_ylabel("MAE на LT2, мин")
    ax.set_title(f"Кросс-target предсказуемость по субъектам (v0011 EMG+NIRS+HRV)\n"
                 f"Spearman ρ={rho:+.2f}, p={p:.3f}, n={len(merged)}")
    ax.grid(alpha=0.3)
    ax.plot([0, max(merged[["mae_lt1", "mae_lt2"]].max())],
            [0, max(merged[["mae_lt1", "mae_lt2"]].max())],
            "k--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    EDA.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    print("Сбор ковариат субъектов…")
    cov = load_subject_covariates()
    print(f"  {len(cov)} субъектов, {cov['trained'].sum()} тренированных (time_to_lt2 > {TRAINED_THRESHOLD_MIN} мин)")

    print("Сборка длинной таблицы…")
    long = build_long_table(cov)
    out_long = EDA / "08_per_subject_table.csv"
    long.to_csv(out_long, index=False)
    print(f"  → {out_long.name} ({long.shape})")

    print("Trained vs untrained…")
    tu = trained_vs_untrained_table(long)
    out_tu = EDA / "09_trained_vs_untrained.csv"
    tu.to_csv(out_tu, index=False)
    print(f"  → {out_tu.name}")
    print(tu.to_string(index=False))

    print("\nГрафики…")
    plot_hist_mae(long, PLOTS / "hist_mae_by_model.png")
    plot_scatter(long, "hrv_hr_baseline_bpm", "HR baseline @30W, bpm",
                 PLOTS / "scatter_mae_vs_hr_baseline.png")
    plot_scatter(long, "time_to_lt2_min", "time_to_lt2, мин",
                 PLOTS / "scatter_mae_vs_time_to_lt2.png")
    plot_scatter(long, "age", "Возраст, лет",
                 PLOTS / "scatter_mae_vs_age.png")
    plot_scatter(long, "bmi", "BMI",
                 PLOTS / "scatter_mae_vs_bmi.png")
    plot_trained_groups(long, PLOTS / "mae_by_trained_group.png")
    plot_lt1_vs_lt2_per_subject(long, PLOTS / "lt1_vs_lt2_mae_per_subject.png")
    print(f"  Графики в {PLOTS}")


if __name__ == "__main__":
    main()
