"""Визуализация поверх кэша (matplotlib + seaborn).

Все функции принимают DataFrame-ы из cache/ и пишут фигуру в указанную
директорию (формат — из cfg.figure_formats). Никогда не читают raw artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # safe default для headless-серверов
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:  # pragma: no cover
    sns = None  # type: ignore[assignment]
    _HAS_SEABORN = False

from analysis.schemas import AnalysisConfig


# ─── saver ─────────────────────────────────────────────────────────────────

def _save(fig, name: str, cfg: AnalysisConfig, subdir: str = "") -> list[Path]:
    """Сохраняет фигуру во всех форматах из cfg.figure_formats."""
    out_dir = cfg.figures_dir / subdir if subdir else cfg.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in cfg.figure_formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=cfg.figure_dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _setup_style() -> None:
    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="paper")
    else:
        plt.rcParams.update({"figure.autolayout": True})


# ─── boxplots / violin ─────────────────────────────────────────────────────

def boxplot_subject_mae_by(subject_metrics: pd.DataFrame, *, group_col: str,
                           target: str, cfg: AnalysisConfig,
                           metric: str = "mae") -> list[Path]:
    """Subject-MAE по группам (модальность / архитектура / семейство)."""
    _setup_style()
    df = subject_metrics[subject_metrics["target"] == target]
    if df.empty:
        return []
    fig, ax = plt.subplots(figsize=(max(6, df[group_col].nunique() * 0.8), 4))
    if _HAS_SEABORN:
        sns.boxplot(data=df, x=group_col, y=metric, ax=ax)
    else:
        groups = sorted(df[group_col].unique())
        data = [df.loc[df[group_col] == g, metric].to_numpy() for g in groups]
        ax.boxplot(data, labels=groups)
    ax.set_title(f"Subject {metric.upper()} by {group_col} ({target.upper()})")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, f"box_{metric}_by_{group_col}_{target}", cfg, subdir="box")


def violin_residuals(preds_best: pd.DataFrame, *, target: str,
                     cfg: AnalysisConfig, top_n: int = 8) -> list[Path]:
    """Violin распределения residuals для top_n моделей по target."""
    _setup_style()
    df = preds_best[preds_best["target"] == target]
    if df.empty:
        return []
    # выбираем модели по медиане |residual|
    med = df.groupby("model_id")["abs_err"].median().sort_values()
    models = med.head(top_n).index.tolist()
    sub = df[df["model_id"].isin(models)]

    fig, ax = plt.subplots(figsize=(max(8, top_n * 0.9), 4))
    if _HAS_SEABORN:
        sns.violinplot(data=sub, x="model_id", y="residual",
                       order=models, inner="box", ax=ax)
    else:
        data = [sub.loc[sub["model_id"] == m, "residual"].to_numpy() for m in models]
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_title(f"Residual distribution — top {top_n} ({target.upper()})")
    ax.set_ylabel("y_pred - y_true [sec]")
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    return _save(fig, f"violin_residual_{target}", cfg, subdir="violin")


# ─── heatmap architecture × modality ───────────────────────────────────────

def heatmap_arch_x_modality(model_summary: pd.DataFrame, *, target: str,
                            cfg: AnalysisConfig,
                            metric: str = "lt_mae_median_policy_mean") -> list[Path]:
    _setup_style()
    df = model_summary[model_summary["target"] == target]
    if df.empty:
        return []
    pivot = df.pivot_table(
        index="architecture_id", columns="feature_set", values=metric,
        aggfunc="mean")
    if pivot.empty:
        return []
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.1),
                                     max(4, pivot.shape[0] * 0.35)))
    if _HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis_r",
                    cbar_kws={"label": metric}, ax=ax)
    else:
        im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
        fig.colorbar(im, ax=ax, label=metric)
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_title(f"{metric} ({target.upper()})")
    return _save(fig, f"heatmap_{metric}_{target}", cfg, subdir="heatmap")


# ─── Bland-Altman ──────────────────────────────────────────────────────────

def bland_altman_top_n(lt_point_metrics: pd.DataFrame, model_summary: pd.DataFrame,
                       *, target: str, cfg: AnalysisConfig,
                       policy: str = "median", top_n: int = 5) -> list[Path]:
    """Bland-Altman: lt_hat_<policy>_sec vs lt_true_sec, top_n моделей."""
    _setup_style()
    pol_col = {
        "median": "lt_hat_median_sec",
        "crossing": "lt_hat_crossing_sec",
        "stable_median": "lt_hat_stable_median_sec",
    }[policy]

    df_ms = model_summary[model_summary["target"] == target].sort_values(
        f"lt_mae_{policy}_policy_mean").head(top_n)
    model_ids = df_ms["model_id"].tolist()
    paths: list[Path] = []
    for mid in model_ids:
        sub = lt_point_metrics[lt_point_metrics["model_id"] == mid]
        if sub.empty:
            continue
        diff = sub[pol_col].to_numpy() - sub["lt_true_sec"].to_numpy()
        mean = (sub[pol_col].to_numpy() + sub["lt_true_sec"].to_numpy()) / 2
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(mean, diff, alpha=0.7)
        mu = float(np.mean(diff)); sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        ax.axhline(mu, color="red", ls="--", label=f"mean={mu:.1f}")
        ax.axhline(mu + 1.96 * sd, color="grey", ls=":")
        ax.axhline(mu - 1.96 * sd, color="grey", ls=":")
        ax.set_xlabel("(lt_hat + lt_true) / 2 [sec]")
        ax.set_ylabel("lt_hat - lt_true [sec]")
        ax.set_title(f"Bland-Altman {mid} ({target}, policy={policy})")
        ax.legend()
        paths.extend(_save(fig, f"bland_{mid}_{target}_{policy}", cfg,
                           subdir="bland_altman"))
    return paths


# ─── scatter pred vs gt + error histogram ──────────────────────────────────

def scatter_pred_vs_gt(preds_best: pd.DataFrame, *, model_id: str,
                       cfg: AnalysisConfig) -> list[Path]:
    _setup_style()
    sub = preds_best[preds_best["model_id"] == model_id]
    if sub.empty:
        return []
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sub["y_true"], sub["y_pred"], alpha=0.4, s=10)
    lim_lo = min(sub["y_true"].min(), sub["y_pred"].min())
    lim_hi = max(sub["y_true"].max(), sub["y_pred"].max())
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8)
    ax.set_xlabel("y_true"); ax.set_ylabel("y_pred")
    ax.set_title(f"Pred vs GT — {model_id}")
    return _save(fig, f"scatter_{model_id}", cfg, subdir="scatter")


def error_histogram(preds_best: pd.DataFrame, *, model_id: str,
                    cfg: AnalysisConfig, bins: int = 60) -> list[Path]:
    _setup_style()
    sub = preds_best[preds_best["model_id"] == model_id]
    if sub.empty:
        return []
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sub["residual"], bins=bins, edgecolor="black", alpha=0.8)
    ax.axvline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("residual = y_pred - y_true [sec]"); ax.set_ylabel("count")
    ax.set_title(f"Error histogram — {model_id}")
    return _save(fig, f"hist_{model_id}", cfg, subdir="hist")


# ─── pairwise p-value heatmap ─────────────────────────────────────────────

def pairwise_pvalue_heatmap(comparison_df: pd.DataFrame, *, name: str,
                            cfg: AnalysisConfig) -> list[Path]:
    """Тепловая карта p-values по группам сравнения."""
    _setup_style()
    if comparison_df.empty:
        return []
    pivot = comparison_df.pivot_table(
        index="condition_a", columns="condition_b",
        values="wilcoxon_pvalue", aggfunc="mean")
    if pivot.empty:
        return []
    fig, ax = plt.subplots(figsize=(max(5, pivot.shape[1] * 0.9),
                                     max(4, pivot.shape[0] * 0.6)))
    if _HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm_r",
                    vmin=0, vmax=1, ax=ax)
    else:
        im = ax.imshow(pivot.values, cmap="coolwarm_r", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_title(f"Wilcoxon p-values — {name}")
    return _save(fig, f"pairwise_{name}", cfg, subdir="pairwise")


# ─── orchestrate plotting ──────────────────────────────────────────────────

# ─── conformal / stability ──────────────────────────────────────────────────

def conformal_coverage_plot(conformal_summary: pd.DataFrame, *, target: str,
                            cfg: AnalysisConfig,
                            policy: str = "median") -> list[Path]:
    """Empirical coverage vs nominal, по моделям (одна точка на модель)."""
    _setup_style()
    df = conformal_summary[
        (conformal_summary["target"] == target)
        & (conformal_summary["policy"] == policy)
    ]
    if df.empty:
        return []
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["nominal_coverage"], df["empirical_coverage"],
               s=18, alpha=0.5, label=f"models (policy={policy})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="nominal=empirical")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage (1 - α)")
    ax.set_ylabel("empirical coverage (leave-one-subject-out)")
    ax.set_title(f"Conformal calibration — {target.upper()} ({policy})")
    ax.legend(loc="upper left")
    return _save(fig, f"conformal_coverage_{target}_{policy}", cfg, subdir="conformal")


def conformal_width_distribution(conformal_summary: pd.DataFrame, *, target: str,
                                 cfg: AnalysisConfig,
                                 policy: str = "median",
                                 alpha: float = 0.2) -> list[Path]:
    """Распределение mean_interval_width_sec по моделям."""
    _setup_style()
    df = conformal_summary[
        (conformal_summary["target"] == target)
        & (conformal_summary["policy"] == policy)
        & (np.isclose(conformal_summary["alpha"], alpha))
    ]
    if df.empty:
        return []
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["mean_interval_width_sec"], bins=30, edgecolor="black", alpha=0.8)
    ax.set_xlabel("mean interval width [sec]")
    ax.set_ylabel("number of models")
    ax.set_title(f"Conformal width — {target.upper()} ({policy}, α={alpha})")
    return _save(fig, f"conformal_width_{target}_{policy}_a{int(round(alpha*100)):03d}",
                 cfg, subdir="conformal")


def training_mae_trajectories(training_dynamics: pd.DataFrame, *, target: str,
                              cfg: AnalysisConfig, top_n: int = 6) -> list[Path]:
    """Кривые train_mae по эпохам для top_n моделей выбранного target."""
    _setup_style()
    df = training_dynamics[training_dynamics["target"] == target]
    if df.empty:
        return []
    # выбираем top_n моделей по сходимости (по min final_train_mae)
    finals = (df.sort_values(["model_id", "fold_id", "epoch"], kind="stable")
              .groupby("model_id").tail(1)
              .groupby("model_id")["train_mae"].mean().sort_values())
    ids = finals.head(top_n).index.tolist()
    fig, ax = plt.subplots(figsize=(7, 5))
    for mid in ids:
        sub = df[df["model_id"] == mid]
        for fold, gf in sub.groupby("fold_id"):
            ax.plot(gf["epoch"], gf["train_mae"], alpha=0.3, lw=0.7)
        agg = sub.groupby("epoch")["train_mae"].mean()
        ax.plot(agg.index, agg.values, lw=2, label=mid)
    ax.set_xlabel("epoch"); ax.set_ylabel("train MAE [sec]")
    ax.set_title(f"Train MAE trajectories — top {top_n} ({target.upper()})")
    ax.legend(fontsize=7, loc="upper right")
    return _save(fig, f"train_mae_traj_{target}", cfg, subdir="training")


def plot_all(*, subject_metrics: pd.DataFrame,
             lt_point_metrics: pd.DataFrame,
             model_summary: pd.DataFrame,
             preds_best: pd.DataFrame,
             comparisons: dict[str, pd.DataFrame],
             cfg: AnalysisConfig,
             conformal_summary: pd.DataFrame | None = None,
             training_dynamics: pd.DataFrame | None = None,
             ) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for target in sorted(model_summary["target"].unique()):
        out[f"box_mae_modality_{target}"] = boxplot_subject_mae_by(
            subject_metrics, group_col="feature_set", target=target, cfg=cfg)
        out[f"box_mae_arch_{target}"] = boxplot_subject_mae_by(
            subject_metrics, group_col="architecture_id", target=target, cfg=cfg)
        out[f"box_mae_family_{target}"] = boxplot_subject_mae_by(
            subject_metrics, group_col="family", target=target, cfg=cfg)
        out[f"violin_{target}"] = violin_residuals(
            preds_best, target=target, cfg=cfg)
        out[f"heatmap_lt_{target}"] = heatmap_arch_x_modality(
            model_summary, target=target, cfg=cfg,
            metric="lt_mae_median_policy_mean")
        out[f"heatmap_mae_{target}"] = heatmap_arch_x_modality(
            model_summary, target=target, cfg=cfg, metric="mae_mean")
        out[f"bland_{target}"] = bland_altman_top_n(
            lt_point_metrics, model_summary, target=target, cfg=cfg,
            policy="median", top_n=5)

        # scatter/hist на топ-3 моделях
        top3 = (model_summary[model_summary["target"] == target]
                .sort_values("lt_mae_median_policy_mean").head(3))
        for mid in top3["model_id"]:
            out[f"scatter_{mid}"] = scatter_pred_vs_gt(preds_best, model_id=mid, cfg=cfg)
            out[f"hist_{mid}"] = error_histogram(preds_best, model_id=mid, cfg=cfg)

    for name, df in comparisons.items():
        out[f"pairwise_{name}"] = pairwise_pvalue_heatmap(df, name=name, cfg=cfg)

    # conformal / training stability
    if conformal_summary is not None and not conformal_summary.empty:
        for target in sorted(conformal_summary["target"].unique()):
            out[f"conformal_cov_{target}"] = conformal_coverage_plot(
                conformal_summary, target=target, cfg=cfg, policy="median")
            for alpha in sorted(conformal_summary["alpha"].unique()):
                out[f"conformal_width_{target}_a{int(round(alpha*100)):03d}"] = (
                    conformal_width_distribution(
                        conformal_summary, target=target, cfg=cfg,
                        policy="median", alpha=alpha))
    if training_dynamics is not None and not training_dynamics.empty:
        for target in sorted(training_dynamics["target"].unique()):
            out[f"train_traj_{target}"] = training_mae_trajectories(
                training_dynamics, target=target, cfg=cfg)
    return out
