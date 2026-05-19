"""Экспорт таблиц и рисунков для главы результатов диссертации.

Модуль читает только уже посчитанные кэши и comparison-таблицы analysis-слоя.
Никаких raw-артефактов обучения и никакого чтения stdout здесь нет.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from analysis.schemas import AnalysisConfig

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:  # pragma: no cover
    sns = None  # type: ignore[assignment]
    _HAS_SEABORN = False


def _setup_style() -> None:
    """Единый спокойный стиль для диссертационных рисунков."""
    base = {
        "figure.autolayout": False,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
    plt.rcParams.update(base)
    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="paper")


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str,
                 cfg: AnalysisConfig) -> list[Path]:
    """Сохраняет фигуру во всех форматах thesis-слоя."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in cfg.figure_formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _to_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return path


def _human_phase_split(series: pd.Series) -> pd.Series:
    mapping = {
        "split": "По фазам цикла",
        "full_cycles": "По полному циклу",
        "full_window": "По полному окну",
        "": "По фазам цикла",
    }
    return series.fillna("split").astype(str).map(lambda x: mapping.get(x, x))


def _family_palette() -> dict[str, str]:
    return {
        "Lin": "#1f77b4",
        "LSTM": "#d62728",
        "TCN": "#2ca02c",
    }


def _modality_palette() -> dict[str, str]:
    """Единая палитра модальностей для рисунков интерпретации."""
    return {
        "EMG": "#d95f02",
        "NIRS": "#1b9e77",
        "HRV": "#7570b3",
        "Interaction": "#e7298a",
        "Kinematics": "#66a61e",
    }


def _human_family(series: pd.Series) -> pd.Series:
    return series.replace({"Lin": "Классическое"})


def _float_to_ru(value: str) -> str:
    """Приводит строковое число к русской записи с запятой."""
    return value.replace(".", ",")


def _short_model_label(model_name: str) -> str:
    """Собирает короткое человекочитаемое обозначение модели."""
    if model_name.startswith("ElasticNet("):
        alpha = re.search(r"alpha=([0-9.]+)", model_name)
        l1_ratio = re.search(r"l1_ratio=([0-9.]+)", model_name)
        alpha_str = _float_to_ru(alpha.group(1)) if alpha else "?"
        l1_str = _float_to_ru(l1_ratio.group(1)) if l1_ratio else "?"
        return f"ElasticNet(α={alpha_str}; l1_ratio={l1_str})"
    if model_name.startswith("Ridge("):
        alpha = re.search(r"alpha=([0-9.]+)", model_name)
        alpha_str = _float_to_ru(alpha.group(1)) if alpha else "?"
        return f"Ridge(α={alpha_str})"
    if model_name.startswith("HuberRegressor("):
        eps = re.search(r"epsilon=([0-9.]+)", model_name)
        eps_str = _float_to_ru(eps.group(1)) if eps else "?"
        return f"HuberRegressor(ε={eps_str})"
    if model_name.startswith("GradientBoostingRegressor("):
        depth = re.search(r"max_depth=([0-9.]+)", model_name)
        trees = re.search(r"n_estimators=([0-9.]+)", model_name)
        depth_str = depth.group(1) if depth else "?"
        trees_str = trees.group(1) if trees else "?"
        return (
            "GradientBoostingRegressor("
            f"max_depth={depth_str}; n_estimators={trees_str})"
        )
    return model_name


def build_thesis_tables(model_summary: pd.DataFrame,
                        cfg: AnalysisConfig) -> dict[str, Path]:
    """Собирает компактные таблицы для подразделов 3.2–3.5."""
    out: dict[str, Path] = {}
    sheets: dict[str, pd.DataFrame] = {}
    out_dir = cfg.thesis_tables_dir

    top_cols = [
        "rank", "family", "feature_set", "with_abs", "phase_split",
        "lt_mae_median_policy_mean", "lt_bias_median_policy_mean",
        "catastrophic_rate_mean", "r2_median",
    ]
    for target in ("lt1", "lt2"):
        sub = model_summary[model_summary["target"] == target].copy()
        sub = sub.sort_values("lt_mae_median_policy_mean").copy()
        sub.insert(0, "rank", np.arange(1, len(sub) + 1))
        sub = (
            sub.groupby("family", as_index=False, sort=False)
               .head(1)
               .reset_index(drop=True)
        )
        slim = sub[top_cols].copy()
        slim["family"] = slim["family"].replace({"Lin": "Классическое"})
        slim["phase_split"] = _human_phase_split(slim["phase_split"])
        slim = slim.rename(columns={
            "family": "Семейство",
            "feature_set": "Модальности",
            "with_abs": "С абс. признаками",
            "phase_split": "Режим ЭМГ-признаков",
            "lt_mae_median_policy_mean": "Ошибка LT, с",
            "lt_bias_median_policy_mean": "Смещение LT, с",
            "catastrophic_rate_mean": "Доля грубых ошибок",
            "r2_median": "Медиана R²",
            "rank": "Место в общем рейтинге",
        })
        csv_path = out_dir / f"top_{target}_family_overview.csv"
        out[f"top_{target}_family_overview_csv"] = _to_csv(slim, csv_path)
        sheets[f"top_{target}_family"] = slim

    out["thesis_tables_workbook"] = _to_excel(
        out_dir / "thesis_tables.xlsx", sheets)
    return out


def build_finalists_table(cfg: AnalysisConfig) -> dict[str, Path]:
    """Собирает компактную таблицу финальных моделей классического семейства."""
    out: dict[str, Path] = {}
    path = cfg.tables_dir / "selected_for_shap.csv"
    if not path.exists():
        return out
    df = pd.read_csv(path)
    if df.empty:
        return out
    df = df[df["family"] == "Lin"].copy()
    if df.empty:
        return out
    df["model_name_short"] = df["model_name"].map(_short_model_label)
    df["phase_split"] = _human_phase_split(df["phase_split"])
    df.loc[~df["feature_set"].astype(str).str.contains("EMG", na=False), "phase_split"] = (
        "Не используется"
    )
    df["target"] = df["target"].map({"lt1": "LT1", "lt2": "LT2"})
    slim = df[[
        "target",
        "model_name_short",
        "feature_set",
        "phase_split",
        "lt_mae_median_policy_mean",
        "catastrophic_rate_mean",
        "r2_median",
    ]].copy()
    slim = slim.rename(columns={
        "target": "Порог",
        "model_name_short": "Модель",
        "feature_set": "Модальности",
        "phase_split": "Режим ЭМГ-признаков",
        "lt_mae_median_policy_mean": "Ошибка LT, с",
        "catastrophic_rate_mean": "Доля грубых ошибок",
        "r2_median": "Медиана R²",
    })
    csv_path = cfg.thesis_tables_dir / "finalists_classic_summary.csv"
    out["finalists_classic_summary_csv"] = _to_csv(slim, csv_path)

    workbook_path = cfg.thesis_tables_dir / "thesis_tables.xlsx"
    if workbook_path.exists():
        existing = pd.read_excel(workbook_path, sheet_name=None)
    else:
        existing = {}
    existing["classic_finalists"] = slim
    out["thesis_tables_workbook"] = _to_excel(workbook_path, existing)
    return out


def _prepare_modality_delta_plot_data(
    modality_cmp: pd.DataFrame,
) -> pd.DataFrame:
    """Готовит усреднённые сдвиги по модальностям для наглядного графика."""
    df = modality_cmp.copy()
    if df.empty:
        return df
    parts = df["group_key"].str.split("|", expand=True)
    for idx in range(parts.shape[1]):
        kv = parts[idx].str.split("=", n=1, expand=True)
        if kv.shape[1] == 2:
            df[kv[0].iloc[0]] = kv[1]
    df = df[df["metric"] == "abs_lt_err_median_sec"].copy()
    df = df[df["condition_a"].isin(["EMG", "EMG+NIRS"])]
    df = df[df["condition_b"].isin(["EMG+NIRS", "EMG+NIRS+HRV"])]
    pair_mask = (
        ((df["condition_a"] == "EMG") & (df["condition_b"] == "EMG+NIRS")) |
        ((df["condition_a"] == "EMG+NIRS") & (df["condition_b"] == "EMG+NIRS+HRV")) |
        ((df["condition_a"] == "EMG") & (df["condition_b"] == "EMG+NIRS+HRV"))
    )
    df = df[pair_mask].copy()
    if df.empty:
        return df
    df["family"] = _human_family(df["family"])
    df["comparison"] = df["condition_a"] + " → " + df["condition_b"]
    grouped = (
        df.groupby(["target", "family", "comparison"], as_index=False)
          .agg(delta_mean_sec=("delta_mean", "mean"))
    )
    return grouped


def plot_modality_effects(modality_cmp: pd.DataFrame,
                          cfg: AnalysisConfig) -> list[Path]:
    """График направлений эффекта модальности для LT1 и LT2."""
    _setup_style()
    plot_df = _prepare_modality_delta_plot_data(modality_cmp)
    if plot_df.empty:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    palette = {
        "Классическое": _family_palette()["Lin"],
        "LSTM": _family_palette()["LSTM"],
        "TCN": _family_palette()["TCN"],
    }
    order = ["EMG → EMG+NIRS", "EMG+NIRS → EMG+NIRS+HRV", "EMG → EMG+NIRS+HRV"]
    for ax, target, title in zip(axes, ["lt1", "lt2"], ["LT1", "LT2"]):
        sub = plot_df[plot_df["target"] == target].copy()
        if _HAS_SEABORN:
            sns.barplot(
                data=sub,
                x="comparison",
                y="delta_mean_sec",
                hue="family",
                order=order,
                palette=palette,
                ax=ax,
            )
        else:  # pragma: no cover
            families = ["Классическое", "LSTM", "TCN"]
            x = np.arange(len(order))
            width = 0.25
            for idx, fam in enumerate(families):
                fam_sub = sub[sub["family"] == fam].set_index("comparison")
                vals = [fam_sub["delta_mean_sec"].get(label, np.nan) for label in order]
                ax.bar(x + (idx - 1) * width, vals, width=width, label=fam,
                       color=palette[fam])
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=18, ha="right")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(title)
        ax.set_ylabel("Сдвиг ошибки, с" if target == "lt1" else "")
        ax.tick_params(axis="x", rotation=18)
        ax.legend(title="Семейство")
    return _save_figure(fig, cfg.thesis_figures_dir, "modality_effects_lt1_lt2", cfg)


def plot_family_comparison(model_summary: pd.DataFrame,
                           cfg: AnalysisConfig) -> list[Path]:
    """Боксплоты качества по семействам моделей."""
    _setup_style()
    df = model_summary.copy()
    if df.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    palette = {
        "Классическое": _family_palette()["Lin"],
        "LSTM": _family_palette()["LSTM"],
        "TCN": _family_palette()["TCN"],
    }
    for ax, target, ylabel in zip(
        axes,
        ["lt1", "lt2"],
        ["Ошибка локализации порога, с", ""],
    ):
        sub = df[df["target"] == target].copy()
        sub["family"] = _human_family(sub["family"])
        if _HAS_SEABORN:
            sns.boxplot(
                data=sub,
                x="family",
                y="lt_mae_median_policy_mean",
                hue="family",
                order=["Классическое", "LSTM", "TCN"],
                palette=palette,
                dodge=False,
                ax=ax,
            )
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        else:  # pragma: no cover
            groups = ["Классическое", "LSTM", "TCN"]
            data = [
                sub.loc[sub["family"] == group, "lt_mae_median_policy_mean"].to_numpy()
                for group in groups
            ]
            ax.boxplot(data, labels=groups)
        ax.set_xlabel("LT1" if target == "lt1" else "LT2")
        ax.set_ylabel(ylabel)
    return _save_figure(fig, cfg.thesis_figures_dir, "family_comparison_lt1_lt2", cfg)


def _rank_corr_points(ranking_primary: pd.DataFrame) -> pd.DataFrame:
    """Сводит ранги LT1/LT2 по архитектурам в один датафрейм."""
    cols = ["architecture_id", "family", "target", "rank"]
    df = ranking_primary[cols].copy()
    pivot = (
        df.pivot_table(
            index=["architecture_id", "family"],
            columns="target",
            values="rank",
            aggfunc="min",
        )
        .reset_index()
    )
    pivot = pivot.dropna(subset=["lt1", "lt2"]).copy()
    return pivot


def plot_lt1_lt2_rank_correlation(ranking_primary: pd.DataFrame,
                                  cfg: AnalysisConfig) -> list[Path]:
    """Диаграмма рассеяния рангов LT1 и LT2 по архитектурам."""
    _setup_style()
    df = _rank_corr_points(ranking_primary)
    if df.empty:
        return []
    df["family"] = _human_family(df["family"])
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {
        "Классическое": _family_palette()["Lin"],
        "LSTM": _family_palette()["LSTM"],
        "TCN": _family_palette()["TCN"],
    }
    for family, sub in df.groupby("family"):
        ax.scatter(
            sub["lt1"],
            sub["lt2"],
            s=35,
            alpha=0.8,
            label=family,
            color=palette.get(family),
        )
    lim = max(df["lt1"].max(), df["lt2"].max()) + 1
    ax.plot([1, lim], [1, lim], color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Ранг по LT1")
    ax.set_ylabel("Ранг по LT2")
    ax.legend(title="Семейство")
    return _save_figure(fig, cfg.thesis_figures_dir, "lt1_lt2_rank_correlation", cfg)


def _prepare_phase_split_top_counts(ranking_primary: pd.DataFrame,
                                    top_n: int = 30) -> pd.DataFrame:
    """Готовит состав верхней части рейтинга по режимам ЭМГ-признаков.

    Берутся только классические модели и только наборы признаков, содержащие
    ЭМГ-компоненту, чтобы не смешивать эффект режима с NIRS-only или HRV-only
    постановками.
    """
    df = ranking_primary.copy()
    if df.empty:
        return df
    df = df[df["family"] == "Lin"].copy()
    df = df[df["feature_set"].fillna("").str.contains("EMG")].copy()
    df["phase_split"] = _human_phase_split(df["phase_split"])
    counts: list[pd.DataFrame] = []
    for target in ("lt1", "lt2"):
        sub = df[df["target"] == target].sort_values("rank").head(top_n).copy()
        if sub.empty:
            continue
        grouped = (
            sub.groupby("phase_split", as_index=False)
               .size()
               .rename(columns={"size": "n_models"})
        )
        grouped["target"] = target.upper()
        grouped["share_pct"] = grouped["n_models"] / grouped["n_models"].sum() * 100.0
        counts.append(grouped)
    if not counts:
        return pd.DataFrame()
    return pd.concat(counts, ignore_index=True)


def plot_phase_split_top_counts(ranking_primary: pd.DataFrame,
                                cfg: AnalysisConfig) -> list[Path]:
    """Показывает, какие режимы ЭМГ доминируют в верхней части рейтинга."""
    _setup_style()
    df = _prepare_phase_split_top_counts(ranking_primary)
    if df.empty:
        return []

    order = ["По фазам цикла", "По полному циклу", "По полному окну"]
    palette = {
        "По фазам цикла": "#8c8c8c",
        "По полному циклу": "#4c78a8",
        "По полному окну": "#f58518",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, target in zip(axes, ["LT1", "LT2"]):
        sub = df[df["target"] == target].copy()
        sub["phase_split"] = pd.Categorical(
            sub["phase_split"], categories=order, ordered=True)
        sub = sub.sort_values("phase_split")
        if _HAS_SEABORN:
            sns.barplot(
                data=sub,
                x="phase_split",
                y="share_pct",
                hue="phase_split",
                order=order,
                palette=palette,
                dodge=False,
                ax=ax,
            )
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        else:  # pragma: no cover
            ax.bar(
                sub["phase_split"],
                sub["share_pct"],
                color=[palette.get(x, "#999999") for x in sub["phase_split"]],
            )
        for _, row in sub.iterrows():
            ax.text(
                order.index(row["phase_split"]),
                row["share_pct"] + 1.5,
                f"{int(row['n_models'])}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xlabel(target)
        ax.set_ylabel("Доля в верхних 30 моделях, %" if target == "LT1" else "")
        ax.tick_params(axis="x", rotation=18)
        ax.set_ylim(0, 60)
    return _save_figure(fig, cfg.thesis_figures_dir, "phase_split_top30_lt1_lt2", cfg)


def plot_trusted_selection_counts(cfg: AnalysisConfig) -> list[Path]:
    """Компактная схема стадий trusted-отбора для классических моделей."""
    adm_path = cfg.tables_dir / "admissibility.csv"
    fin_path = cfg.tables_dir / "finalists.csv"
    if not adm_path.exists() or not fin_path.exists():
        return []
    adm = pd.read_csv(adm_path)
    fin = pd.read_csv(fin_path)
    adm = adm[adm["family"] == "Lin"].copy()
    fin = fin[fin["family"] == "Lin"].copy()
    if adm.empty:
        return []

    rows = []
    for target, label in [("lt1", "LT1"), ("lt2", "LT2")]:
        sub = adm[adm["target"] == target]
        rows.extend([
            {"target": label, "stage": "Все модели", "count": len(sub)},
            {"target": label, "stage": "Прошли фильтр", "count": int(sub["admissible"].sum())},
            {"target": label, "stage": "Финальные модели", "count": len(fin[fin["target"] == target])},
        ])
    plot_df = pd.DataFrame(rows)

    _setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    palette = {"LT1": "#4c78a8", "LT2": "#f58518"}
    order = ["Все модели", "Прошли фильтр", "Финальные модели"]
    if _HAS_SEABORN:
        sns.barplot(
            data=plot_df,
            x="stage",
            y="count",
            hue="target",
            order=order,
            palette=palette,
            ax=ax,
        )
    else:  # pragma: no cover
        x = np.arange(len(order))
        width = 0.34
        for idx, target in enumerate(["LT1", "LT2"]):
            sub = plot_df[plot_df["target"] == target].set_index("stage")
            vals = [sub["count"].get(stage, 0) for stage in order]
            ax.bar(x + (idx - 0.5) * width, vals, width=width,
                   color=palette[target], label=target)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + max(plot_df["count"]) * 0.015,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("")
    ax.set_ylabel("Число моделей")
    ax.legend(title="Порог")
    return _save_figure(fig, cfg.thesis_figures_dir, "trusted_selection_counts", cfg)


def plot_shap_finalist_modalities(cfg: AnalysisConfig) -> list[Path]:
    """Агрегированный вклад модальностей для классических finalist-моделей."""
    path = cfg.tables_dir / "shap_modality_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    plot_df = (
        df.groupby(["target", "model_id", "modality"], as_index=False)["abs_shap_share"]
          .mean()
          .groupby(["target", "modality"], as_index=False)["abs_shap_share"]
          .mean()
    )
    plot_df["target"] = plot_df["target"].map({"lt1": "LT1", "lt2": "LT2"})
    order = ["EMG", "NIRS", "HRV", "Interaction", "Kinematics"]
    palette = _modality_palette()

    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, target in zip(axes, ["LT1", "LT2"]):
        sub = plot_df[plot_df["target"] == target].copy()
        sub["modality"] = pd.Categorical(sub["modality"], categories=order, ordered=True)
        sub = sub.sort_values("modality")
        if _HAS_SEABORN:
            sns.barplot(
                data=sub,
                x="modality",
                y="abs_shap_share",
                hue="modality",
                order=order,
                palette=palette,
                dodge=False,
                ax=ax,
            )
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        else:  # pragma: no cover
            ax.bar(sub["modality"], sub["abs_shap_share"],
                   color=[palette[m] for m in sub["modality"]])
        ax.set_xlabel(target)
        ax.set_ylabel("Средняя доля |SHAP|" if target == "LT1" else "")
        ax.tick_params(axis="x", rotation=18)
        ax.set_ylim(0, 0.75)
    return _save_figure(fig, cfg.thesis_figures_dir, "shap_finalist_modalities", cfg)


def plot_captum_modalities(cfg: AnalysisConfig) -> list[Path]:
    """Сопоставляет модальные вклады трёх методов Captum для двух NN-моделей."""
    path = cfg.captum_method_consistency_path
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    if df.empty:
        return []
    plot_df = df.melt(
        id_vars=["model_id", "architecture_id", "target", "modality"],
        value_vars=["integrated_gradients", "gradient_shap", "occlusion"],
        var_name="method",
        value_name="share",
    )
    method_map = {
        "integrated_gradients": "Integrated Gradients",
        "gradient_shap": "GradientShap",
        "occlusion": "Маскирование",
    }
    plot_df["method"] = plot_df["method"].map(method_map)
    title_map = {
        "LSTM3_31f2b090": "LSTM3 (LT1)",
        "TCN3_77409486": "TCN3 (LT2)",
    }
    order_methods = ["Integrated Gradients", "GradientShap", "Маскирование"]
    order_mods = ["EMG", "NIRS", "HRV", "Interaction", "Kinematics"]
    palette = _modality_palette()

    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, model_id in zip(axes, ["LSTM3_31f2b090", "TCN3_77409486"]):
        sub = plot_df[plot_df["model_id"] == model_id].copy()
        pivot = (
            sub.pivot_table(index="method", columns="modality", values="share", aggfunc="mean")
               .reindex(order_methods)
               .fillna(0.0)
        )
        bottom = np.zeros(len(pivot))
        for mod in order_mods:
            vals = pivot[mod].to_numpy() if mod in pivot.columns else np.zeros(len(pivot))
            ax.bar(
                pivot.index,
                vals,
                bottom=bottom,
                color=palette[mod],
                label=mod,
                width=0.7,
            )
            bottom += vals
        ax.set_xlabel(title_map.get(model_id, model_id))
        ax.tick_params(axis="x", rotation=18)
        ax.set_ylabel("Доля суммарного вклада" if model_id == "LSTM3_31f2b090" else "")
        ax.set_ylim(0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Модальность", loc="upper center",
               ncol=5, bbox_to_anchor=(0.5, 1.04), frameon=False)
    return _save_figure(fig, cfg.thesis_figures_dir, "captum_modalities", cfg)


def _load_predictions_selected(cache_dir: Path,
                               *,
                               target: str,
                               architecture_id: str,
                               model_id: str) -> pd.DataFrame:
    """Загружает траектории предсказаний одной модели из partitioned cache."""
    base = cache_dir / "predictions_selected"
    dset = ds.dataset(str(base), format="parquet", partitioning="hive")
    table = dset.to_table(
        filter=(
            (ds.field("target") == target)
            & (ds.field("architecture_id") == architecture_id)
        )
    )
    df = table.to_pandas()
    df = df[df["model_id"] == model_id].copy()
    if df.empty:
        return df
    df["time_min"] = df["sample_end_sec"] / 60.0
    return df.sort_values(["subject_id", "sample_end_sec"], kind="stable")


def _plot_single_or_compare_trajectories(*,
                                         true_df: pd.DataFrame,
                                         pred_df_b: pd.DataFrame | None,
                                         out_path: Path,
                                         cfg: AnalysisConfig,
                                         label_a: str,
                                         label_b: str | None = None) -> list[Path]:
    """Строит сетку 6×3 траекторий с одной или двумя модельными кривыми."""
    _setup_style()
    subjects = sorted(true_df["subject_id"].unique())
    if pred_df_b is not None:
        common = sorted(set(subjects) & set(pred_df_b["subject_id"].unique()))
        subjects = common
        true_df = true_df[true_df["subject_id"].isin(common)].copy()
        pred_df_b = pred_df_b[pred_df_b["subject_id"].isin(common)].copy()
    ncols = 6
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.25 * nrows),
        sharex=False, sharey=False,
    )
    axes = np.atleast_2d(axes)

    col_true = "#1f77b4"
    col_a = "#d62728"
    col_b = "#2ca02c"

    for idx, subj in enumerate(subjects):
        ax = axes[idx // ncols][idx % ncols]
        g = true_df[true_df["subject_id"] == subj].copy()
        t = g["time_min"].to_numpy()
        y_true = g["y_true"].to_numpy() / 60.0
        y_a = g["y_pred"].to_numpy() / 60.0

        ax.plot(t, y_true, color=col_true, lw=1.5, label="истинная траектория")
        ax.plot(t, y_a, color=col_a, lw=1.0, alpha=0.9, label=label_a)

        series_min = [np.nanmin(y_true), np.nanmin(y_a)]
        series_max = [np.nanmax(y_true), np.nanmax(y_a)]

        if pred_df_b is not None:
            g_b = pred_df_b[pred_df_b["subject_id"] == subj].copy()
            if not g_b.empty:
                y_b = g_b["y_pred"].to_numpy() / 60.0
                t_b = g_b["time_min"].to_numpy()
                ax.plot(t_b, y_b, color=col_b, lw=1.0, alpha=0.9, label=label_b)
                series_min.append(np.nanmin(y_b))
                series_max.append(np.nanmax(y_b))

        ax.axhline(0.0, color="#666666", ls=":", lw=0.8)
        zero_idx = int(np.argmin(np.abs(y_true)))
        if abs(y_true[zero_idx]) < 5.0 / 60.0:
            ax.axvline(t[zero_idx], color=col_true, ls="--", lw=0.7, alpha=0.6)

        lo = min(series_min)
        hi = max(series_max)
        pad = max(0.3, 0.08 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(str(subj), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, lw=0.4)

    for j in range(len(subjects), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.supxlabel("Время теста, мин", fontsize=10)
    fig.supylabel("Время до LT, мин", fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3 if pred_df_b is not None else 2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    return _save_figure(fig, out_path.parent, out_path.stem, cfg)


def plot_trajectory_comparisons(cfg: AnalysisConfig) -> list[Path]:
    """Строит narrative-сравнение траекторий для LT1 и LT2."""
    out_paths: list[Path] = []

    # LT1: raw и trusted совпадают -> одна модель.
    lt1_df = _load_predictions_selected(
        cfg.cache_dir, target="lt1",
        architecture_id="Lin10", model_id="Lin10_dc410fe4",
    )
    if not lt1_df.empty:
        out_paths.extend(
            _plot_single_or_compare_trajectories(
                true_df=lt1_df,
                pred_df_b=None,
                out_path=cfg.thesis_figures_dir / "trajectory_compare_lt1.png",
                cfg=cfg,
                label_a="лучшая и финальная модель",
            )
        )

    # LT2: raw-лидер по LT и trusted-лидер различаются.
    lt2_raw = _load_predictions_selected(
        cfg.cache_dir, target="lt2",
        architecture_id="TCN1", model_id="TCN1_aa09c186",
    )
    lt2_trusted = _load_predictions_selected(
        cfg.cache_dir, target="lt2",
        architecture_id="Lin12", model_id="Lin12_ecfa2a6a",
    )
    if not lt2_raw.empty and not lt2_trusted.empty:
        out_paths.extend(
            _plot_single_or_compare_trajectories(
                true_df=lt2_raw,
                pred_df_b=lt2_trusted,
                out_path=cfg.thesis_figures_dir / "trajectory_compare_lt2.png",
                cfg=cfg,
                label_a="исходный лидер",
                label_b="финальная модель",
            )
        )
    return out_paths


def export_thesis_artifacts(model_summary: pd.DataFrame,
                            comparisons: dict[str, pd.DataFrame],
                            ranking_primary: pd.DataFrame,
                            cfg: AnalysisConfig) -> dict[str, Path | list[Path]]:
    """Главная точка входа thesis-слоя для подразделов 3.2–3.9."""
    out: dict[str, Path | list[Path]] = {}
    out.update(build_thesis_tables(model_summary, cfg))
    out.update(build_finalists_table(cfg))

    modality_cmp = comparisons.get("modality_within_family", pd.DataFrame())
    out["fig_modality_effects"] = plot_modality_effects(modality_cmp, cfg)
    out["fig_family_comparison"] = plot_family_comparison(model_summary, cfg)
    out["fig_lt1_lt2_rank_correlation"] = plot_lt1_lt2_rank_correlation(
        ranking_primary, cfg)
    out["fig_phase_split_top_counts"] = plot_phase_split_top_counts(
        ranking_primary, cfg)
    out["fig_trusted_selection_counts"] = plot_trusted_selection_counts(cfg)
    out["fig_shap_finalist_modalities"] = plot_shap_finalist_modalities(cfg)
    out["fig_captum_modalities"] = plot_captum_modalities(cfg)
    out["fig_trajectory_comparisons"] = plot_trajectory_comparisons(cfg)
    return out
