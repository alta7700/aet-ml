"""Экспорт таблиц analysis-кэша в CSV / Excel / LaTeX."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from analysis.ranking import (
    rank_composite, rank_primary, rank_secondary, top_n_by_target,
)
from analysis.schemas import AnalysisConfig


# ─── базовые экспортёры ─────────────────────────────────────────────────────

def to_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def to_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def to_excel(workbook_path: Path, sheets: Mapping[str, pd.DataFrame]) -> Path:
    """Один xlsx, каждый ключ — отдельный лист."""
    workbook_path = Path(workbook_path)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as xl:
        for name, df in sheets.items():
            # Excel-листы максимум 31 символ
            df.to_excel(xl, sheet_name=name[:31], index=False)
    return workbook_path


def to_latex_booktabs(df: pd.DataFrame, path: Path, *, caption: str,
                      label: str, float_format: str = "%.2f") -> Path:
    """LaTeX-таблица в стиле booktabs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(
        index=False, escape=False, float_format=float_format,
        caption=caption, label=label, longtable=False,
    )
    path.write_text(latex, encoding="utf-8")
    return path


# ─── стандартные таблицы диссертации ────────────────────────────────────────

_TOP_COLS_PRIMARY = [
    "rank", "model_id", "architecture_id", "family", "target",
    "feature_set", "with_abs", "wavelet_mode",
    "lt_mae_median_policy_mean", "lt_mae_median_policy_std",
    "lt_mae_median_policy_ci_low", "lt_mae_median_policy_ci_high",
    "mae_mean", "catastrophic_rate_mean",
    "composite_score",
]


def export_dissertation_tables(model_summary: pd.DataFrame,
                                comparisons: dict[str, pd.DataFrame],
                                cfg: AnalysisConfig,
                                conformal_summary: pd.DataFrame | None = None,
                                training_summary: pd.DataFrame | None = None,
                                robustness: pd.DataFrame | None = None,
                                ) -> dict[str, Path]:
    """Готовит и сохраняет CSV/XLSX/TEX для стандартного набора таблиц."""
    out: dict[str, Path] = {}
    tables_dir = cfg.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1) Top-N per target по primary
    top_per_target = top_n_by_target(model_summary, n=cfg.top_n,
                                      metric="lt_mae_median_policy_mean")
    sheets: dict[str, pd.DataFrame] = {}
    for target, df in top_per_target.items():
        cols = [c for c in _TOP_COLS_PRIMARY if c in df.columns]
        slim = df[cols]
        out[f"top_{target}_csv"] = to_csv(
            slim, tables_dir / f"top_{target}.csv")
        out[f"top_{target}_tex"] = to_latex_booktabs(
            slim, tables_dir / f"top_{target}.tex",
            caption=f"Top-{cfg.top_n} моделей по {target.upper()}",
            label=f"tab:top_{target}")
        sheets[f"top_{target}"] = slim

    # 2) полный summary (primary + secondary rankings)
    primary = rank_primary(model_summary)
    secondary = rank_secondary(model_summary)
    composite = rank_composite(model_summary)
    out["summary_primary_csv"] = to_csv(primary, tables_dir / "ranking_primary.csv")
    out["summary_secondary_csv"] = to_csv(secondary, tables_dir / "ranking_secondary.csv")
    out["summary_composite_csv"] = to_csv(composite, tables_dir / "ranking_composite.csv")
    sheets["ranking_primary"] = primary
    sheets["ranking_secondary"] = secondary
    sheets["ranking_composite"] = composite

    # 3) сравнения — каждое в свой csv + лист xlsx
    for name, df in comparisons.items():
        if df.empty:
            continue
        out[f"cmp_{name}_csv"] = to_csv(df, tables_dir / f"cmp_{name}.csv")
        sheets[f"cmp_{name}"] = df
        # Не вся таблица влезет в LaTeX как есть — экспортируем сокращённую.
        cols_slim = [
            "comparison_kind", "group_key", "metric",
            "condition_a", "condition_b", "n_subjects",
            "mean_a", "mean_b", "delta_mean",
            "wilcoxon_pvalue", "pvalue_adj", "reject_at_alpha",
            "cohens_d", "cliffs_delta",
        ]
        slim = df[[c for c in cols_slim if c in df.columns]].copy()
        out[f"cmp_{name}_tex"] = to_latex_booktabs(
            slim, tables_dir / f"cmp_{name}.tex",
            caption=f"Сравнение: {name}", label=f"tab:cmp_{name}",
            float_format="%.3f")

    # 4) conformal / training stability / robustness (если есть)
    if conformal_summary is not None and not conformal_summary.empty:
        out["conformal_summary_csv"] = to_csv(
            conformal_summary, tables_dir / "conformal_summary.csv")
        sheets["conformal_summary"] = conformal_summary
    if training_summary is not None and not training_summary.empty:
        out["training_summary_csv"] = to_csv(
            training_summary, tables_dir / "training_summary.csv")
        sheets["training_summary"] = training_summary
    if robustness is not None and not robustness.empty:
        out["robustness_csv"] = to_csv(robustness, tables_dir / "robustness_ranking.csv")
        sheets["robustness_ranking"] = robustness
        # топ-10 в латех
        out["robustness_tex"] = to_latex_booktabs(
            robustness.head(cfg.top_n),
            tables_dir / "robustness_top.tex",
            caption=f"Top-{cfg.top_n} моделей по сводному robustness-скору "
                    f"(point + uncertainty + stability)",
            label="tab:robustness_top",
            float_format="%.2f")

    # 5) единый Excel
    out["workbook"] = to_excel(tables_dir / "analysis.xlsx", sheets)
    return out
