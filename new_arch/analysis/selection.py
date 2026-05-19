"""Отбор trustable top-моделей для интерпретации и финального shortlist.

Логика двухступенчатая:
  1. admissibility-filter по качеству траектории и coverage;
  2. пересечение верхних квантилей LT-ranking и trajectory-ranking.

Идея: primary leaderboard по LT сохраняется отдельно, а shortlist для SHAP и
финальных кандидатов должен состоять только из моделей, которые одновременно:
  - достаточно хороши по LT-time,
  - не провальны по качеству всей траектории.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from analysis.schemas import AnalysisConfig, SelectionTargetRules, TARGETS


def _top_k_from_quantile(n: int, q: float) -> int:
    """Размер shortlist по квантилю; минимум одна модель."""
    if n <= 0:
        return 0
    return max(1, int(math.ceil(n * q)))


def _rules_for(cfg: AnalysisConfig, target: str) -> SelectionTargetRules:
    """Возвращает target-specific правила селекции."""
    return cfg.selection_rules.get(str(target), SelectionTargetRules())


def _rank_lt(df: pd.DataFrame) -> pd.DataFrame:
    """LT-ranking: основная метрика + стабильность + катастрофические ошибки."""
    out = df.sort_values(
        ["lt_mae_median_policy_mean", "lt_mae_median_policy_std", "catastrophic_rate_mean"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    out["lt_rank"] = out.index + 1
    return out


def _rank_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Trajectory-ranking: глобальная ошибка + хвостовые ошибки + устойчивый R²."""
    out = df.sort_values(
        ["mae_mean", "catastrophic_rate_mean", "r2_median", "r2_mean"],
        ascending=[True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    out["trajectory_rank"] = out.index + 1
    return out


def _admissibility_table(summary: pd.DataFrame, target: str,
                         rules: SelectionTargetRules) -> pd.DataFrame:
    """Строит таблицу с pass/fail по каждому guardrail."""
    df = summary[summary["target"] == target].copy()
    if df.empty:
        return df

    mode = str(rules.selection_mode).lower()
    if mode not in {"robust", "strict"}:
        raise ValueError(
            f"selection_mode={rules.selection_mode!r} — допустимы 'robust' | 'strict'")

    df["pass_r2_median"] = df["r2_median"] >= rules.min_r2_median
    df["pass_r2_mean"] = df["r2_mean"] >= rules.min_r2_mean
    if mode == "robust":
        df["pass_r2"] = df["pass_r2_median"]
    else:
        df["pass_r2"] = df["pass_r2_median"] & df["pass_r2_mean"]
    df["pass_catastrophic"] = (
        df["catastrophic_rate_mean"] <= rules.max_catastrophic_rate_mean)
    df["pass_zero_crossing"] = (
        df["zero_crossing_coverage"] >= rules.min_zero_crossing_coverage)
    df["pass_stable_window"] = (
        df["stable_window_coverage"] >= rules.min_stable_window_coverage)
    df["admissible"] = (
        df["pass_r2"]
        & df["pass_catastrophic"]
        & df["pass_zero_crossing"]
        & df["pass_stable_window"]
    )

    reasons: list[str] = []
    for _, row in df.iterrows():
        fail: list[str] = []
        if not bool(row["pass_r2_median"]):
            fail.append(f"r2_med<{rules.min_r2_median:.2f}")
        if mode == "strict" and not bool(row["pass_r2_mean"]):
            fail.append(f"r2_mean<{rules.min_r2_mean:.2f}")
        if not bool(row["pass_catastrophic"]):
            fail.append(
                f"cat>{rules.max_catastrophic_rate_mean:.2f}")
        if not bool(row["pass_zero_crossing"]):
            fail.append(
                f"zcov<{rules.min_zero_crossing_coverage:.2f}")
        if not bool(row["pass_stable_window"]):
            fail.append(
                f"scov<{rules.min_stable_window_coverage:.2f}")
        reasons.append("" if not fail else ";".join(fail))
    df["admissibility_fail_reasons"] = reasons
    df["selection_mode"] = mode
    return df


def _select_finalists(admissible_df: pd.DataFrame,
                      rules: SelectionTargetRules) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Возвращает finalists и метаданные использованного квантиля."""
    if admissible_df.empty:
        return admissible_df.copy(), {
            "selection_quantile_used": rules.lt_top_quantile,
            "selection_mode": "empty_after_admissibility",
            "lt_cutoff_rank": 0,
            "trajectory_cutoff_rank": 0,
        }

    lt_ranked = _rank_lt(admissible_df)
    traj_ranked = _rank_trajectory(admissible_df)
    merged = lt_ranked.merge(
        traj_ranked[["model_id", "trajectory_rank"]],
        on="model_id",
        how="left",
    )

    n = len(merged)
    for quantile, mode in (
        (rules.lt_top_quantile, "primary_quantile"),
        (rules.fallback_quantile, "fallback_quantile"),
    ):
        lt_cut = _top_k_from_quantile(n, quantile)
        tr_cut = _top_k_from_quantile(n, rules.trajectory_top_quantile
                                      if mode == "primary_quantile"
                                      else rules.fallback_quantile)
        finalists = merged[
            (merged["lt_rank"] <= lt_cut)
            & (merged["trajectory_rank"] <= tr_cut)
        ].copy()
        if not finalists.empty:
            finalists = finalists.sort_values(
                ["lt_rank", "trajectory_rank", "lt_mae_median_policy_mean"],
                kind="stable",
            ).reset_index(drop=True)
            return finalists, {
                "selection_quantile_used": quantile,
                "selection_mode": mode,
                "lt_cutoff_rank": lt_cut,
                "trajectory_cutoff_rank": tr_cut,
            }

    # Редкий случай: даже fallback-пересечение пусто.
    return merged.head(0).copy(), {
        "selection_quantile_used": rules.fallback_quantile,
        "selection_mode": "empty_after_intersection",
        "lt_cutoff_rank": _top_k_from_quantile(len(merged), rules.fallback_quantile),
        "trajectory_cutoff_rank": _top_k_from_quantile(len(merged), rules.fallback_quantile),
    }


def build_selection_tables(model_summary: pd.DataFrame,
                           cfg: AnalysisConfig) -> dict[str, pd.DataFrame]:
    """Строит таблицы raw/admissible/finalists/selected_for_shap по target."""
    if cfg.selection_family:
        summary_scope = model_summary[
            model_summary["family"] == cfg.selection_family].copy()
    else:
        summary_scope = model_summary.copy()

    raw_rows: list[pd.DataFrame] = []
    admissibility_rows: list[pd.DataFrame] = []
    finalists_rows: list[pd.DataFrame] = []
    shap_rows: list[pd.DataFrame] = []

    for target in TARGETS:
        rules = _rules_for(cfg, target)
        target_df = summary_scope[summary_scope["target"] == target].copy()
        if target_df.empty:
            continue

        raw = _rank_lt(target_df)
        raw["selection_target"] = target
        raw["selection_table"] = "raw_lt"
        raw_rows.append(raw)

        admissibility = _admissibility_table(summary_scope, target, rules)
        admissibility["selection_target"] = target
        admissibility_rows.append(admissibility)

        admissible_only = admissibility[admissibility["admissible"]].copy()
        finalists, meta = _select_finalists(admissible_only, rules)
        if not finalists.empty:
            for key, value in meta.items():
                finalists[key] = value
            finalists["selection_target"] = target
            finalists_rows.append(finalists)

            selected = finalists.sort_values(
                ["lt_rank", "trajectory_rank", "lt_mae_median_policy_mean"],
                kind="stable",
            ).copy()
            selected["selection_target"] = target
            selected["selection_table"] = "selected_for_shap"
            shap_rows.append(selected)

    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    return {
        "raw_lt": _concat(raw_rows),
        "admissibility": _concat(admissibility_rows),
        "finalists": _concat(finalists_rows),
        "selected_for_shap": _concat(shap_rows),
    }
