"""Ranking models по метрикам model_summary."""

from __future__ import annotations

import pandas as pd


def rank_by(summary: pd.DataFrame, metric: str,
            ascending: bool = True) -> pd.DataFrame:
    """Сортирует summary по метрике, добавляет колонку ``rank``."""
    if metric not in summary.columns:
        raise KeyError(f"Колонка {metric!r} отсутствует в model_summary")
    out = summary.sort_values(metric, ascending=ascending, kind="stable").reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


def rank_primary(summary: pd.DataFrame) -> pd.DataFrame:
    """Главный ranking: lt_mae_median_policy_mean ascending."""
    return rank_by(summary, "lt_mae_median_policy_mean", ascending=True)


def rank_secondary(summary: pd.DataFrame) -> pd.DataFrame:
    """Вспомогательный ranking: mae_mean ascending."""
    return rank_by(summary, "mae_mean", ascending=True)


def rank_composite(summary: pd.DataFrame) -> pd.DataFrame:
    return rank_by(summary, "composite_score", ascending=True)


def top_n_by_target(summary: pd.DataFrame, n: int = 10,
                    metric: str = "lt_mae_median_policy_mean") -> dict[str, pd.DataFrame]:
    """Top-N для каждого target."""
    out: dict[str, pd.DataFrame] = {}
    for target, g in summary.groupby("target"):
        ranked = rank_by(g, metric, ascending=True)
        out[str(target)] = ranked.head(n).reset_index(drop=True)
    return out


# ─── Robustness ranking ─────────────────────────────────────────────────────

def _zscore(s: pd.Series) -> pd.Series:
    """z-нормировка с защитой от вырожденного std."""
    import numpy as np
    mu = s.mean(skipna=True); sd = s.std(skipna=True, ddof=1)
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def build_robustness_ranking(model_summary: pd.DataFrame, *,
                             policy: str = "median",
                             primary_alpha: float = 0.2,
                             ) -> pd.DataFrame:
    """Отдельный robustness-ranking: точечная точность + uncertainty + stability.

    Это НЕ замена primary ranking. Это сводный взгляд "одной цифрой" поверх
    point performance, conformal coverage/width и training stability. Каждый
    суб-скор z-нормирован и хранится отдельно для трассировки.
    """
    import numpy as np
    if model_summary.empty:
        return pd.DataFrame()

    aaa = f"{int(round(primary_alpha * 100)):03d}"
    cov_col = f"conformal_coverage_{policy}_policy_at_alpha_{aaa}"
    wid_col = f"conformal_width_{policy}_policy_at_alpha_{aaa}_mean"

    df = model_summary.copy()
    nominal = 1.0 - primary_alpha
    if cov_col in df.columns:
        df["conformal_undercoverage_penalty_020"] = (
            (nominal - df[cov_col]).clip(lower=0))
    else:
        df["conformal_undercoverage_penalty_020"] = np.nan

    # суб-скоры (меньше = лучше)
    df["point_score"] = (
        _zscore(df["lt_mae_median_policy_mean"]).fillna(0)
        + 0.5 * _zscore(df["catastrophic_rate_mean"]).fillna(0))
    if wid_col in df.columns and cov_col in df.columns:
        df["uncertainty_score"] = (
            _zscore(df[wid_col]).fillna(0)
            + 1.0 * _zscore(df["conformal_undercoverage_penalty_020"]).fillna(0))
    else:
        df["uncertainty_score"] = 0.0
    if "training_instability_mean" in df.columns:
        # инверсия converged_rate: чем больше rate тем лучше → минусуем z.
        cv = _zscore(df["converged_rate"]).fillna(0) if "converged_rate" in df.columns else 0.0
        df["stability_score"] = (
            _zscore(df["training_instability_mean"]).fillna(0) - cv)
    else:
        df["stability_score"] = 0.0

    # robustness = взвешенная сумма; равные веса по дефолту
    df["robustness_total"] = (
        df["point_score"] + df["uncertainty_score"] + df["stability_score"])
    df["robustness_rank"] = df["robustness_total"].rank(
        method="min", ascending=True).astype(int)

    keep = [
        "model_id", "architecture_id", "family", "target",
        "feature_set", "with_abs", "wavelet_mode",
        "lt_mae_median_policy_mean", "catastrophic_rate_mean",
        cov_col, wid_col,
        "conformal_undercoverage_penalty_020",
        "training_instability_mean", "converged_rate",
        "point_score", "uncertainty_score", "stability_score",
        "robustness_total", "robustness_rank",
    ]
    # Стандартизуем имена для экспорта.
    rename_map = {
        cov_col: "conformal_coverage_at_alpha_020",
        wid_col: "conformal_width_at_alpha_020",
    }
    keep_existing = [c for c in keep if c in df.columns]
    out = df[keep_existing].rename(columns=rename_map)
    out = out.sort_values("robustness_rank", kind="stable").reset_index(drop=True)
    return out
