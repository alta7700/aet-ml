"""Layer 4: парные сравнения групп моделей (modality, ABS, family, ...).

Каждое сравнение строится на subject-aligned векторах primary-метрики.
Когда внутри одной группы несколько моделей соответствуют одному условию
(например, две LSTM-модели в EMG+NIRS), берётся среднее по моделям внутри
субъекта — это устойчивее к выборам внутри семейства, чем "лучшая модель".
"""

from __future__ import annotations

import warnings
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.schemas import AnalysisConfig, COMPARISON_COLUMNS
from analysis.statistics import paired_test

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:  # pragma: no cover
    multipletests = None  # type: ignore[assignment]


def _apply_multitest_correction(df: pd.DataFrame, *, method: str,
                                alpha: float) -> pd.DataFrame:
    """Корректирует Wilcoxon p-values методом ``method`` внутри метрики.

    Семья поправки — (comparison_kind × metric): primary и secondary
    дорожки корректируются независимо, потому что отвечают на разные
    научные вопросы. comparison_kind = одна аналитическая ось.

    NaN p-values исключаются из коррекции, в выход возвращаются как NaN.
    """
    if df.empty:
        for col in ("pvalue_adj", "correction_method", "reject_at_alpha"):
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        return df

    if multipletests is None:
        raise ImportError(
            "statsmodels не установлен — нужен для multitest correction. "
            "uv sync --extra analysis")

    df = df.copy()
    df["pvalue_adj"] = np.nan
    df["correction_method"] = method
    df["reject_at_alpha"] = False

    for metric, grp_idx in df.groupby("metric").groups.items():
        sub = df.loc[grp_idx, "wilcoxon_pvalue"]
        mask = sub.notna() & np.isfinite(sub.to_numpy(dtype=float))
        if mask.sum() == 0:
            continue
        pvals = sub[mask].to_numpy(dtype=float)
        reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
        df.loc[sub.index[mask], "pvalue_adj"] = p_adj
        df.loc[sub.index[mask], "reject_at_alpha"] = reject

    return df


# ─── helpers ────────────────────────────────────────────────────────────────

def _per_subject_mean(metric_df: pd.DataFrame, *, group_keys: list[str],
                      condition_col: str, metric: str) -> pd.DataFrame:
    """Группа × условие × subject → среднее метрики по моделям.

    Возвращает long DataFrame: ``group_keys + [condition_col, subject_id, metric]``.
    """
    cols = group_keys + [condition_col, "subject_id", metric]
    sub = metric_df[cols].copy()
    grouped = (
        sub.groupby(group_keys + [condition_col, "subject_id"],
                    observed=True, dropna=False)[metric]
        .mean()
        .reset_index()
    )
    return grouped


def _pair_subject_align(per_subj: pd.DataFrame, *, group_keys: list[str],
                        condition_col: str, metric: str,
                        cond_a: str, cond_b: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Берёт две condition внутри одной группы, возвращает (vec_a, vec_b, n_subj)."""
    a = per_subj[per_subj[condition_col] == cond_a]
    b = per_subj[per_subj[condition_col] == cond_b]
    merged = a.merge(b, on=group_keys + ["subject_id"],
                     suffixes=("_a", "_b"))
    if merged.empty:
        return np.array([]), np.array([]), 0
    return (
        merged[f"{metric}_a"].to_numpy(),
        merged[f"{metric}_b"].to_numpy(),
        int(len(merged)),
    )


def _row(comparison_kind: str, group_key: str, metric: str,
         cond_a: str, cond_b: str, res) -> dict:
    return {
        "comparison_kind": comparison_kind,
        "group_key": group_key,
        "metric": metric,
        "condition_a": cond_a,
        "condition_b": cond_b,
        **asdict(res),
    }


def _build_pairwise(metric_df: pd.DataFrame, *, comparison_kind: str,
                    group_keys: list[str], condition_col: str,
                    metric: str, cfg: AnalysisConfig,
                    allowed_conditions: list[str] | None = None) -> pd.DataFrame:
    """Универсальный построитель: для каждой группы — все пары условий."""
    if condition_col not in metric_df.columns:
        warnings.warn(
            f"{comparison_kind}: колонка {condition_col!r} отсутствует, skip")
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    per_subj = _per_subject_mean(
        metric_df, group_keys=group_keys, condition_col=condition_col, metric=metric)
    if per_subj.empty:
        warnings.warn(f"{comparison_kind}: пустой набор после группировки")
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    rows: list[dict] = []
    for keys, g in per_subj.groupby(group_keys, observed=True, dropna=False):
        keys_t = (keys,) if not isinstance(keys, tuple) else keys
        gkey = "|".join(f"{k}={v}" for k, v in zip(group_keys, keys_t))
        conds = sorted(g[condition_col].dropna().unique().tolist())
        if allowed_conditions is not None:
            conds = [c for c in conds if c in allowed_conditions]
        if len(conds) < 2:
            continue
        for a, b in combinations(conds, 2):
            va, vb, n = _pair_subject_align(
                g, group_keys=group_keys, condition_col=condition_col,
                metric=metric, cond_a=a, cond_b=b)
            if n < 2:
                continue
            res = paired_test(va, vb,
                              bootstrap_n=cfg.bootstrap_n,
                              ci_level=cfg.ci_level,
                              seed=cfg.bootstrap_seed)
            rows.append(_row(comparison_kind, gkey, metric, a, b, res))

    df = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    return df


# ─── обязательные сравнения ─────────────────────────────────────────────────

def family_overall(metric_df: pd.DataFrame, *, metric: str,
                   cfg: AnalysisConfig) -> pd.DataFrame:
    return _build_pairwise(
        metric_df,
        comparison_kind="family_overall",
        group_keys=["target"],
        condition_col="family",
        metric=metric, cfg=cfg)


def modality_within_family(metric_df: pd.DataFrame, *, metric: str,
                            cfg: AnalysisConfig) -> pd.DataFrame:
    return _build_pairwise(
        metric_df,
        comparison_kind="modality_within_family",
        group_keys=["family", "target", "with_abs"],
        condition_col="feature_set",
        metric=metric, cfg=cfg)


def abs_within_arch(metric_df: pd.DataFrame, *, metric: str,
                    cfg: AnalysisConfig) -> pd.DataFrame:
    """Сравнение with_abs True/False внутри (arch, target, feature_set)."""
    df = metric_df.copy()
    df["with_abs_str"] = df["with_abs"].map(
        {True: "abs", False: "no_abs"})
    return _build_pairwise(
        df,
        comparison_kind="abs_within_arch",
        group_keys=["architecture_id", "target", "feature_set"],
        condition_col="with_abs_str",
        metric=metric, cfg=cfg)


def wavelet_within_family(metric_df: pd.DataFrame, *, metric: str,
                          cfg: AnalysisConfig) -> pd.DataFrame:
    return _build_pairwise(
        metric_df,
        comparison_kind="wavelet_within_family",
        group_keys=["family", "target", "feature_set"],
        condition_col="wavelet_mode",
        metric=metric, cfg=cfg)


def lt1_vs_lt2_per_model_class(metric_df: pd.DataFrame, *, metric: str,
                                cfg: AnalysisConfig) -> pd.DataFrame:
    """LT1 vs LT2 для одной и той же конфигурации (model_name + features)."""
    return _build_pairwise(
        metric_df,
        comparison_kind="lt1_vs_lt2_per_model_class",
        group_keys=["model_name", "feature_set", "with_abs"],
        condition_col="target",
        metric=metric, cfg=cfg)


# ─── условные сравнения ─────────────────────────────────────────────────────

def stateful_vs_stateless(metric_df: pd.DataFrame, *, metric: str,
                          cfg: AnalysisConfig) -> pd.DataFrame:
    """Сравнение Stateful vs Stateless внутри LSTM."""
    df = metric_df[metric_df["family"] == "LSTM"].copy()
    if df.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    df["state_kind"] = df["model_name"].astype(str).map(
        lambda s: "stateful" if "Stateful" in s else "stateless")
    if df["state_kind"].nunique() < 2:
        warnings.warn("stateful_vs_stateless: нет обоих режимов, skip")
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    return _build_pairwise(
        df,
        comparison_kind="stateful_vs_stateless",
        group_keys=["target", "feature_set", "with_abs"],
        condition_col="state_kind",
        metric=metric, cfg=cfg)


def attention_vs_plain(metric_df: pd.DataFrame, *, metric: str,
                       cfg: AnalysisConfig) -> pd.DataFrame:
    """Attention LSTM vs обычный LSTM."""
    df = metric_df[metric_df["family"] == "LSTM"].copy()
    if df.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    df["attn_kind"] = df["model_name"].astype(str).map(
        lambda s: "attention" if "Attention" in s else "plain")
    if df["attn_kind"].nunique() < 2:
        warnings.warn("attention_vs_plain: нет обоих режимов, skip")
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    return _build_pairwise(
        df,
        comparison_kind="attention_vs_plain",
        group_keys=["target", "feature_set", "with_abs"],
        condition_col="attn_kind",
        metric=metric, cfg=cfg)


def stride_within_arch(metric_df: pd.DataFrame, *, metric: str,
                       cfg: AnalysisConfig) -> pd.DataFrame:
    """Сравнение разных stride_sec внутри (family, target, feature_set, with_abs)."""
    df = metric_df.copy()
    df["stride_str"] = df["stride_sec"].astype(int).astype(str)
    if df["stride_str"].nunique() < 2:
        warnings.warn("stride_within_arch: один stride на всю выборку, skip")
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    return _build_pairwise(
        df,
        comparison_kind="stride_within_arch",
        group_keys=["family", "target", "feature_set", "with_abs"],
        condition_col="stride_str",
        metric=metric, cfg=cfg)


# ─── dispatcher ─────────────────────────────────────────────────────────────

REGISTRY = {
    "family_overall": family_overall,
    "modality_within_family": modality_within_family,
    "abs_within_arch": abs_within_arch,
    "wavelet_within_family": wavelet_within_family,
    "lt1_vs_lt2_per_model_class": lt1_vs_lt2_per_model_class,
    "stateful_vs_stateless": stateful_vs_stateless,
    "attention_vs_plain": attention_vs_plain,
    "stride_within_arch": stride_within_arch,
}


def build_all_comparisons(lt_point_metrics_df: pd.DataFrame,
                          subject_metrics_df: pd.DataFrame,
                          cfg: AnalysisConfig) -> dict[str, Path]:
    """Запускает обязательные + условные сравнения, пишет parquet'ы."""
    cfg.comparisons_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # primary метрика берётся из lt_point_metrics — это abs_lt_err_median_sec
    # на (model_id, subject_id). secondary метрика из subject_metrics — mae.
    primary_metric = "abs_lt_err_median_sec"
    secondary_metric = "mae"

    runs = [
        (cfg.comparisons_required, False),
        (cfg.comparisons_optional, True),
    ]
    for names, is_optional in runs:
        for name in names:
            if name not in REGISTRY:
                warnings.warn(f"Unknown comparison: {name}, skip")
                continue
            fn = REGISTRY[name]
            # Primary
            try:
                df_p = fn(lt_point_metrics_df, metric=primary_metric, cfg=cfg)
            except Exception as exc:
                if is_optional:
                    warnings.warn(f"{name} (primary) пропущен: {exc}")
                    df_p = pd.DataFrame(columns=COMPARISON_COLUMNS)
                else:
                    raise
            # Secondary
            try:
                df_s = fn(subject_metrics_df, metric=secondary_metric, cfg=cfg)
            except Exception as exc:
                if is_optional:
                    warnings.warn(f"{name} (secondary) пропущен: {exc}")
                    df_s = pd.DataFrame(columns=COMPARISON_COLUMNS)
                else:
                    raise

            combined = pd.concat([df_p, df_s], ignore_index=True)
            if combined.empty and is_optional:
                warnings.warn(f"{name}: нет данных, skip запись")
                continue
            # Поправка на множественные сравнения внутри (comparison_kind × metric).
            combined = _apply_multitest_correction(
                combined,
                method=cfg.multitest_correction,
                alpha=cfg.multitest_alpha,
            )
            # Привести порядок колонок к каноничной схеме.
            extra = [c for c in combined.columns if c not in COMPARISON_COLUMNS]
            combined = combined[[c for c in COMPARISON_COLUMNS
                                 if c in combined.columns] + extra]
            out_path = cfg.comparisons_dir / f"{name}.parquet"
            combined.to_parquet(out_path, index=False)
            written[name] = out_path
    return written
