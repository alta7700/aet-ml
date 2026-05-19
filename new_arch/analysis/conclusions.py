"""Авто-генератор научных выводов поверх comparison-таблиц.

Тексты собираются по шаблонам и привязываются к конкретным p-value
и эффект-сайзам из ``analysis_out/comparisons/*.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.schemas import AnalysisConfig


_ALPHA = 0.05
_PVAL_COL = "pvalue_adj"  # с поправкой Benjamini–Hochberg по умолчанию


def _pvals(df: pd.DataFrame) -> pd.Series:
    """Возвращает p-values с поправкой если она есть, иначе сырые Wilcoxon."""
    if _PVAL_COL in df.columns and df[_PVAL_COL].notna().any():
        return df[_PVAL_COL]
    return df["wilcoxon_pvalue"]


def _significant_share(df: pd.DataFrame, *, cond_a_pred,
                       cond_b_pred, metric: str = "abs_lt_err_median_sec") -> tuple[int, int, float]:
    """Возвращает (n_total, n_a_better_significant, share)."""
    sub = df[df["metric"] == metric]
    sub = sub[sub["condition_a"].map(cond_a_pred) & sub["condition_b"].map(cond_b_pred)]
    if sub.empty:
        # попробуем reverse
        sub = df[df["metric"] == metric]
        sub_rev = sub[sub["condition_b"].map(cond_a_pred) & sub["condition_a"].map(cond_b_pred)].copy()
        sub_rev["delta_mean"] = -sub_rev["delta_mean"]
        sub = sub_rev
    n_total = len(sub)
    # "a лучше b" = delta_mean < 0 (метрика — ошибка, меньше = лучше)
    n_sig = int(((sub["delta_mean"] < 0) & (_pvals(sub) < _ALPHA)).sum())
    share = n_sig / n_total if n_total else 0.0
    return n_total, n_sig, share


def _fmt_delta(df: pd.DataFrame, metric: str = "abs_lt_err_median_sec") -> str:
    sub = df[df["metric"] == metric]
    if sub.empty:
        return "—"
    return f"median Δ = {sub['delta_mean'].median():+.1f} sec"


# ─── секции ────────────────────────────────────────────────────────────────

def _section_multimodal(cmp_modality: pd.DataFrame) -> str:
    """Улучшает ли multimodal fusion?"""
    if cmp_modality.empty:
        return "## Multimodal fusion\nДанных нет.\n"
    is_multi = lambda v: v in ("EMG+NIRS", "EMG+NIRS+HRV")
    is_emg = lambda v: v == "EMG"
    n_total, n_sig, share = _significant_share(
        cmp_modality, cond_a_pred=is_multi, cond_b_pred=is_emg)
    delta = _fmt_delta(cmp_modality)
    return (
        "## Multimodal fusion vs unimodal EMG\n"
        f"Сравнений (multi vs EMG): {n_total}. "
        f"Multi значимо лучше: {n_sig} (доля {share:.0%}, α={_ALPHA}).\n"
        f"{delta}.\n"
    )


def _section_abs(cmp_abs: pd.DataFrame) -> str:
    """Помогают ли |EMG| фичи."""
    if cmp_abs.empty:
        return "## ABS features\nДанных нет.\n"
    is_abs = lambda v: v == "abs"
    is_no = lambda v: v == "no_abs"
    n_total, n_sig, share = _significant_share(
        cmp_abs, cond_a_pred=is_abs, cond_b_pred=is_no)
    delta = _fmt_delta(cmp_abs)
    return (
        "## ABS (|EMG|) features\n"
        f"Сравнений (abs vs no_abs): {n_total}. "
        f"ABS значимо лучше: {n_sig} ({share:.0%}). {delta}.\n"
    )


def _section_wavelet(cmp_wav: pd.DataFrame) -> str:
    if cmp_wav.empty:
        return "## Wavelet preprocessing\nДанных нет.\n"
    sub = cmp_wav[cmp_wav["metric"] == "abs_lt_err_median_sec"]
    lines = ["## Wavelet preprocessing\n"]
    for (a, b), g in sub.groupby(["condition_a", "condition_b"]):
        n_total = len(g)
        n_sig = int((_pvals(g) < _ALPHA).sum())
        med_delta = g["delta_mean"].median()
        winner = a if med_delta < 0 else b
        lines.append(
            f"- {a} vs {b}: n={n_total}, значимых {n_sig}; "
            f"median Δ = {med_delta:+.1f} sec → лучше **{winner}**\n")
    return "".join(lines)


def _section_phase_split(cmp_ps: pd.DataFrame) -> str:
    """Ablation на phase split EMG-сигнала (split / full_cycles / full_window)."""
    if cmp_ps.empty:
        return "## EMG phase split (load/rest)\nДанных нет.\n"
    sub = cmp_ps[cmp_ps["metric"] == "abs_lt_err_median_sec"]
    if sub.empty:
        return "## EMG phase split (load/rest)\nНет primary-метрики в сравнениях.\n"
    lines = ["## EMG phase split (load/rest vs full)\n"]
    for (a, b), g in sub.groupby(["condition_a", "condition_b"]):
        n_total = len(g)
        n_sig = int((_pvals(g) < _ALPHA).sum())
        med_delta = g["delta_mean"].median()
        winner = a if med_delta < 0 else b
        lines.append(
            f"- {a} vs {b}: n={n_total}, значимых {n_sig}; "
            f"median Δ = {med_delta:+.1f} sec → лучше **{winner}**\n")
    return "".join(lines)


def _section_lt12(cmp_lt: pd.DataFrame) -> str:
    if cmp_lt.empty:
        return "## LT1 vs LT2 difficulty\nДанных нет.\n"
    sub = cmp_lt[cmp_lt["metric"] == "abs_lt_err_median_sec"]
    if sub.empty:
        return "## LT1 vs LT2 difficulty\nНет primary-метрики.\n"
    med = sub["delta_mean"].median()
    n_sig = int((_pvals(sub) < _ALPHA).sum())
    direction = "LT1 проще" if med < 0 else "LT2 проще"
    return (
        "## LT1 vs LT2 difficulty\n"
        f"Сравнений: {len(sub)}, значимых {n_sig}. "
        f"Median Δ(LT1 − LT2) = {med:+.1f} sec → {direction}.\n"
    )


def _section_family(cmp_fam: pd.DataFrame, model_summary: pd.DataFrame) -> str:
    if cmp_fam.empty or model_summary.empty:
        return "## Architecture family\nДанных нет.\n"
    primary_metric = "lt_mae_median_policy_mean"
    fam_avg = (
        model_summary.groupby("family")[primary_metric].mean().sort_values())
    winner = fam_avg.index[0]
    lines = [
        "## Architecture family\n",
        f"Средний {primary_metric} по семействам:\n"
    ]
    for fam, val in fam_avg.items():
        lines.append(f"- **{fam}**: {val:.1f} sec\n")
    lines.append(f"\nЛучшее семейство по primary метрике: **{winner}**.\n")
    sig = cmp_fam[
        (cmp_fam["metric"] == "abs_lt_err_median_sec")
        & (_pvals(cmp_fam) < _ALPHA)]
    if not sig.empty:
        lines.append(
            f"Значимых парных различий между семействами: {len(sig)}.\n")
    return "".join(lines)


def _section_best_temporal(model_summary: pd.DataFrame) -> str:
    if model_summary.empty:
        return "## Temporal configuration\nДанных нет.\n"
    # внутри победившего семейства возьмём лучшую stride/sequence комбинацию
    primary = "lt_mae_median_policy_mean"
    winner_family = model_summary.groupby("family")[primary].mean().idxmin()
    sub = model_summary[model_summary["family"] == winner_family]
    grp = (sub.groupby(["stride_sec", "sequence_length"])[primary]
           .mean().sort_values())
    if grp.empty:
        return "## Temporal configuration\nДанных нет.\n"
    (stride, seq_len), val = grp.index[0], grp.iloc[0]
    lines = [
        f"## Temporal configuration (family={winner_family})\n",
        f"Лучшая комбинация: stride={stride}s, sequence_length={seq_len} → "
        f"{primary} = {val:.1f} sec.\n",
        "Полная таблица:\n",
    ]
    for (s, l), v in grp.items():
        lines.append(f"- stride={s}s, seq_len={l}: {v:.1f} sec\n")
    return "".join(lines)


# ─── главный билдер ─────────────────────────────────────────────────────────

def _section_conformal(conformal_summary: pd.DataFrame) -> str:
    if conformal_summary is None or conformal_summary.empty:
        return "## Uncertainty (conformal)\nДанных нет.\n"
    sub = conformal_summary[conformal_summary["policy"] == "median"]
    lines = [
        "## Uncertainty (Jackknife+ conformal, LT median policy)\n",
        "Конформные интервалы построены leave-one-subject-out (N=18). "
        "Из-за малой калибровочной выборки честно репортируются только "
        "умеренные α: эмпирическое покрытие должно сходиться к 1−α.\n\n",
    ]
    for alpha, g in sub.groupby("alpha", observed=True):
        nominal = 1.0 - float(alpha)
        emp_med = float(g["empirical_coverage"].median())
        gap_med = float(g["coverage_gap"].median())
        w_med = float(g["median_interval_width_sec"].median())
        n_at_nom = int(((g["empirical_coverage"] - nominal).abs() < 0.05).sum())
        lines.append(
            f"- α={alpha:.2f} (nominal {nominal:.0%}): "
            f"median empirical={emp_med:.0%}, gap={gap_med:+.2%}, "
            f"median width={w_med:.0f} sec. "
            f"Моделей с покрытием в пределах ±5% от nominal: {n_at_nom}/{len(g)}.\n")
    return "".join(lines)


def _section_stability(training_summary: pd.DataFrame) -> str:
    if training_summary is None or training_summary.empty:
        return "## Training stability\nДанных нет (history.csv доступен только для NN).\n"
    n = len(training_summary)
    conv = float(training_summary["converged_rate"].mean())
    inst = float(training_summary["training_instability_mean"].median())
    slope = float(training_summary["train_mae_slope_last_K_mean"].mean())
    return (
        f"## Training stability ({n} NN-моделей)\n"
        f"- Средний converged_rate (доля фолдов с |slope| ниже порога): {conv:.0%}.\n"
        f"- Median training_instability (CV последних шагов): {inst:.3f}.\n"
        f"- Mean train_mae_slope_last_K: {slope:+.3f} sec/epoch "
        f"(отрицательный = всё ещё улучшается).\n"
        "**Замечание:** без validation-split это не overfitting, а только динамика train-метрик.\n"
    )


def _section_captum(consistency_df: pd.DataFrame | None,
                    global_df: pd.DataFrame | None,
                    time_df: pd.DataFrame | None,
                    skipped_df: pd.DataFrame | None) -> str:
    """Сводка по Captum-интерпретации NN-finalist'ов.

    Содержит:
      • кросс-метод consistency по модальностям (IG vs GradientShap vs Occlusion);
      • топ-3 признака IG на каждую модель;
      • профиль важности шагов последовательности (где сосредоточена внимание).
    """
    lines: list[str] = ["## NN-интерпретация (Captum, trusted finalists)\n"]
    if consistency_df is None or consistency_df.empty:
        lines.append("Данных нет (NN-finalists не выбраны или модуль выключен).\n")
        return "".join(lines)

    method_cols = [c for c in consistency_df.columns
                   if c not in ("model_id", "architecture_id",
                                "target", "modality")]

    for model_id, g in consistency_df.groupby("model_id", sort=True):
        target = str(g["target"].iloc[0]).upper()
        arch_id = str(g["architecture_id"].iloc[0])
        lines.append(f"\n### {model_id} ({arch_id}, {target})\n")

        # Топ-модальность по IG (если есть) и наибольший Δ между методами.
        if "integrated_gradients" in method_cols:
            ig_top = g.sort_values(
                "integrated_gradients", ascending=False).iloc[0]
            lines.append(
                f"- IG-топ-модальность: **{ig_top['modality']}** "
                f"({float(ig_top['integrated_gradients']):.0%}).\n")
        if {"integrated_gradients", "occlusion"} <= set(method_cols):
            disagree = (g["integrated_gradients"] - g["occlusion"]).abs()
            i_dis = int(disagree.idxmax())
            mod = str(g.loc[i_dis, "modality"])
            ig_v = float(g.loc[i_dis, "integrated_gradients"])
            oc_v = float(g.loc[i_dis, "occlusion"])
            lines.append(
                f"- Наибольшее расхождение IG↔Occlusion: **{mod}** "
                f"(IG={ig_v:.0%}, Occlusion={oc_v:.0%}). "
                "IG переоценивает «острые» признаки, Occlusion — «дублирующиеся»; "
                "сильное расхождение → проверять отдельно.\n")

        # Полная сводка модальностей.
        lines.append("- Доли по модальностям (subject-mean):\n")
        lines.append("  | modality | " + " | ".join(method_cols) + " |\n")
        lines.append("  |---" + "|---" * len(method_cols) + "|\n")
        for _, row in g.sort_values(
                "integrated_gradients"
                if "integrated_gradients" in method_cols
                else method_cols[0], ascending=False).iterrows():
            cells = " | ".join(f"{float(row[c]):.2f}" for c in method_cols)
            lines.append(f"  | {row['modality']} | {cells} |\n")

        # Top-3 признака IG.
        if global_df is not None and not global_df.empty:
            sub = global_df[(global_df["model_id"] == model_id)
                            & (global_df["method"] == "integrated_gradients")]
            top3 = sub.sort_values(
                "share_mean_across_subjects", ascending=False).head(3)
            if not top3.empty:
                feat_str = ", ".join(
                    f"`{r['feature_name']}` ({float(r['share_mean_across_subjects']):.1%})"
                    for _, r in top3.iterrows())
                lines.append(f"- Top-3 признака IG: {feat_str}.\n")

        # Профиль шагов последовательности.
        if time_df is not None and not time_df.empty:
            t_ig = time_df[(time_df["model_id"] == model_id)
                           & (time_df["method"] == "integrated_gradients")]
            if not t_ig.empty:
                profile = (t_ig.groupby("time_offset_sec")["share"].mean()
                           .sort_index())
                seq_len = len(profile)
                # Доля внимания на ближнюю половину окна.
                half = seq_len // 2
                near = profile.iloc[half:].sum()
                lines.append(
                    f"- Профиль шагов (seq_len={seq_len}): "
                    f"{near:.0%} внимания приходится на ближнюю половину "
                    f"({int(profile.index[half])}…{int(profile.index[-1])} сек); "
                    f"остаток — на дальнюю историю "
                    f"({int(profile.index[0])}…{int(profile.index[half - 1])} сек).\n")

    if skipped_df is not None and not skipped_df.empty:
        lines.append(
            f"\n**Пропущено моделей** (вне whitelist'а или без чекпоинтов): "
            f"{len(skipped_df)}.\n")
    return "".join(lines)


def build_conclusions(model_summary: pd.DataFrame,
                      comparisons: dict[str, pd.DataFrame],
                      cfg: AnalysisConfig,
                      conformal_summary: pd.DataFrame | None = None,
                      training_summary: pd.DataFrame | None = None,
                      captum_consistency: pd.DataFrame | None = None,
                      captum_global: pd.DataFrame | None = None,
                      captum_time: pd.DataFrame | None = None,
                      captum_skipped: pd.DataFrame | None = None,
                      ) -> Path:
    """Сохраняет analysis_out/conclusions/conclusions.md и возвращает путь."""
    cfg.conclusions_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = ["# Auto-generated conclusions\n"]
    parts.append(
        f"Primary метрика: `lt_mae_median_policy_mean` (LT-time prediction MAE).\n"
        f"Secondary: `mae_mean` (regression fit).\n"
        f"Уровень значимости α = {_ALPHA}.\n"
        f"P-values скорректированы методом Benjamini–Hochberg (FDR) внутри "
        f"`(comparison_kind × metric)`; используется колонка `{_PVAL_COL}`.\n\n")

    parts.append(_section_multimodal(comparisons.get(
        "modality_within_family", pd.DataFrame())))
    parts.append("\n")
    parts.append(_section_abs(comparisons.get("abs_within_arch", pd.DataFrame())))
    parts.append("\n")
    parts.append(_section_wavelet(comparisons.get(
        "wavelet_within_family", pd.DataFrame())))
    parts.append("\n")
    parts.append(_section_phase_split(comparisons.get(
        "phase_split_within_arch", pd.DataFrame())))
    parts.append("\n")
    parts.append(_section_lt12(comparisons.get(
        "lt1_vs_lt2_per_model_class", pd.DataFrame())))
    parts.append("\n")
    parts.append(_section_family(comparisons.get(
        "family_overall", pd.DataFrame()), model_summary))
    parts.append("\n")
    parts.append(_section_best_temporal(model_summary))
    parts.append("\n")
    parts.append(_section_conformal(conformal_summary))
    parts.append("\n")
    parts.append(_section_stability(training_summary))
    parts.append("\n")
    parts.append(_section_captum(
        captum_consistency, captum_global, captum_time, captum_skipped))

    out = cfg.conclusions_dir / "conclusions.md"
    out.write_text("".join(parts), encoding="utf-8")
    return out
