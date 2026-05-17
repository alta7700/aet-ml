"""Сборщики слоёв analysis-кэша.

Layer 1: predictions_best (best-epoch predictions, денормализованные).
Layer 2a: subject_metrics (regression-fit).
Layer 2b: lt_point_metrics (LT-time prediction, 3 policy).
Layer 3: model_summary (агрегаты по субъектам + bootstrap CI + composite).
Layer 4: comparisons (см. также analysis/statistics.py).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.loader import DiscoveryResult
from analysis.metrics import compute_all_subject_metrics
from analysis.schemas import (
    AnalysisConfig,
    LT_POINT_METRIC_COLUMNS,
    LT_POLICIES,
    MODEL_META_COLUMNS,
    MODEL_SUMMARY_COLUMNS,
    PREDICTIONS_SELECTED_COLUMNS,
    SUBJECT_METRIC_COLUMNS,
)


# ─── Layer 1: best-epoch predictions ────────────────────────────────────────

def select_epoch(predictions_df: pd.DataFrame, *,
                 policy: str = "last") -> tuple[int, float]:
    """Возвращает (selected_epoch, subject_mean_mae_at_epoch) по policy.

    Policy:
      "last"  — max(epoch) этой конкретной модели; нет test peeking, primary.
                Разные модели могут иметь разный max_epochs — каждой берём свой last.
      "best"  — argmin subject-mean MAE по эпохам; ВНИМАНИЕ: contains test
                selection bias (выбор гиперпараметра по test), отчётность
                как oracle ceiling. Tie-break: бо́льшая эпоха.
    """
    if predictions_df.empty:
        raise ValueError("predictions_df пуст")

    err = (predictions_df["y_true"] - predictions_df["y_pred"]).abs()
    tmp = predictions_df.assign(__abs_err=err)
    subject_mae = (
        tmp.groupby(["epoch", "subject_id"], observed=True)["__abs_err"]
        .mean()
        .reset_index()
    )
    per_epoch = subject_mae.groupby("epoch")["__abs_err"].mean()

    if policy == "last":
        epoch = int(per_epoch.index.max())
        return epoch, float(per_epoch.loc[epoch])

    if policy == "best":
        per_epoch_sorted = per_epoch.sort_values(kind="stable")
        best_value = per_epoch_sorted.iloc[0]
        eps = 1e-12
        ties = per_epoch_sorted[(per_epoch_sorted - best_value).abs() <= eps].index.tolist()
        return int(max(ties)), float(best_value)

    raise ValueError(f"Unknown epoch_policy={policy!r}; supported: 'last' | 'best'")


# Backwards-compat алиас на случай внешних вызовов.
def select_best_epoch(predictions_df: pd.DataFrame) -> tuple[int, float]:
    """DEPRECATED: используйте select_epoch(..., policy='best')."""
    return select_epoch(predictions_df, policy="best")


def build_predictions_selected(disc: DiscoveryResult, cfg: AnalysisConfig,
                                exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Собирает Layer 1.

    Для каждого model_id:
      1. читает predictions parquet;
      2. определяет best_epoch;
      3. фильтрует строки этой эпохи;
      4. добавляет производные колонки и архитектурную мету;
      5. дописывает в партиционированный dataset.

    Возвращает best_epochs_df (для cache/best_epochs.parquet).
    """
    excl = set(exclude)
    out_dir = cfg.predictions_selected_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # подготовим lookup model_id → ряд models_df
    models_df = disc.models_df.set_index("model_id", drop=False)

    policy = cfg.epoch_policy
    selection_rows: list[dict] = []

    for model_id, path in disc.predictions_paths.items():
        if model_id in excl:
            continue
        if model_id not in models_df.index:
            warnings.warn(f"{model_id}: нет записи в models.csv, пропуск")
            continue

        meta = models_df.loc[model_id]
        df = pd.read_parquet(path)

        selected_epoch, mae_at = select_epoch(df, policy=policy)
        # На случай "best" фиксируем рядом и last для трассировки (и наоборот).
        last_epoch = int(df["epoch"].max())
        selection_rows.append({
            "model_id": model_id,
            "selected_epoch": selected_epoch,
            "epoch_policy": policy,
            "subject_mean_mae_at_selected": mae_at,
            "last_epoch": last_epoch,
            "n_epochs_total": int(df["epoch"].nunique()),
            "n_samples_at_selected": int((df["epoch"] == selected_epoch).sum()),
            "predictions_path": str(path),
            "predictions_mtime": float(Path(path).stat().st_mtime),
        })

        # фильтр и денормализация
        sub = df[df["epoch"] == selected_epoch].copy()
        sub["architecture_id"] = str(meta["architecture_id"])
        sub["target"] = str(meta["target"])
        sub["residual"] = sub["y_pred"] - sub["y_true"]
        sub["abs_err"] = sub["residual"].abs()
        sub["lt_true_sec"] = sub["sample_end_sec"] + sub["y_true"]
        sub["lt_hat_sec"] = sub["sample_end_sec"] + sub["y_pred"]
        sub["selected_epoch"] = int(selected_epoch)

        sub = sub[PREDICTIONS_SELECTED_COLUMNS].copy()

        # запись в партицию: target=…/architecture_id=…
        part_dir = (
            out_dir
            / f"target={sub['target'].iloc[0]}"
            / f"architecture_id={sub['architecture_id'].iloc[0]}"
        )
        part_dir.mkdir(parents=True, exist_ok=True)
        part_path = part_dir / f"part-{model_id}.parquet"
        # колонки target/architecture_id убираем из тела файла —
        # они закодированы в пути партиции
        body = sub.drop(columns=["target", "architecture_id"])
        body.to_parquet(part_path, index=False)

    selection_df = pd.DataFrame(selection_rows)
    cfg.epoch_selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_df.to_parquet(cfg.epoch_selection_path, index=False)
    return selection_df


def read_predictions_selected(cfg: AnalysisConfig) -> pd.DataFrame:
    """Читает партиционированный Layer 1 целиком.

    Партиционирующие колонки (``target``, ``architecture_id``) восстанавливаются
    из путей партиций.
    """
    base = cfg.predictions_selected_dir
    if not base.exists():
        raise FileNotFoundError(
            f"Layer 1 не построен: {base} не существует. Запустите build-cache.")
    # pyarrow dataset API: hive-партиционирование target=… / architecture_id=…
    import pyarrow.dataset as pa_ds
    table = pa_ds.dataset(str(base), format="parquet", partitioning="hive").to_table()
    df = table.to_pandas()
    # упорядочим колонки согласно схеме
    cols = [c for c in PREDICTIONS_SELECTED_COLUMNS if c in df.columns]
    other = [c for c in df.columns if c not in cols]
    return df[cols + other]


# ─── Layer 2a: subject_metrics ──────────────────────────────────────────────

def build_subject_metrics(preds_best: pd.DataFrame, disc: DiscoveryResult,
                          cfg: AnalysisConfig) -> pd.DataFrame:
    """Считает Layer 2a: одна строка на (model_id, subject_id)."""
    meta_df = disc.models_df.set_index("model_id", drop=False)
    rows: list[dict] = []

    for (model_id, subject_id), g in preds_best.groupby(
            ["model_id", "subject_id"], sort=True, observed=True):
        if model_id not in meta_df.index:
            continue
        meta = meta_df.loc[model_id]
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        m = compute_all_subject_metrics(
            y_true, y_pred, threshold_sec=cfg.catastrophic_threshold_sec)
        row = {
            "model_id": model_id,
            "subject_id": subject_id,
            **{k: meta[k] for k in MODEL_META_COLUMNS},
            **m,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # упорядочить колонки
    df = df[[c for c in SUBJECT_METRIC_COLUMNS if c in df.columns]]
    cfg.subject_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.subject_metrics_path, index=False)
    return df


# ─── Layer 2b: lt_point_metrics ─────────────────────────────────────────────

def _subject_lt_estimates(g: pd.DataFrame, stable_thr_sec: float) -> dict:
    """Три policy LT-времени для окон одного субъекта."""
    n = len(g)
    # base
    lt_true = float(np.median(g["lt_true_sec"].to_numpy()))

    # ── policy 1: median по всем lt_hat
    lt_hat_med = float(np.median(g["lt_hat_sec"].to_numpy()))

    # ── policy 2: crossing
    # Первый sample, где y_pred <= 0 → его sample_end_sec.
    g_sorted = g.sort_values("sample_index", kind="stable")
    y_pred = g_sorted["y_pred"].to_numpy()
    end = g_sorted["sample_end_sec"].to_numpy()
    crossed = np.where(y_pred <= 0.0)[0]
    if crossed.size > 0:
        lt_hat_cross = float(end[crossed[0]])
        zero_found = True
    else:
        # пересечения нет → последний sample
        lt_hat_cross = float(end[-1] + y_pred[-1])
        zero_found = False

    # ── policy 3: stable_median
    mask = np.abs(g_sorted["y_pred"].to_numpy()) <= stable_thr_sec
    if mask.any():
        lt_hat_stable = float(np.median(
            g_sorted.loc[mask, "lt_hat_sec"].to_numpy()))
        stable_n = int(mask.sum())
    else:
        # fallback на median по всем
        lt_hat_stable = lt_hat_med
        stable_n = 0

    return {
        "lt_true_sec": lt_true,
        "lt_hat_median_sec": lt_hat_med,
        "lt_err_median_sec": lt_hat_med - lt_true,
        "abs_lt_err_median_sec": abs(lt_hat_med - lt_true),
        "lt_hat_crossing_sec": lt_hat_cross,
        "lt_err_crossing_sec": lt_hat_cross - lt_true,
        "abs_lt_err_crossing_sec": abs(lt_hat_cross - lt_true),
        "zero_crossing_found": zero_found,
        "lt_hat_stable_median_sec": lt_hat_stable,
        "lt_err_stable_median_sec": lt_hat_stable - lt_true,
        "abs_lt_err_stable_median_sec": abs(lt_hat_stable - lt_true),
        "stable_window_count": stable_n,
        "n_samples": n,
    }


def build_lt_point_metrics(preds_best: pd.DataFrame, disc: DiscoveryResult,
                           cfg: AnalysisConfig) -> pd.DataFrame:
    """Считает Layer 2b: LT-time predictions per (model_id, subject_id)."""
    meta_df = disc.models_df.set_index("model_id", drop=False)
    rows: list[dict] = []
    for (model_id, subject_id), g in preds_best.groupby(
            ["model_id", "subject_id"], sort=True, observed=True):
        if model_id not in meta_df.index:
            continue
        meta = meta_df.loc[model_id]
        est = _subject_lt_estimates(g, cfg.stable_window_threshold_sec)
        row = {
            "model_id": model_id,
            "subject_id": subject_id,
            **{k: meta[k] for k in MODEL_META_COLUMNS},
            **est,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[[c for c in LT_POINT_METRIC_COLUMNS if c in df.columns]]
    cfg.lt_point_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.lt_point_metrics_path, index=False)
    return df


# ─── Layer 3: model_summary ─────────────────────────────────────────────────

def _bootstrap_mean_ci(x: np.ndarray, *, B: int, ci: float,
                      rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap percentile CI для среднего."""
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(B, len(x)))
    boots = x[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def _agg_subjects(values: np.ndarray) -> dict:
    """mean/std/median/worst/best на векторе по субъектам (worst = max, best = min)."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "worst": float(np.max(values)),
        "best": float(np.min(values)),
    }


def build_model_summary(subject_metrics_df: pd.DataFrame,
                        lt_point_metrics_df: pd.DataFrame,
                        disc: DiscoveryResult,
                        cfg: AnalysisConfig,
                        best_epochs_df: pd.DataFrame | None = None,
                        conformal_summary_df: pd.DataFrame | None = None,
                        training_summary_df: pd.DataFrame | None = None,
                        ) -> pd.DataFrame:
    """Собирает Layer 3 + composite_score."""
    meta_df = disc.models_df.set_index("model_id", drop=False)
    be = (best_epochs_df.set_index("model_id")
          if best_epochs_df is not None and not best_epochs_df.empty
          else None)
    # обратная совместимость: если в best_epochs только старая колонка "best_epoch"
    if be is not None and "selected_epoch" not in be.columns and "best_epoch" in be.columns:
        be = be.rename(columns={"best_epoch": "selected_epoch"})
    rng = np.random.default_rng(cfg.bootstrap_seed)

    rows: list[dict] = []

    common_ids = (
        set(subject_metrics_df["model_id"]) & set(lt_point_metrics_df["model_id"]))

    for model_id in sorted(common_ids):
        sm = subject_metrics_df[subject_metrics_df["model_id"] == model_id]
        lp = lt_point_metrics_df[lt_point_metrics_df["model_id"] == model_id]
        if model_id not in meta_df.index:
            continue
        meta = meta_df.loc[model_id]

        # дорожка A
        mae_vec = sm["mae"].to_numpy()
        rmse_vec = sm["rmse"].to_numpy()
        bias_vec = sm["bias"].to_numpy()
        r2_vec = sm["r2"].to_numpy()
        cat_vec = sm["catastrophic_rate"].to_numpy()
        prn_vec = sm["pearson_r"].to_numpy()

        mae_a = _agg_subjects(mae_vec)
        mae_ci = _bootstrap_mean_ci(
            mae_vec, B=cfg.bootstrap_n, ci=cfg.ci_level, rng=rng)
        rmse_a = _agg_subjects(rmse_vec)
        bias_a = _agg_subjects(bias_vec)

        row = {
            "model_id": model_id,
            **{k: meta[k] for k in MODEL_META_COLUMNS},
            "selected_epoch": int(be.loc[model_id, "selected_epoch"]) if be is not None
                            and model_id in be.index else -1,
            "epoch_policy": cfg.epoch_policy,
            "n_subjects": int(len(sm)),

            "mae_mean": mae_a["mean"],
            "mae_std": mae_a["std"],
            "mae_median": mae_a["median"],
            "mae_worst": mae_a["worst"],
            "mae_best": mae_a["best"],
            "mae_ci_low": mae_ci[0], "mae_ci_high": mae_ci[1],

            "rmse_mean": rmse_a["mean"], "rmse_std": rmse_a["std"],
            "bias_mean": bias_a["mean"], "bias_std": bias_a["std"],

            "r2_mean": float(np.nanmean(r2_vec)),
            "pearson_r_mean": float(np.nanmean(prn_vec)),
            "catastrophic_rate_mean": float(np.mean(cat_vec)),
        }

        # дорожка B — три policy
        for policy in LT_POLICIES:
            col = f"abs_lt_err_{policy}_sec" if policy != "median" else "abs_lt_err_median_sec"
            # имена в LT_POINT_METRIC_COLUMNS используют формат abs_lt_err_{policy}_sec,
            # но для median мы оставили "abs_lt_err_median_sec".
            col = {
                "median": "abs_lt_err_median_sec",
                "crossing": "abs_lt_err_crossing_sec",
                "stable_median": "abs_lt_err_stable_median_sec",
            }[policy]
            err_col = {
                "median": "lt_err_median_sec",
                "crossing": "lt_err_crossing_sec",
                "stable_median": "lt_err_stable_median_sec",
            }[policy]
            vec = lp[col].to_numpy()
            vec_signed = lp[err_col].to_numpy()
            ag = _agg_subjects(vec)
            ci = _bootstrap_mean_ci(
                vec, B=cfg.bootstrap_n, ci=cfg.ci_level, rng=rng)
            row[f"lt_mae_{policy}_policy_mean"] = ag["mean"]
            row[f"lt_mae_{policy}_policy_std"] = ag["std"]
            row[f"lt_bias_{policy}_policy_mean"] = float(np.mean(vec_signed))
            row[f"lt_mae_{policy}_policy_ci_low"] = ci[0]
            row[f"lt_mae_{policy}_policy_ci_high"] = ci[1]

        row["zero_crossing_coverage"] = float(lp["zero_crossing_found"].mean())
        row["stable_window_coverage"] = float(
            (lp["stable_window_count"] > 0).mean())

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = _add_composite_score(summary, cfg)

    # Денормализация conformal-метрик (если есть): на каждый alpha и policy
    # одна колонка с coverage и одна с шириной.
    if conformal_summary_df is not None and not conformal_summary_df.empty:
        for (policy, alpha), g in conformal_summary_df.groupby(
                ["policy", "alpha"], observed=True):
            aaa = f"{int(round(float(alpha) * 100)):03d}"
            cov_col = f"conformal_coverage_{policy}_policy_at_alpha_{aaa}"
            wid_col = f"conformal_width_{policy}_policy_at_alpha_{aaa}_mean"
            gap_col = f"conformal_coverage_gap_{policy}_policy_at_alpha_{aaa}"
            mapping_cov = dict(zip(g["model_id"], g["empirical_coverage"]))
            mapping_wid = dict(zip(g["model_id"], g["mean_interval_width_sec"]))
            mapping_gap = dict(zip(g["model_id"], g["coverage_gap"]))
            summary[cov_col] = summary["model_id"].map(mapping_cov)
            summary[wid_col] = summary["model_id"].map(mapping_wid)
            summary[gap_col] = summary["model_id"].map(mapping_gap)

    # Денормализация training stability (если есть).
    if training_summary_df is not None and not training_summary_df.empty:
        ts = training_summary_df.set_index("model_id")
        for col in ("training_instability_mean", "converged_rate",
                    "train_mae_slope_last_K_mean", "final_train_mae_mean"):
            if col in ts.columns:
                summary[col] = summary["model_id"].map(ts[col])

    # порядок колонок: канонические сначала, conformal/stability сами по
    # себе денормализованы — оставляем как есть после summary[MODEL_SUMMARY_COLUMNS].
    known = [c for c in MODEL_SUMMARY_COLUMNS if c in summary.columns]
    extra = [c for c in summary.columns if c not in known]
    summary = summary[known + extra]

    cfg.model_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(cfg.model_summary_path, index=False)
    return summary


# Backwards-compat aliases.
def build_predictions_best(disc, cfg, exclude=()):  # noqa: D401
    """DEPRECATED: используйте build_predictions_selected."""
    return build_predictions_selected(disc, cfg, exclude=exclude)


def read_predictions_best(cfg):  # noqa: D401
    """DEPRECATED: используйте read_predictions_selected."""
    return read_predictions_selected(cfg)


def _add_composite_score(summary: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Z-нормировка по совместной таблице + взвешенная сумма."""
    if summary.empty:
        summary["composite_score"] = []
        return summary
    w = cfg.composite_weights

    def z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd

    score = (
        w.lt_mae_median_policy_mean * z(summary["lt_mae_median_policy_mean"]).fillna(0)
        + w.lt_mae_median_policy_std * z(summary["lt_mae_median_policy_std"]).fillna(0)
        + w.catastrophic_rate_mean * z(summary["catastrophic_rate_mean"]).fillna(0)
        + w.abs_lt_bias_median_policy_mean
            * z(summary["lt_bias_median_policy_mean"].abs()).fillna(0)
    )
    summary["composite_score"] = score
    return summary
