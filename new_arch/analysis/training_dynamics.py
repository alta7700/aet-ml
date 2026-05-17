"""Training stability / convergence diagnostics из history.csv.

Без validation-сплита это НЕ overfitting в строгом смысле — это анализ
динамики обучения по train-метрикам: насколько быстро/гладко модель
сходится, сколько фолдов вышли на плато, какая внутренняя стабильность
траектории. Все термины в коде и таблицах называются именно так.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.loader import DiscoveryResult
from analysis.schemas import (
    AnalysisConfig,
    TRAINING_DYNAMICS_COLUMNS,
    TRAINING_SUMMARY_COLUMNS,
)


# ─── Чтение history.csv ─────────────────────────────────────────────────────

def _read_history_for_model(model_dir: Path) -> pd.DataFrame | None:
    """Читает history.csv для одной модели. Возвращает None если нет (Linear)."""
    path = model_dir / "history.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"{model_dir.name}: history.csv не читается ({exc})")
        return None


# ─── Производные на per-epoch dataframe ─────────────────────────────────────

def _add_per_epoch_derivatives(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Δ-производные и rolling means внутри fold'а."""
    df = df.sort_values(["fold_id", "epoch"], kind="stable").copy()
    df["train_loss_delta"] = df.groupby("fold_id")["train_loss"].diff()
    df["train_mae_delta"] = df.groupby("fold_id")["train_mae"].diff()
    df["train_loss_rolling_mean"] = (
        df.groupby("fold_id")["train_loss"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    df["train_mae_rolling_mean"] = (
        df.groupby("fold_id")["train_mae"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    return df


# ─── Per-fold scalars (для дальнейшей агрегации) ────────────────────────────

def _slope_last_k(values: np.ndarray, k: int) -> float:
    """Наклон линейной регрессии последних k точек (в единицах метрики / epoch)."""
    if len(values) < 2:
        return float("nan")
    take = values[-k:] if len(values) >= k else values
    x = np.arange(len(take), dtype=float)
    if len(take) < 2:
        return float("nan")
    a, _ = np.polyfit(x, take, deg=1)
    return float(a)


def _instability(values: np.ndarray, k: int) -> float:
    """Coefficient of variation абсолютных delta на последних k шагах."""
    if len(values) < 3:
        return float("nan")
    take = np.diff(values[-k:]) if len(values) >= k else np.diff(values)
    take = np.abs(take)
    mean_v = float(np.mean(take))
    if mean_v <= 1e-12:
        return 0.0
    return float(np.std(take, ddof=1) / mean_v)


def _per_fold_features(history: pd.DataFrame, *, window: int,
                       relative_threshold: float) -> pd.DataFrame:
    """Сворачивает history.csv в одну строку на fold.

    Сходимость считается относительно: |slope|·K / mean(train_mae_last_K)
    меньше ``relative_threshold`` (default 0.05 → менее 5% изменения за окно).
    """
    out: list[dict] = []
    for fold_id, g in history.groupby("fold_id", observed=True):
        g = g.sort_values("epoch", kind="stable")
        train_mae = g["train_mae"].to_numpy(dtype=float)
        train_loss = g["train_loss"].to_numpy(dtype=float)
        mae_slope = _slope_last_k(train_mae, window)
        loss_slope = _slope_last_k(train_loss, window)

        # относительный масштаб
        tail = train_mae[-window:] if len(train_mae) >= window else train_mae
        base = float(np.mean(tail)) if tail.size else float("nan")
        if np.isnan(mae_slope) or not np.isfinite(base) or base <= 0:
            relative_change = float("nan")
            converged = False
        else:
            relative_change = abs(mae_slope) * min(window, len(train_mae)) / base
            converged = bool(relative_change < relative_threshold)

        out.append({
            "fold_id": fold_id,
            "n_epochs": int(len(g)),
            "final_train_mae": float(train_mae[-1]) if train_mae.size else float("nan"),
            "final_train_loss": float(train_loss[-1]) if train_loss.size else float("nan"),
            "train_mae_slope_last_K": mae_slope,
            "train_loss_slope_last_K": loss_slope,
            "relative_change_last_K": relative_change,
            "training_instability": _instability(train_mae, window),
            "converged": converged,
        })
    return pd.DataFrame(out)


# ─── Builders ───────────────────────────────────────────────────────────────

def build_training_dynamics(disc: DiscoveryResult,
                            cfg: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит две таблицы.

    1. ``training_dynamics`` — per (model_id × fold_id × epoch) с производными.
    2. ``training_summary``  — per model_id с per-fold агрегатами (mean/std/...).

    Linear-модели имеют epoch=0 и одну точку — для них summary заполняется
    NaN'ами по slope/instability/converged (метрики неопределены), но строка
    создаётся для единообразия (n_folds, final_train_mae_mean записываются).
    """
    meta_df = disc.models_df.set_index("model_id", drop=False)
    window = cfg.stability_window_epochs
    rel_thr = cfg.convergence_relative_threshold

    dyn_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for model_id, mdir in disc.model_dirs.items():
        if model_id not in meta_df.index:
            continue
        history = _read_history_for_model(mdir)
        if history is None:
            continue  # Linear или повреждённый history.csv

        meta = meta_df.loc[model_id]
        history = history.copy()
        history["model_id"] = model_id
        history["architecture_id"] = str(meta["architecture_id"])
        history["family"] = str(meta["family"])
        history["target"] = str(meta["target"])

        # производные на per-epoch уровне
        dyn = _add_per_epoch_derivatives(
            history[["model_id", "architecture_id", "family", "target",
                     "fold_id", "epoch", "train_loss", "train_mae", "lr"]],
            window=window)
        dyn_rows.append(dyn)

        # per-fold scalars и сворачивание в одну строку на model_id
        fold_feat = _per_fold_features(
            history, window=window, relative_threshold=rel_thr)
        summary_rows.append({
            "model_id": model_id,
            "architecture_id": str(meta["architecture_id"]),
            "family": str(meta["family"]),
            "target": str(meta["target"]),
            "selected_epoch": int(history["epoch"].max()),
            "n_folds": int(len(fold_feat)),
            "n_epochs_mean": float(fold_feat["n_epochs"].mean())
                              if not fold_feat.empty else float("nan"),
            "final_train_mae_mean": float(fold_feat["final_train_mae"].mean()),
            "final_train_mae_std": float(fold_feat["final_train_mae"].std(ddof=1))
                                    if len(fold_feat) > 1 else 0.0,
            "final_train_loss_mean": float(fold_feat["final_train_loss"].mean()),
            "train_mae_slope_last_K_mean": float(np.nanmean(
                fold_feat["train_mae_slope_last_K"])),
            "train_loss_slope_last_K_mean": float(np.nanmean(
                fold_feat["train_loss_slope_last_K"])),
            "training_instability_mean": float(np.nanmean(
                fold_feat["training_instability"])),
            "converged_rate": float(fold_feat["converged"].mean()),
            "relative_change_last_K_mean": float(np.nanmean(
                fold_feat["relative_change_last_K"])),
            "stability_window_epochs": window,
            "convergence_relative_threshold": rel_thr,
        })

    dynamics_df = (pd.concat(dyn_rows, ignore_index=True)
                   if dyn_rows else pd.DataFrame(columns=TRAINING_DYNAMICS_COLUMNS))
    summary_df = pd.DataFrame(summary_rows,
                              columns=TRAINING_SUMMARY_COLUMNS)

    # порядок колонок в dynamics
    cols = [c for c in TRAINING_DYNAMICS_COLUMNS if c in dynamics_df.columns]
    extra = [c for c in dynamics_df.columns if c not in cols]
    dynamics_df = dynamics_df[cols + extra]

    cfg.training_dynamics_path.parent.mkdir(parents=True, exist_ok=True)
    dynamics_df.to_parquet(cfg.training_dynamics_path, index=False)
    summary_df.to_parquet(cfg.training_summary_path, index=False)
    return dynamics_df, summary_df
