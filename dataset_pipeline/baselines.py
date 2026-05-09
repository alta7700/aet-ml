"""Честные baseline-модели для оценки sequential регрессоров.

Обязательная часть пайплайна оценки — запускается для каждой
(feature_set, target) пары рядом с основным зоопарком моделей.

Два baseline:
  MeanPredictor       — предсказывает медиану таргета из обучающих субъектов.
                        Измеряет межсубъектную вариабельность.
  FirstWindowPredictor — та же архитектура модели, но обученная ТОЛЬКО на первом
                         окне каждого субъекта и предсказывающая фиксированное
                         значение для всего теста.
                         Если модель ≤ FirstWindow — sequential tracking не добавляет ценности.

Stability score:
  std(window_start_sec + y_pred) per subject, усреднённое.
  ≈ 0   → модель = таймер (фиксирует полную длительность с первого окна).
  большой → модель реально обновляется по ходу теста.

Verdicts (gap = raw_mae − fw_mae):
  gap < −0.2  → ✓ sequential tracking значимо лучше FirstWindow
  |gap| ≤ 0.2 → ~ нет значимой онлайн-ценности
  gap > +0.2  → ⚠ sequential tracking хуже FirstWindow
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


# ─── MeanPredictor ────────────────────────────────────────────────────────────

def mean_predictor_loso(df: pd.DataFrame,
                         target_col: str) -> dict[str, dict]:
    """LOSO MeanPredictor: медиана таргета из train-субъектов.

    Предсказывает время_до_порога = median(train_LT) − elapsed_sec.
    """
    subjects = sorted(df["subject_id"].unique())
    result: dict[str, dict] = {}

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        train_lt = train.groupby("subject_id")[target_col].first()
        pred_val = float(train_lt.median())

        t_sec  = test["window_start_sec"].values
        y_true = test[target_col].values
        y_pred = np.maximum(pred_val - t_sec, 0.0)

        result[test_s] = {"t_sec": t_sec, "y_pred": y_pred, "y_true": y_true}

    return result


# ─── FirstWindowPredictor ────────────────────────────────────────────────────

def first_window_predictor_loso(df: pd.DataFrame,
                                 feat_cols: list[str],
                                 target_col: str,
                                 model_factory) -> dict[str, dict]:
    """LOSO FirstWindowPredictor: модель обучается только на первых окнах.

    Для каждого тестового субъекта предсказывает одно фиксированное значение
    (predicted_LT_abs), из которого вычитается текущее elapsed_sec.
    Отражает ценность «начального снимка» без последующего обновления.
    """
    subjects = sorted(df["subject_id"].unique())
    result: dict[str, dict] = {}

    for test_s in subjects:
        train_full = df[df["subject_id"] != test_s]
        test_full  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        train_first = train_full.groupby("subject_id").first().reset_index()
        test_first  = test_full.iloc[[0]]

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        mdl = model_factory()

        X_tr = sc.fit_transform(imp.fit_transform(train_first[feat_cols].values))
        X_te = sc.transform(imp.transform(test_first[feat_cols].values))
        mdl.fit(X_tr, train_first[target_col].values)

        pred_duration = float(mdl.predict(X_te)[0])
        t_sec  = test_full["window_start_sec"].values
        y_true = test_full[target_col].values
        y_pred = np.maximum(pred_duration - t_sec, 0.0)

        result[test_s] = {
            "t_sec":         t_sec,
            "y_pred":        y_pred,
            "y_true":        y_true,
            "pred_duration": pred_duration,
        }

    return result


# ─── Метрики ─────────────────────────────────────────────────────────────────

def metrics_from_per_subject(per_subject: dict,
                              kalman_fn=None,
                              sigma_p: float = 0.0,
                              sigma_obs: float = 0.0) -> dict:
    """MAE (мин), R², ρ, stability_std (с) из per-subject словаря.

    kalman_fn: callable(y_pred, sigma_p, sigma_obs) → smoothed.
    Если None — оценивает сырые предсказания.
    """
    all_pred, all_true, stab_stds = [], [], []

    for d in per_subject.values():
        y_pred = d["y_pred"].copy()
        y_true = d["y_true"]
        t_sec  = d["t_sec"]

        if kalman_fn is not None:
            y_pred = kalman_fn(y_pred, sigma_p, sigma_obs)

        all_pred.append(y_pred)
        all_true.append(y_true)
        stab_stds.append(float(np.std(t_sec + y_pred)))

    y_p = np.concatenate(all_pred)
    y_t = np.concatenate(all_true)

    return {
        "mae_min":       round(mean_absolute_error(y_t, y_p) / 60.0, 4),
        "r2":            round(r2_score(y_t, y_p), 3),
        "rho":           round(float(spearmanr(y_t, y_p).statistic), 3),
        "stability_std": round(float(np.mean(stab_stds)), 1),
    }


def verdict(gap: float) -> str:
    """Вердикт по gap = raw_mae_min − first_window_mae_min."""
    if gap < -0.2:
        return "✓ sequential значимо лучше FirstWindow"
    if gap > 0.2:
        return "⚠ sequential хуже FirstWindow (нет онлайн-ценности)"
    return "~ нет значимой разницы с FirstWindow"


# ─── Публичный API ────────────────────────────────────────────────────────────

def run_honest_baselines(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    raw_per_subject: dict,
    model_factory,
    kalman_fn=None,
    sigma_p: float = 5.0,
    sigma_obs_ref: float = 150.0,
) -> dict:
    """Запускает оба baseline и возвращает сводный dict.

    Параметры
    ----------
    raw_per_subject : результат loso_raw_per_subject() из основного скрипта
    model_factory   : та же factory, что использовалась для raw_per_subject
    kalman_fn       : функция Kalman (из v0011_modality_ablation)
    sigma_obs_ref   : σ_obs для справочной Kalman-метрики (обычно 150)

    Возвращает
    ----------
    dict с ключами:
      mean_pred, first_win   — метрики baseline-ов
      raw                    — метрики сырой модели
      kalman_ref             — метрики с kalman_fn(sigma_p, sigma_obs_ref)
      gap_raw_vs_fw          — raw_mae − fw_mae
      verdict                — строка-вердикт
      stability_raw          — stability_std сырой модели
      stability_fw           — stability_std FirstWindowPredictor
    """
    # MeanPredictor
    mean_ps = mean_predictor_loso(df, target_col)
    mean_m  = metrics_from_per_subject(mean_ps)

    # FirstWindowPredictor
    fw_ps = first_window_predictor_loso(df, feat_cols, target_col, model_factory)
    fw_m  = metrics_from_per_subject(fw_ps)

    # Сырая модель
    raw_m = metrics_from_per_subject(raw_per_subject)

    # Kalman (справочная σ)
    if kalman_fn is not None:
        kal_m = metrics_from_per_subject(
            raw_per_subject, kalman_fn, sigma_p, sigma_obs_ref)
    else:
        kal_m = raw_m

    gap = raw_m["mae_min"] - fw_m["mae_min"]

    return {
        "mean_pred":       mean_m,
        "first_win":       fw_m,
        "raw":             raw_m,
        "kalman_ref":      kal_m,
        "gap_raw_vs_fw":   round(gap, 4),
        "verdict":         verdict(gap),
        "stability_raw":   raw_m["stability_std"],
        "stability_fw":    fw_m["stability_std"],
    }


def format_honest_block(result: dict,
                         feature_set: str,
                         target: str) -> str:
    """Форматирует честный блок для report.md."""
    r = result
    lines = [
        f"### Honest baselines — {feature_set} / {target.upper()}",
        "",
        "| Модель | MAE (мин) | Stability std (с) |",
        "|---|---|---|",
        f"| MeanPredictor       | {r['mean_pred']['mae_min']:.3f} | — |",
        f"| FirstWindowPredictor | {r['first_win']['mae_min']:.3f} "
        f"| {r['stability_fw']:.0f} |",
        f"| Raw model (no Kalman) | {r['raw']['mae_min']:.3f} "
        f"| {r['stability_raw']:.0f} |",
        f"| Kalman (σ_obs=ref)  | {r['kalman_ref']['mae_min']:.3f} | — |",
        "",
        f"gap (raw − FirstWin): {r['gap_raw_vs_fw']:+.3f} мин",
        f"**{r['verdict']}**",
        "",
    ]
    return "\n".join(lines)
