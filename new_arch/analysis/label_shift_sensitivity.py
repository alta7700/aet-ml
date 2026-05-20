"""Анализ чувствительности LT-метрик к сдвигу лактатных меток.

Логика: капиллярный лактат запаздывает относительно внутримышечного на ~60–90 с.
Применяем сдвиг label_shift_sec к y_true (y_true_shifted = y_true - shift),
пересчитываем LT-метрики для топ-моделей из selected_for_shap.csv — без
переобучения, только post-hoc на сохранённых предсказаниях Layer 1.

Использование:
    cd new_arch
    uv run python -m analysis.label_shift_sensitivity \
        --config analysis/config.toml \
        [--shifts 0 45 90 120] \
        [--out-name sensitivity_label_shift]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.schemas import AnalysisConfig, load_config
from analysis.aggregation import read_predictions_selected


# ── вспомогательная функция вычисления LT-метрик при заданном сдвиге ────────

def _lt_metrics_for_shift(preds: pd.DataFrame,
                           shift_sec: float,
                           stable_thr_sec: float) -> pd.DataFrame:
    """Пересчитывает LT-метрики для всех моделей при сдвиге shift_sec.

    y_true_s = y_true - shift_sec (предполагаем, что истинный LT наступил раньше)
    Возвращает DataFrame: model_id, target, shift_sec, lt_mae_median, lt_mae_crossing,
    lt_bias_median, stable_coverage, zero_crossing_coverage.
    """
    df = preds.copy()
    df["y_true_s"] = df["y_true"] - shift_sec
    df["lt_true_sec_s"] = df["lt_true_sec"] - shift_sec

    rows: list[dict] = []

    for (model_id, target), grp in df.groupby(["model_id", "target"],
                                               sort=True, observed=True):
        subj_rows: list[dict] = []

        for subject_id, g in grp.groupby("subject_id", sort=True, observed=True):
            g_s = g.sort_values("sample_index", kind="stable")
            y_true_s = g_s["y_true_s"].to_numpy()
            y_pred = g_s["y_pred"].to_numpy()
            end = g_s["sample_end_sec"].to_numpy()

            lt_true = float(np.median(g_s["lt_true_sec_s"].to_numpy()))

            # policy 1: median
            lt_hat_med = float(np.median(g_s["lt_hat_sec"].to_numpy()))

            # policy 2: crossing
            crossed = np.where(y_pred <= 0.0)[0]
            if crossed.size > 0:
                lt_hat_cross = float(end[crossed[0]])
                zero_found = True
            else:
                lt_hat_cross = float(end[-1] + y_pred[-1])
                zero_found = False

            # policy 3: stable_median
            mask = np.abs(y_pred) <= stable_thr_sec
            lt_hat_stable = (float(np.median(g_s.loc[mask, "lt_hat_sec"].to_numpy()))
                             if mask.any() else lt_hat_med)

            subj_rows.append({
                "subject_id": subject_id,
                "lt_true_sec": lt_true,
                "abs_lt_err_median": abs(lt_hat_med - lt_true),
                "lt_err_median": lt_hat_med - lt_true,
                "abs_lt_err_crossing": abs(lt_hat_cross - lt_true),
                "lt_err_crossing": lt_hat_cross - lt_true,
                "abs_lt_err_stable": abs(lt_hat_stable - lt_true),
                "zero_crossing_found": zero_found,
                "stable_found": bool(mask.any()),
            })

        sub_df = pd.DataFrame(subj_rows)
        rows.append({
            "model_id": model_id,
            "target": target,
            "shift_sec": shift_sec,
            "lt_mae_median": sub_df["abs_lt_err_median"].mean(),
            "lt_bias_median": sub_df["lt_err_median"].mean(),
            "lt_mae_crossing": sub_df["abs_lt_err_crossing"].mean(),
            "lt_bias_crossing": sub_df["lt_err_crossing"].mean(),
            "lt_mae_stable": sub_df["abs_lt_err_stable"].mean(),
            "zero_crossing_coverage": sub_df["zero_crossing_found"].mean(),
            "stable_coverage": sub_df["stable_found"].mean(),
            "n_subjects": len(sub_df),
        })

    return pd.DataFrame(rows)


# ── основная функция ─────────────────────────────────────────────────────────

def run(cfg: AnalysisConfig,
        shifts: list[float],
        out_name: str = "sensitivity_label_shift") -> pd.DataFrame:
    """Запускает sensitivity sweep по shifts и сохраняет результаты."""

    # Топ-модели из selected_for_shap
    shap_path = cfg.out_root / "tables" / "selected_for_shap.csv"
    if not shap_path.exists():
        raise FileNotFoundError(
            f"Файл {shap_path} не найден. Сначала запустите основной pipeline.")
    selected_ids = set(pd.read_csv(shap_path)["model_id"].tolist())
    print(f"Топ-моделей из selected_for_shap: {len(selected_ids)}")

    # Загружаем Layer 1 только для нужных моделей
    all_preds = read_predictions_selected(cfg)
    preds = all_preds[all_preds["model_id"].isin(selected_ids)].copy()
    if preds.empty:
        raise RuntimeError("Предсказания для выбранных моделей не найдены в Layer 1.")
    print(f"Загружено строк предсказаний: {len(preds)} "
          f"({preds['model_id'].nunique()} моделей)")

    # Sweep по сдвигам
    results: list[pd.DataFrame] = []
    for shift in shifts:
        part = _lt_metrics_for_shift(
            preds, shift_sec=shift,
            stable_thr_sec=cfg.stable_window_threshold_sec)
        results.append(part)
        print(f"  shift={int(shift):4d}s  "
              + "  ".join(
                  f"{row['target'].upper()} {row['model_id'][:12]} "
                  f"mae_med={row['lt_mae_median']:.0f}s"
                  for _, row in part.iterrows()))

    out_df = pd.concat(results, ignore_index=True)

    # Добавляем имена моделей для читаемости таблицы
    meta_cols = pd.read_csv(shap_path)[["model_id", "model_name", "feature_set",
                                         "family"]].drop_duplicates("model_id")
    out_df = out_df.merge(meta_cols, on="model_id", how="left")

    # Сохраняем
    tables_dir = cfg.out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{out_name}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\nСохранено: {csv_path}")

    # Сводная таблица для диссертации: строки=shift, столбцы=модели
    pivot = out_df.pivot_table(
        index="shift_sec",
        columns=["target", "model_id"],
        values="lt_mae_median",
        aggfunc="first",
    ).round(1)
    tex_path = tables_dir / f"{out_name}.tex"
    pivot.to_latex(tex_path, float_format="%.1f", na_rep="—")
    print(f"LaTeX pivot: {tex_path}")

    # Краткая сводка в консоль
    print("\n── Сводка MAE(median policy) по сдвигам ──")
    summary = out_df.groupby(["target", "shift_sec"])["lt_mae_median"].agg(
        ["mean", "min", "max"]).round(1)
    print(summary.to_string())

    return out_df


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity анализ LT-метрик к сдвигу лактатных меток.")
    parser.add_argument("--config", default="analysis/config.toml")
    parser.add_argument("--shifts", nargs="+", type=float,
                        default=[0, 45, 90, 120],
                        help="Список сдвигов в секундах (default: 0 45 90 120)")
    parser.add_argument("--out", default=None,
                        help="Переопределить out_root из конфига")
    parser.add_argument("--out-name", default="sensitivity_label_shift",
                        help="Базовое имя выходных файлов")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.out:
        cfg.out_root = Path(args.out).resolve()
    cfg.out_root = Path(cfg.out_root).resolve()
    cfg.results_root = Path(cfg.results_root).resolve()

    run(cfg, shifts=args.shifts, out_name=args.out_name)


if __name__ == "__main__":
    main()
