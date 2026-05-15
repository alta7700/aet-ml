"""Conformal-интервалы LOSO split-conformal для noabs-победителей.

Компаньон final_shap_conformal.py: для тех же четырёх финалистов
(lt2/HRV, lt2/ENH, lt1/EN, lt1/ENH) исключает абсолютные признаки
из EXCLUDE_ABS и повторно прогоняет LOSO с Ridge α=1000, после чего
строит conformal-интервалы. Дописывает строки в coverage_summary.csv
с тегами вида lt2_HRV_noabs.

SHAP в noabs **не запускается** — интерпретация в тексте остаётся
на with_abs (для LT2) и на отдельных SVR-расчётах (для LT1).

Запуск:
    PYTHONPATH=. uv run python scripts/final_conformal_noabs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Переиспользуем реализованную инфраструктуру.
from final_shap_conformal import (  # noqa: E402
    CONFORMAL_ALPHAS,
    CONF_DIR,
    FINALISTS,
    TARGET_COL,
    load_subject_meta,
    loso_predictions,
    run_conformal,
    plot_conformal,
)
from v0011_modality_ablation import (  # noqa: E402
    EXCLUDE_ABS,
    prepare_data,
    get_feature_cols,
)

DATASET_DIR = ROOT / "dataset"


def main() -> None:
    CONF_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка датасета…")
    df_raw = pd.read_parquet(DATASET_DIR / "merged_features_ml.parquet")
    sp = pd.read_parquet(DATASET_DIR / "session_params.parquet")
    meta = load_subject_meta()

    df_by_target = {
        t: prepare_data(df_raw, sp, t).dropna(subset=[TARGET_COL[t]])
        for t in ("lt1", "lt2")
    }

    all_conf = []
    summary_rows: list[dict] = []

    for target, fset, tag in FINALISTS:
        tag_na = f"{tag}_noabs"
        print(f"\n[{tag_na}] target={target}, feature_set={fset} (noabs)")
        df = df_by_target[target]
        full_cols = get_feature_cols(df, fset)
        feat_cols = [c for c in full_cols if c not in EXCLUDE_ABS]
        n_removed = len(full_cols) - len(feat_cols)
        print(f"  признаков: {len(feat_cols)} (исключено {n_removed} абсолютных)")
        if n_removed == 0:
            print("  Внимание: в этом наборе нечего исключать (нет абс.) — пропуск.")
            continue

        loso = loso_predictions(df, feat_cols, TARGET_COL[target])
        mae = mean_absolute_error(loso["y_true"], loso["y_pred"]) / 60.0
        print(f"  LOSO raw MAE noabs = {mae:.3f} мин")

        conf = run_conformal(loso, meta, tag_na)
        all_conf.append(conf)

        for alpha in CONFORMAL_ALPHAS:
            sub = conf[conf["alpha"] == alpha]
            summary_rows.append({
                "tag": tag_na, "target": target, "feature_set": fset,
                "variant": "noabs",
                "alpha": alpha, "target_coverage": 1 - alpha,
                "mean_empirical_coverage": round(sub["empirical_coverage"].mean(), 4),
                "mean_halfwidth_min": round(sub["q_halfwidth_min"].mean(), 3),
                "halfwidth_trained_min": round(
                    sub[sub["trained"] == 1]["q_halfwidth_min"].mean(), 3),
                "halfwidth_untrained_min": round(
                    sub[sub["trained"] == 0]["q_halfwidth_min"].mean(), 3),
            })

    # Объединяем с уже существующей сводкой for_abs (если есть)
    out_summary_path = CONF_DIR / "coverage_summary.csv"
    summary_df = pd.DataFrame(summary_rows)
    if out_summary_path.exists():
        existing = pd.read_csv(out_summary_path)
        if "variant" not in existing.columns:
            existing["variant"] = "with_abs"
        # Удаляем потенциальные дубли по tag (на случай повторного запуска)
        existing = existing[~existing["tag"].isin(summary_df["tag"])]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(out_summary_path, index=False)
    print("\n" + "=" * 70)
    print("Объединённая coverage_summary (with_abs + noabs):")
    print(summary_df.to_string(index=False))

    # Пересоздаём сводный график со всеми тегами
    conf_all = pd.concat(all_conf, ignore_index=True) if all_conf else pd.DataFrame()
    # Подмешиваем with_abs-данные из существующих файлов
    wa_frames = []
    for _, _, tag in FINALISTS:
        p = CONF_DIR / tag / "intervals.csv"
        if p.exists():
            wa_frames.append(pd.read_csv(p))
    if wa_frames:
        wa_all = pd.concat(wa_frames, ignore_index=True)
        conf_all = pd.concat([wa_all, conf_all], ignore_index=True)

    if not conf_all.empty:
        plot_conformal(conf_all, CONF_DIR / "conformal_summary.png")
        print(f"\n→ {CONF_DIR / 'conformal_summary.png'} (8 моделей)")


if __name__ == "__main__":
    main()
