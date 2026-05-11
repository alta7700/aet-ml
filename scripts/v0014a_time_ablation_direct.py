"""v0014a — Ablation по модальностям без прямых носителей времени/мощности

Версия:    v0014a
Дата:      2026-05-11
Предыдущая версия: v0011_modality_ablation.py
Результаты: results/v0014a/

Что изменено:
  Тот же зоопарк и LOSO что в v0011, но из всех наборов признаков исключены
  четыре прямых носителя времени и мощности:
    - smo2_from_running_max   (накопленное падение SmO₂ с начала теста)
    - hhb_from_running_min    (накопленный рост HHb с начала теста)
    - smo2_rel_drop_pct       (% падения SmO₂ от базового уровня покоя)
    - feat_rr_per_watt        (RR / текущая мощность — прямой носитель ступени)

  Абсолютные уровни (trainred_*_mean, hrv_mean_rr_ms) оставлены.
  Kalman-постобработка сохранена для сопоставимости с v0011.

Гипотеза:
  Убрав прямые носители времени, оцениваем вклад «знания о протоколе»
  в точность модели. Сравниваем с v0011 по Δ MAE.

Воспроизведение:
  uv run python scripts/v0014a_time_ablation_direct.py
  uv run python scripts/v0014a_time_ablation_direct.py --no-shap
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Импортируем весь зоопарк, LOSO, Kalman и утилиты из v0011
from scripts.v0011_modality_ablation import (
    _build_zoo,
    prepare_data,
    get_feature_cols as _get_feature_cols_v0011,
    run_zoo,
    kalman_smooth,
    compute_shap,
    write_report,
    _loso_to_per_subject,
)
from dataset_pipeline.common import DEFAULT_DATASET_DIR
from dataset_pipeline.baselines import run_honest_baselines, format_honest_block

OUT_DIR = _ROOT / "results" / "v0014a"

# Признаки-носители времени/мощности — исключаются из всех наборов
EXCLUDE = frozenset([
    "smo2_from_running_max",
    "hhb_from_running_min",
    "smo2_rel_drop_pct",
    "feat_rr_per_watt",
])


def get_feature_cols(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Обёртка над v0011: те же наборы, минус EXCLUDE."""
    cols = _get_feature_cols_v0011(df, feature_set)
    return [c for c in cols if c not in EXCLUDE]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0014a — ablation без прямых носителей времени")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--feature-set", nargs="+",
                   choices=["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default=["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"])
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--sigma-p",   type=float, default=15.0)
    p.add_argument("--sigma-obs", type=float, default=150.0)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("v0014a — ABLATION БЕЗ ПРЯМЫХ НОСИТЕЛЕЙ ВРЕМЕНИ/МОЩНОСТИ")
    print(f"Исключены: {sorted(EXCLUDE)}")
    print("=" * 70)

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shap_dir = OUT_DIR / "shap"

    zoo = _build_zoo()
    print(f"Зоопарк: {len(zoo)} конфигураций")

    # Загружаем референс v0011 для сравнения
    ref_path = _ROOT / "results" / "v0011" / "best_per_set.csv"
    ref = pd.read_csv(ref_path) if ref_path.exists() else pd.DataFrame()

    targets_cfg = {
        "lt2": {"col": "target_time_to_lt2_center_sec", "label": "lt2"},
        "lt1": {"col": "target_time_to_lt1_sec",        "label": "lt1"},
    }
    if args.target != "both":
        targets_cfg = {k: v for k, v in targets_cfg.items() if k == args.target}

    all_records: list[dict] = []
    honest_records: list[dict] = []
    honest_md_blocks: list[str] = []

    for tgt_name, tgt_cfg in targets_cfg.items():
        print(f"\n{'═'*70}")
        print(f"ТАРГЕТ: {tgt_name.upper()}")
        print(f"{'═'*70}")

        df_prep = prepare_data(df_raw, session_params, tgt_name)
        target_col = tgt_cfg["col"]
        if target_col not in df_prep.columns:
            continue
        df_prep_tgt = df_prep.dropna(subset=[target_col])

        for fset in args.feature_set:
            feat_cols = get_feature_cols(df_prep_tgt, fset)
            if not feat_cols:
                print(f"  [{fset}] — нет признаков, пропуск")
                continue

            excluded_here = [c for c in _get_feature_cols_v0011(df_prep_tgt, fset)
                             if c in EXCLUDE]
            print(f"  [{fset}] исключено {len(excluded_here)}: {excluded_here}")

            records = run_zoo(df_prep_tgt, feat_cols, target_col,
                              fset, tgt_name, zoo,
                              args.sigma_p, args.sigma_obs,
                              n_jobs=args.n_jobs)
            all_records.extend(records)

            best_rec = min(
                (r for r in records if r.get("kalman_mae_min") is not None),
                key=lambda r: r["kalman_mae_min"],
                default=None,
            )
            if best_rec is not None:
                _loso_best = best_rec.get("_loso")
                if _loso_best is not None:
                    fset_tag = fset.replace("+", "_")
                    np.save(OUT_DIR / f"ypred_{tgt_name}_{fset_tag}.npy", _loso_best["y_pred"])
                    np.save(OUT_DIR / f"ytrue_{tgt_name}_{fset_tag}.npy", _loso_best["y_true"])

                best_factory = next(
                    (c["factory"] for c in zoo if c["name"] == best_rec["model"]),
                    None,
                )
                if best_factory is not None:
                    print(f"  [{fset}] honest baselines...", end=" ", flush=True)
                    raw_ps = _loso_to_per_subject(
                        df_prep_tgt, feat_cols, target_col, best_factory)
                    hb = run_honest_baselines(
                        df_prep_tgt, feat_cols, target_col,
                        raw_ps, best_factory,
                        kalman_fn=kalman_smooth,
                        sigma_p=args.sigma_p,
                        sigma_obs_ref=args.sigma_obs,
                    )
                    print(f"gap={hb['gap_raw_vs_fw']:+.3f}  {hb['verdict']}")
                    honest_records.append({
                        "feature_set": fset, "target": tgt_name,
                        "best_model": best_rec["model"],
                        "mae_raw": hb["raw"]["mae_min"],
                        "mae_kalman_ref": hb["kalman_ref"]["mae_min"],
                        "gap_raw_vs_fw": hb["gap_raw_vs_fw"],
                        "verdict": hb["verdict"],
                    })
                    honest_md_blocks.append(format_honest_block(hb, fset, tgt_name))

    summary_rows = [{k: v for k, v in r.items() if k != "_loso"} for r in all_records]
    summary_df = pd.DataFrame(summary_rows).dropna(subset=["kalman_mae_min"])
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\n  → summary.csv ({len(summary_df)} строк)")

    best_df = (summary_df
               .sort_values("kalman_mae_min")
               .groupby(["feature_set", "target"], sort=False)
               .first()
               .reset_index())
    best_df.to_csv(OUT_DIR / "best_per_set.csv", index=False)

    # Вывод итогов с дельтой к v0011
    print("\n" + "═" * 70)
    print("ИТОГИ v0014a vs v0011 (Kalman MAE мин, Δ = v0014a − v0011):")
    print("═" * 70)
    for tgt in ["lt2", "lt1"]:
        sub = best_df[best_df["target"] == tgt].sort_values("kalman_mae_min")
        if sub.empty:
            continue
        print(f"\n  {tgt.upper()}:")
        for _, row in sub.iterrows():
            ref_row = ref[(ref["feature_set"] == row["feature_set"]) &
                          (ref["target"] == tgt)]
            ref_mae = float(ref_row["kalman_mae_min"].iloc[0]) if not ref_row.empty else float("nan")
            delta = row["kalman_mae_min"] - ref_mae
            sign = "+" if delta >= 0 else ""
            print(f"    {row['feature_set']:<16s}  {row['model']:<28s}  "
                  f"kalman={row['kalman_mae_min']:.3f}  Δ={sign}{delta:.3f} мин")

    write_report(summary_df, best_df, OUT_DIR, "\n".join(honest_md_blocks))

    if not args.no_shap:
        print("\n" + "─" * 70)
        print("SHAP для лучших моделей:")
        for _, best_row in best_df.iterrows():
            fset = best_row["feature_set"]
            tgt_name = best_row["target"]
            target_col = targets_cfg[tgt_name]["col"]
            df_prep = prepare_data(df_raw, session_params, tgt_name)
            df_prep_tgt = df_prep.dropna(subset=[target_col])
            feat_cols = get_feature_cols(df_prep_tgt, fset)
            best_factory = next(
                (c["factory"] for c in zoo if c["name"] == best_row["model"]),
                None,
            )
            if best_factory is None:
                continue
            from scripts.v0011_modality_ablation import loso_predict
            loso_res = loso_predict(df_prep_tgt, feat_cols, target_col, best_factory)
            compute_shap(df_prep_tgt, feat_cols, target_col,
                         fset, tgt_name, loso_res, shap_dir)

    print(f"\n✅ Готово. Результаты: {OUT_DIR}")


if __name__ == "__main__":
    main()
