"""linear_runner — обучение одной классической Lin*-архитектуры с LOSO.

Сохраняет стандартизированные артефакты new_arch:
  results/{architecture_id}/models.csv           (upsert строки этой модели)
  results/{architecture_id}/{model_id}/predictions_{model_id}.parquet
  results/{architecture_id}/{model_id}/model_{model_id}_fold-{fold_id}.joblib  ×N

Пример запуска:
  PYTHONPATH=. uv run python new_arch/linear_runner.py \
      --architecture Lin1 --target lt1 --feature-set EMG
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from new_arch.architectures import build_estimator, get_architecture
from new_arch.common_lib import (
    ExperimentMetadata, arch_dir, build_fold_id, model_dir,
    save_model_checkpoint, save_models_csv, save_predictions_parquet,
)
from new_arch.dataset_pipeline.common import DEFAULT_DATASET_DIR
from new_arch.features import get_feature_cols, prepare_data

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

TARGET_COLS = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="linear runner (LOSO)")
    p.add_argument("--architecture", required=True,
                   help="architecture_id, например Lin1")
    p.add_argument("--target", choices=["lt1", "lt2"], default="lt1")
    p.add_argument("--feature-set",
                   choices=["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default="EMG+NIRS+HRV")
    p.add_argument("--with-abs", dest="with_abs", action="store_true", default=True,
                   help="включать абсолютные признаки NIRS/HRV (по умолчанию)")
    p.add_argument("--no-abs", dest="with_abs", action="store_false",
                   help="исключить абсолютные признаки NIRS/HRV")
    p.add_argument("--wavelet-mode",
                   choices=["none", "dwt", "cwt", "wavelet_features", "wavelet_cnn"],
                   default="none")
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    p.add_argument("--session-params", type=Path,
                   default=DEFAULT_DATASET_DIR / "session_params.parquet")
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return p.parse_args()


def _loso_fold(arch, df_subj_tr: pd.DataFrame, df_subj_te: pd.DataFrame,
               feat_cols: list[str], target_col: str):
    """Одна LOSO-итерация: возвращает (fitted_pipeline_components, y_pred)."""
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    mdl = build_estimator(arch)

    X_tr = sc.fit_transform(imp.fit_transform(df_subj_tr[feat_cols].values))
    X_te = sc.transform(imp.transform(df_subj_te[feat_cols].values))
    mdl.fit(X_tr, df_subj_tr[target_col].values)
    y_pred = mdl.predict(X_te)

    # Сохраняем все три компонента (imp+sc+mdl), чтобы checkpoint был самодостаточным.
    pipeline = {"imputer": imp, "scaler": sc, "model": mdl}
    return pipeline, y_pred


def run(args: argparse.Namespace) -> None:
    arch = get_architecture(args.architecture)
    if arch.family != "Lin":
        raise SystemExit(
            f"linear_runner работает только с family='Lin', "
            f"получено {arch.family!r} для {arch.architecture_id}"
        )

    print(f"[{arch.architecture_id}] {arch.architecture_name}")

    meta = ExperimentMetadata.from_arch(
        arch,
        target=args.target, feature_set=args.feature_set,
        with_abs=args.with_abs, wavelet_mode=args.wavelet_mode,
    )
    print(f"  model_id={meta.model_id}")
    print(f"  target={meta.target}  feature_set={meta.feature_set}  "
          f"with_abs={meta.with_abs}  wavelet_mode={meta.wavelet_mode}")

    md = model_dir(args.results_root, meta)
    md.mkdir(parents=True, exist_ok=True)
    save_models_csv(meta, args.results_root)
    print(f"  → {arch_dir(args.results_root, arch.architecture_id) / 'models.csv'}")

    df_raw = pd.read_parquet(args.dataset)
    session_params = pd.read_parquet(args.session_params) if args.session_params.exists() else pd.DataFrame()

    df_prep = prepare_data(df_raw, session_params, meta.target)
    target_col = TARGET_COLS[meta.target]
    df_prep = df_prep.dropna(subset=[target_col])
    feat_cols = get_feature_cols(df_prep, meta.feature_set, with_abs=meta.with_abs)
    if not feat_cols:
        raise SystemExit(f"Пустой feature_set для {meta.feature_set}")
    print(f"  n_subjects={df_prep['subject_id'].nunique()}  "
          f"n_features={len(feat_cols)}  n_windows={len(df_prep)}")

    subjects = sorted(df_prep["subject_id"].unique())
    pred_rows: list[pd.DataFrame] = []

    t0 = time.perf_counter()
    for test_s in subjects:
        fold_id = build_fold_id(test_s)
        tr = df_prep[df_prep["subject_id"] != test_s]
        te = df_prep[df_prep["subject_id"] == test_s].sort_values("window_start_sec").reset_index(drop=True)

        pipeline, y_pred = _loso_fold(arch, tr, te, feat_cols, target_col)
        save_model_checkpoint(pipeline, md, meta, fold_id)

        n = len(te)
        sample_start = te["window_start_sec"].astype(float).values
        sample_end = sample_start + float(meta.window_size_sec)
        fold_df = pd.DataFrame({
            "model_id": meta.model_id,
            "fold_id": fold_id,
            "subject_id": te["subject_id"].values,
            "window_size_sec": meta.window_size_sec,
            "sequence_length": meta.sequence_length,
            "stride_sec": meta.stride_sec,
            "sample_stride_sec": meta.sample_stride_sec,
            "sample_index": np.arange(n, dtype=np.int64),
            "sample_start_sec": sample_start,
            "sample_end_sec": sample_end,
            "y_true": te[target_col].astype(float).values,
            "y_pred": np.asarray(y_pred, dtype=float),
        })
        pred_rows.append(fold_df)

        mae_min = float(np.mean(np.abs(fold_df["y_true"] - fold_df["y_pred"]))) / 60.0
        print(f"  fold {fold_id:<22s}  n={n:<4d}  MAE={mae_min:.3f} мин")

    elapsed = time.perf_counter() - t0
    preds = pd.concat(pred_rows, ignore_index=True)
    save_predictions_parquet(preds, md, meta)

    overall_mae_min = float(np.mean(np.abs(preds["y_true"] - preds["y_pred"]))) / 60.0
    print(f"\n  Всего {len(subjects)} folds, {elapsed:.1f}s; "
          f"overall MAE={overall_mae_min:.3f} мин")
    print(f"  → {md}/predictions_{meta.model_id}.parquet")
    print(f"  → {md}/model_{meta.model_id}_fold-*.joblib  ({len(subjects)} штук)")


if __name__ == "__main__":
    run(parse_args())
