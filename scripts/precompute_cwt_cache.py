"""
Предрасчёт CWT-кеша для всего датасета.

Запускать один раз перед обучением v03xx. Результат — dataset/cwt_cache.npz.

Отличие от предыдущей версии:
  - Применяет полный feature engineering из v0011_modality_ablation:
    per-subject z-norm EMG + кинематика (z_vl_*, z_<kin>_*),
    running NIRS (smo2_from_running_max, hhb_from_running_min, smo2_rel_drop_pct),
    interaction-признаки (feat_smo2_x_rr, feat_smo2_x_dfa, feat_rr_per_watt).
  - Охватывает все признаки, запрашиваемые наборами EMG / EMG+NIRS / EMG+NIRS+HRV.
  - Работает с полным датасетом без фильтрации по таргету.
  - Сохраняет оригинальные parquet-индексы в row_ids — совместимо с prepare_data
    (которая теперь тоже не делает reset_index).

Формат кэша:
  cwt       — (N, n_features, n_scales) float32, |CWT|-коэффициенты
  row_ids   — (N,) оригинальные индексы df (для lookup в CwtCache.get)
  feat_cols — (n_features,) имена признаков
  scales    — использованные масштабы

Запуск:
  PYTHONPATH=. python3 scripts/precompute_cwt_cache.py
"""

import sys
from pathlib import Path

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import pywt
import time
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from scripts.v0011_modality_ablation import (
    NIRS_FEATURES,
    RUNNING_NIRS_FEATURES,
    HRV_FEATURES,
    INTERACTION_FEATURES,
    KINEMATICS_FEATURES,
    _get_emg_raw_cols,
    _add_subject_z,
    _add_running_nirs,
    _add_interactions,
)

SCALES  = [1, 2, 4, 8, 16]
WAVELET = "morl"
CACHE_PATH = DEFAULT_DATASET_DIR / "cwt_cache.npz"


def _cwt_one_feature(args):
    """CWT для одного признака — запускается из пула потоков."""
    f, col, scales, wavelet = args
    coeffs, _ = pywt.cwt(col.astype(np.float64), scales, wavelet)
    return f, np.abs(coeffs).T.astype(np.float32)


def compute_cwt_matrix(X: np.ndarray, scales, wavelet) -> np.ndarray:
    """Вычисляет CWT параллельно по признакам через joblib.

    Возвращает (n_rows, n_features, n_scales) float32.
    """
    n_f = X.shape[1]
    cwt_full = np.zeros((len(X), n_f, len(scales)), dtype=np.float32)
    args = [(f, X[:, f], scales, wavelet) for f in range(n_f)]
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_cwt_one_feature)(a) for a in args
    )
    for f, cwt_f in results:
        cwt_full[:, f, :] = cwt_f
    return cwt_full


def main():
    print("Загружаем датасет...")
    df = pd.read_parquet(DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()
    print(f"  {len(df)} строк, {df['subject_id'].nunique()} субъектов")

    # Сортируем по субъекту и времени — без reset_index, чтобы оригинальные
    # parquet-метки сохранились и совпадали с lookup в CwtCache.get().
    df = df.sort_values(["subject_id", "window_start_sec"])

    print("Применяем feature engineering...")

    # Per-subject z-norm ЭМГ по baseline (stage_index=0, 60 Вт)
    emg_raw = _get_emg_raw_cols(df)
    df, z_emg_cols = _add_subject_z(df, emg_raw, prefix="z_")
    print(f"  z-norm EMG: {len(z_emg_cols)} признаков ({z_emg_cols[0]}..{z_emg_cols[-1]})")

    # Per-subject z-norm кинематики по тому же baseline
    kin_present = [c for c in KINEMATICS_FEATURES if c in df.columns]
    df, z_kin_cols = _add_subject_z(df, kin_present, prefix="z_")
    print(f"  z-norm кинематика: {len(z_kin_cols)} признаков")

    # Running NIRS (накопленные от начала теста, порядок важен)
    df = _add_running_nirs(df, session_params)
    print(f"  running NIRS: {RUNNING_NIRS_FEATURES}")

    # Interaction-признаки (NIRS × HRV)
    if "trainred_smo2_mean" in df.columns and "hrv_mean_rr_ms" in df.columns:
        df = _add_interactions(df)
        print(f"  interactions: {INTERACTION_FEATURES}")

    # Итоговый набор признаков — ровно то, что запрашивают модели
    feat_cols = (
        z_emg_cols
        + z_kin_cols
        + [c for c in NIRS_FEATURES        if c in df.columns]
        + [c for c in RUNNING_NIRS_FEATURES if c in df.columns]
        + [c for c in HRV_FEATURES          if c in df.columns]
        + [c for c in INTERACTION_FEATURES  if c in df.columns]
    )
    print(f"\nИтого признаков для CWT: {len(feat_cols)}")

    # Глобальная импутация медианой (в фолдах — своя, но разница поглощается
    # нормализацией внутри CwtCache.get при каждом lookup)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(df[feat_cols].values).astype(np.float32)
    print(f"  X_imp: {X_imp.shape}")

    print(f"\nСчитаем CWT (wavelet={WAVELET}, scales={SCALES})...")
    t0 = time.time()
    cwt_matrix = compute_cwt_matrix(X_imp, SCALES, WAVELET)
    elapsed = time.time() - t0
    print(f"  Готово за {elapsed:.1f}с  →  shape {cwt_matrix.shape}")
    print(f"  Размер в памяти: {cwt_matrix.nbytes / 1e6:.1f} МБ")

    print(f"\nСохраняем в {CACHE_PATH}...")
    np.savez_compressed(
        CACHE_PATH,
        cwt       = cwt_matrix,
        row_ids   = df.index.values,   # оригинальные parquet-метки (0..N-1)
        scales    = np.array(SCALES),
        feat_cols = np.array(feat_cols),
    )
    size_mb = CACHE_PATH.stat().st_size / 1e6
    print(f"  Сохранено: {size_mb:.1f} МБ")
    print(f"\nГотово. Залейте на сервер:")
    print(f"  rsync -avz dataset/cwt_cache.npz ml:~/dissertation/dataset/")


if __name__ == "__main__":
    main()
