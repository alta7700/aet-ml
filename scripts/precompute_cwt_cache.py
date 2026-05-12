"""
Предрасчёт CWT для всего датасета merged_features_ml.parquet.
Запускать один раз локально. Результат сохраняется в dataset/cwt_cache.npz.

Формат кэша:
  cwt_cache[i]  — CWT для строки с оригинальным индексом df.index[i]
  row_ids       — массив оригинальных индексов (для выравнивания с df.index)
  feature_names — список признаков (чтобы проверить совпадение при загрузке)
  scales        — использованные масштабы
  wavelet       — имя вейвлета
"""

import sys
from pathlib import Path
# Добавляем корень проекта и папку scripts для импорта модулей
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pywt
import time
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from dataset_pipeline.common import DEFAULT_DATASET_DIR

# ─── Параметры (должны совпадать с конфигом v0107) ───────────────────────────
SCALES  = [1, 2, 4, 8, 16]
WAVELET = "morl"

# Признаки — всё что используется в моделях (EMG+NIRS+HRV с abs)
EXCLUDE_COLS = {"window_id", "subject_id", "subject_name", "source_h5_path",
                "window_start_sec", "window_end_sec", "session_id",
                "target_time_to_lt2_center_sec", "target_time_to_lt1_sec"}

CACHE_PATH = DEFAULT_DATASET_DIR / "cwt_cache.npz"


def _cwt_one_feature(args):
    """CWT для одного признака — запускается из пула потоков."""
    f, col, scales, wavelet = args
    coeffs, _ = pywt.cwt(col.astype(np.float64), scales, wavelet)
    return f, np.abs(coeffs).T.astype(np.float32)


def compute_cwt_matrix(X: np.ndarray, scales, wavelet) -> np.ndarray:
    """Вычисляет CWT параллельно по признакам.
    Возвращает (n_rows, n_features, n_scales) float32."""
    n, n_f, n_sc = len(X), X.shape[1], len(scales)
    cwt_full = np.zeros((n, n_f, n_sc), dtype=np.float32)
    args = [(f, X[:, f], scales, wavelet) for f in range(n_f)]
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_cwt_one_feature)(a) for a in args)
    for f, cwt_f in results:
        cwt_full[:, f, :] = cwt_f
    return cwt_full


def main():
    print("Загружаем датасет...")
    df = pd.read_parquet(DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    print(f"  {len(df)} строк, {len(df.columns)} колонок")

    # Берём все числовые признаки (те же что используют модели)
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE_COLS and df[c].dtype in (np.float32, np.float64, float)]
    print(f"  {len(feat_cols)} признаков для CWT")

    # Глобальная импутация медианой (приближение — в фолдах своя импутация,
    # но разница поглощается внутренней нормализацией датасета)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(df[feat_cols].values).astype(np.float32)
    print(f"  X_imp: {X_imp.shape}")

    print(f"\nСчитаем CWT (wavelet={WAVELET}, scales={SCALES})...")
    t0 = time.time()
    cwt_matrix = compute_cwt_matrix(X_imp, SCALES, WAVELET)
    elapsed = time.time() - t0
    print(f"  Готово за {elapsed:.1f}с  →  shape {cwt_matrix.shape}")
    print(f"  Размер: {cwt_matrix.nbytes / 1e6:.1f} МБ")

    # Сохраняем с метаданными для проверки при загрузке
    print(f"\nСохраняем в {CACHE_PATH}...")
    np.savez_compressed(
        CACHE_PATH,
        cwt        = cwt_matrix,          # (n_rows, n_features, n_scales)
        row_ids    = df.index.values,     # оригинальные индексы датасета
        scales     = np.array(SCALES),
        feat_cols  = np.array(feat_cols), # для проверки совпадения
    )
    size_mb = CACHE_PATH.stat().st_size / 1e6
    print(f"  Сохранено: {size_mb:.1f} МБ")
    print("\nГотово. Теперь перекиньте cwt_cache.npz на GPU-сервер:")
    print(f"  rsync -avz dataset/cwt_cache.npz ml:~/dataset/")


if __name__ == "__main__":
    main()
