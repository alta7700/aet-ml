"""
Предрасчёт CWT-кеша из сырого EMG-сигнала.

Для каждого 30-секундного окна из merged_features_ml.parquet читает сырой
ЭМГ-сигнал (VL_dist, VL_prox) из finaltest.h5 и вычисляет CWT с шкалами,
покрывающими диапазон 25–522 Гц (физиологически значимый для ЭМГ усталости).

По каждой шкале и каждому каналу сохраняет:
  - mean |CWT|  — средняя амплитуда в частотной полосе
  - std  |CWT|  — разброс внутри окна (маркер нестационарности)

Итого на строку: 2 канала × 7 шкал × 2 стат = 28 признаков.

Запуск:
  PYTHONPATH=. python3 scripts/precompute_cwt_cache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

import h5py
import numpy as np
import pandas as pd
import pywt
from joblib import Parallel, delayed

from dataset_pipeline.common import DEFAULT_DATASET_DIR

# ─── Константы ────────────────────────────────────────────────────────────────

# Шкалы CWT для Morlet при ~1926 Гц: покрывают 25–522 Гц
# scale = fc × fs / f  (fc_morl ≈ 0.8125)
SCALES   = [3, 6, 10, 16, 25, 40, 63]
WAVELET  = "morl"

WINDOW_SEC   = 30.0     # длина одного окна датасета
MIN_COVERAGE = 0.75     # минимальная доля ожидаемых сэмплов в окне

EMG_CHANNELS = ["trigno.vl.avanti", "trigno.rf.avanti"]
CHANNEL_NAMES = ["vl_dist", "vl_prox"]

CACHE_PATH = DEFAULT_DATASET_DIR / "cwt_cache.npz"


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _load_emg_for_subject(h5_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Загружает сырой ЭМГ-сигнал из h5 и переводит timestamps в секунды."""
    result = {}
    with h5py.File(h5_path, "r") as f:
        anchor_ms = float(f["channels/moxy.smo2/timestamps"][0])
        for ch_key, ch_name in zip(EMG_CHANNELS, CHANNEL_NAMES):
            path = f"channels/{ch_key}"
            if path not in f:
                result[ch_name] = (np.array([]), np.array([]))
                continue
            ts_sec = (f[f"{path}/timestamps"][:].astype(np.float64) - anchor_ms) / 1000.0
            vals   = f[f"{path}/values"][:].astype(np.float32)
            result[ch_name] = (ts_sec, vals)
    return result


def _cwt_window(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CWT одного окна; возвращает (mean_per_scale, std_per_scale), shape (n_scales,)."""
    if len(signal) == 0:
        nan = np.full(len(SCALES), np.nan, dtype=np.float32)
        return nan, nan
    coeffs, _ = pywt.cwt(signal.astype(np.float64), SCALES, WAVELET)  # (n_scales, T)
    abs_c = np.abs(coeffs).astype(np.float32)
    return abs_c.mean(axis=1), abs_c.std(axis=1)


def _process_subject(subject_id: str, h5_path: str,
                     windows_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Обрабатывает все окна одного субъекта.

    Возвращает:
      row_ids  — (n_windows,)     оригинальные индексы parquet
      cwt_mean — (n_windows, n_channels, n_scales)
      cwt_std  — (n_windows, n_channels, n_scales)
    """
    emg = _load_emg_for_subject(h5_path)

    n  = len(windows_df)
    nc = len(CHANNEL_NAMES)
    ns = len(SCALES)

    row_ids  = windows_df.index.values
    cwt_mean = np.full((n, nc, ns), np.nan, dtype=np.float32)
    cwt_std  = np.full((n, nc, ns), np.nan, dtype=np.float32)

    for row_pos, (parquet_idx, row) in enumerate(windows_df.iterrows()):
        t_start = float(row["window_start_sec"])
        t_end   = t_start + WINDOW_SEC

        for ch_pos, ch_name in enumerate(CHANNEL_NAMES):
            ts, vals = emg[ch_name]
            if len(ts) == 0:
                continue

            # Оцениваем ожидаемое число сэмплов по средней частоте канала
            fs_est = len(ts) / max(ts[-1] - ts[0], 1.0)
            expected = fs_est * WINDOW_SEC

            mask = (ts >= t_start) & (ts < t_end)
            sig  = vals[mask]

            if len(sig) < MIN_COVERAGE * expected:
                continue

            cwt_mean[row_pos, ch_pos], cwt_std[row_pos, ch_pos] = _cwt_window(sig)

    n_ok = int(np.isfinite(cwt_mean[:, 0, 0]).sum())
    print(f"  [{subject_id}] {n_ok}/{n} окон валидны")
    return row_ids, cwt_mean, cwt_std


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Загружаем датасет и таблицу субъектов...")
    df  = pd.read_parquet(DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    sub = pd.read_parquet(DEFAULT_DATASET_DIR / "subjects.parquet")[
        ["subject_id", "source_h5_path"]
    ]
    print(f"  Строк: {len(df)}, субъектов: {df['subject_id'].nunique()}")

    # Строим список задач: (subject_id, h5_path, windows_df)
    tasks = []
    for _, srow in sub.iterrows():
        sid     = srow["subject_id"]
        h5_path = srow["source_h5_path"]
        wdf     = df[df["subject_id"] == sid].sort_values("window_start_sec")
        if len(wdf) == 0 or not Path(h5_path).exists():
            print(f"  [SKIP] {sid}: нет окон или h5-файл не найден")
            continue
        tasks.append((sid, h5_path, wdf))

    # Вычисляем псевдочастоты для лога
    fc = pywt.central_frequency(WAVELET)
    fs_ref = 1926.0
    print(f"\nШкалы CWT (wavelet={WAVELET}, fc={fc:.4f}):")
    for s in SCALES:
        print(f"  scale={s:3d}: ~{fc*fs_ref/s:.0f} Гц")

    print(f"\nОбрабатываем {len(tasks)} субъектов параллельно...")
    t0 = time.time()

    results = Parallel(n_jobs=-1, backend="loky", verbose=0)(
        delayed(_process_subject)(sid, h5, wdf)
        for sid, h5, wdf in tasks
    )

    elapsed = time.time() - t0
    print(f"\nВычисление завершено за {elapsed:.1f}с")

    # Собираем в единые массивы, восстанавливая порядок parquet
    all_row_ids  = np.concatenate([r[0] for r in results])
    all_cwt_mean = np.concatenate([r[1] for r in results], axis=0)
    all_cwt_std  = np.concatenate([r[2] for r in results], axis=0)

    # Сортируем по row_id, чтобы порядок совпадал с parquet
    order = np.argsort(all_row_ids)
    all_row_ids  = all_row_ids[order]
    all_cwt_mean = all_cwt_mean[order]
    all_cwt_std  = all_cwt_std[order]

    n_nan = int(np.isnan(all_cwt_mean[:, 0, 0]).sum())
    print(f"Строк с NaN (нехватка сигнала): {n_nan}/{len(all_row_ids)}")
    print(f"Форма cwt_mean: {all_cwt_mean.shape}  "
          f"({len(CHANNEL_NAMES)} канала × {len(SCALES)} шкал × N строк)")

    print(f"\nСохраняем в {CACHE_PATH}...")
    np.savez_compressed(
        CACHE_PATH,
        cwt_mean     = all_cwt_mean,          # (N, n_channels, n_scales)
        cwt_std      = all_cwt_std,            # (N, n_channels, n_scales)
        row_ids      = all_row_ids,
        scales       = np.array(SCALES),
        channel_names= np.array(CHANNEL_NAMES),
    )
    size_mb = CACHE_PATH.stat().st_size / 1e6
    print(f"  Сохранено: {size_mb:.1f} МБ")
    print(f"\nГотово. Залейте на сервер:")
    print(f"  rsync -avz dataset/cwt_cache.npz ml:~/dissertation/dataset/")


if __name__ == "__main__":
    main()
