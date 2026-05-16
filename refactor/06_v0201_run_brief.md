# Задание для Sonnet-агента: запуск v0201 на GPU-сервере

## Контекст (зачем это нужно)

В диссертации все нейросетевые модели (v0101–v0107) на этой выборке (n=18,
LOSO) **схлопываются к константному предсказанию**: `std(ŷ)/std(y)`
медианно 0,03–0,12 у LSTM/TCN/Wavelet против 0,83–0,86 у классики и
ансамбля. Подозреваемая причина — таргет в обучение подавался без
нормализации (диапазон ±600 сек), а `HuberLoss(δ=60 c)` на типичных
ошибках в сотни секунд работает в линейном (≈MAE) режиме, что в SGD
известно своим тяготением к медианному (≈ постоянному) предсказанию.

Скрипт [`scripts/v0201_wavelet_tcn_ynorm.py`](../scripts/v0201_wavelet_tcn_ynorm.py)
— это **точная копия `v0106b_wavelet_tcn.py`** с двумя изменениями:

1. таргет нормируется per-fold (StandardScaler на y_tr), модель обучается
   и предсказывает в стандартизированных единицах; на инференсе
   предсказания обратно денормализуются в секунды;
2. `nn.HuberLoss(delta=60.0)` заменён на `nn.MSELoss()` (на нормированной
   шкале MSE адекватен, не вырождается в MAE-режим).

Архитектура, гиперпараметры, разбиение train/val, ранняя остановка,
формат сохранения, расчёт Калман-grid — **полностью идентичны v0106b**,
чтобы любое отличие метрик можно было однозначно приписать
нормализации таргета.

## Критерий успеха диагностики

После прогона достаточно убедиться, что на ENH (EMG+NIRS+HRV)
конфигурации:

- `std(ŷ)/std(y)` (медиана по субъектам LOSO) поднялся с ~0,05 (v0106b)
  до значения **существенно выше 0,3**;
- `loso_neg_r2_share` упал с ~0,76 (v0106b) хотя бы до 0,3–0,5.

Если оба условия выполнены — гипотеза «коллапс из-за необработанного
таргета + HuberLoss» подтверждена, можно докладывать в текст.
Если нет — нужно копать в другую сторону (не делайте этот вывод
самостоятельно — сообщите числа автору).

## Что запустить

После того как окружение поднято и `uv sync` отработал:

### Шаг 1 — быстрая проверка на одной конфигурации (ENH, оба порога)

```bash
PYTHONPATH=. uv run python scripts/v0201_wavelet_tcn_ynorm.py \
    --feature-set EMG+NIRS+HRV \
    --target both
```

Это пройдёт ~2–3 часа на GPU, обработает обе целевые переменные
(LT1 и LT2) на самой богатой комбинации признаков, сохранит:

- `results/v0201/summary.csv`
- `results/v0201/per_subject.csv`
- `results/v0201/ypred_lt1_EMG_NIRS_HRV.npy`,
  `results/v0201/ytrue_lt1_EMG_NIRS_HRV.npy`
- то же для lt2
- аналогичные файлы в `results/v0201/noabs/` (вариант без абсолютных
  уровней NIRS/HRV — он тоже считается автоматически вторым проходом
  по данным)

### Шаг 2 — посчитать `std_ratio` и сравнить с v0106b

```bash
PYTHONPATH=. uv run python << 'PYEOF'
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from pathlib import Path
from dataset_pipeline.common import DEFAULT_DATASET_DIR

df = pd.read_parquet(DEFAULT_DATASET_DIR / 'merged_features_ml.parquet')

def idx_for(target):
    if target == 'lt2':
        d = df[df['window_valid_all_required'] == 1] \
              .dropna(subset=['target_time_to_lt2_center_sec'])
    else:
        d = df[df['target_time_to_lt1_usable'] == 1]
    return (d.sort_values(['subject_id', 'window_start_sec'])
              [['subject_id', 'window_start_sec']].reset_index(drop=True))

print(f"{'version':<8} {'target':<5} {'std_ratio_med':<14} "
      f"{'neg_r2_share':<13} {'raw_mae_min':<12}")
for v in ['v0106b', 'v0201']:
    for tgt in ['lt1', 'lt2']:
        yp = np.load(f'results/{v}/ypred_{tgt}_EMG_NIRS_HRV.npy')
        yt = np.load(f'results/{v}/ytrue_{tgt}_EMG_NIRS_HRV.npy')
        idx = idx_for(tgt)
        if len(idx) != len(yp):
            ps = pd.read_csv(f'results/{v}/per_subject.csv')
            ps = ps[(ps.target == tgt) & (ps.feature_set == 'EMG+NIRS+HRV')]
            kept = sorted(ps.subject_id.unique())
            idx = idx[idx.subject_id.isin(kept)].reset_index(drop=True)
        ratios, neg = [], 0
        for s, _ in idx.groupby('subject_id'):
            m = idx.subject_id.values == s
            if np.std(yt[m]) > 1e-6:
                ratios.append(np.std(yp[m]) / np.std(yt[m]))
            ss_res = np.sum((yt[m] - yp[m]) ** 2)
            ss_tot = np.sum((yt[m] - np.mean(yt[m])) ** 2)
            if ss_tot > 0 and 1 - ss_res / ss_tot < 0:
                neg += 1
        mae = float(np.mean(np.abs(yt - yp))) / 60.0
        print(f"{v:<8} {tgt:<5} "
              f"{np.median(ratios):<14.3f} "
              f"{neg / len(set(idx.subject_id)):<13.2f} "
              f"{mae:<12.3f}")
PYEOF
```

### Шаг 3 — вернуть автору сводку

В одном сообщении (без воды) сообщить:

- цифры из таблицы шага 2 (4 строки);
- сработала ли диагностика по критерию выше;
- любые ошибки/варнинги, которые встретились по ходу прогона.

## Чего НЕ делать

- **Не редактировать** `scripts/v0201_wavelet_tcn_ynorm.py` — изменения
  должны остаться минимальными (нормировка таргета + MSE).
- **Не запускать v0106b повторно** — его артефакты уже есть в
  `results/v0106b/` и используются для сравнения.
- **Не запускать v0201 на остальных feature_set (EMG, NIRS, EMG+NIRS)**
  без явного указания автора — это тратит GPU-время впустую, для
  диагностики достаточно ENH.
- **Не делать содержательных выводов про диссертацию** — только
  сообщить числа, решение принимает автор.
- Не трогать другие файлы в `results/`, `disser-ml-text/`, `refactor/`.

## Полезные пути

- скрипт прогона: `scripts/v0201_wavelet_tcn_ynorm.py`
- референс (старая версия): `scripts/v0106b_wavelet_tcn.py` (для diff)
- датасет: `dataset/merged_features_ml.parquet` (путь в
  `dataset_pipeline.common.DEFAULT_DATASET_DIR`)
- CWT-кэш (ускоряет прогон в ~2–3 раза, если есть):
  `dataset/cwt_cache.npz`
