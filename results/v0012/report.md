# v0012 — Body Composition: отчёт

## Лучшие модели по (набор признаков, таргет)

| feature_set     | target   | model            |   raw_mae_min |   kalman_mae_min |    r2 |   rho |
|:----------------|:---------|:-----------------|--------------:|-----------------:|------:|------:|
| BodyComp        | lt1      | EN(α=1.0,l1=0.2) |        5.4972 |           5.4665 | 0.004 | 0.086 |
| EMG             | lt1      | Ridge(α=1000)    |        3.0467 |           2.8957 | 0.677 | 0.849 |
| EMG+BC          | lt1      | GBM(n=50,d=2)    |        2.9794 |           2.7194 | 0.665 | 0.82  |
| EMG+NIRS        | lt1      | Ridge(α=1000)    |        2.8971 |           2.7499 | 0.73  | 0.869 |
| EMG+NIRS+BC     | lt1      | EN(α=1.0,l1=0.2) |        2.9797 |           2.8507 | 0.717 | 0.858 |
| EMG+NIRS+HRV    | lt1      | GBM(n=50,d=2)    |        2.3107 |           2.2774 | 0.811 | 0.905 |
| EMG+NIRS+HRV+BC | lt1      | GBM(n=100,d=2)   |        2.1464 |           2.0864 | 0.833 | 0.912 |

## Сравнение с v0011

| Набор | Таргет | MAE v0011 | MAE v0012 | Δ MAE |
|---|---|---|---|---|
| EMG | lt1 | 2.896 | 2.896 | 0.000 |
| EMG+NIRS | lt1 | 2.750 | 2.750 | 0.000 |
| EMG+NIRS+HRV | lt1 | 2.277 | 2.277 | 0.000 |

## Честные baselines (MeanPredictor, FirstWindowPredictor)

### Honest baselines — BodyComp / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.251 | — |
| FirstWindowPredictor | 4.147 | 142 |
| Raw model (no Kalman) | 5.497 | 330 |
| Kalman (σ_obs=ref)  | 5.466 | — |

gap (raw − FirstWin): +1.350 мин
**⚠ sequential хуже FirstWindow (нет онлайн-ценности)**


### Honest baselines — EMG+BC / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.251 | — |
| FirstWindowPredictor | 4.137 | 161 |
| Raw model (no Kalman) | 2.979 | 157 |
| Kalman (σ_obs=ref)  | 2.719 | — |

gap (raw − FirstWin): -1.157 мин
**✓ sequential значимо лучше FirstWindow**


### Honest baselines — EMG+NIRS+BC / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.251 | — |
| FirstWindowPredictor | 3.663 | 140 |
| Raw model (no Kalman) | 2.980 | 137 |
| Kalman (σ_obs=ref)  | 2.851 | — |

gap (raw − FirstWin): -0.684 мин
**✓ sequential значимо лучше FirstWindow**


### Honest baselines — EMG+NIRS+HRV+BC / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.251 | — |
| FirstWindowPredictor | 4.545 | 177 |
| Raw model (no Kalman) | 2.146 | 95 |
| Kalman (σ_obs=ref)  | 2.086 | — |

gap (raw − FirstWin): -2.398 мин
**✓ sequential значимо лучше FirstWindow**


### Honest baselines — EMG / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.251 | — |
| FirstWindowPredictor | 4.262 | 146 |
| Raw model (no Kalman) | 3.047 | 162 |
| Kalman (σ_obs=ref)  | 2.896 | — |

gap (raw − FirstWin): -1.215 мин
**✓ sequential значимо лучше FirstWindow**


### Honest baselines — EMG+NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.251 | — |
| FirstWindowPredictor | 4.205 | 146 |
| Raw model (no Kalman) | 2.897 | 144 |
| Kalman (σ_obs=ref)  | 2.750 | — |

gap (raw − FirstWin): -1.308 мин
**✓ sequential значимо лучше FirstWindow**


### Honest baselines — EMG+NIRS+HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.251 | — |
| FirstWindowPredictor | 4.836 | 201 |
| Raw model (no Kalman) | 2.311 | 108 |
| Kalman (σ_obs=ref)  | 2.277 | — |

gap (raw − FirstWin): -2.525 мин
**✓ sequential значимо лучше FirstWindow**

## Защита от переобучения

- Статические BC-признаки одинаковы для всех окон субъекта.
- При N=14 (LT2) каждый субъект в тесте видит BC-вектор, не участвовавший в обучении,
  но корреляция BC с LT может быть подобрана на 13 субъектах → переобучение.
- Ориентир: gap (raw − FirstWindow) для BC-наборов не должен ухудшаться vs v0011.
- Per-subject Δ MAE: если улучшение на 1–2 субъектах — не надёжно.
