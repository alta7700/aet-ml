# v0011 — Ablation по модальностям
Дата: 2026-05-12  
Метрика: MAE (мин) после фильтра Калмана, LOSO CV.

## LT2

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | GBM(n=50,d=3) | 19 | 103 | 2.136 | **2.095** | 0.853 | 0.925 |
| **HRV** | EN(α=0.1,l1=0.2) | 19 | 7 | 2.264 | **2.240** | 0.816 | 0.911 |
| **EMG+NIRS** | EN(α=1.0,l1=0.2) | 19 | 94 | 2.996 | **2.841** | 0.647 | 0.827 |
| **EMG** | EN(α=1.0,l1=0.2) | 19 | 79 | 3.133 | **2.993** | 0.625 | 0.823 |
| **NIRS** | SVR(C=10,ε=0.1) | 19 | 15 | 4.500 | **4.434** | 0.282 | 0.515 |

## LT1

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | EN(α=1.0,l1=0.5) | 14 | 103 | 2.274 | **2.174** | 0.818 | 0.917 |
| **EMG+NIRS** | EN(α=1.0,l1=0.9) | 14 | 94 | 2.675 | **2.476** | 0.750 | 0.882 |
| **HRV** | Huber(ε=2.0) | 14 | 7 | 2.594 | **2.662** | 0.742 | 0.861 |
| **EMG** | GBM(n=100,d=2) | 14 | 79 | 2.831 | **2.672** | 0.701 | 0.859 |
| **NIRS** | EN(α=1.0,l1=0.5) | 14 | 15 | 4.471 | **4.362** | 0.306 | 0.620 |

## Полная таблица (все конфиги)

### LT2

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.095 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.132 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.157 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.172 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.182 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.189 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.207 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.219 |
| EMG+NIRS+HRV | Ridge(α=100) | 2.222 |
| HRV | EN(α=0.1,l1=0.2) | 2.240 |
| HRV | EN(α=0.1,l1=0.5) | 2.240 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.241 |
| HRV | EN(α=1.0,l1=0.9) | 2.243 |
| HRV | Ridge(α=100) | 2.248 |
| HRV | EN(α=0.1,l1=0.9) | 2.264 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.267 |
| HRV | EN(α=0.01,l1=0.2) | 2.268 |
| HRV | EN(α=0.01,l1=0.5) | 2.273 |
| HRV | Ridge(α=10) | 2.277 |
| HRV | EN(α=0.01,l1=0.9) | 2.281 |

### LT1

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.174 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.174 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.196 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.204 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.277 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.297 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.342 |
| EMG+NIRS+HRV | GBM(n=100,d=2) | 2.401 |
| EMG+NIRS+HRV | Ridge(α=100) | 2.410 |
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.429 |
| EMG+NIRS | EN(α=1.0,l1=0.9) | 2.476 |
| EMG+NIRS | EN(α=1.0,l1=0.5) | 2.485 |
| EMG+NIRS | Ridge(α=1000) | 2.487 |
| EMG+NIRS+HRV | GBM(n=200,d=2) | 2.494 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.496 |
| EMG+NIRS | EN(α=1.0,l1=0.2) | 2.509 |
| EMG+NIRS | GBM(n=50,d=2) | 2.526 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.528 |
| EMG+NIRS | EN(α=0.1,l1=0.2) | 2.540 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.9) | 2.549 |

## Honest baselines

### Honest baselines — EMG / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.869 | — |
| FirstWindowPredictor | 4.970 | 132 |
| Raw model (no Kalman) | 3.133 | 138 |
| Kalman (σ_obs=ref)  | 2.993 | — |

gap (raw − FirstWin): -1.836 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — NIRS / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.869 | — |
| FirstWindowPredictor | 3.865 | 141 |
| Raw model (no Kalman) | 4.500 | 210 |
| Kalman (σ_obs=ref)  | 4.434 | — |

gap (raw − FirstWin): +0.635 мин
**⚠ sequential хуже FirstWindow (нет онлайн-ценности)**

### Honest baselines — HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.869 | — |
| FirstWindowPredictor | 2.891 | 110 |
| Raw model (no Kalman) | 2.264 | 87 |
| Kalman (σ_obs=ref)  | 2.240 | — |

gap (raw − FirstWin): -0.627 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.869 | — |
| FirstWindowPredictor | 4.738 | 125 |
| Raw model (no Kalman) | 2.996 | 135 |
| Kalman (σ_obs=ref)  | 2.841 | — |

gap (raw − FirstWin): -1.743 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.869 | — |
| FirstWindowPredictor | 3.618 | 118 |
| Raw model (no Kalman) | 2.136 | 87 |
| Kalman (σ_obs=ref)  | 2.095 | — |

gap (raw − FirstWin): -1.482 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.023 | — |
| FirstWindowPredictor | 4.491 | 171 |
| Raw model (no Kalman) | 2.831 | 154 |
| Kalman (σ_obs=ref)  | 2.672 | — |

gap (raw − FirstWin): -1.659 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.023 | — |
| FirstWindowPredictor | 3.421 | 151 |
| Raw model (no Kalman) | 4.471 | 237 |
| Kalman (σ_obs=ref)  | 4.362 | — |

gap (raw − FirstWin): +1.049 мин
**⚠ sequential хуже FirstWindow (нет онлайн-ценности)**

### Honest baselines — HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.023 | — |
| FirstWindowPredictor | 4.002 | 172 |
| Raw model (no Kalman) | 2.594 | 115 |
| Kalman (σ_obs=ref)  | 2.662 | — |

gap (raw − FirstWin): -1.408 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.023 | — |
| FirstWindowPredictor | 4.208 | 154 |
| Raw model (no Kalman) | 2.675 | 150 |
| Kalman (σ_obs=ref)  | 2.476 | — |

gap (raw − FirstWin): -1.534 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.023 | — |
| FirstWindowPredictor | 4.380 | 151 |
| Raw model (no Kalman) | 2.274 | 117 |
| Kalman (σ_obs=ref)  | 2.174 | — |

gap (raw − FirstWin): -2.106 мин
**✓ sequential значимо лучше FirstWindow**

