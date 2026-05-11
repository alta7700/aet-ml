# v0011 — Ablation по модальностям
Дата: 2026-05-11  
Метрика: MAE (мин) после фильтра Калмана, LOSO CV.

## LT2

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | GBM(n=50,d=3) | 19 | 107 | 2.197 | **2.162** | 0.846 | 0.924 |
| **HRV** | EN(α=0.1,l1=0.2) | 19 | 7 | 2.264 | **2.240** | 0.816 | 0.911 |
| **EMG+NIRS** | EN(α=1.0,l1=0.2) | 19 | 97 | 3.008 | **2.855** | 0.645 | 0.826 |
| **EMG** | EN(α=1.0,l1=0.2) | 19 | 79 | 3.133 | **2.993** | 0.625 | 0.823 |
| **NIRS** | SVR(C=100,ε=0.1) | 19 | 18 | 4.128 | **3.962** | 0.371 | 0.619 |

## LT1

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | EN(α=1.0,l1=0.5) | 14 | 107 | 2.268 | **2.178** | 0.821 | 0.916 |
| **EMG+NIRS** | GBM(n=50,d=2) | 14 | 97 | 2.616 | **2.449** | 0.758 | 0.877 |
| **HRV** | Huber(ε=2.0) | 14 | 7 | 2.594 | **2.662** | 0.742 | 0.861 |
| **EMG** | GBM(n=100,d=2) | 14 | 79 | 2.831 | **2.672** | 0.701 | 0.859 |
| **NIRS** | GBM(n=50,d=2) | 14 | 18 | 4.169 | **3.968** | 0.393 | 0.631 |

## Полная таблица (все конфиги)

### LT2

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.162 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.178 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.195 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.212 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.231 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.231 |
| HRV | EN(α=0.1,l1=0.2) | 2.240 |
| HRV | EN(α=0.1,l1=0.5) | 2.240 |
| HRV | EN(α=1.0,l1=0.9) | 2.243 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.244 |
| HRV | Ridge(α=100) | 2.248 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.263 |
| HRV | EN(α=0.1,l1=0.9) | 2.264 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.264 |
| HRV | EN(α=0.01,l1=0.2) | 2.268 |
| HRV | EN(α=0.01,l1=0.5) | 2.273 |
| HRV | Ridge(α=10) | 2.277 |
| HRV | EN(α=0.01,l1=0.9) | 2.281 |
| HRV | Ridge(α=1) | 2.285 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.288 |

### LT1

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.178 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.183 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.187 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.213 |
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.236 |
| EMG+NIRS+HRV | GBM(n=100,d=2) | 2.254 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.264 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.275 |
| EMG+NIRS+HRV | GBM(n=200,d=2) | 2.278 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.286 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.307 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.374 |
| EMG+NIRS+HRV | Ridge(α=100) | 2.440 |
| EMG+NIRS | GBM(n=50,d=2) | 2.449 |
| EMG+NIRS | EN(α=1.0,l1=0.5) | 2.491 |
| EMG+NIRS | EN(α=1.0,l1=0.2) | 2.507 |
| EMG+NIRS | Ridge(α=1000) | 2.508 |
| EMG+NIRS | GBM(n=100,d=2) | 2.538 |
| EMG+NIRS+HRV | SVR(C=100,ε=1.0) | 2.547 |
| EMG+NIRS+HRV | SVR(C=100,ε=0.1) | 2.549 |

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
| FirstWindowPredictor | 3.869 | 131 |
| Raw model (no Kalman) | 4.128 | 181 |
| Kalman (σ_obs=ref)  | 3.962 | — |

gap (raw − FirstWin): +0.259 мин
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
| FirstWindowPredictor | 4.749 | 125 |
| Raw model (no Kalman) | 3.008 | 134 |
| Kalman (σ_obs=ref)  | 2.855 | — |

gap (raw − FirstWin): -1.741 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.869 | — |
| FirstWindowPredictor | 3.706 | 119 |
| Raw model (no Kalman) | 2.197 | 86 |
| Kalman (σ_obs=ref)  | 2.162 | — |

gap (raw − FirstWin): -1.509 мин
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
| FirstWindowPredictor | 4.148 | 161 |
| Raw model (no Kalman) | 4.169 | 201 |
| Kalman (σ_obs=ref)  | 3.968 | — |

gap (raw − FirstWin): +0.020 мин
**~ нет значимой разницы с FirstWindow**

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
| FirstWindowPredictor | 4.564 | 168 |
| Raw model (no Kalman) | 2.616 | 139 |
| Kalman (σ_obs=ref)  | 2.449 | — |

gap (raw − FirstWin): -1.949 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.023 | — |
| FirstWindowPredictor | 4.354 | 151 |
| Raw model (no Kalman) | 2.268 | 113 |
| Kalman (σ_obs=ref)  | 2.178 | — |

gap (raw − FirstWin): -2.086 мин
**✓ sequential значимо лучше FirstWindow**

