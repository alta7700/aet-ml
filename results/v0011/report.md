# v0011 — Ablation по модальностям
Дата: 2026-05-10  
Метрика: MAE (мин) после фильтра Калмана, LOSO CV.

## LT2

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | EN(α=1.0,l1=0.9) | 17 | 107 | 2.203 | **2.125** | 0.825 | 0.913 |
| **HRV** | EN(α=1.0,l1=0.9) | 17 | 7 | 2.409 | **2.385** | 0.796 | 0.905 |
| **EMG+NIRS** | EN(α=1.0,l1=0.2) | 17 | 97 | 2.989 | **2.834** | 0.628 | 0.817 |
| **EMG** | EN(α=1.0,l1=0.2) | 17 | 79 | 3.138 | **3.002** | 0.607 | 0.816 |
| **NIRS** | SVR(C=100,ε=1.0) | 17 | 18 | 4.088 | **3.917** | 0.374 | 0.607 |

## LT1

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | Ridge(α=1000) | 12 | 107 | 2.139 | **2.043** | 0.823 | 0.915 |
| **EMG+NIRS** | EN(α=1.0,l1=0.9) | 12 | 97 | 2.582 | **2.386** | 0.755 | 0.883 |
| **HRV** | GBM(n=50,d=3) | 12 | 7 | 2.476 | **2.457** | 0.774 | 0.894 |
| **EMG** | EN(α=1.0,l1=0.2) | 12 | 79 | 2.851 | **2.704** | 0.697 | 0.864 |
| **NIRS** | SVR(C=10,ε=1.0) | 12 | 18 | 4.180 | **4.047** | 0.388 | 0.677 |

## Полная таблица (все конфиги)

### LT2

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.125 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.146 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.176 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.200 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.242 |
| EMG+NIRS+HRV | Ridge(α=100) | 2.251 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.286 |
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.345 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.363 |
| HRV | EN(α=1.0,l1=0.9) | 2.385 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.388 |
| HRV | EN(α=0.1,l1=0.2) | 2.398 |
| HRV | EN(α=0.1,l1=0.5) | 2.416 |
| HRV | Ridge(α=1000) | 2.430 |
| HRV | Ridge(α=100) | 2.432 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.9) | 2.440 |
| HRV | EN(α=0.1,l1=0.9) | 2.460 |
| HRV | EN(α=0.01,l1=0.2) | 2.465 |
| HRV | EN(α=0.01,l1=0.5) | 2.471 |
| HRV | Ridge(α=10) | 2.475 |

### LT1

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | Ridge(α=1000) | 2.043 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.048 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.064 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.156 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.164 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.193 |
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.220 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.239 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.257 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.261 |
| EMG+NIRS+HRV | GBM(n=100,d=2) | 2.280 |
| EMG+NIRS+HRV | Ridge(α=100) | 2.313 |
| EMG+NIRS+HRV | GBM(n=200,d=2) | 2.331 |
| EMG+NIRS | EN(α=1.0,l1=0.9) | 2.386 |
| EMG+NIRS | EN(α=0.1,l1=0.2) | 2.421 |
| EMG+NIRS | Ridge(α=1000) | 2.444 |
| EMG+NIRS | EN(α=1.0,l1=0.5) | 2.453 |
| EMG+NIRS | EN(α=0.1,l1=0.5) | 2.455 |
| HRV | GBM(n=50,d=3) | 2.457 |
| HRV | GBM(n=100,d=3) | 2.481 |

## Honest baselines

### Honest baselines — EMG / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.727 | — |
| FirstWindowPredictor | 4.717 | 122 |
| Raw model (no Kalman) | 3.138 | 133 |
| Kalman (σ_obs=ref)  | 3.002 | — |

gap (raw − FirstWin): -1.579 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — NIRS / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.727 | — |
| FirstWindowPredictor | 3.742 | 132 |
| Raw model (no Kalman) | 4.088 | 183 |
| Kalman (σ_obs=ref)  | 3.917 | — |

gap (raw − FirstWin): +0.346 мин
**⚠ sequential хуже FirstWindow (нет онлайн-ценности)**

### Honest baselines — HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.727 | — |
| FirstWindowPredictor | 2.755 | 112 |
| Raw model (no Kalman) | 2.409 | 90 |
| Kalman (σ_obs=ref)  | 2.385 | — |

gap (raw − FirstWin): -0.346 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.727 | — |
| FirstWindowPredictor | 4.494 | 116 |
| Raw model (no Kalman) | 2.989 | 129 |
| Kalman (σ_obs=ref)  | 2.834 | — |

gap (raw − FirstWin): -1.505 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.727 | — |
| FirstWindowPredictor | 3.904 | 119 |
| Raw model (no Kalman) | 2.203 | 90 |
| Kalman (σ_obs=ref)  | 2.125 | — |

gap (raw − FirstWin): -1.701 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.907 | — |
| FirstWindowPredictor | 3.899 | 140 |
| Raw model (no Kalman) | 2.851 | 147 |
| Kalman (σ_obs=ref)  | 2.704 | — |

gap (raw − FirstWin): -1.048 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.907 | — |
| FirstWindowPredictor | 3.843 | 169 |
| Raw model (no Kalman) | 4.180 | 210 |
| Kalman (σ_obs=ref)  | 4.047 | — |

gap (raw − FirstWin): +0.337 мин
**⚠ sequential хуже FirstWindow (нет онлайн-ценности)**

### Honest baselines — HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.907 | — |
| FirstWindowPredictor | 3.952 | 174 |
| Raw model (no Kalman) | 2.476 | 109 |
| Kalman (σ_obs=ref)  | 2.457 | — |

gap (raw − FirstWin): -1.477 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.907 | — |
| FirstWindowPredictor | 4.103 | 144 |
| Raw model (no Kalman) | 2.582 | 134 |
| Kalman (σ_obs=ref)  | 2.386 | — |

gap (raw − FirstWin): -1.520 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.907 | — |
| FirstWindowPredictor | 3.798 | 145 |
| Raw model (no Kalman) | 2.139 | 104 |
| Kalman (σ_obs=ref)  | 2.043 | — |

gap (raw − FirstWin): -1.659 мин
**✓ sequential значимо лучше FirstWindow**

