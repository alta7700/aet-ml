# v0011 — Ablation по модальностям
Дата: 2026-05-08  
Метрика: MAE (мин) после фильтра Калмана, LOSO CV.

## LT2

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | EN(α=0.1,l1=0.5) | 14 | 107 | 1.992 | **1.859** | 0.858 | 0.929 |
| **EMG+NIRS** | EN(α=1.0,l1=0.2) | 14 | 97 | 3.249 | **3.117** | 0.609 | 0.804 |
| **EMG** | EN(α=1.0,l1=0.2) | 14 | 79 | 3.324 | **3.198** | 0.600 | 0.811 |
| **NIRS** | SVR(C=10,ε=0.1) | 14 | 18 | 4.357 | **4.251** | 0.301 | 0.560 |

## LT1

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | GBM(n=50,d=2) | 10 | 107 | 2.311 | **2.277** | 0.811 | 0.905 |
| **EMG+NIRS** | Ridge(α=1000) | 10 | 97 | 2.897 | **2.750** | 0.730 | 0.869 |
| **EMG** | Ridge(α=1000) | 10 | 79 | 3.047 | **2.896** | 0.677 | 0.849 |
| **NIRS** | SVR(C=10,ε=1.0) | 10 | 18 | 4.185 | **4.063** | 0.428 | 0.717 |

## Полная таблица (все конфиги)

### LT2

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 1.859 |
| EMG+NIRS+HRV | Ridge(α=100) | 1.866 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 1.881 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 1.953 |
| EMG+NIRS+HRV | GBM(n=200,d=2) | 1.984 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.9) | 2.043 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.049 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.066 |
| EMG+NIRS+HRV | GBM(n=100,d=2) | 2.071 |
| EMG+NIRS+HRV | EN(α=0.01,l1=0.2) | 2.098 |
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.141 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.192 |
| EMG+NIRS+HRV | EN(α=0.01,l1=0.5) | 2.192 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.211 |
| EMG+NIRS+HRV | Ridge(α=10) | 2.251 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.275 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.386 |
| EMG+NIRS+HRV | EN(α=0.01,l1=0.9) | 2.524 |
| EMG+NIRS+HRV | SVR(C=100,ε=0.1) | 2.664 |
| EMG+NIRS+HRV | SVR(C=100,ε=1.0) | 2.665 |

### LT1

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.277 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.278 |
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.279 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.307 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.320 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.321 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.322 |
| EMG+NIRS+HRV | GBM(n=100,d=2) | 2.339 |
| EMG+NIRS+HRV | GBM(n=200,d=2) | 2.462 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.508 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.619 |
| EMG+NIRS | Ridge(α=1000) | 2.750 |
| EMG+NIRS | EN(α=1.0,l1=0.5) | 2.751 |
| EMG+NIRS | EN(α=1.0,l1=0.2) | 2.763 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.784 |
| EMG+NIRS+HRV | Ridge(α=100) | 2.787 |
| EMG+NIRS | EN(α=1.0,l1=0.9) | 2.848 |
| EMG | Ridge(α=1000) | 2.896 |
| EMG | EN(α=1.0,l1=0.5) | 2.898 |
| EMG | EN(α=1.0,l1=0.2) | 2.912 |

## Honest baselines

### Honest baselines — EMG / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.788 | — |
| FirstWindowPredictor | 5.404 | 127 |
| Raw model (no Kalman) | 3.324 | 144 |
| Kalman (σ_obs=ref)  | 3.198 | — |

gap (raw − FirstWin): -2.079 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — NIRS / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.788 | — |
| FirstWindowPredictor | 3.813 | 149 |
| Raw model (no Kalman) | 4.357 | 193 |
| Kalman (σ_obs=ref)  | 4.251 | — |

gap (raw − FirstWin): +0.543 мин
**⚠ sequential хуже FirstWindow (нет онлайн-ценности)**

### Honest baselines — EMG+NIRS / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.788 | — |
| FirstWindowPredictor | 4.980 | 109 |
| Raw model (no Kalman) | 3.249 | 137 |
| Kalman (σ_obs=ref)  | 3.117 | — |

gap (raw − FirstWin): -1.731 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.788 | — |
| FirstWindowPredictor | 4.387 | 119 |
| Raw model (no Kalman) | 1.992 | 92 |
| Kalman (σ_obs=ref)  | 1.859 | — |

gap (raw − FirstWin): -2.395 мин
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

### Honest baselines — NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.251 | — |
| FirstWindowPredictor | 4.227 | 174 |
| Raw model (no Kalman) | 4.185 | 213 |
| Kalman (σ_obs=ref)  | 4.063 | — |

gap (raw − FirstWin): -0.042 мин
**~ нет значимой разницы с FirstWindow**

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

