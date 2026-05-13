# v0011 — Ablation по модальностям
Дата: 2026-05-13  
Метрика: MAE (мин) после фильтра Калмана, LOSO CV.

## LT2

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | EN(α=1.0,l1=0.9) | 18 | 107 | 2.248 | **2.175** | 0.826 | 0.919 |
| **HRV** | EN(α=0.1,l1=0.2) | 18 | 7 | 2.247 | **2.218** | 0.817 | 0.911 |
| **EMG+NIRS** | EN(α=1.0,l1=0.2) | 18 | 97 | 2.941 | **2.796** | 0.657 | 0.835 |
| **EMG** | EN(α=1.0,l1=0.5) | 18 | 79 | 3.078 | **2.923** | 0.641 | 0.832 |
| **NIRS** | SVR(C=100,ε=0.1) | 18 | 18 | 4.220 | **4.052** | 0.333 | 0.615 |

## Полная таблица (все конфиги)

### LT2

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.175 |
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.183 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.188 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.208 |
| HRV | EN(α=0.1,l1=0.2) | 2.218 |
| HRV | EN(α=0.1,l1=0.5) | 2.220 |
| HRV | EN(α=1.0,l1=0.9) | 2.221 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.222 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.229 |
| HRV | Ridge(α=100) | 2.231 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.243 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.244 |
| HRV | EN(α=0.1,l1=0.9) | 2.249 |
| HRV | EN(α=0.01,l1=0.2) | 2.253 |
| HRV | EN(α=0.01,l1=0.5) | 2.258 |
| HRV | Ridge(α=10) | 2.262 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.264 |
| EMG+NIRS+HRV | Ridge(α=100) | 2.265 |
| HRV | EN(α=0.01,l1=0.9) | 2.266 |
| HRV | Ridge(α=1) | 2.270 |

## Honest baselines

### Honest baselines — EMG / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.756 | — |
| FirstWindowPredictor | 4.919 | 128 |
| Raw model (no Kalman) | 3.070 | 140 |
| Kalman (σ_obs=ref)  | 2.923 | — |

gap (raw − FirstWin): -1.849 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — NIRS / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.756 | — |
| FirstWindowPredictor | 3.857 | 137 |
| Raw model (no Kalman) | 4.220 | 182 |
| Kalman (σ_obs=ref)  | 4.052 | — |

gap (raw − FirstWin): +0.363 мин
**⚠ sequential хуже FirstWindow (нет онлайн-ценности)**

### Honest baselines — HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.756 | — |
| FirstWindowPredictor | 3.062 | 107 |
| Raw model (no Kalman) | 2.247 | 92 |
| Kalman (σ_obs=ref)  | 2.218 | — |

gap (raw − FirstWin): -0.815 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.756 | — |
| FirstWindowPredictor | 4.903 | 117 |
| Raw model (no Kalman) | 2.941 | 134 |
| Kalman (σ_obs=ref)  | 2.796 | — |

gap (raw − FirstWin): -1.962 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.756 | — |
| FirstWindowPredictor | 4.279 | 118 |
| Raw model (no Kalman) | 2.248 | 95 |
| Kalman (σ_obs=ref)  | 2.175 | — |

gap (raw − FirstWin): -2.031 мин
**✓ sequential значимо лучше FirstWindow**

