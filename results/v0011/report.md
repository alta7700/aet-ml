# v0011 — Ablation по модальностям
Дата: 2026-05-13  
Метрика: MAE (мин) после фильтра Калмана, LOSO CV.

## LT1

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | SVR(C=100,ε=1.0) | 18 | 107 | 3.287 | **3.100** | 0.574 | 0.791 |
| **EMG+NIRS** | SVR(C=100,ε=1.0) | 18 | 97 | 3.517 | **3.287** | 0.520 | 0.745 |
| **EMG** | GBM(n=100,d=2) | 18 | 79 | 3.767 | **3.544** | 0.437 | 0.710 |
| **NIRS** | SVR(C=100,ε=1.0) | 18 | 18 | 4.142 | **3.853** | 0.371 | 0.719 |
| **HRV** | SVR(C=10,ε=0.1) | 18 | 7 | 4.014 | **3.948** | 0.379 | 0.654 |

## Полная таблица (все конфиги)

### LT1

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | SVR(C=100,ε=1.0) | 3.100 |
| EMG+NIRS+HRV | SVR(C=100,ε=0.1) | 3.100 |
| EMG+NIRS+HRV | SVR(C=10,ε=0.1) | 3.220 |
| EMG+NIRS+HRV | SVR(C=10,ε=1.0) | 3.221 |
| EMG+NIRS | SVR(C=100,ε=1.0) | 3.287 |
| EMG+NIRS | SVR(C=100,ε=0.1) | 3.287 |
| EMG+NIRS | SVR(C=10,ε=0.1) | 3.409 |
| EMG+NIRS | SVR(C=10,ε=1.0) | 3.410 |
| EMG | GBM(n=100,d=2) | 3.544 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 3.591 |
| EMG | GBM(n=50,d=3) | 3.608 |
| EMG | GBM(n=50,d=2) | 3.612 |
| EMG | GBM(n=100,d=3) | 3.626 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 3.631 |
| EMG | SVR(C=100,ε=1.0) | 3.643 |
| EMG | SVR(C=100,ε=0.1) | 3.643 |
| EMG | GBM(n=200,d=2) | 3.665 |
| EMG | GBM(n=200,d=3) | 3.686 |
| EMG+NIRS+HRV | Ridge(α=1000) | 3.714 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 3.830 |

## Honest baselines

### Honest baselines — EMG / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 5.085 | — |
| FirstWindowPredictor | 4.993 | 245 |
| Raw model (no Kalman) | 3.767 | 151 |
| Kalman (σ_obs=ref)  | 3.544 | — |

gap (raw − FirstWin): -1.226 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 5.085 | — |
| FirstWindowPredictor | 5.090 | 271 |
| Raw model (no Kalman) | 4.142 | 178 |
| Kalman (σ_obs=ref)  | 3.853 | — |

gap (raw − FirstWin): -0.948 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 5.085 | — |
| FirstWindowPredictor | 5.084 | 269 |
| Raw model (no Kalman) | 4.014 | 188 |
| Kalman (σ_obs=ref)  | 3.948 | — |

gap (raw − FirstWin): -1.070 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 5.085 | — |
| FirstWindowPredictor | 5.124 | 271 |
| Raw model (no Kalman) | 3.517 | 152 |
| Kalman (σ_obs=ref)  | 3.287 | — |

gap (raw − FirstWin): -1.607 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 5.085 | — |
| FirstWindowPredictor | 5.129 | 272 |
| Raw model (no Kalman) | 3.287 | 131 |
| Kalman (σ_obs=ref)  | 3.100 | — |

gap (raw − FirstWin): -1.842 мин
**✓ sequential значимо лучше FirstWindow**

