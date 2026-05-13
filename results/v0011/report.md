# v0011 — Ablation по модальностям
Дата: 2026-05-13  
Метрика: MAE (мин) после фильтра Калмана, LOSO CV.

## LT1

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | EN(α=1.0,l1=0.2) | 14 | 107 | 2.279 | **2.231** | 0.816 | 0.915 |
| **EMG+NIRS** | GBM(n=50,d=2) | 14 | 97 | 2.608 | **2.469** | 0.754 | 0.874 |
| **HRV** | Huber(ε=1.35) | 14 | 7 | 2.543 | **2.622** | 0.753 | 0.874 |
| **EMG** | EN(α=1.0,l1=0.5) | 14 | 79 | 2.857 | **2.731** | 0.711 | 0.868 |
| **NIRS** | GBM(n=50,d=2) | 14 | 18 | 4.380 | **4.162** | 0.361 | 0.614 |

## Полная таблица (все конфиги)

### LT1

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.231 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.240 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.260 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.347 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.391 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.412 |
| EMG+NIRS+HRV | GBM(n=100,d=2) | 2.449 |
| EMG+NIRS+HRV | GBM(n=200,d=2) | 2.458 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.459 |
| EMG+NIRS | GBM(n=50,d=2) | 2.469 |
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.492 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.510 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.511 |
| EMG+NIRS+HRV | Ridge(α=100) | 2.522 |
| EMG+NIRS | GBM(n=100,d=2) | 2.544 |
| EMG+NIRS | GBM(n=200,d=2) | 2.568 |
| EMG+NIRS | EN(α=1.0,l1=0.5) | 2.593 |
| EMG+NIRS | EN(α=1.0,l1=0.2) | 2.603 |
| EMG+NIRS | Ridge(α=1000) | 2.613 |
| HRV | Huber(ε=1.35) | 2.622 |

## Honest baselines

### Honest baselines — EMG / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.800 | — |
| FirstWindowPredictor | 4.566 | 150 |
| Raw model (no Kalman) | 2.857 | 152 |
| Kalman (σ_obs=ref)  | 2.731 | — |

gap (raw − FirstWin): -1.709 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.800 | — |
| FirstWindowPredictor | 4.046 | 141 |
| Raw model (no Kalman) | 4.380 | 203 |
| Kalman (σ_obs=ref)  | 4.162 | — |

gap (raw − FirstWin): +0.334 мин
**⚠ sequential хуже FirstWindow (нет онлайн-ценности)**

### Honest baselines — HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.800 | — |
| FirstWindowPredictor | 3.907 | 162 |
| Raw model (no Kalman) | 2.543 | 114 |
| Kalman (σ_obs=ref)  | 2.622 | — |

gap (raw − FirstWin): -1.365 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.800 | — |
| FirstWindowPredictor | 4.340 | 160 |
| Raw model (no Kalman) | 2.608 | 136 |
| Kalman (σ_obs=ref)  | 2.469 | — |

gap (raw − FirstWin): -1.732 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.800 | — |
| FirstWindowPredictor | 4.430 | 149 |
| Raw model (no Kalman) | 2.279 | 114 |
| Kalman (σ_obs=ref)  | 2.231 | — |

gap (raw − FirstWin): -2.151 мин
**✓ sequential значимо лучше FirstWindow**

