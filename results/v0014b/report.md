# v0014b — Ablation: только внутриоконные тренды (без абсолютных уровней)
Дата: 2026-05-12  
Метрика: MAE (мин) после фильтра Калмана, LOSO CV.  
N = 19 субъектов.

## Что исключено

**v0014a (прямые носители времени/мощности):**
`smo2_from_running_max`, `hhb_from_running_min`, `smo2_rel_drop_pct`, `feat_rr_per_watt`

**v0014b (дополнительно — абсолютные уровни, монотонные по тесту):**
`trainred_smo2_mean`, `trainred_hhb_mean`, `trainred_hbdiff_mean`, `trainred_thb_mean`,
`hrv_mean_rr_ms`, `feat_smo2_x_rr`

Остаются только **внутриоконные динамические признаки**: ЭМГ-амплитуда и частоты (z-norm per-subject), NIRS-slope/std/drop/dsmo2_dt/dhhb_dt, HRV-вариабельность (sdnn, rmssd, dfa_alpha1 и др.).

## Ключевые выводы

### 1. Прямые носители протокола были лишними (v0014a)
Убрав smo2_from_running_max и аналоги, **полная модель EMG+NIRS+HRV даже улучшилась** (LT2: 2.095 vs 2.163 в v0011, Δ=−0.067). Эти признаки не несли уникальной информации — только кодировали время.

### 2. hrv_mean_rr_ms несёт реальную физиологию (v0014b)
После исключения абсолютного RR-интервала HRV-набор потерял **0.604 мин** (LT2: 2.845 vs 2.241). Он не просто кодирует время — замедление сердца отражает реальный физиологический статус.

### 3. EMG+NIRS без абсолютных уровней приблизился к полной модели
**Главный результат v0014b:** EMG+NIRS (2.664 LT2, 2.291 LT1) vs EMG+NIRS+HRV (2.423 LT2, 2.275 LT1).  
Разрыв сократился до **0.24 мин по LT2 и 0.016 мин по LT1** — против 0.5+ мин в v0011.  
Когда убираем "временной шум" из абсолютных уровней, ЭМГ+NIRS несут почти всю полезную физиологическую информацию без HRV.

### 4. NIRS без абсолютных уровней ведёт себя лучше по LT2
NIRS (v0014b): 4.000 мин vs NIRS (v0014a): 4.434 мин — Δ=−0.434. Абсолютные уровни SmO₂/HHb вносили шум для одиночного NIRS-набора.



## LT2

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | EN(α=0.1,l1=0.5) | 19 | 97 | 2.581 | **2.423** | 0.761 | 0.882 |
| **EMG+NIRS** | GBM(n=100,d=2) | 19 | 90 | 2.888 | **2.664** | 0.672 | 0.834 |
| **HRV** | Huber(ε=1.5) | 19 | 6 | 2.908 | **2.845** | 0.667 | 0.844 |
| **EMG** | EN(α=1.0,l1=0.2) | 19 | 79 | 3.133 | **2.993** | 0.625 | 0.823 |
| **NIRS** | SVR(C=10,ε=0.1) | 19 | 11 | 4.091 | **4.000** | 0.422 | 0.653 |

## LT1

| Набор | Лучшая модель | n субъектов | Признаков | RAW MAE | Kalman MAE | R² | ρ |
|---|---|---|---|---|---|---|---|
| **EMG+NIRS+HRV** | GBM(n=100,d=2) | 14 | 97 | 2.430 | **2.275** | 0.773 | 0.887 |
| **EMG+NIRS** | GBM(n=100,d=2) | 14 | 90 | 2.520 | **2.291** | 0.763 | 0.883 |
| **EMG** | GBM(n=100,d=2) | 14 | 79 | 2.831 | **2.672** | 0.701 | 0.859 |
| **HRV** | Huber(ε=2.0) | 14 | 6 | 3.002 | **3.010** | 0.651 | 0.830 |
| **NIRS** | SVR(C=10,ε=1.0) | 14 | 11 | 3.953 | **3.837** | 0.458 | 0.699 |

## Полная таблица (все конфиги)

### LT2

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.423 |
| EMG+NIRS+HRV | Ridge(α=100) | 2.425 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.428 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.434 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.9) | 2.440 |
| EMG+NIRS+HRV | EN(α=0.01,l1=0.2) | 2.457 |
| EMG+NIRS+HRV | EN(α=0.01,l1=0.5) | 2.473 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.478 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.482 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.489 |
| EMG+NIRS+HRV | Ridge(α=10) | 2.494 |
| EMG+NIRS+HRV | GBM(n=200,d=2) | 2.505 |
| EMG+NIRS+HRV | EN(α=0.01,l1=0.9) | 2.522 |
| EMG+NIRS+HRV | Huber(ε=2.0) | 2.524 |
| EMG+NIRS+HRV | GBM(n=100,d=2) | 2.535 |
| EMG+NIRS+HRV | Ridge(α=1) | 2.542 |
| EMG+NIRS+HRV | Huber(ε=1.5) | 2.567 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.576 |
| EMG+NIRS+HRV | Huber(ε=1.35) | 2.587 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.599 |

### LT1

| Набор | Модель | Kalman MAE |
|---|---|---|
| EMG+NIRS+HRV | GBM(n=100,d=2) | 2.275 |
| EMG+NIRS+HRV | GBM(n=50,d=2) | 2.290 |
| EMG+NIRS | GBM(n=100,d=2) | 2.291 |
| EMG+NIRS | GBM(n=50,d=2) | 2.293 |
| EMG+NIRS+HRV | GBM(n=200,d=2) | 2.304 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.5) | 2.332 |
| EMG+NIRS | GBM(n=200,d=2) | 2.335 |
| EMG+NIRS+HRV | Ridge(α=1000) | 2.343 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.2) | 2.344 |
| EMG+NIRS+HRV | EN(α=1.0,l1=0.9) | 2.376 |
| EMG+NIRS+HRV | GBM(n=50,d=3) | 2.407 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.2) | 2.428 |
| EMG+NIRS | GBM(n=200,d=3) | 2.428 |
| EMG+NIRS | GBM(n=100,d=3) | 2.435 |
| EMG+NIRS | GBM(n=50,d=3) | 2.442 |
| EMG+NIRS+HRV | GBM(n=100,d=3) | 2.446 |
| EMG+NIRS | EN(α=1.0,l1=0.9) | 2.455 |
| EMG+NIRS+HRV | EN(α=0.1,l1=0.5) | 2.460 |
| EMG+NIRS+HRV | GBM(n=200,d=3) | 2.472 |
| EMG+NIRS | Ridge(α=1000) | 2.485 |

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
| FirstWindowPredictor | 3.830 | 140 |
| Raw model (no Kalman) | 4.091 | 211 |
| Kalman (σ_obs=ref)  | 4.000 | — |

gap (raw − FirstWin): +0.261 мин
**⚠ sequential хуже FirstWindow (нет онлайн-ценности)**

### Honest baselines — HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.869 | — |
| FirstWindowPredictor | 4.786 | 128 |
| Raw model (no Kalman) | 2.908 | 135 |
| Kalman (σ_obs=ref)  | 2.845 | — |

gap (raw − FirstWin): -1.878 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.869 | — |
| FirstWindowPredictor | 4.612 | 122 |
| Raw model (no Kalman) | 2.888 | 147 |
| Kalman (σ_obs=ref)  | 2.664 | — |

gap (raw − FirstWin): -1.724 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT2

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 3.869 | — |
| FirstWindowPredictor | 5.747 | 133 |
| Raw model (no Kalman) | 2.581 | 130 |
| Kalman (σ_obs=ref)  | 2.423 | — |

gap (raw − FirstWin): -3.167 мин
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
| FirstWindowPredictor | 3.895 | 181 |
| Raw model (no Kalman) | 3.953 | 223 |
| Kalman (σ_obs=ref)  | 3.837 | — |

gap (raw − FirstWin): +0.058 мин
**~ нет значимой разницы с FirstWindow**

### Honest baselines — HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.023 | — |
| FirstWindowPredictor | 4.061 | 143 |
| Raw model (no Kalman) | 3.002 | 153 |
| Kalman (σ_obs=ref)  | 3.010 | — |

gap (raw − FirstWin): -1.060 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.023 | — |
| FirstWindowPredictor | 4.540 | 178 |
| Raw model (no Kalman) | 2.520 | 142 |
| Kalman (σ_obs=ref)  | 2.291 | — |

gap (raw − FirstWin): -2.020 мин
**✓ sequential значимо лучше FirstWindow**

### Honest baselines — EMG+NIRS+HRV / LT1

| Модель | MAE (мин) | Stability std (с) |
|---|---|---|
| MeanPredictor       | 4.023 | — |
| FirstWindowPredictor | 4.335 | 175 |
| Raw model (no Kalman) | 2.430 | 132 |
| Kalman (σ_obs=ref)  | 2.275 | — |

gap (raw − FirstWin): -1.905 мин
**✓ sequential значимо лучше FirstWindow**

