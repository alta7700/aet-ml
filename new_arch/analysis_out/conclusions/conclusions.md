# Auto-generated conclusions
Primary метрика: `lt_mae_median_policy_mean` (LT-time prediction MAE).
Secondary: `mae_mean` (regression fit).
Уровень значимости α = 0.05.
P-values скорректированы методом Benjamini–Hochberg (FDR) внутри `(comparison_kind × metric)`; используется колонка `pvalue_adj`.

## Multimodal fusion vs unimodal EMG
Сравнений (multi vs EMG): 20. Multi значимо лучше: 0 (доля 0%, α=0.05).
median Δ = +21.4 sec.

## ABS (|EMG|) features
Сравнений (abs vs no_abs): 378. ABS значимо лучше: 0 (0%). median Δ = +7.0 sec.

## Wavelet preprocessing
- cwt vs none: n=6, значимых 0; median Δ = +0.6 sec → лучше **none**

## LT1 vs LT2 difficulty
Сравнений: 336, значимых 0. Median Δ(LT1 − LT2) = +21.3 sec → LT2 проще.

## Architecture family
Средний lt_mae_median_policy_mean по семействам:
- **LSTM**: 178.0 sec
- **Lin**: 182.2 sec
- **TCN**: 187.1 sec

Лучшее семейство по primary метрике: **LSTM**.

## Temporal configuration (family=LSTM)
Лучшая комбинация: stride=5s, sequence_length=6 → lt_mae_median_policy_mean = 162.5 sec.
Полная таблица:
- stride=5s, seq_len=6: 162.5 sec
- stride=5s, seq_len=0: 169.8 sec
- stride=15s, seq_len=12: 173.6 sec
- stride=15s, seq_len=0: 181.4 sec
- stride=30s, seq_len=6: 182.6 sec
- stride=15s, seq_len=6: 182.7 sec
- stride=30s, seq_len=0: 183.3 sec
- stride=30s, seq_len=12: 188.3 sec

## Uncertainty (Jackknife+ conformal, LT median policy)
Конформные интервалы построены leave-one-subject-out (N=18). Из-за малой калибровочной выборки честно репортируются только умеренные α: эмпирическое покрытие должно сходиться к 1−α.

- α=0.20 (nominal 80%): median empirical=89%, gap=+8.89%, median width=778 sec. Моделей с покрытием в пределах ±5% от nominal: 0/780.
- α=0.30 (nominal 70%): median empirical=78%, gap=+7.78%, median width=547 sec. Моделей с покрытием в пределах ±5% от nominal: 0/780.

## Training stability (200 NN-моделей)
- Средний converged_rate (доля фолдов с |slope| ниже порога): 37%.
- Median training_instability (CV последних шагов): 0.736.
- Mean train_mae_slope_last_K: -0.002 sec/epoch (отрицательный = всё ещё улучшается).
**Замечание:** без validation-split это не overfitting, а только динамика train-метрик.
