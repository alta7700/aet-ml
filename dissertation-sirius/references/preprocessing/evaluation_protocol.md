# Протокол оценки: честные baselines и sequential tracking

> Файл: `preprocessing/evaluation_protocol.md`
> Модуль: `dataset_pipeline/baselines.py`
> Обязателен для: всех версий v0011+

---

## Мотивация

Регрессор «время до порога» работает на временных рядах. Риск: модель может
предсказывать полную длительность теста с первого окна и затем просто
отсчитывать таймер — без реального физиологического отслеживания.
Эта ситуация возникает при агрессивном Kalman-сглаживании (σ_obs ≫ σ_p).

Для честной оценки каждая версия обязана сравниваться с двумя
наивными baseline-ами.

---

## Два обязательных baseline

### MeanPredictor

Для тестового субъекта предсказывает медиану LT (в секундах) из обучающих
субъектов. Затем вычитает elapsed_sec:

```
y_pred[t] = max(median_train_LT − window_start_sec[t], 0)
```

**Что измеряет:** межсубъектную вариабельность порога.
Если модель не бьёт MeanPredictor — она не несёт никакой ценности.

### FirstWindowPredictor

Та же архитектура модели, но:
- Обучается **только на первом окне** каждого субъекта
- Для тестового субъекта предсказывает одно фиксированное число (predicted_LT_abs)
- Применяет это число для всех окон: `y_pred[t] = max(predicted_LT_abs − t, 0)`

**Что измеряет:** ценность «начального снимка» без последующего обновления.
Если полная sequential модель не бьёт FirstWindowPredictor — online-слежение
не добавляет ценности (модель = таймер).

---

## Метрика честности: gap

```
gap = MAE_raw_sequential − MAE_FirstWindow
```

| gap | Вердикт |
|---|---|
| < −0.2 мин | ✓ Sequential tracking значимо лучше FirstWindow |
| −0.2 … +0.2 | ~ Нет значимой онлайн-ценности |
| > +0.2 мин | ⚠ Sequential хуже FirstWindow (нет смысла в online) |

Используется **raw MAE** (без Kalman) — чтобы сглаживание не маскировало
отсутствие реального tracking.

---

## Stability score

```
stability_std[subj] = std(window_start_sec[t] + y_pred[t])
```

Усреднённое по всем субъектам.

| Значение | Интерпретация |
|---|---|
| ≈ 0 с | Модель = чистый таймер |
| > 100 с | Модель активно обновляет предсказание |

Для сравнения: FirstWindowPredictor имеет stability_std ≈ 100–200 с
(ненулевое из-за шума первой точки).

---

## Влияние σ_obs Kalman на честность

| σ_obs | Kalman gain | Поведение |
|---|---|---|
| 30–75 с | ~0.1–0.3 | Модель реально обновляется, честный tracking |
| 150 с | ~0.05 | Умеренное сглаживание, ещё честно |
| 500–1000 с | ~0.01 | Почти таймер — маскирует плохой tracking |

**Рекомендация:** основная метрика в тексте диссертации — σ_obs=150.
σ_obs=500 приводить как upper bound с объяснением.

---

## Результаты v0011 (для справки)

| Модальность | gap (raw−FW) | Вердикт |
|---|---|---|
| EMG | −2.08 мин | ✓ sequential значимо лучше |
| NIRS / LT2 | +0.54 мин | ⚠ sequential хуже FirstWindow |
| NIRS / LT1 | −0.04 мин | ~ нет значимой разницы |
| EMG+NIRS | −1.73 мин | ✓ sequential значимо лучше |
| EMG+NIRS+HRV | −2.40 мин | ✓ sequential значимо лучше |

**Вывод:** NIRS как самостоятельная модальность не несёт онлайн-ценности.
В комбинации с EMG+HRV — полезна через корреляцию с HRV-признаками
(interaction features).

---

## Результаты v0012 (body composition, LT2)

Добавлены статические признаки состава тела: BMI, body_fat_pct, phase_angle,
dominant_leg_circumference, leg_fat_pct, muscle_to_fat_leg (из биоимпеданса).

| Набор | MAE мин | Δ vs база | gap (raw−FW) | Вердикт |
|---|---|---|---|---|
| BodyComp only | 5.38 | — | +0.83 | ⚠ хуже FirstWindow |
| EMG+BC | 3.075 | −0.12 | −2.23 | ✓ |
| EMG+NIRS+BC | 2.730 | −0.39 | −2.20 | ✓ |
| EMG+NIRS+HRV+BC | 1.901 | **+0.04** | −2.44 | ✓ |

**Вывод по v0012:**
- BC-признаки одни неинформативны: хуже MeanPredictor (5.38 мин > 3.79 мин).
- Эффект добавления BC нестабилен: LT2 и LT1 дают противоположные знаки Δ для одних
  и тех же наборов (EMG+NIRS: −0.39 для LT2, +0.10 для LT1; EMG+NIRS+HRV: +0.04 для LT2, −0.19 для LT1).
- Per-subject Δ сильно разбросана (±1 мин): нет устойчивого вклада BC.
- Итог: N=14 недостаточно для надёжной оценки статических признаков. BC не включать в итоговую модель.
- Честные baselines для BC-наборов остаются ✓ (gap < −0.2 мин) — online-слежение сохраняется.

---

## Интеграция в пайплайн

Модуль: `dataset_pipeline/baselines.py`

```python
from dataset_pipeline.baselines import run_honest_baselines, format_honest_block

hb = run_honest_baselines(
    df, feat_cols, target_col,
    raw_per_subject=raw_ps,      # dict[subject_id → {t_sec, y_pred, y_true}]
    model_factory=best_factory,
    kalman_fn=kalman_smooth,
    sigma_p=5.0,
    sigma_obs_ref=150.0,
)
# hb содержит: mean_pred, first_win, raw, kalman_ref, gap_raw_vs_fw, verdict
```

Запускается **автоматически** для лучшей модели каждой (feature_set, target)
пары. Результаты сохраняются в `honest_baselines.csv` и `report.md`.

---

## Связанные файлы

- `dataset_pipeline/baselines.py` — реализация MeanPredictor, FirstWindowPredictor
- `scripts/v0011c_honest_baselines.py` — полный диагностический скрипт с кривой MAE vs σ_obs
- `results/v0011/honest_baselines.csv` — результаты для v0011
- `scripts/v0012_body_comp.py` — аблация с признаками состава тела
- `results/v0012/honest_baselines.csv` — результаты v0012
- `results/v0012/body_comp_delta.csv` — per-subject Δ MAE при добавлении BC
