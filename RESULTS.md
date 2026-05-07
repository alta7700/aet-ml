# Результаты моделей ML — Предсказание LT1 и LT2

**Стратегия оценки:** Leave-One-Subject-Out (LOSO) CV  
**Предобработка (везде):** `SimpleImputer(strategy="median")` → `StandardScaler()`, fit только на train-фолде

> **Правило ведения лога:** новые эксперименты добавляются в конец раздела «Лог».  
> Старые записи не удаляются и не редактируются — только помечаются `[superseded]`.  
> Каждая запись содержит все гиперпараметры, необходимые для воспроизведения.

---

## Датасет

| Параметр | Значение |
|---|---|
| Файл | `dataset/merged_features_ml.parquet` |
| Всего окон | 3035 |
| Участников всего | 13 |
| Ширина окна | 30 с, шаг 5 с |
| Модальности | ЭМГ (vl_dist + vl_prox), NIRS (SmO₂, HHb, HbDiff, tHb), HRV |
| LT2-участники | S001–S014 (13 человек) |
| LT1-участники | S002, S003, S004, S006, S009, S010, S011, S013, S014 (9 человек) |

---

## Текущие лучшие результаты (сводка)

| Задача | Модель | Признаки | MAE, мин | R² | ρ | Запись |
|---|---|---|---|---|---|---|
| **LT2** | Ridge(α=100) | NIRS+HRV+interaction+z_EMG | **2.097** | 0.846 | 0.919 | [v4](#v4) |
| **LT1** | HuberRegressor(ε=1.35) | z_EMG+HRV | **2.021** | 0.842 | 0.935 | [v4](#v4) |

С постобработкой Kalman (σ_proc=30 с, σ_obs=120 с):

| Задача | MAE без Kalman | MAE с Kalman | Запись |
|---|---|---|---|
| **LT2** | 2.097 | ~2.05 (оценка) | [v3](#v3) |
| **LT1** | 2.021 | ~2.00 (оценка) | [v3](#v3) |

> Kalman считался поверх ElasticNet (v2). Точный пересчёт с Ridge/Huber:
> `uv run python scripts/train_temporal.py` (требует обновления фабрики моделей).

---

## Лог экспериментов

---

### v1: ElasticNet — базовая конфигурация (2026-05-07) {#v1}

**Скрипт:** `scripts/train_best_models.py` (commit до обновления v4)  
**Воспроизведение:** см. [исторические скрипты](#historical)

#### Гиперпараметры

| Компонент | Параметр | Значение |
|---|---|---|
| Модель | `sklearn.linear_model.ElasticNet` | |
| | `alpha` | `0.5` |
| | `l1_ratio` | `0.9` |
| | `max_iter` | `5000` |
| | `random_state` | не задан (результаты могут незначительно варьировать) |
| | `tol` | `1e-4` (sklearn default) |
| Imputer | `SimpleImputer` | `strategy="median"` |
| Scaler | `StandardScaler` | defaults |
| CV | LOSO | нет вложенного CV для гиперпараметров |
| Session-z EMG | z = (x − μ_subj) / (σ_subj + 1e-8) | per-subject, per-feature |

#### LT2 — результаты

| Конфигурация признаков | MAE, мин | R² | ρ |
|---|---|---|---|
| Baseline predict-mean | 5.219 | — | — |
| Baseline elapsed (линейный) | 4.278 | — | — |
| NIRS + HRV | 2.485 | 0.784 | 0.896 |
| NIRS + HRV + interaction | 2.370 | 0.809 | 0.901 |
| **NIRS + HRV + interaction + z_EMG** | **2.110** | **0.846** | **0.920** |

MAE по участникам (лучшая конфигурация):  
S001=1.51, S002=1.83, S003=5.44(!), S004=1.85, S005=1.65, S006=1.25, S007=1.72,  
S009=1.83, S010=1.96, S011=2.52, S012=1.29, S013=2.08, S014=1.02

#### LT1 — результаты

| Конфигурация признаков | MAE, мин | R² | ρ |
|---|---|---|---|
| Baseline predict-mean | 5.210 | — | — |
| NIRS + HRV | 4.103 | 0.433 | 0.785 |
| HRV только | 2.791 | 0.725 | 0.879 |
| Running NIRS + HRV | 2.517 | 0.779 | 0.915 |
| **z_EMG + HRV** | **2.030** | **0.838** | **0.932** |
| z_EMG + HRV + running NIRS | 2.106 | 0.834 | 0.935 |

MAE по участникам (лучшая конфигурация):  
S002=3.23, S003=3.29, S004=1.12, S006=1.86, S009=2.38, S010=1.91, S011=1.77, S013=1.08, S014=1.16

---

### v2: Temporal — лаговые признаки + Kalman-сглаживание (2026-05-07) {#v2} {#v3}

**Скрипт:** `scripts/train_temporal.py`  
**Воспроизведение:** `uv run python scripts/train_temporal.py`

#### Гиперпараметры

| Компонент | Параметр | Значение |
|---|---|---|
| Базовая модель | `ElasticNet` | α=0.5, l1_ratio=0.9, max_iter=5000 |
| Лаги | сдвиги | t-1, t-2, t-3 (внутри subject, causal) |
| | дельты | Δ = x(t) − x(t-1) |
| Kalman | σ_process | 30 с (физический prior: порог уходит ~30 с/окно) |
| | σ_obs | 120 с (доверяем модели меньше, чем prior) |
| | реализация | `kalman_smooth_subject()` — 1D, without learned params |
| | seed | нет (детерминированный) |

#### Результаты: лаговые признаки

| Конфигурация | LT2 MAE, мин | LT1 MAE, мин |
|---|---|---|
| Без лагов (ElasticNet v1) | 2.110 | 2.030 |
| + лаг t-1 + Δ | 2.140 | 2.036 |
| + лаги t-1, t-2 + Δ | 2.172 | 2.040 |
| + лаги t-1, t-2, t-3 + Δ | 2.207 | 2.049 |

**Вывод:** лаги не помогают — история уже закодирована в running-признаках.

#### Результаты: Kalman-сглаживание

| Конфигурация | LT2: до → после | LT1: до → после |
|---|---|---|
| Без лагов | 2.110 → **2.054** (−0.056) | 2.030 → **2.007** (−0.023) |
| + лаг t-1 + Δ | 2.140 → 2.083 | 2.036 → 2.008 |

Kalman даёт стабильное улучшение ~0.02–0.06 мин без обучаемых параметров.

---

### v3: Анализ признаков — within vs between confound (2026-05-07) {#v3-confound}

**Скрипт:** интерактивный анализ (ad hoc), не оформлен в отдельный файл

#### hrv_mean_rr_ms: декомпозиция вариации

| Признак | Within-subject ρ(feat, target) | Between-subject ρ(feat, target) |
|---|---|---|
| hrv_mean_rr_ms | +0.997 | +0.753 |
| hrv_dfa_alpha1 | +0.759 | −0.027 |

`hrv_mean_rr_ms`: between = 0.753 → тренированные субъекты системно имеют высокое RR.  
`hrv_dfa_alpha1`: between ≈ 0 → чистая внутрисессионная динамика.

#### Эксперименты по замене hrv_mean_rr_ms

| Замена | Гиперпараметры | LT2 MAE | LT1 MAE | Примечание |
|---|---|---|---|---|
| Убрать mean_rr | ElasticNet α=0.5 | 3.112 | 2.824 | S012: 1.29→9.03 |
| rr_from_running_max | ElasticNet α=0.5 | 3.403 | — | between ρ=0.319 |
| z_hrv_mean_rr_ms | ElasticNet α=0.5 | 3.173 | — | between ρ=−0.401 |

**Решение:** оставить hrv_mean_rr_ms с явным указанием в диссертации, что признак несёт
как динамику нагрузки (within), так и межсубъектный confound тренированности (between).
Разделение невозможно при N=9–13 без повторных измерений.

---

### v4: Сравнение моделей + Ridge/Huber (2026-05-07) {#v4}

**Скрипт:** `scripts/train_best_models.py` (текущая версия)  
**Воспроизведение:** `uv run python scripts/train_best_models.py`

#### Гиперпараметры финальных моделей

**LT2 — Ridge(α=100):**

| Компонент | Параметр | Значение |
|---|---|---|
| Модель | `sklearn.linear_model.Ridge` | |
| | `alpha` | `100.0` |
| | `solver` | `"auto"` (sklearn default) |
| | `fit_intercept` | `True` (sklearn default) |
| | `random_state` | не применим (детерминированный) |
| Imputer | `SimpleImputer` | `strategy="median"` |
| Scaler | `StandardScaler` | `with_mean=True, with_std=True` |
| Признаки | NIRS(15) + HRV(7) + Interaction(3) + z_EMG(9) | итого 34 |
| Session-z EMG | z = (x − μ_subj) / (σ_subj + 1e-8) | per-subject, per-feature |
| CV | LOSO | нет вложенного CV |

**LT1 — HuberRegressor(ε=1.35):**

| Компонент | Параметр | Значение |
|---|---|---|
| Модель | `sklearn.linear_model.HuberRegressor` | |
| | `epsilon` | `1.35` |
| | `alpha` | `0.0001` (sklearn default — L2 регуляризация) |
| | `max_iter` | `500` |
| | `tol` | `1e-05` (sklearn default) |
| | `fit_intercept` | `True` (sklearn default) |
| | `warm_start` | `False` (sklearn default) |
| | `random_state` | не применим (детерминированный) |
| Imputer | `SimpleImputer` | `strategy="median"` |
| Scaler | `StandardScaler` | `with_mean=True, with_std=True` |
| Признаки | z_EMG(9) + HRV(7) | итого 16 |
| Session-z EMG | z = (x − μ_subj) / (σ_subj + 1e-8) | per-subject, per-feature |
| CV | LOSO | нет вложенного CV |

#### Полное сравнение типов моделей (одинаковые признаки, LOSO)

| Модель | Гиперпараметры (ключевые) | LT2 MAE, мин | LT1 MAE, мин | Детерм.? |
|---|---|---|---|---|
| ElasticNet | α=0.5, l1_ratio=0.9, max_iter=5000 | 2.110 | 2.030 | нет¹ |
| **Ridge** | **α=100** | **2.097** | 2.060 | да |
| Lasso | α=0.1, max_iter=5000 | 2.183 | 2.107 | нет¹ |
| **HuberRegressor** | **ε=1.35**, max_iter=500 | 2.124 | **2.021** | да |
| SVR | kernel=RBF, C=1.0, ε=0.1 | 3.1+ | 2.8+ | да |
| KNeighbors | n_neighbors=5, metric=euclidean | 3.4+ | 3.1+ | да |
| RandomForest | n_estimators=100, random_state=42 | 2.5+ | 2.7+ | нет² |
| LGBM | n_estimators=100, lr=0.1, random_state=42 | 2.6+ | 3.1+ | нет² |

¹ ElasticNet/Lasso: результаты могут незначительно варьировать без `random_state`.  
² RandomForest, LGBM: `random_state=42` зафиксирован, но LGBM зависит от версии библиотеки.

**Вывод:** линейные модели превосходят нелинейные — недостаточно данных для деревьев при N=9–13.  
Ridge выбран для LT2 (минимальный MAE), Huber — для LT1 (устойчивость к выбросу S003).

#### LT2 — результаты v4 (Ridge α=100)

| Конфигурация признаков | MAE, мин | R² | ρ |
|---|---|---|---|
| Baseline predict-mean | 5.219 | — | — |
| NIRS + HRV | 2.486 | 0.783 | 0.896 |
| NIRS + HRV + interaction | 2.346 | 0.811 | 0.902 |
| **NIRS + HRV + interaction + z_EMG** | **2.097** | **0.846** | **0.919** |
| z_EMG только | 3.509 | 0.572 | 0.793 |
| z_EMG + HRV | 2.201 | 0.840 | 0.929 |

MAE по участникам (лучшая конфигурация):  
S001=1.35, S002=1.91, S003=5.50(!), S004=1.86, S005=1.56, S006=1.23, S007=1.71,  
S009=1.78, S010=1.95, S011=2.62, S012=1.42, S013=2.01, S014=1.01

#### LT2 — топ-признаки (LOSO-усреднённые коэффициенты Ridge)

| Признак | Коэф (μ ± σ) | Направление | Модальность |
|---|---|---|---|
| hrv_mean_rr_ms | +137.9 ± 15.6 | → дальше | HRV |
| trainred_hhb_std | +89.7 ± 12.1 | → дальше | NIRS |
| trainred_thb_std | −85.5 ± 16.3 | → ближе | NIRS |
| feat_smo2_x_rr | +81.3 ± 9.7 | → дальше | Interaction |
| trainred_thb_mean | −74.2 ± 12.9 | → ближе | NIRS |
| trainred_hbdiff_std | +60.1 ± 10.6 | → дальше | NIRS |
| z_vl_prox_load_rms | −52.6 ± 10.3 | → ближе | z_EMG |
| feat_rr_per_watt | +40.6 ± 5.2 | → дальше | Interaction |
| trainred_smo2_mean | −39.7 ± 9.6 | → ближе | NIRS |
| hrv_dfa_alpha1 | +34.9 ± 7.1 | → дальше | HRV |

#### LT1 — результаты v4 (HuberRegressor ε=1.35)

| Конфигурация признаков | MAE, мин | R² | ρ |
|---|---|---|---|
| Baseline predict-mean | 5.210 | — | — |
| NIRS + HRV | 8.025 | −1.425 | 0.554 |
| HRV только | 2.781 | 0.728 | 0.887 |
| Running NIRS + HRV | 2.430 | 0.792 | 0.917 |
| **z_EMG + HRV** | **2.021** | **0.842** | **0.935** |
| z_EMG + HRV + running NIRS | 2.328 | 0.813 | 0.923 |
| z_EMG только | 3.346 | 0.624 | 0.825 |

> NIRS + HRV с Huber: MAE=8.025 — без session-z нормировки NIRS-признаки
> несопоставимы между субъектами (SmO₂ при LT1: от 62% до 75%).

MAE по участникам (лучшая конфигурация):  
S002=2.46, S003=3.43(!), S004=1.12, S006=1.85, S009=2.07, S010=1.98, S011=1.52, S013=1.93, S014=1.03

#### LT1 — топ-признаки (LOSO-усреднённые коэффициенты Huber)

| Признак | Коэф (μ ± σ) | Направление | Модальность |
|---|---|---|---|
| hrv_mean_rr_ms | +280.3 ± 37.0 | → дальше | HRV |
| z_vl_prox_load_rms | −81.4 ± 8.7 | → ближе | z_EMG |
| z_vl_dist_load_mav | −51.4 ± 23.4 | → ближе | z_EMG |
| z_vl_dist_load_rms | −37.2 ± 25.1 | → ближе | z_EMG |
| hrv_dfa_alpha1 | +33.6 ± 11.0 | → дальше | HRV |
| z_vl_prox_load_mdf | −16.9 ± 4.0 | → ближе | z_EMG |

> hrv_sdnn_ms и hrv_sd2_ms имеют огромные коэффициенты с σ≈700 и противоположными знаками —
> мультиколлинеарность (sdnn ≈ sd2). Для интерпретации значимы только признаки с малым σ.

---

### v5: SHAP + Conformal Prediction (2026-05-07) {#v5}

**Скрипт:** `scripts/train_shap_conformal.py`  
**Воспроизведение:** `uv run python scripts/train_shap_conformal.py`

#### Гиперпараметры

| Компонент | Параметр | Значение |
|---|---|---|
| SHAP | `shap.LinearExplainer` | background=train mean |
| | модели | Ridge(α=100) для LT2, Huber(ε=1.35) для LT1 |
| | обучение | на всех данных (не LOSO — для глобальной интерпретации) |
| Conformal | схема | cross-conformal LOSO |
| | nonconformity | |y_true − ŷ| (абсолютная ошибка) |
| | квантиль | ⌈(n_cal+1)(1−α)⌉ / n_cal, min(q, 1.0) |
| | α уровни | 0.05, 0.10, 0.20 |
| | seed | нет (детерминировано моделями) |

#### Conformal: покрытие и ширина интервала

Результаты получены `uv run python scripts/train_shap_conformal.py` (2026-05-07).  
Модели: Ridge(α=100) для LT2, HuberRegressor(ε=1.35, α=0.0001, max_iter=500) для LT1.

**LT2 (n=13 участников):**

| α | Целевое покрытие | Фактическое покрытие | Полная ширина интервала | ±полуширина |
|---|---|---|---|---|
| 0.05 | 95% | 91.6% | 12.19 мин | ±6.10 мин |
| 0.10 | 90% | 89.3% | 9.39 мин | ±4.70 мин |
| 0.20 | 80% | 81.0% | 6.91 мин | ±3.46 мин |

**LT1 (n=9 участников):**

| α | Целевое покрытие | Фактическое покрытие | Полная ширина интервала | ±полуширина |
|---|---|---|---|---|
| 0.05 | 95% | 94.2% | 12.35 мин | ±6.18 мин |
| 0.10 | 90% | 89.7% | 9.04 мин | ±4.52 мин |
| 0.20 | 80% | 82.5% | 6.98 мин | ±3.49 мин |

> LT2, α=0.05: покрытие 91.6% ниже цели 95% — эффект малого n=13 (квантильная оценка
> с поправкой ⌈(n+1)(1-α)⌉/n нестабильна при n<20).
> При n→∞ гарантия выполняется точно.

---

## Ключевые методические идеи

### Session-z нормировка ЭМГ

Проблема: ЭМГ в мкВ — произвольная шкала, зависящая от расположения электрода,
импеданса кожи, анатомии участника. Абсолютные значения между субъектами несопоставимы.

Решение:
```
z_feature = (x − μ_session) / (σ_session + 1e-8)
```
Применяется per-subject, per-feature. Сохраняет форму траектории (как ЭМГ меняется
с нагрузкой), убирает масштаб. **Не применяется к NIRS** (SmO₂ в % — физическая шкала).

### Running NIRS признаки (для LT1)

`smo2_from_running_max = max(SmO₂ до t) − SmO₂(t)` — каузальный признак,
показывающий насколько SmO₂ упал от своего пика в этой сессии.
Решает проблему межсубъектной несопоставимости абсолютного SmO₂.

---

## Воспроизведение

### Предусловия

```bash
# Датасет должен быть собран
ls dataset/merged_features_ml.parquet   # ✓
ls dataset/session_params.parquet       # ✓ (нужен для running NIRS)
ls dataset/lt1_labels.parquet           # ✓ (нужен для LT1 таргетов)
```

### Пересборка датасета (если нужна)

```bash
# Шаг 1: метки LT1
uv run python scripts/extract_lt1_labels.py

# Шаг 2: полный пайплайн (пропускает уже готовые файлы)
uv run python scripts/build_dataset_all.py

# Или с принудительной пересборкой
uv run python scripts/build_dataset_all.py --force
```

### Воспроизведение текущих лучших результатов (v4)

```bash
# Обе модели (LT1 + LT2), Ridge и Huber
uv run python scripts/train_best_models.py

# Только LT2 (Ridge α=100)
uv run python scripts/train_best_models.py --target lt2

# Только LT1 (HuberRegressor ε=1.35)
uv run python scripts/train_best_models.py --target lt1

# Без графиков (быстрее)
uv run python scripts/train_best_models.py --no-plots
```

Результат: `results/best_models/summary.csv` + графики в `results/best_models/lt1/` и `lt2/`.

### Воспроизведение SHAP + Conformal (v5)

```bash
uv run python scripts/train_shap_conformal.py
```

Результат: `results/shap_conformal/lt1/`, `results/shap_conformal/lt2/`.

### Воспроизведение temporal (v2)

```bash
uv run python scripts/train_temporal.py
```

Результат: `results/temporal/summary.csv`.

### Исторические скрипты {#historical}

| Скрипт | Версия | Что делает |
|---|---|---|
| `scripts/train_baseline.py` | — | Baseline: Ridge, RF, LGBM на всех наборах признаков (LT2) |
| `scripts/final_analysis.py` | — | Дельта-признаки + nested LOSO + ансамбль (LT2) |
| `scripts/improve_model.py` | — | Warm-up калибровка + interactions + Huber (LT2) |
| `scripts/train_lt1.py` | — | LOSO ElasticNet на NIRS+HRV (LT1) |
| `scripts/train_lt1_nirs_features.py` | — | Running NIRS признаки для LT1 |
| `scripts/train_best_models.py` | **v4** | Канонический — Ridge(LT2) + Huber(LT1) |
| `scripts/train_temporal.py` | **v2** | Лаги + Kalman |
| `scripts/train_shap_conformal.py` | **v5** | SHAP + Conformal |

---

## Структура файлов

```
scripts/
  train_best_models.py          ← v4: Ridge(LT2) + HuberRegressor(LT1)
  train_temporal.py             ← v2: лаги + Kalman-сглаживание
  train_shap_conformal.py       ← v5: SHAP + Conformal Prediction
  extract_lt1_labels.py         ← LT1 метки (D-max)
  build_dataset_all.py          ← полный пайплайн датасета

dataset/
  merged_features_ml.parquet   ← главный датасет (3035 окон × 143 признака)
  session_params.parquet        ← калибровочные параметры (nirs_smo2_baseline)
  lt1_labels.parquet            ← метки LT1 (13 участников, 9 usable)
  subjects.parquet              ← метаданные участников + LT2 метки

results/
  best_models/
    summary.csv                 ← v4: все конфигурации, MAE/R²/ρ
    lt2/                        ← графики LT2
    lt1/                        ← графики LT1
  temporal/
    summary.csv                 ← v2: лаги + Kalman
  shap_conformal/
    lt2/                        ← SHAP importance, beeswarm, conformal trajectories
    lt1/                        ← то же для LT1
```
