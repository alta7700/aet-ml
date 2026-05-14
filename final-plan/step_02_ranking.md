# Шаг 01. Единый ranker и Top-K кандидатов

## Цель
Из `summary_all_versions.csv` + всех `per_subject.csv` собрать одну широкую таблицу с производными метриками и построить Top-K по нескольким критериям. Пересечение топов = список кандидатов в «лучшие модели».

## Входы
- `results/summary_all_versions.csv`
- `results/<version>/per_subject.csv` для всех 9 версий
- Маппинг семейств архитектур (зафиксирован ниже)

## Семьи архитектур (для группировки C)
- `linear` = `v0011` (sklearn-зоопарк, baseline)
- `lstm` = `v0101`, `v0104` (Attention-LSTM)
- `tcn` = `v0102`, `v0105` (Monotonic TCN)
- `wavelet` = `v0103`, `v0106a`, `v0106b`, `v0106c`
- `ensemble` = `v0107`

> Если эта группировка неточна — поправить в скрипте `final_build_ranking.py`, переменная `FAMILY_MAP`.

## Производные метрики (на каждую строку `version × variant × target × feature_set`)
- `loso_mae_mean`, `loso_mae_median`, `loso_mae_std`, `loso_mae_iqr`
- `loso_r2_median`
- `loso_neg_r2_share` — доля субъектов с R²<0
- `loso_n_subjects` — сколько реально записано (16/17/18)
- `gap_mae = loso_mae_mean − raw_mae_min`
- `kalman_gain = raw_mae_min − kalman_mae_min`

Для v0011 (зоопарк): для каждой `(target, fset, variant)` берётся внутренняя модель с минимальным `loso_mae_median`; её имя сохраняется в колонку `inner_model`.

## Top-K (K=10)
По каждому из 6 критериев (внутри группы):
1. `raw_mae_min` ↑
2. `kalman_mae_min` ↑
3. `loso_mae_median` ↑
4. `loso_mae_std` ↑ (стабильность)
5. `rho` ↓
6. `r2` ↓

### Три уровня группировки
- **A. По `(target, feature_set)`** — главный топ: яблоки с яблоками.
- **B. По `(target, version)`** — какая модальность лучше для каждой архитектуры.
- **C. По `(target, feature_set, family)`** — какая архитектура лучше при фиксированной модальности.

## Пересечения
Внутри группы A: модель попала в Top-10 по ≥3 из 6 критериев → кандидат. Лента кандидатов сохраняется в `results/final/05_candidates_intersection.csv`.

## Артефакты
- `scripts/final_build_ranking.py`
- `results/final/01_ranking_wide.csv`
- `results/final/02_topk_by_target_fset.md`
- `results/final/03_topk_by_target_version.md`
- `results/final/04_topk_by_target_fset_family.md`
- `results/final/05_candidates_intersection.csv`
- `results/final/plots/stability_vs_mae.png` — scatter `loso_mae_median` × `loso_mae_std`, кандидаты подписаны
- `results/final/plots/kalman_gain_by_version.png`
- Подпись каждого файла — в `results/final/README.md`

## Критерий завершённости
- Скрипт детерминирован, перезапуск даёт идентичный CSV.
- Хотя бы 3–5 моделей попали в пересечение Top-K по ≥3 критериям в группе A.
- В `README.md` для `results/final/` есть короткие описания каждого артефакта.

---

## Отчёт о выполнении

- **Дата**: 2026-05-14
- **Скрипт**: `scripts/final_build_ranking.py`
- **Артефакты**:
  - `results/final/01_ranking_wide.csv` — 164 строки × 25 колонок
  - `results/final/02_topk_by_target_fset.md` — главный топ
  - `results/final/03_topk_by_target_version.md`
  - `results/final/04_topk_by_target_fset_family.md`
  - `results/final/05_candidates_intersection.csv` — 87 строк
  - `results/final/plots/stability_vs_mae.png`, `kalman_gain_by_version.png`
  - `results/final/README.md`

- **Технические правки**:
  - v0107 имеет нестандартную схему `per_subject.csv` (ансамбль с 3 экспертами: `mae_ens_min`, `mae_tcn_min`, `mae_lin_min`). Для рейтинга используется `mae_ens_min`, `inner_model='ensemble'`, `r2=NaN`.
  - v0011: variant в per_subject не различается, агрегаты на `(target, fset)` привязываются к обоим summary-строкам.
  - В `summary_all_versions.csv` 164 строки (а не 138, как казалось из ранней проверки — файл был перегенерирован).

- **Распределение n_hits** (модель попала в Top-10 по N критериям из 6):
  - 3 топа: 18 моделей; 4 топа: 39; 5 топов: 23; **6 топов: 7**.

- **Победители n_hits=6** (попали во ВСЕ 6 критериев Top-10, все `target=lt2`):
  - **v0011 HRV Ridge(α=1000)** — `loso_mae_median = 1.86 мин`, ρ=0.91, R²=0.82, n=18 ⭐
  - **v0011 EMG+NIRS+HRV EN(α=1.0, l1=0.9)** — `loso_mae_median = 1.89 мин`, ρ=0.91, R²=0.83, n=18 ⭐
  - v0106b NIRS (wavelet, обе варианты), v0106b EMG+NIRS with_abs — `loso_mae_median ≈ 3.8–4.6 мин`

- **Главное наблюдение**:
  Для **lt2** линейная модель v0011 с признаками HRV (DFA-α1, RMSSD и проч.) даёт `MAE = 1.86 мин` — **в 1.3 раза лучше** лучшего ансамбля NN (v0107 EMG+NIRS+HRV with_abs: 2.38 мин). При n=18 это ожидаемо: HRV-признаки сильно скоррелированы с приближением LT2, а линейная регрессия с регуляризацией не оставляет NN места.

  Для **lt1** ни одна модель не попала в n_hits ≥ 5: критерии «MAE / стабильность / ρ / R²» дают разных лидеров. Текущие модели для lt1 не доминируют — нужно отдельно решать, по какому критерию ранжировать.

- **Семейства-победители (число строк с n_hits ≥ 5)**:
  | Семейство  | Hits |
  |------------|------|
  | linear (v0011) | 10 |
  | wavelet (v0106b/a/c) | 9 |
  | ensemble (v0107) | 5 |
  | tcn (v0105) | 5 |
  | lstm (v0104) | 1 |

- **Замечания для следующих шагов**:
  - Шорт-лист 87 мягкий (min_hits=3 из 6). Для шага 04 (Wilcoxon) — фильтровать по `n_hits ≥ 5` (30 моделей).
  - В lt2 явный фаворит — v0011 HRV. Это нужно перепроверить на отсутствие утечки: HRV-фичи каузальны (trailing 120 с), но при сильной корреляции возможна подсказка через DFA-α1.
  - Для lt1 — нужен отдельный анализ. Возможный путь: выбрать «главную» метрику (ρ или loso_mae_median) и фильтровать только по ней.

- **Следующий шаг**: step_03 — ablation `with_abs vs noabs`.

---

## Доп-итерация: Top-5 по более узким группам (по запросу)

**Запрос**: построить Top-5 (а не 10) отдельно для:
- 4 групп `(target, variant)`: `lt1_with_abs`, `lt1_noabs`, `lt2_with_abs`, `lt2_noabs`.
- Модальностей внутри каждого target (без привязки к variant — варианты остаются видимыми в одной таблице).

**Скрипт**: `scripts/final_topk_views.py` (читает готовый `01_ranking_wide.csv`).

**Артефакты**:
- `results/final/06_top5_by_target_variant.md`
- `results/final/07_top5_by_target_modality.md`

Аналитика — ниже в этом отчёте по факту запуска.

### Результаты доп-итерации

- **Скрипт**: `scripts/final_topk_views.py`
- **Артефакты**: `06_top5_by_target_variant.md`, `07_top5_by_target_modality.md`

**Главные строки Top-5 по `loso_mae_median`** (только lt2 — см. оговорку ниже):

| target/variant | 1 место | 2 место | 3 место |
|----------------|---------|---------|---------|
| lt2 / with_abs | v0011 HRV Ridge(α=1000) — **1.86 мин**, ρ=0.91, R²=0.82 | v0011 EMG+NIRS+HRV EN — **1.89 мин**, ρ=0.91, R²=0.83 | v0107 EMG+NIRS+HRV ensemble — 2.38 мин, ρ=0.85, R²=0.57 |
| lt2 / noabs    | v0011 HRV Ridge(α=1000) — 1.86 мин, ρ=0.85, R²=0.68   | v0011 EMG+NIRS+HRV EN — 1.89 мин, ρ=0.88, R²=0.58   | v0107 EMG+NIRS+HRV ensemble — 2.52 мин, ρ=0.87, R²=0.37 |

**По модальностям (best per fset, lt2)**:

| feature_set | best model | loso_mae_median | ρ |
|-------------|-----------|------------------|---|
| EMG          | v0011 SVR(C=10, ε=0.1)        | 2.67 мин | 0.67 |
| NIRS         | v0011 GBM(n=50, d=2)          | 3.19 мин | 0.72 (noabs) / 0.62 (with_abs) |
| HRV          | v0011 Ridge(α=1000)           | **1.86 мин** | 0.91 |
| EMG+NIRS     | v0011 SVR(C=10, ε=1.0)        | 2.88 мин | 0.73 / 0.70 |
| EMG+NIRS+HRV | v0011 EN(α=1.0, l1=0.9)       | **1.89 мин** | 0.91 / 0.88 |

**КРИТИЧЕСКОЕ ОТКРЫТИЕ — пропуск данных для lt1**:

При проверке выяснилось, что **во всех 10 версиях `per_subject.csv` содержит только `target=lt2`**. Для `lt1` LOSO-метрики по субъектам отсутствуют. Это значит:
- Для `lt1` мы можем строить топы только по 4 критериям из 6 (`raw_mae_min`, `kalman_mae_min`, `rho`, `r2`) — `loso_mae_median` и `loso_mae_std` недоступны.
- Шаг 04 (paired Wilcoxon) и шаг 05 (два эксперта) для `lt1` **невозможны без переобучения**.
- Утверждение шага 02 «для lt1 нет лидера в n_hits≥5» **некорректно**: лидера действительно нет, но потому что 4 из 6 критериев недоступны, а не потому, что модели расходятся в оценках.

В файлах `06_*.md` и `07_*.md` строки lt1 в топах по loso-метрикам — это первые 5 строк с NaN (pandas сортирует NaN в конец, и если все NaN — порядок не меняется). Корректно для lt1 интерпретировать только топы по `raw_mae_min`, `kalman_mae_min`, `rho`, `r2`.

**Что делать**: решение пользователя (запрошено отдельно).

- **Решение и реализация (по результату обсуждения)**:
  Замечено, что в `results/<version>/{ypred,ytrue}_lt1_*.npy` предсказания для lt1 **есть** для всех 10 версий — просто их не агрегировали в per_subject. Порядок в npy: `sorted(subject_id)` × `sort_values(window_start_sec)` после `prepare_data + dropna(target_col)` (одинаков для всех скриптов, см. `scripts/v0011_modality_ablation.py:loso_predict` и аналог в v0101).

  Написан скрипт **`scripts/final_rebuild_per_subject_lt1.py`**, который:
  1. Воспроизводит `prepare_data` (импорт из v0011_modality_ablation).
  2. Восстанавливает границы субъектов в порядке npy.
  3. Считает MAE_min и R² по каждому субъекту → пишет `results/<version>/per_subject_full.csv`.
  4. Для NN-версий: lt1 строки берутся из npy, lt2 — из существующего `per_subject.csv` (сливаются).
  5. Для v0011: npy в корне = best внутренняя модель зоопарка на (target, fset); `variant=with_abs` (для noabs npy не сохранён — variant=with_abs привязывается к обеим строкам summary, как и в шаге 02).

  **Результаты восстановления**:
  | Версия | lt1 rows | lt2 rows |
  |--------|----------|----------|
  | v0101 | 144 ✓ | 144 |
  | v0102 | 144 ✓ | 128 |
  | v0103 | 144 ✓ | 136 |
  | v0104 | 144 ✓ | 136 |
  | v0105 | 144 ✓ | 128 |
  | v0106a/b/c | 144 ✓ | 136 |
  | v0107 | 144 ✓ | 136 |
  | v0011 | 90 ✓ | 90 |

  Покрытие после пересборки: **lt1 = 82/82, lt2 = 82/82**.

  Скрипты `final_build_ranking.py` и `final_topk_views.py` обновлены: приоритетно читают `per_subject_full.csv`. Артефакты в `results/final/` перезаписаны.

  Для lt2 в npy у NN модели число окон 3553/3634 vs ожидаемое 3651 (17–98 окон расходится из-за внутренних dropna). Это **не влияет** на анализ, так как для lt2 используются исходные per_subject.csv (где субъекты записаны напрямую). Документировано как известный артефакт.

### Победители lt1 (после восстановления)

**Top-5 lt1 / with_abs по `loso_mae_median`**:

| версия / fset       | MAE мин | ρ    | R²   |
|---------------------|---------|------|------|
| v0011 / EMG+NIRS         | **2.11** | 0.74 | 0.52 |
| v0011 / EMG+NIRS+HRV     | **2.23** | 0.79 | 0.57 |
| v0011 / EMG              | 2.94    | 0.71 | 0.44 |
| v0011 / HRV              | 3.51    | 0.65 | 0.38 |
| v0011 / NIRS             | 3.62    | 0.72 | 0.37 |

**Лучший NN для lt1**: v0107 ensemble EMG+NIRS+HRV (noabs) — `loso_mae_median = 2.71 мин`, ρ=0.78.

**Главное наблюдение по разнице lt1 vs lt2**:
- Для **lt1** ведущая модальность — **EMG+NIRS** (2.11 мин). HRV отдельно даёт лишь 3.5 мин.
- Для **lt2** ведущая модальность — **HRV** (1.86 мин). EMG+NIRS — 2.88 мин.

Физиологически правдоподобно: до LT1 (аэробный порог) лактат ещё не накопился, и автономный тонус (HRV) меняется слабо, а вот ЭМГ и оксигенация мышц уже сдвигаются. До LT2 (анаэробный порог) HRV резко смещается из-за подключения симпатической ветви → лучший предиктор именно у HRV.

- **Следующий шаг**: step_03 — ablation `with_abs vs noabs`.



