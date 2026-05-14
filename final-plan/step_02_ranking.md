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
