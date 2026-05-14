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
_Заполняется по факту выполнения._

- Дата начала:
- Дата завершения:
- Скрипт: 
- Сделано:
- Что не сделано / отклонения от плана:
- Найденные кандидаты:
- Следующий шаг:
