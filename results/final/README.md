# results/final/

Финальные артефакты для диссертации. Создаются скриптами из ../scripts/final_*.py.

## Шаг 02 — ранжирование моделей

- **01_ranking_wide.csv** — широкая таблица по всем 138 строкам summary с
  производными метриками LOSO (loso_mae_median/std/iqr, gap, kalman_gain и т.д.).
- **02_topk_by_target_fset.md** — главные топы: внутри каждой `(target, feature_set)`
  10 лучших по 6 критериям. Это «яблоки с яблоками».
- **03_topk_by_target_version.md** — внутри `(target, version)`: какая модальность
  лучше для каждой архитектуры.
- **04_topk_by_target_fset_family.md** — внутри `(target, feature_set, family)`:
  какая архитектура лучше при фиксированной модальности.
- **05_candidates_intersection.csv** — модели, попавшие в Top-10 одновременно по
  ≥3 из 6 критериев в группе (target, feature_set). Это шорт-лист для шага 04.

### Критерии Top-K
- `raw_mae_min` ↑   — MAE по всем окнам (взвешена, техническая)
- `kalman_mae_min` ↑ — MAE после калмановского сглаживания
- `loso_mae_median` ↑ — медиана MAE по субъектам LOSO (главная)
- `loso_mae_std` ↑   — стабильность по субъектам
- `rho` ↓           — Spearman, ранговое качество
- `r2` ↓            — R²

### Графики
- plots/stability_vs_mae.png — scatter точность vs стабильность, кандидаты подписаны.
- plots/kalman_gain_by_version.png — средний выигрыш от Калмана по версиям.

## Важные оговорки
- В v0011 per_subject имеет колонку `model` (зоопарк sklearn), variant не различается.
  Для агрегатов LOSO выбирается лучший внутренний model по медиане MAE на (target, fset);
  его имя в колонке `inner_model`.
- per_subject у NN-версий неполный: v0101=18, v0103–v0107=17, v0102/v0105=16 субъектов.
  Колонка `loso_n_subjects` — фактическое число.
