# results/final/ — финальные артефакты для диссертации

Собрано скриптами `scripts/final_*.py`. План работ — в `final-plan/`.
Окружение и порядок воспроизведения — в `ENV.md`.

## Краткие выводы (для раздела «Результаты»)

1. **Минимальный достаточный набор сенсоров зависит от порога.**
   LT2 — достаточно ВСР (HRV ≈ EMG+NIRS+HRV, p=0.97). LT1 — достаточно
   ЭМГ+NIRS (≈ EMG+NIRS+HRV, p=0.44). Мультимодальность не даёт значимого
   выигрыша.
2. **Линейная регрессия (v0011) статистически доминирует** над всеми NN
   (paired Wilcoxon + Holm): победитель в 6/8 конфигураций, не уступает нигде.
3. **Лучшие модели**: LT2 — v0011/HRV (loso MAE 1.86 мин, ρ=0.91, R²=0.82);
   LT1 — v0011/EMG+NIRS (loso MAE 2.11 мин, ρ=0.74, R²=0.52).
4. **Абсолютные признаки не нужны для NN** (86% конфигураций эквивалентны),
   но **нужны для линейной модели** (hrv_mean_rr_ms — 44.6% веса).
5. **«Два эксперта»**: симметричная гипотеза отвергнута (trained-спец на n=6
   недообучен), асимметричная подтверждена (спец для нетренированных +
   generalist для тренированных: oracle p=0.034).
6. **Conformal**: LT2 интервалы практичны (±3.5 мин при 80%), LT1 — широки
   (±9 мин при 90%).
7. **Слабость LT1 структурирована**: 79% окон с ошибкой >7 мин — тренированные
   субъекты. Это один и тот же феномен в шагах 02a/02b/05/06.

## dissertation_tables/ — готовые таблицы для текста

| Файл | Содержание |
|------|------------|
| `table_main_results.csv`   | Главная таблица «Результаты» — финалисты обоих таргетов |
| `table_by_modality.csv`    | Лучшая модель на каждой модальности |
| `table_significance.csv`   | Сводка статистических тестов (ablation, cross-modality, pairwise) |
| `table_two_experts.csv`    | Сравнение систем «два эксперта» vs generalist |
| `table_conformal.csv`      | Покрытие и ширина conformal-интервалов |

## Артефакты по шагам

### Шаг 02 — ранжирование
- `01_ranking_wide.csv` — широкая таблица (164×25) с производными LOSO-метриками
- `02_topk_by_target_fset.md`, `03_topk_by_target_version.md`,
  `04_topk_by_target_fset_family.md` — Top-10 по трём уровням группировки
- `05_candidates_intersection.csv` — пересечение топов
- `06_top5_by_target_variant.md`, `07_top5_by_target_modality.md` — узкие Top-5
- `per_subject_all.csv` — длинная сводка per-subject (5212 строк) + сверка со старыми
- `per_subject_rebuild_report.csv` — статусы пересборки per_subject из npy
- `plots/stability_vs_mae.png`, `plots/kalman_gain_by_version.png`

### Шаг 02a — распределение per-subject ошибок
- `eda/per_subject_dist/08_per_subject_table.csv`
- `eda/per_subject_dist/09_trained_vs_untrained.csv` — Mann-Whitney trained vs untrained
- `eda/per_subject_dist/plots/` — гистограммы, scatter с ковариатами

### Шаг 02b — анализ ошибки по времени до порога
- `eda/time_resolved/10_error_by_time_bins.csv`
- `eda/time_resolved/plots/` — кривые ошибки по бинам

### Шаг 03 — ablation with_abs vs noabs
- `11_abs_vs_noabs.csv` — paired Wilcoxon, 72 конфигурации
- `11b_v0011_hrv_coefs.csv` — коэффициенты линейной модели (абсолют vs тренд)
- `plots/abs_vs_noabs_volcano.png`

### Шаг 04 — парные сравнения
- `12_pairwise_wilcoxon.csv` — все пары версий, Wilcoxon + Holm
- `13_pairwise_win_counts.csv` — счёт значимых побед/поражений
- `14_champion_cross_modality.csv` — кросс-модальные сравнения чемпионов
- `plots/pairwise/` — heatmap p_holm по группам

### Шаг 05 — два эксперта (LT1)
- `two_experts/subject_meta.csv` — разметка trained/untrained + ковариаты
- `two_experts/15_per_subject_predictions.csv` — предсказания всех систем
- `two_experts/16_system_comparison.csv` — сравнение систем + Wilcoxon
- `two_experts/17_router_metrics.csv` — важность признаков маршрутизатора
- `two_experts/plots/systems_comparison.png`

### Шаг 06 — SHAP + conformal
- `shap/<tag>/` — beeswarm, bar, shap_importance.csv, shap_by_group.csv
- `conformal/<tag>/intervals.csv` — per-subject интервалы
- `conformal/coverage_summary.csv`, `conformal/conformal_summary.png`

## Важные оговорки
- Все модели обучены LOSO, n=18 (некоторые NN — n=16–17, см. `loso_n_subjects`).
- `loso_mae_median` (по субъектам) систематически ниже window-weighted MAE —
  при цитировании указывать, какая метрика.
- v0011 per_subject не различает variant; для него `noabs` приравнен к `with_abs`.
- npy для lt2 у части NN короче ожидаемого на 17–98 окон (внутренние dropna) —
  это учтено через список субъектов из per_subject.csv.
