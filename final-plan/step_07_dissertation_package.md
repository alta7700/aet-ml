# Шаг 06. Упаковка результатов для диссертации

## Цель
Собрать `results/final/` так, чтобы из него можно было прямо переносить материалы в раздел «Результаты» текста диссертации без дополнительной обработки.

## Структура `results/final/`
```
results/final/
  README.md                           # навигация: что есть что, для чего
  ENV.md                              # версии пакетов, сиды
  01_ranking_wide.csv
  02_topk_by_target_fset.md
  03_topk_by_target_version.md
  04_topk_by_target_fset_family.md
  05_candidates_intersection.csv
  06_abs_vs_noabs.csv
  07_pairwise_wilcoxon.csv
  two_experts/
    subject_meta.csv
    08_per_group_mae.csv
    09_specialists_loso.csv
    10_router_metrics.csv
    11_system_vs_generalist.csv
  shap/...
  conformal/...
  plots/
    stability_vs_mae.png
    kalman_gain_by_version.png
    abs_vs_noabs_paired.png
    pairwise_heatmap_*.png
    final_leaderboard.png             # итоговая таблица-картинка для слайдов
  dissertation_tables/                # готовые таблицы для .docx
    table_main_results.csv            # главная таблица «Результаты»
    table_ablation_abs.csv
    table_significance.csv
    table_two_experts.csv
```

## Главная таблица «Результаты»
Колонки (для каждой модели-финалиста):
- `version`, `architecture_family`
- `feature_set`, `target`
- `loso_mae_median` ± IQR
- `rho`
- `r2_median`
- `kalman_mae_min` (для сравнения с альтернативной фильтрацией)
- `n_subjects`
- `significance_vs_baseline` — символ из шага 03

## README в `results/final/`
Один файл, в котором по каждому артефакту:
- что это (1 строка);
- из чего собрано (источник);
- как читать (как сортировать, что значат столбцы);
- куда в тексте диссертации идёт.

## Критерий завершённости
- Любой человек (научрук, оппонент) открывает `results/final/README.md` и за 5 минут понимает, что лежит и в каком разделе диссертации применяется.
- Все таблицы для текста — в `dissertation_tables/` в готовом виде, ничего не пересчитывается на лету.

---

## Отчёт о выполнении
_Заполняется по факту._

- Дата:
- Что упаковано:
- Что осталось вне `final/` и почему:
- Финальный список таблиц для текста:
