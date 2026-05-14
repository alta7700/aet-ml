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

- **Дата**: 2026-05-14
- **Скрипт**: `scripts/final_package.py`
- **Что упаковано**:
  - `results/final/dissertation_tables/` — 5 готовых таблиц для текста (без запятых в полях, готовы к переносу);
  - `results/final/ENV.md` — версии пакетов, детерминизм, порядок воспроизведения всех скриптов шагов 01–07;
  - `results/final/README.md` — перезаписан: краткие выводы (7 пунктов) + полная навигация по артефактам всех шагов + оговорки.

- **Финальный список таблиц** (`results/final/dissertation_tables/`):
  | Файл | Содержание |
  |------|------------|
  | `table_main_results.csv` | 6 финалистов (по 3 на target): победитель + мультимодальный + лучший NN |
  | `table_by_modality.csv` | Лучшая модель на каждой из 5 модальностей × 2 target |
  | `table_significance.csv` | 5 строк: ablation, 3 кросс-модальных сравнения, доминирование v0011 |
  | `table_two_experts.csv` | 5 систем: generalist / oracle_sym / router_sym / oracle_asym / router_asym |
  | `table_conformal.csv` | Покрытие и ширина интервалов, 4 модели × 2 уровня α |

- **Что осталось вне `final/` и почему**:
  - `results/v0001…v0014`, `baseline`, `temporal`, `experiment_grid`, `improve_model` — устаревшие итерации, в диссертацию не идут (решение пользователя: из линейных оставлен только v0011).
  - `results/shap_conformal/` — старый, заменён на `results/final/shap/` и `results/final/conformal/`.
  - Артефакты моделей (npy) остаются в `results/<version>/` — это источник для пересборки, не финальный результат.

- **Проверка критерия завершённости**:
  - `README.md` открывается и за 5 минут даёт картину: 7 кратких выводов сверху, навигация по шагам, оговорки внизу. ✓
  - Все таблицы в `dissertation_tables/` — готовые CSV, ничего не пересчитывается. ✓

### Статус всего плана
Шаги 01–07 выполнены. Все отчёты заполнены. `results/final/` готов к переносу в раздел «Результаты» диссертации.

