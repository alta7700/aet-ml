# Доработка раздела «Результаты» ML-части диссертации

## Контекст

В ходе ревизии текстов 07–12 в `disser-ml-text/` обнаружено три проблемы:

1. **Ложное утверждение об отсутствии артефактов.** В [text_07:93](disser-ml-text/text_07_rez_ranzhirovanie.md) заявлено, что для версии v0011 в варианте `noabs` «артефакты предсказаний не были сохранены». Проверка показала: в `results/v0011/noabs/` есть полные `ypred_*.npy` и `ytrue_*.npy` для всех 5 модальностей × 2 таргета, длины массивов идентичны `with_abs`. Из-за ошибочного утверждения из анализа выпало 8 групп `target × feature_set` категории noabs (попарные сравнения и линейный ablation), что ослабило доказательную базу подраздела 3.3 и сделало «одноточечным» подраздел 3.4.2.

2. **SHAP победителей LT1 был на «не той» модели.** В [text_10](disser-ml-text/text_10_rez_interpretaciya_nadezhnost.md) таблицы 3.15–3.16 построены на Ridge α=1000 (LinearExplainer), хотя фактический победитель LT1 — SVR с RBF-ядром. Это явное методологическое ограничение, зафиксированное в [text_12:53](disser-ml-text/text_12_zaklyuchenie_ml.md) (4.4-7) и в направлениях дальнейших исследований (4.5-5). Пользователь 2026-05-15 добавил `results/final/shap/lt1_EN_svr/` и `lt1_ENH_svr/` через **PermutationExplainer на SVR RBF** — ограничение теперь снимается.

3. **Отсутствуют графики, описанные в тексте.** Рисунки 3.6, 3.7, 3.8 присутствуют как ссылки в [text_09:26-30](disser-ml-text/text_09_rez_dva_eksperta.md) и [text_09:53](disser-ml-text/text_09_rez_dva_eksperta.md), но PNG не созданы в `results/final/eda/`. Скрипты `final_per_subject_dist.py` и `final_time_resolved.py` существуют.

**Цель доработки** — закрыть все три пробела, привести таблицы и рисунки в соответствие с реально доступными данными, обновить выводы в обсуждении и заключении.

**Принципиальные решения по итогам уточнения:**
- Объём — **полный**: метод + графики 3.6–3.8 + новые иллюстрации.
- Conformal — расширить на noabs-победителей (Таблица 3.17).
- SVR-SHAP — **заменить** Ridge-проекцию в основном тексте; Ridge-вариант уходит в Приложение А (отдельный файл `text_A2_shap_ridge.md`).
- Новые графики: calibration scatter ypred×ytrue, abs vs noabs per-subject scatter, modality×MAE bar для noabs. Session trajectories — отменить.

---

## Блок 1. Закрыть пробел v0011/noabs

### 1.1 Восстановить per-subject MAE для v0011/noabs

- **Скрипт:** [scripts/final_per_subject_from_npy.py](scripts/final_per_subject_from_npy.py) — уже умеет восстанавливать per-subject из ypred/ytrue + sort by (subject_id, window_start_sec); сейчас не включает v0011/noabs.
- **Действия:** добавить v0011/noabs в список конфигураций (5 модальностей × 2 таргета = 10 строк × 18 subj).
- **Артефакты:**
  - `results/v0011/noabs/per_subject.csv` (новый, ~180 строк)
  - дополнить `results/v0011/noabs/summary.csv` (сейчас только LT1) строками LT2 — пересчётом window-weighted raw MAE / R² / ρ из `ypred_*`/`ytrue_*`
  - обновить `results/final/per_subject_all.csv` (добавить v0011/noabs)
- **Зависимости:** требуется привязка окно→subject_id. В `scripts/v0011_modality_ablation.py:246` `prepare_data()` уже её восстанавливает; функцию следует использовать как источник subject-id-вектора длины N_windows.

### 1.2 Paired Wilcoxon v0011: with_abs vs noabs (10 пар)

- **Скрипт:** [scripts/final_abs_vs_noabs.py](scripts/final_abs_vs_noabs.py) — сейчас покрывает только нейросетевые v0101–v0107.
- **Действия:** расширить на v0011 (победитель в каждой комбинации `feature_set × target`), применить операциональную границу эквивалентности из раздела 2.11.3 (Δ < 0,3 мин И p ≥ 0,05).
- **Артефакты:**
  - `results/final/11_abs_vs_noabs.csv` (+10 строк для v0011)
  - новая таблица 3.7-bis: «Ablation линейного семейства, paired Wilcoxon per-subject» — на 10 пар вместо одной точки
  - volcano-companion: `results/final/plots/abs_vs_noabs_volcano_v0011.png`

### 1.3 Полные попарные Уилкоксоны в категории noabs

- **Скрипт:** [scripts/final_pairwise_wilcoxon.py](scripts/final_pairwise_wilcoxon.py) — сейчас уже считает попарные сравнения для всех версий, но v0011 noabs выпадает из-за отсутствия per_subject (закрывается шагом 1.1).
- **Действия:** перезапустить после шага 1.1; пересчитать таблицу win-counts.
- **Артефакты:**
  - `results/final/12_pairwise_wilcoxon.csv` (+ ~ 96 новых пар: 10 версий × 9 версий × 8 групп noabs, в которых теперь участвует v0011)
  - `results/final/13_pairwise_win_counts.csv` — расширенная сводка
  - Таблица 3.4: дополнить noabs-аналогом (8 групп target × feature_set)
  - Тепловые карты для noabs: `plots/pairwise/pairwise_lt2_HRV_noabs.png` и т.п. — две новые в основной текст (для победителей категории noabs)

### 1.4 Кросс-модальные сравнения чемпионов в noabs

- **Скрипт:** новая логика в `final_analysis.py` (или отдельный `final_champion_cross_modality.py`, если есть).
- **Действия:** для каждого таргета выделить лучшую модель в noabs-категории по каждой модальности (линейное семейство + нейросетевые), провести парные Уилкоксоны HRV vs EMG+NIRS, HRV vs EMG+NIRS+HRV, EMG+NIRS vs EMG+NIRS+HRV — по аналогии с Таблицей 3.3.
- **Артефакт:** `results/final/14_champion_cross_modality.csv` (+ noabs-строки), новая Таблица 3.3-bis или расширение Таблицы 3.3.

### 1.5 Modality × MAE bar для noabs

- **Скрипт:** [scripts/final_plot_modality_bar.py](scripts/final_plot_modality_bar.py) — взять как шаблон, дублировать для варианта noabs.
- **Артефакт:** `results/final/plots/modality_vs_mae_noabs.png` — добавить как Рис. 3.2-bis или как вторую панель Рис. 3.2.

### 1.6 Conformal split-LOSO для noabs-победителей

- **Скрипт:** [scripts/final_shap_conformal.py](scripts/final_shap_conformal.py) — содержит conformal-логику; сейчас построен на 4 with_abs-финалистах.
- **Действия:** добавить 4 noabs-финалиста (по победителю каждого `target × {HRV, EMG+NIRS, EMG+NIRS+HRV, EMG+NIRS}` из Таблиц 3.1/3.2 noabs); посчитать coverage и полуширину при α ∈ {0,1; 0,2}, разбивку trained/untrained.
- **Артефакты:**
  - `results/final/conformal/coverage_summary.csv` (+ 8 строк)
  - Таблица 3.17 — расширенная (16 строк вместо 8)
  - Рисунок 3.13 — пересоздать с noabs-точками

---

## Блок 2. Интегрировать новые SVR-SHAP

### 2.1 Перенос SVR-SHAP в основной текст 3.7

- **Источники:** `results/final/shap/lt1_EN_svr/shap_importance.csv` (топ-N по mean |SHAP|), `results/final/shap/lt1_EN_svr/shap_by_group.csv`, аналогичные пары для `lt1_ENH_svr/`. Beeswarm и bar PNG уже сгенерированы.
- **Действия в [text_10_rez_interpretaciya_nadezhnost.md](disser-ml-text/text_10_rez_interpretaciya_nadezhnost.md):**
  - **3.7.3 LT1 / EMG+NIRS:** Таблицы 3.15 и 3.16 — заменить значения mean |SHAP| и доли группы на SVR-PermutationExplainer. Дополнительно добавить колонку «согласие с Ridge-проекцией» (топ-10 пересечение).
  - **Конец 3.7.3:** мультимодальный LT1 — обновить процентовку EMG/NIRS/HRV/взаимодействий по `lt1_ENH_svr/shap_by_group.csv`.
  - **Преамбула 3.7 (line 3):** переписать обоснование выбора модели интерпретации — для LT2-конфигураций остаётся Ridge LinearExplainer, для LT1 — SVR PermutationExplainer.
  - **Рисунки 3.11, 3.12** — обновить ссылки на PNG из новых `lt1_*_svr` подпапок.

### 2.2 Снять ограничение 4.4-7 и переформулировать 4.5-5 в text_12

- **Действия в [text_12_zaklyuchenie_ml.md](disser-ml-text/text_12_zaklyuchenie_ml.md):**
  - Удалить пункт 4.4-7 целиком (ограничение снято).
  - Перенумеровать оставшиеся пункты 4.4.
  - Удалить или переформулировать направление 4.5-5: вместо «пересчитать SHAP на SVR» — «расширение интерпретации на нейросетевые архитектуры (DeepExplainer / Integrated Gradients)».

### 2.3 Обновить раздел 3.9.5 в text_11

- **Действия в [text_11_obsuzhdenie.md](disser-ml-text/text_11_obsuzhdenie.md):**
  - В подразделе 3.9.5 («Различная структура SHAP-вкладов»): убрать пометки про «линейную проекцию SVR через Ridge», обновить численные доли по SVR-SHAP.
  - В подразделе 3.9.1: подтвердить диссоциацию модальностей теперь уже на SVR-SHAP (если численная картина согласуется).

### 2.4 Создать Приложение А-2 с Ridge-вариантом

- **Новый файл:** [disser-ml-text/text_A2_shap_ridge_lt1.md](disser-ml-text/text_A2_shap_ridge_lt1.md) — перенос текущих Таблиц 3.15/3.16 на Ridge с пометкой «линейная проекция, для согласованности с LT2». Краткое сопоставление с SVR (топ-10 пересечение).
- В text_10 — одна ссылка: «Сравнение с линейной Ridge-проекцией — см. Приложение А-2».

---

## Блок 3. Создать недостающие и новые графики

### 3.1 Рисунки 3.6 и 3.7 (per-subject распределения)

- **Скрипт:** [scripts/final_per_subject_dist.py](scripts/final_per_subject_dist.py) — судя по существующему `results/final/eda/per_subject_dist/09_trained_vs_untrained.csv`, частично исполнен.
- **Действия:** запустить визуализационную часть, обеспечить сохранение в указанные в тексте пути:
  - `results/final/eda/per_subject_dist/plots/hist_mae_by_model.png` (Рис. 3.6)
  - `results/final/eda/per_subject_dist/plots/mae_by_trained_group.png` (Рис. 3.7)

### 3.2 Рисунок 3.8 (error by time bins)

- **Скрипт:** [scripts/final_time_resolved.py](scripts/final_time_resolved.py).
- **Действия:** запустить, обеспечить сохранение:
  - `results/final/eda/time_resolved/plots/error_vs_time_to_threshold_lt1.png`
  - `results/final/eda/time_resolved/plots/error_vs_time_to_threshold_lt2.png`

### 3.3 Новые графики (без описания в текущем тексте → добавить упоминания)

**3.3.1 Calibration scatter ypred×ytrue для финалистов:**
- Новый скрипт `scripts/final_plot_calibration.py`.
- 4 финалиста with_abs + 4 noabs, в panel layout 2×4; диагональ y=x, фон — плотность точек.
- Артефакт: `results/final/plots/calibration_finalists.png` (Рис. 3.14 в text_10, после Таблицы 3.17).

**3.3.2 abs vs noabs per-subject scatter:**
- Новый скрипт `scripts/final_plot_abs_noabs_per_subject.py`.
- 5 модальностей × 2 таргета = 10 панелей; точки — 18 испытуемых, ось X — MAE_with_abs, ось Y — MAE_noabs, диагональ y=x.
- Артефакт: `results/final/plots/abs_vs_noabs_per_subject.png` (Рис. 3.5-bis в text_08, после Таблицы 3.6).

**3.3.3 Modality × MAE bar для noabs** — см. 1.5.

---

## Блок 4. Синхронизация текстов

### 4.1 [text_07_rez_ranzhirovanie.md](disser-ml-text/text_07_rez_ranzhirovanie.md)
- **Строка 93** — удалить утверждение «артефакты не были сохранены»; вместо него — «полные результаты попарных сравнений для категории noabs приведены в таблице 3.4 ниже».
- **Таблица 3.4** — добавить 8 строк с noabs-категорией.
- **После Таблицы 3.4** — переформулировать обобщающий абзац: «39+N значимых побед на M+8 групп».
- **Рисунки 3.3–3.4** — оставить with_abs; добавить ссылки на 2 новых noabs-тепловых карты (Рис. 3.3-bis, 3.4-bis).

### 4.2 [text_08_rez_ablation.md](disser-ml-text/text_08_rez_ablation.md)
- **Раздел 3.4.2** — переписать: заменить «единственная точка HRV/LT2» на полную Таблицу 3.7-bis с paired Wilcoxon per-subject для 10 пар линейного семейства; обновить вердикт 3.4.3.
- **Добавить Рис. 3.5-bis** (abs vs noabs per-subject scatter).

### 4.3 [text_09_rez_dva_eksperta.md](disser-ml-text/text_09_rez_dva_eksperta.md)
- **Строка 62** — исправить смысловую нестыковку: «победитель LT1» → «базовая модель для двух специалистов (Ridge α=1000 на EMG+NIRS, раздел 2.8.9)».
- Подтвердить, что Рис. 3.6/3.7/3.8 теперь существуют по указанным путям (после Блока 3.1–3.2).

### 4.4 [text_10_rez_interpretaciya_nadezhnost.md](disser-ml-text/text_10_rez_interpretaciya_nadezhnost.md)
- **Раздел 3.7** — переработка по Блоку 2.1.
- **Таблица 3.17** — расширить на 8 строк noabs-финалистов.
- **Рис. 3.13** — пересоздать со всеми 8 моделями.
- **Добавить Рис. 3.14** (calibration scatter).

### 4.5 [text_11_obsuzhdenie.md](disser-ml-text/text_11_obsuzhdenie.md)
- **3.9.2** — обновить формулировку доминирования линейных: «39 значимых побед в 6 из 8 групп with_abs и N значимых побед в M из 8 групп noabs».
- **3.9.3** — добавить, что ablation линейного семейства теперь имеет полную статистическую сводку (10 пар), а не одну точку.
- **3.9.5** — обновить по Блоку 2.3.

### 4.6 [text_12_zaklyuchenie_ml.md](disser-ml-text/text_12_zaklyuchenie_ml.md)
- **4.1.2** — обновить «39 значимых побед, 0 поражений в 6 из 8 групп with_abs»; добавить noabs-составляющую.
- **4.1.3** — переписать SHAP-абзац под SVR-результаты.
- **4.4** — удалить пункт 7; перенумеровать.
- **4.5** — удалить пункт 5; перенумеровать; на освободившееся место — новая задача «расширение интерпретации на нейросети».

### 4.7 Новый файл-приложение
- [disser-ml-text/text_A2_shap_ridge_lt1.md](disser-ml-text/text_A2_shap_ridge_lt1.md) — Ridge-вариант SHAP LT1 (Блок 2.4).

---

## Критические файлы для модификации

**Скрипты (запуск/расширение):**
- `scripts/final_per_subject_from_npy.py` — расширить на v0011/noabs
- `scripts/final_abs_vs_noabs.py` — расширить на v0011 (10 пар)
- `scripts/final_pairwise_wilcoxon.py` — перезапустить после 1.1
- `scripts/final_shap_conformal.py` — расширить на noabs-финалистов
- `scripts/final_plot_modality_bar.py` — клон под noabs
- `scripts/final_per_subject_dist.py` — запустить визуализационную часть
- `scripts/final_time_resolved.py` — запустить визуализационную часть
- **Новые:** `scripts/final_plot_calibration.py`, `scripts/final_plot_abs_noabs_per_subject.py`

**Таблицы:**
- `results/final/11_abs_vs_noabs.csv` (+10 строк)
- `results/final/12_pairwise_wilcoxon.csv` (+ ~96 пар)
- `results/final/13_pairwise_win_counts.csv` — пересчёт
- `results/final/14_champion_cross_modality.csv` (+ noabs)
- `results/final/conformal/coverage_summary.csv` (+8 строк)
- `results/final/per_subject_all.csv` (+ v0011/noabs)
- `results/final/dissertation_tables/table_main_results.csv` — добавить noabs-аналоги (опционально)
- `results/final/dissertation_tables/table_by_modality.csv` — добавить колонки noabs (опционально)

**Тексты:**
- text_07, text_08, text_09, text_10, text_11, text_12 — правки по Блоку 4
- text_A2_shap_ridge_lt1.md — новый файл

---

## Верификация

1. **Закрытие пробела v0011/noabs:**
   - Проверить: `results/v0011/noabs/per_subject.csv` существует, содержит 5 × 2 × 18 = 180 строк.
   - Проверить: в `results/final/12_pairwise_wilcoxon.csv` есть строки `v0011 × v01XX` для всех 8 noabs-групп.
   - Проверить: в text_07 в Таблице 3.4 есть строки `LT1, EMG, noabs`, `LT2, HRV, noabs` и т.д.

2. **SVR-SHAP в тексте:**
   - Проверить: в Таблице 3.15 — топ-10 признаков совпадают с `results/final/shap/lt1_EN_svr/shap_importance.csv`.
   - Проверить: в text_12 пункт 4.4-7 отсутствует, пункт 4.5-5 отсутствует.
   - Проверить: text_A2_shap_ridge_lt1.md существует и содержит Ridge-вариант.

3. **Графики:**
   - `ls results/final/eda/per_subject_dist/plots/` → есть hist_mae_by_model.png, mae_by_trained_group.png
   - `ls results/final/eda/time_resolved/plots/` → есть error_vs_time_to_threshold_lt{1,2}.png
   - `ls results/final/plots/` → есть calibration_finalists.png, abs_vs_noabs_per_subject.png, modality_vs_mae_noabs.png
   - Все упомянутые в текстах рисунки 3.1–3.14 имеют существующие PNG-файлы.

4. **Сквозная согласованность:**
   - В text_07/text_12 числа «39 значимых побед» / «N в категории noabs» совпадают по всем упоминаниям.
   - В text_10 преамбула раздела 3.7 согласована со снятым ограничением в 4.4.
   - В text_09:62 «базовая модель» вместо «победитель LT1» (правка по итогам предыдущей сессии).

5. **Чистовой прогон скриптов:**
   - `python scripts/final_per_subject_from_npy.py` → no errors, файлы созданы
   - `python scripts/final_abs_vs_noabs.py` → no errors, CSV обновлён
   - `python scripts/final_pairwise_wilcoxon.py` → no errors
   - `python scripts/final_shap_conformal.py --variant noabs` → no errors
   - `python scripts/final_plot_modality_bar.py --variant noabs` → PNG создан
   - `python scripts/final_per_subject_dist.py` → PNG созданы
   - `python scripts/final_time_resolved.py` → PNG созданы
   - `python scripts/final_plot_calibration.py` → PNG создан
   - `python scripts/final_plot_abs_noabs_per_subject.py` → PNG создан

## Последовательность исполнения

Жёсткий порядок зависимостей:

```
1.1 per_subject noabs ──► 1.2 ablation v0011 ──► 1.3 pairwise ──► 1.4 cross-modality
                     └──► 1.5 bar plot
                     └──► 1.6 conformal noabs ──► 3.3.1 calibration
                                                 └──► 3.3.2 abs vs noabs scatter

2.1 SVR-SHAP в text_10 ──► 2.2 text_12 cleanup ──► 2.3 text_11 update
                     └──► 2.4 Приложение А-2

3.1 Рис. 3.6/3.7  ─┐
3.2 Рис. 3.8       ├─► 4.x синхронизация текстов
3.3.* новые        ─┘
```

Блоки 1, 2, 3 — независимы и могут идти параллельно; блок 4 (тексты) — финальный, после всех артефактов.