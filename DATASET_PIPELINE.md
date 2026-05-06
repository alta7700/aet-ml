# Пайплайн подготовки датасета

## Назначение

Этот документ фиксирует **пайплайн подготовки датасета**, а не пайплайн
обучения модели.

На текущем этапе цель такая:

- собрать воспроизводимый датасет из `finaltest.h5`;
- поддержать два типа целевых переменных:
  - бинарная классификация `до / после LT2`;
  - регрессия `time_to_lt2`;
- сохранить все пропуски и все признаки качества данных явно, без скрытой
  импутации на этапе сборки.

Главное ограничение:

- каждое окно должно быть **каузальным**;
- текущее окно может использовать только данные, доступные **до его правой границы**;
- никакие будущие значения в признаки попадать не должны.

## Источник истины

Единый источник для построения датасета:

- `data/<subject>/finaltest.h5`

Из него берутся:

- физиологические каналы;
- метаданные испытуемого;
- `subject_id`;
- метаданные `LT2`;
- quality-поля для `LT2`.

Промежуточные `json` и старые `h5`-файлы не считаются источником истины для
датасета.

## Операционный маппинг ЭМГ-каналов

В сыром `finaltest.h5` имена каналов остаются исходными:

- `trigno.vl.avanti`
- `trigno.rf.avanti`

Но на уровне датасета вводится **операционный маппинг**:

- `trigno.vl.avanti` → `VL_dist`
- `trigno.rf.avanti` → `VL_prox`

То же правило распространяется на гироскопы этих датчиков:

- `trigno.vl.avanti.gyro.*` → `VL_dist_gyro_*`
- `trigno.rf.avanti.gyro.*` → `VL_prox_gyro_*`

Важно:

- это **алиасы уровня датасета**, а не переименование каналов в `HDF5`;
- исходный файл не переписывается;
- provenance остаётся прозрачным.

## Базовая единица датасета

Одна строка датасета = одно каузальное окно.

Базовые параметры окна:

- длина окна: `30 с`
- шаг окна: `5 с`

Для всех модальностей базовое окно одно и то же, кроме `HRV`:

- `EMG`, `IMU`, `NIRS`: признаки по самому окну `30 с`
- `HRV`: признаки по trailing-истории длиной `120 с`, заканчивающейся в правой
  границе текущего окна

То есть:

- `window_start_sec`
- `window_end_sec`
- `window_center_sec`

а для `HRV` дополнительно:

- `hrv_context_start_sec = window_end_sec - 120`
- `hrv_context_end_sec = window_end_sec`

## Границы анализа

В основную выборку входят только рабочие окна:

- от начала первой рабочей части после калибровочного старта;
- до `stop_time_sec`.

`Recovery` из основной выборки исключается.

Причина:

- recovery — это отдельный физиологический режим;
- смешивать его с задачей определения порога неправильно.

## Структура датасета

На выходе должны получиться отдельные таблицы.

### 1. Таблица испытуемых

Файл:

- `dataset/subjects.parquet`

Одна строка = один испытуемый.

Поля:

- `subject_id`
- `subject_name`
- `source_h5_path`
- `sex`
- `age`
- `height`
- `weight`
- `body_fat_mass`
- `skeletal_muscle_mass`
- `dominant_leg_lean_mass`
- `dominant_leg_fat_mass`
- `phase_angle`
- `dominant_leg_circumference`
- `stop_time_sec`
- `lt2_power_w`
- `lt2_time_center_sec`
- `lt2_interval_start_sec`
- `lt2_interval_end_sec`
- `lt2_interval_start_power_w`
- `lt2_interval_end_power_w`
- `lt2_refined_valid`
- `lt2_refined_time_sec`
- `lt2_refined_window_start_sec`
- `lt2_refined_window_end_sec`
- `lt2_power_label_quality`
- `lt2_time_label_quality`

Для summary-метаданных с пропусками дополнительно создаются флаги:

- `<field>_is_missing`

Пример:

- `phase_angle_is_missing`

### 2. Таблица окон

Файл:

- `dataset/windows.parquet`

Одна строка = одно окно.

Поля:

- `window_id`
- `subject_id`
- `window_start_sec`
- `window_end_sec`
- `window_center_sec`
- `window_duration_sec`
- `current_power_w`
- `window_power_mode_w`
- `stage_index`
- `elapsed_sec`
- `is_work_phase`

Определения:

- `current_power_w` — мощность в **правой границе окна**
- `window_power_mode_w` — мощность, покрывающая наибольшую часть окна
- `elapsed_sec` — время от начала **рабочей части** теста (старта первой рабочей ступени) до правой границы окна; не зависит от длины baseline (30 Вт)

### 3. Таблица ЭМГ и кинематики

Файл:

- `dataset/features_emg_kinematics.parquet`

Ключи:

- `window_id`
- `subject_id`

### 4. Таблица NIRS

Файл:

- `dataset/features_nirs.parquet`

Ключи:

- `window_id`
- `subject_id`

### 5. Таблица HRV

Файл:

- `dataset/features_hrv.parquet`

Ключи:

- `window_id`
- `subject_id`

### 6. Таблица таргетов

Файл:

- `dataset/targets.parquet`

### 7. Таблица QC

Файл:

- `dataset/qc_windows.parquet`

### 8. Слитая таблица для classical ML

Файл:

- `dataset/merged_features_ml.parquet`

## Признаки на окно

## ЭМГ + кинематика

Используется текущий метод фазовой разметки:

- PCA гироскопа по первой минуте / `30 Вт`
- детекция циклов педалирования
- уточнение onset по ЭМГ
- нормализация ЭМГ по первой минуте / `30 Вт`

Для каждого окна формируются 4 потока:

- `VL_dist_load`
- `VL_dist_rest`
- `VL_prox_load`
- `VL_prox_rest`

Для каждого потока считаются `16` признаков.

### Временная область

- `rms`
- `mav`
- `wl`
- `zcr`

### Частотная область

- `mdf`
- `mnf`
- `p_low`
- `p_mid`
- `p_high`
- `ratio_mid_high`

### Вейвлеты

- `e_d2`
- `e_d3`
- `e_d4`
- `e_d5`
- `wavelet_entropy`
- `ratio_low_high`

Итого:

- `4 потока × 16 = 64` базовых ЭМГ-признака

### Производные prox-dist признаки

С учётом принятого маппинга `VL_dist / VL_prox` дополнительно считаются
разностные признаки по двум фазам:

- `delta_rms_prox_dist_load`
- `ratio_rms_prox_dist_load`
- `delta_mdf_prox_dist_load`
- `delta_we_prox_dist_load`
- `delta_rms_prox_dist_rest`
- `ratio_rms_prox_dist_rest`
- `delta_mdf_prox_dist_rest`
- `delta_we_prox_dist_rest`

Итого:

- `8` производных ЭМГ-признаков

### Кинематика

На каждое окно:

- `cadence_mean_rpm`
- `cadence_cv`
- `load_duration_ms`
- `rest_duration_ms`
- `load_rest_ratio`

Итого:

- `5` кинематических признаков

### Всего ЭМГ + кинематика

- `64 + 8 + 5 = 77` признаков

## NIRS

Для анализа используются **только каналы Train.Red**.

Каналы:

- `train.red.smo2`
- `train.red.hhb.unfiltered`
- `train.red.hbdiff`
- `train.red.thb.unfiltered`

Базовый набор признаков на окно:

### `train.red.smo2`

- `trainred_smo2_mean`
- `trainred_smo2_drop`
- `trainred_smo2_slope`
- `trainred_dsmo2_dt`
- `trainred_smo2_std`

Нормализация `train.red.smo2` фиксируется так же, как и для `EMG`:

- baseline = первая минута теста / ступень `30 Вт`
- `trainred_smo2_drop = baseline - smo2`

### `train.red.hhb.unfiltered`

- `trainred_hhb_mean`
- `trainred_hhb_slope`
- `trainred_dhhb_dt`
- `trainred_hhb_std`

### `train.red.hbdiff`

- `trainred_hbdiff_mean`
- `trainred_hbdiff_slope`
- `trainred_hbdiff_std`

### `train.red.thb.unfiltered`

- `trainred_thb_mean`
- `trainred_thb_slope`
- `trainred_thb_std`

### Всего NIRS

- `15` признаков

Важно:

- `moxy.smo2` и `moxy.thb` в базовый датасет не включаются;
- они остаются в `HDF5`, но не используются в основном feature pipeline.

## HRV

Для `HRV` используется trailing-context длиной `120 с`, заканчивающийся в
`window_end_sec`.

> ⚠️ **Тип окна DFA-α1:** при расчёте LT2 в `create_finaltest_h5.py` DFA-α1
> считается на **центрированных** 120-секундных окнах (офлайн, пост-фактум).
> В `features_hrv.parquet` DFA-α1 должен считаться на **trailing** 120-секундных
> окнах (каузально: только данные до `window_end_sec`). Значения
> `lt2_hrvt2_time_sec` из `finaltest.h5` attrs — это ground truth по LT2,
> а не готовые признаки для ML. Напрямую их в датасет не переносить.

Признаки:

- `mean_rr`
- `sdnn`
- `rmssd`
- `dfa_alpha1`
- `sd1`
- `sd2`
- `sd1_sd2_ratio`

Итого:

- `7` признаков

## Общие поля окна

На уровне окна сохраняются общие контекстные поля:

- `current_power_w`
- `window_power_mode_w`
- `stage_index`
- `elapsed_sec`

Смысл:

- они входят в датасет как **служебный и контекстный слой**;
- это не означает, что они обязательно пойдут в baseline-модель.

## Что такое context-модель

На этом этапе модель **не строится**, но термин фиксируем заранее.

`Context-модель` — это будущая версия модели, которая получает не только
физиологические признаки, но и признаки протокольного контекста:

- `current_power_w`
- `window_power_mode_w`
- `elapsed_sec`
- `stage_index`

То есть:

- **baseline physiology-only** модель использует только сигнальные признаки;
- **context-модель** использует сигнальные признаки + контекст теста.

На этапе сборки датасета это различие важно только тем, что контекстные поля
нужно сохранить в таблице окон.

## Таргеты

Конечная цель — определение `time_to_lt2` на каждом окне.

Поэтому в таблице таргетов сохраняются оба семейства целей:

### Регрессия

- `target_time_to_lt2_center_sec = lt2_time_center_sec - window_end_sec`
- `target_time_to_lt2_refined_sec = lt2_refined_time_sec - window_end_sec`

Здесь:

- `target_time_to_lt2_center_sec` есть всегда;
- `target_time_to_lt2_refined_sec` использовать как полноценную цель можно
  только если `lt2_time_label_quality` равен `high` или `medium`.

### Бинарная классификация

Бинарный таргет нельзя делать “в лоб” по одной точке, потому что `LT2` у нас
имеет интервал неопределённости.

Поэтому вводятся поля:

- `target_binary_label`
- `target_binary_valid`
- `target_in_coarse_lt2_interval`
- `target_in_refined_lt2_window`

Правила:

- `target_binary_label = 0`, если `window_end_sec < lt2_interval_start_sec`
- `target_binary_label = 1`, если `window_start_sec >= lt2_interval_end_sec`
- иначе окно считается **неоднозначным** для бинарной задачи:
  - `target_binary_valid = 0`
  - `target_binary_label = NaN`

Это принципиально: окна, попавшие в сам переходный интервал, не надо
насильно маркировать как “до” или “после”.

## Обработка пропусков и невалидности

На этапе подготовки датасета **не выполняется финальная импутация**.

Все пропуски должны быть сохранены явно.

### Пропуски в summary

Если summary-поле отсутствует:

- значение поля = `NaN`
- отдельный флаг `<field>_is_missing = 1`

### Невалидный канал целиком

Если весь канал признан невалидным:

- все соответствующие признаки = `NaN`
- флаг модальности = `0`

Примеры:

- `emg_valid = 0`
- `nirs_valid = 0`
- `hrv_valid = 0`
- `kinematics_valid = 0`

### Локальные выпадения внутри сессии

Если датчик отвалился временно:

- признаки становятся `NaN` только в затронутых окнах;
- соседние окна, где покрытие достаточно, остаются валидными.

## QC и покрытие окна

Базовое правило для модальности:

- окно считается валидным, если покрытие по времени `>= 80%`

### EMG / гироскоп

Для непрерывных сигналов покрытие считается по sample-rate:

- `coverage = observed_samples / expected_samples`

Окно валидно, если:

- `coverage >= 0.8`

### HRV

Для `HRV` используется trailing-окно `120 с`.

Окно считается валидным, если одновременно:

- покрытие по длительности валидных RR-интервалов `>= 80%`
- доля артефактов `<= 5%`

Детекция проблем в `RR`:

1. аномальные интервалы определяются так же, как в текущем методе:
   - физиологические пределы;
   - локальная медианная проверка;
   - артефактная доля окна.
2. разрыв дополнительно определяется по рассогласованию между:
   - длительностью `RR_i`;
   - разницей `timestamp[i+1] - timestamp[i]`

Если временная дельта заметно длиннее ожидаемого RR, лишняя часть считается
выпадением канала.

Практическое правило:

- `rr_gap_sec = max(0, delta_t_next_sec - rr_i_sec)`
- сумма таких gap-вкладов уменьшает `rr_coverage_fraction`

### NIRS

Для `NIRS` окно считается валидным, если покрытие `>= 80%`.

Разрыв `Train.Red` определяется не по фиксированному числу сэмплов, а по
временной шкале самого канала:

1. для сессии считается `median_dt_sec` по `train.red.smo2`
2. gap начинается, если:
   - `dt > max(1.0 c, 5 * median_dt_sec)`

Обоснование:

- типичный шаг `Train.Red` значительно меньше 1 секунды;
- разрыв в 1 секунду уже означает потерю нескольких подряд измерений;
- множитель `5 * median_dt` защищает от ложных срабатываний на более редких
  сессиях.

Вклад gap в пропуск:

- `nirs_gap_sec = max(0, dt - median_dt_sec)`

Сумма таких вкладов по окну уменьшает `nirs_coverage_fraction`.

## QC-поля окна

В `qc_windows.parquet` должны быть как минимум:

- `window_id`
- `subject_id`
- `emg_coverage_fraction`
- `kinematics_coverage_fraction`
- `nirs_coverage_fraction`
- `hrv_coverage_fraction`
- `hrv_artifact_fraction`
- `cycles_count`
- `emg_valid`
- `kinematics_valid`
- `nirs_valid`
- `hrv_valid`
- `window_valid_any`
- `window_valid_all_required`

## Порядок сборки

Порядок реализации builder-скриптов:

1. `subjects`
2. `windows`
3. `targets`
4. `features_emg_kinematics`
5. `features_nirs`
6. `features_hrv`
7. `qc_windows`
8. `merged_features_ml`

Именно такой порядок удобен, потому что:

- сначала фиксируется индекс испытуемых;
- потом временной индекс окон;
- потом таргеты;
- потом независимые модальные признаки;
- потом QC;
- и только потом общий merge.

## Что уже зафиксировано

- источник истины: `finaltest.h5`
- цель подготовки: датасет, а не обучение модели
- базовое окно: `30 с`, шаг `5 с`
- `HRV`-контекст: trailing `120 с`
- `Recovery` исключается
- `Train.Red` используется как основной `NIRS`
- baseline-нормализация `Train.Red SmO2` делается по первой минуте / `30 Вт`
- `Moxy` в базовый датасет не включается
- `VL_dist / VL_prox` задаются через операционный маппинг каналов
- бинарный таргет строится только вне coarse LT2-интервала
- базовый порог валидности покрытия: `80%`
