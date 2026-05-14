# Шаг 01. Базальные HRV-признаки на калибровке 30 Вт

## Цель
Добавить в `dataset/session_params.parquet` две новые колонки:
- `hrv_hr_baseline_bpm` — средняя ЧСС на ступени 30 Вт;
- `hrv_rmssd_baseline_ms` — RMSSD на ступени 30 Вт.

Эти признаки нужны для маршрутизатора в шаге 05 (вариант «б+в» = базальные показатели нагрузки 30 Вт + антропометрия). 30 Вт исключена из ML-окон, утечки нет.

## Что не делаем (решение пользователя)
- `cadence_baseline` — не нужен (на 30 Вт сопротивления почти нет, кадане шумен).
- Реального покоя (вариант «а») в записях нет — заменяем калибровкой 30 Вт.
- Признаки первого окна нагрузки (вариант «б» в моём предыдущем понимании) не используем — создавали бы утечку, плюс шумны и не усреднены.

## Что есть
- `dataset_pipeline/hrv.py:load_rr_signal` — уже умеет читать `zephyr.rr` из HDF5.
- `dataset_pipeline/emg_kinematics.py:_load_nirs_baseline_mean` — образец, как считать baseline по маске стадии 30 Вт. Использует `BASELINE_STAGE_INDEX = 0` из common.
- `dataset_pipeline/emg_kinematics.py:_build_session_params_row` — собирает строку `session_params.parquet`, добавляем сюда 2 поля.
- `scripts/build_dataset_all.py` — сборщик, перезапустим с `--force` только нужный шаг.

## Реализация

### 1. Добавить функцию в `hrv.py`
```python
def compute_hrv_baseline(
    source_h5_path: Path,
    baseline_start_sec: float,
    baseline_end_sec: float,
) -> tuple[float, float]:
    """Возвращает (hr_bpm, rmssd_ms) на интервале калибровки 30 Вт."""
```
- Загружает RR через `load_rr_signal`.
- Фильтрует по `baseline_start_sec ≤ t ≤ baseline_end_sec`.
- Применяет `correct_rr_window` (артефакты).
- Если валидных RR < 30 (≤30 с записи) → возвращает `(nan, nan)` и пишет warning.
- HR = 60_000 / mean(rr_ms); RMSSD = sqrt(mean(diff(rr_ms)²)).

### 2. Прокинуть в `emg_kinematics.py`
- В `PhasedSession` добавить `hrv_hr_baseline_bpm`, `hrv_rmssd_baseline_ms`.
- В `_build_phased_session` (или эквивалент) вызвать `compute_hrv_baseline` с границами стадии 0 (так же, как берётся для `_load_nirs_baseline_mean`).
- В `_build_session_params_row` добавить две колонки.

### 3. Перезапустить сборку
```bash
uv run python scripts/build_dataset_all.py --force --only emg_kinematics
# либо если --only нет — просто --force, остальные шаги быстрые
```

### 4. Проверить
- `session_params.parquet.shape == (19, 10)` (было (19, 8)).
- Распределение `hrv_hr_baseline_bpm` по 18 субъектам — sanity: 60–120 bpm.
- Корреляция `hrv_hr_baseline_bpm` с `time_to_lt2` отрицательная (тренированный = низкая ЧСС покоя → больше `time_to_lt2`).

### 5. Обновить `nn-versions-plan.md` / `DATASET_PIPELINE.md`
Один абзац: что и зачем добавлено в session_params.

## Артефакты
- Изменённый `dataset_pipeline/hrv.py` (+ функция `compute_hrv_baseline`).
- Изменённый `dataset_pipeline/emg_kinematics.py` (+ 2 поля в PhasedSession + session_params row).
- Перегенерированный `dataset/session_params.parquet`.
- В этом плане — sanity-таблица (mean/median/std `hrv_hr_baseline_bpm` и `hrv_rmssd_baseline_ms` по выборке).

## Критерий завершённости
- В `session_params.parquet` есть две новые колонки, ни одна не полностью NaN.
- Корреляция HR_baseline и time_to_lt2 имеет ожидаемый знак (ρ < 0).
- ML-пайплайн (`merged_features_ml.parquet`) при этом **не меняется** — мы не трогаем windows и не добавляем поля per-window.

---

## Отчёт о выполнении

- **Дата**: 2026-05-14
- **Что изменено**:
  - `dataset_pipeline/hrv.py` — добавлена `compute_hrv_baseline(source_h5_path, baseline_start_sec, baseline_end_sec) → (hr_bpm, rmssd_ms)`. Минимальная валидная длительность baseline-записи `_MIN_BASELINE_RR_DURATION_SEC = 30 с`, иначе NaN. Применяется штатный `correct_rr_window`.
  - `dataset_pipeline/emg_kinematics.py` — в `PhasedSession` добавлены поля `hrv_hr_baseline_bpm`, `hrv_rmssd_baseline_ms`; в `build_phased_session` вызывается `compute_hrv_baseline` на границах `result.normalization_stage`; в `_build_session_params_row` добавлены 2 колонки.
  - `scripts/patch_session_params_hrv_baseline.py` — одноразовый скрипт обновления существующего `session_params.parquet` без пересборки тяжёлого `features_emg_kinematics.parquet`. Запускался один раз; в будущем `build_dataset_all.py` сам соберёт эти поля при `--force`.
  - `dataset/session_params.parquet` — было `(19, 8)`, стало `(19, 10)`.

- **Sanity по 18 валидным субъектам** (S015 → NaN, у него и так нет EMG/NIRS):
  - `hrv_hr_baseline_bpm`: 65.9–133.6 (mean 96.4)
  - `hrv_rmssd_baseline_ms`: 4.8–28.4 (mean 14.0)

- **Корреляции (Spearman, n=18)**:
  - HR_baseline × time_to_lt2: **ρ = −0.501, p = 0.034** ✓ ожидаемый знак, значимо
  - RMSSD_baseline × time_to_lt2: ρ = +0.193, p = 0.44 (направление верное, не значимо при n=18)

- **Разделение по порогу `time_to_lt2 > 13 мин`** (6 трен. vs 12 нетрен.):
  - HR_baseline: 82.9 vs 103.2 bpm (Δ = 20 уд/мин — сильный разделяющий признак)
  - RMSSD_baseline: 16.6 vs 13.5 мс

- **Замечания**:
  - S005 — выброс: HR=133 bpm, RMSSD=4.8 мс. Стресс на калибровке или артефакт; обработка штатным `correct_rr_window` его пропустила, в выборку входит. Проверить отдельно при анализе маршрутизатора.
  - S015 остаётся NaN — он отсутствует в основном корпусе моделей.
  - Признак HR_baseline уже сам по себе кандидат №1 в маршрутизатор шага 05.

- **Следующий шаг**: step_02 — единый ranker по `summary_all_versions.csv`.
