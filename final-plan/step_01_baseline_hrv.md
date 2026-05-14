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
_Заполняется по факту выполнения._

- Дата начала:
- Дата завершения:
- Что изменено:
- Sanity-таблица:
- Корреляция HR_baseline × time_to_lt2: ρ = , p =
- Замечания:
