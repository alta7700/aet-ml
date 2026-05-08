# Гироскоп: PCA-детекция фаз педалирования

> Файл: `preprocessing/gyroscope_pca.md`
> Канал: Delsys Trigno Avanti, гироскоп 3 оси, 148 Гц
> Датчик приклеен на m. vastus lateralis, не на педаль

---

## PCA-калибровка

**Один раз на участника** — на ступени **30 Вт**:
- Используется последние 60 с ступени 30 Вт (`DEFAULT_CALIBRATION_TAIL_SEC = 60.0`)
- PCA(n_components=1) по трём осям гироскопа → вектор W shape (3,)
- W фиксируется на всю сессию: `gyro_main = gyro_xyz @ W`
- Знак W корректируется так чтобы load-фаза (активная) давала положительный пик

Параметры из кода:
```python
DEFAULT_CALIBRATION_POWER_W   = 30.0   # Вт — ступень калибровки
DEFAULT_CALIBRATION_TAIL_SEC  = 60.0   # с  — хвост ступени для PCA
DEFAULT_CALIBRATION_MARGIN_SEC = 0.0   # с  — отступ от конца ступени
```

---

## Per-stage prominence

Порог для детекции пиков педалирования пересчитывается на **каждой ступени**:

```python
DEFAULT_PEAK_PROMINENCE_STD = 0.5
prominence_i = std(gyro_segment_stage_i) × 0.5
```

Сегмент — запись гироскопа на данной ступени. Prominence используется в `find_peaks`.

---

## Детекция фаз

`detect_phases()` строит список `CyclePhase`-объектов.
Каждый `CyclePhase` содержит `push_start_sec`, `push_end_sec` (= load) и `pull_start_sec`, `pull_end_sec` (= rest) для одного оборота педали.

Минимальный каданс: `DEFAULT_MIN_CADENCE_RPM` (50 об/мин).
Максимальный: `DEFAULT_MAX_CADENCE_RPM`.

---

## Кинематические признаки на окно

Из `CyclePhase`-объектов, попадающих в [window_start, window_end]:

| Признак | Описание |
|---|---|
| `cadence_rpm` | Средний каданс (об/мин) |
| `push_dur_ms` | Средняя длительность load-фазы (мс) |
| `pull_dur_ms` | Средняя длительность rest-фазы (мс) |
| `phase_ratio` | push_dur / pull_dur |
| `cadence_cv` | CV каданса — нестабильность педалирования |
| `load_sampen_30s` | SampEn длительностей load-фаз в 30с-окне |
| `rest_sampen_30s` | SampEn длительностей rest-фаз в 30с-окне |
| `load_duration_cv` | CV длительностей load-фаз |
| `rest_duration_cv` | CV длительностей rest-фаз |

SampEn и timing_cv — маркеры нестабильности нейромышечного ритма при нарастании нагрузки.

---

## Выходные файлы

- `dataset/features_emg_kinematics.parquet` — кинематические признаки включены
- `dataset/session_params.parquet` — `pca_axis` (вектор W) на участника

---

## Связанные файлы

- `preprocessing/emg_pipeline.md` — использует `CyclePhase` для сегментации ЭМГ
