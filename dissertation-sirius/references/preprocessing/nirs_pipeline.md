# NIRS: пайплайн предобработки и извлечения признаков

> Файл: `preprocessing/nirs_pipeline.md`
> Канал: Train.Red, BLE, ~2 Гц
> Переменные: SmO₂ (%), HHb (г/дл), HbDiff (г/дл), THb (г/дл)

---

## Устройство

**Train.Red** — устройство ближней инфракрасной спектроскопии (NIRS).
Крепится на m. vastus lateralis (VL) между двумя ЭМГ-датчиками Delsys.

HDF5-каналы в `finaltest.h5`:
```
channels/train.red.smo2
channels/train.red.hhb.unfiltered
channels/train.red.hbdiff
channels/train.red.thb.unfiltered
```

Временна́я привязка: якорь берётся из `channels/moxy.smo2/timestamps[0]` (legacy-поле).

---

## Базовая линия

SmO₂ baseline = среднее значение SmO₂ на ступени **30 Вт** (весь интервал).
Сохраняется в `session_params.parquet` → `nirs_smo2_baseline_mean`.

Признак `trainred_smo2_drop = baseline − smo2_mean` — нормированное накопленное падение оксигенации.
Сопоставим между участниками, независимо от индивидуального уровня SmO₂ в покое.

---

## Признаки на скользящее окно (15 признаков)

Для каждого окна 30 с берётся сегмент сигнала `[window_start, window_end)`.
Окно считается валидным если NIRS-покрытие ≥ 80%.

### SmO₂ — 5 признаков

| Колонка | Описание |
|---|---|
| `trainred_smo2_mean` | Среднее SmO₂ в окне (%) |
| `trainred_smo2_drop` | `baseline − smo2_mean` — падение от 30 Вт (%) |
| `trainred_smo2_slope` | Наклон SmO₂ (%/с) — ключевой маркер ускорения десатурации |
| `trainred_dsmo2_dt` | Производная: среднее `np.gradient(smo2, t)` |
| `trainred_smo2_std` | Стандартное отклонение SmO₂ в окне |

### HHb — 4 признака

| Колонка | Описание |
|---|---|
| `trainred_hhb_mean` | Среднее HHb в окне |
| `trainred_hhb_slope` | Наклон HHb (г/дл/с) — растёт при нарастании нагрузки |
| `trainred_hhb_std` | Стандартное отклонение HHb |
| `trainred_dhhb_dt` | Производная HHb |

### HbDiff — 3 признака

| Колонка | Описание |
|---|---|
| `trainred_hbdiff_mean` | Среднее HbDiff (O₂Hb − HHb) |
| `trainred_hbdiff_slope` | Наклон HbDiff |
| `trainred_hbdiff_std` | Стандартное отклонение HbDiff |

### THb — 3 признака

| Колонка | Описание |
|---|---|
| `trainred_thb_mean` | Среднее суммарного гемоглобина |
| `trainred_thb_slope` | Наклон THb |
| `trainred_thb_std` | Стандартное отклонение THb |

---

## Выходной файл

`dataset/features_nirs.parquet` — 15 признаков + 3 QC-поля на окно:
- `nirs_valid` (0/1)
- `nirs_coverage_fraction`
- окна с `nirs_valid == 0` дают `nan` во всех признаках

NIRS-признаки используются только для **LT2**. Для LT1-модели NIRS недоступен (не у всех участников).

---

## Параметры

| Параметр | Значение |
|---|---|
| Минимальное покрытие | 80% (0.8) |
| Baseline | SmO₂ mean на 30 Вт (весь интервал) |
| Признаков итого | 15 |
| Таргет | LT2 |

---

## Связанные файлы

- `preprocessing/emg_pipeline.md` — ЭМГ-датчики пространственно совмещены с NIRS
- `preprocessing/gyroscope_pca.md` — синхронизация временных осей
