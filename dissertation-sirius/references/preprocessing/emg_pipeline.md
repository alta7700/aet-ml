# ЭМГ: пайплайн предобработки и извлечения признаков

> Файл: `preprocessing/emg_pipeline.md`
> Конфигурация: 2 × Delsys Trigno Avanti на m. vastus lateralis (VL)
> Каналы: VL-дистальный (`trigno.vl.avanti`) + VL-проксимальный (`trigno.rf.avanti`), 1926 Гц

---

## Конфигурация датчиков

Снизу вверх по ноге: **Delsys Avanti #1** (VL дистальный) → **Train.Red** (NIRS) → **Delsys Avanti #2** (VL проксимальный).
Два ЭМГ-датчика на одной мышце в разных зонах. NIRS пространственно совмещён с ЭМГ.

---

## Препроцессинг

1. **Notch 50 Гц** (Q=30) — сетевая помеха
2. **Bandpass 20–450 Гц** (Butterworth 4-й порядок) — рабочий диапазон sEMG
3. **Нормализация на baseline RMS** — делитель: RMS сигнала на ступени **30 Вт**, отдельно для каждого датчика. Сохраняется в `session_params.parquet` (`emg_vl_dist_baseline_rms`, `emg_vl_prox_baseline_rms`).

---

## Сегментация по фазам педалирования

Каждый оборот педали делится на **load** (активная фаза — педаль под нагрузкой) и **rest** (пассивная фаза — педаль идёт вверх). Границы определяются из `CyclePhase`-объектов, построенных через `detect_phases()` (гироскоп + PCA).

Итого **4 потока** на окно:

| Поток | Описание |
|---|---|
| `vl_dist_load` | VL-дистальный, load-фаза |
| `vl_dist_rest` | VL-дистальный, rest-фаза |
| `vl_prox_load` | VL-проксимальный, load-фаза |
| `vl_prox_rest` | VL-проксимальный, rest-фаза |

---

## Признаки в скользящем окне

### Временна́я область (4 признака на поток)

| Признак | Формула |
|---|---|
| `rms` | √(mean(x²)) |
| `mav` | mean(\|x\|) |
| `wl` | Σ\|x[i+1]−x[i]\| — длина волны |
| `zcr` | Число пересечений нуля |

### Частотная область (6 признаков на поток)

Метод Welch, nperseg=512.

| Признак | Описание |
|---|---|
| `mdf` | Медианная частота (Гц) |
| `mnf` | Средняя частота (Гц) |
| `p_low` | Мощность 20–50 Гц |
| `p_mid` | Мощность 50–150 Гц |
| `p_high` | Мощность 150–450 Гц |
| `ratio_mid_high` | p_mid / p_high |

### Вейвлеты db4, уровень 5 (6 признаков на поток)

db4 — стандарт для sEMG.

| Признак | Диапазон |
|---|---|
| `e_d2` | Энергия d2 (~240–481 Гц) |
| `e_d3` | Энергия d3 (~120–240 Гц) |
| `e_d4` | Энергия d4 (~60–120 Гц) |
| `e_d5` | Энергия d5 (~30–60 Гц) |
| `wavelet_entropy` | −Σ(e_i × log(e_i)) |
| `ratio_low_high` | (e_d4+e_d5) / (e_d2+e_d3) |

### Производные prox−dist (8 признаков)

Для load и rest фазы по 4 признака:

| Признак | Описание |
|---|---|
| `delta_rms_prox_dist` | rms_prox − rms_dist |
| `ratio_rms_prox_dist` | rms_prox / rms_dist |
| `delta_mdf_prox_dist` | mdf_prox − mdf_dist |
| `delta_we_prox_dist` | wavelet_entropy_prox − wavelet_entropy_dist |

### Per-cycle RMS CV (4 признака)

Коэффициент вариации RMS по отдельным циклам педалирования для каждого потока.

---

## Итог в parquet

`dataset/features_emg_kinematics.parquet` — 101 признак на окно:
- 4 потока × 16 признаков = 64 ЭМГ
- 8 prox−dist дифференциальных
- 4 EMG-CV (per-cycle)
- 7 кинематических (из гироскопа)
- 18 вариабельностных (SampEn, timing_cv)

QC-поля: `emg_valid`, `emg_vl_dist_coverage`, `emg_vl_prox_coverage`.

---

## Признаки, используемые в ML (v0004+)

В ML-скриптах к ЭМГ-признакам применяется дополнительная **z-нормировка per-subject per-feature**:
```
z = (x − μ_subj) / (σ_subj + 1e-8)
```
Итоговые колонки с префиксом `z_vl_*`.

Подмножество, вошедшее в лучшие модели:

| Колонка | Описание |
|---|---|
| `z_vl_dist_load_rms` | Нормированный RMS дистального датчика, load |
| `z_vl_dist_load_mdf` | Нормированная MDF дистального, load |
| `z_vl_dist_load_mav` | Нормированный MAV дистального, load |
| `z_vl_dist_rest_rms` | RMS дистального, rest |
| `z_vl_prox_load_rms` | RMS проксимального, load |
| `z_vl_prox_load_mdf` | MDF проксимального, load |
| `z_vl_prox_rest_rms` | RMS проксимального, rest |
| `delta_rms` | prox−dist разность RMS (опц.) |
| `ratio_rms` | prox/dist отношение RMS (опц.) |

ЭМГ-признаки используются для **LT1 и LT2**.

---

## Параметры для М/М

| Параметр | Значение |
|---|---|
| Датчики | 2 × Delsys Trigno Avanti на VL (dist + prox) |
| Частота дискретизации | 1926 Гц |
| Notch | 50 Гц, Q=30 |
| Bandpass | 20–450 Гц, Butterworth 4-й пор. |
| Baseline RMS | ступень 30 Вт (per-датчик) |
| z-нормировка в ML | per-subject, per-feature |
| Вейвлет | db4, level=5 |
| Фазы | load / rest (по гироскопу) |

---

## Связанные файлы

- `preprocessing/gyroscope_pca.md` — детекция фаз load/rest
- `preprocessing/nirs_pipeline.md` — NIRS пространственно совмещён с ЭМГ
