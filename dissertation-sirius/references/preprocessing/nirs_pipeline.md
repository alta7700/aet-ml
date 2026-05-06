# NIRS: пайплайн предобработки и извлечения признаков (SmO₂)

> Файл: `preprocessing/nirs_pipeline.md`  
> Канал: Moxy Monitor, ANT+, 3–4 Гц  
> Переменные: SmO₂ (%), THb (г/дл)

---

## Физиология сигнала

SmO₂ = O₂Hb / (O₂Hb + HHb) — процент насыщения мышечного гемоглобина кислородом.

Поведение при нарастающей нагрузке:
```
Покой / разминка:   SmO₂ ≈ стабильна (65–85%, индивидуально)
До LT1:             медленное падение (S1 — пологий наклон)
LT1 → LT2:         ускорение падения — breakpoint наклона (S2)
После LT2:          резкое падение или плато десатурации
Восстановление:     быстрый возврат к исходному (реоксигенация)
```

Wang et al. (2006): лучший NIRS-маркер LT — ΔHHb (breakpoint у всех 15).
Moxy отдаёт только SmO₂ и THb, поэтому работаем с производными SmO₂.

---

## По THb — брать или нет

Ответ: опционально, низкий приоритет.

Wang 2006: nTHI растёт линейно в ходе теста — гиперемия, не breakpoint.
Van der Zwaard 2016: изменения THb при cwNIRS — частично артефакт
(прибор предполагает фиксированный коэффициент рассеяния).

Решение: включить как один признак для SHAP.
Если SHAP показывает нулевой вклад — подтверждение что неинформативен.

---

## БЛОК 0 — Загрузка

```python
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import linregress

def load_nirs(hdf5_path: str, channel: str = 'moxy') -> dict:
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        time = f[f'{channel}/time'][:]   # секунды от старта
        smo2 = f[f'{channel}/smo2'][:]   # %
        thb  = f[f'{channel}/thb'][:]    # г/дл
    return {'time': time, 'smo2': smo2, 'thb': thb}
```

---

## БЛОК 1 — Препроцессинг SmO₂

```python
def preprocess_smo2(time: np.ndarray, smo2: np.ndarray,
                    fs_target: float = 4.0,
                    smooth_window_sec: float = 10.0,
                    artifact_threshold: float = 5.0) -> dict:
    """
    Шаг 1: детекция и замена артефактов (скачок > threshold % за 1 пакет)
    Шаг 2: интерполяция на равномерную сетку fs_target Гц
    Шаг 3: Savitzky-Golay сглаживание (сохраняет форму кривой)
    """
    # Шаг 1: артефакты
    diff = np.abs(np.diff(smo2))
    artifact_mask = np.zeros(len(smo2), dtype=bool)
    artifact_mask[1:] = diff > artifact_threshold
    valid_idx = np.where(~artifact_mask)[0]
    smo2_clean = np.interp(np.arange(len(smo2)), valid_idx, smo2[valid_idx])

    # Шаг 2: равномерная сетка
    time_regular = np.arange(time[0], time[-1], 1.0 / fs_target)
    interp_fn = interp1d(time, smo2_clean, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    smo2_regular = interp_fn(time_regular)

    # Шаг 3: сглаживание
    window_pts = int(smooth_window_sec * fs_target)
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = max(window_pts, 5)
    smo2_smooth = savgol_filter(smo2_regular, window_length=window_pts,
                                 polyorder=2)

    return {
        'time': time_regular,
        'smo2_raw': smo2_regular,
        'smo2_smooth': smo2_smooth,
        'artifact_fraction': artifact_mask.mean(),
    }
```

---

## БЛОК 2 — Нормализация

```python
def compute_smo2_baseline(smo2_smooth: np.ndarray, time: np.ndarray,
                           step_30W_start_sec: float,
                           step_30W_end_sec: float) -> float:
    """
    Базовая линия = среднее SmO₂ за первую минуту теста, то есть за ступень 30 Вт.

    В этом проекте baseline для NIRS специально унифицирован с EMG:
    и ЭМГ, и Train.Red нормируются на один и тот же ранний опорный участок.
    Это упрощает сопоставление мультимодальных признаков по окнам и убирает
    лишнюю методическую неоднородность между блоками пайплайна.

    Абсолютные значения SmO₂ несопоставимы между испытуемыми
    из-за разной ATT (толщина подкожного жира).
    Нормировка на baseline убирает inter-subject offset.
    """
    start_idx = np.searchsorted(time, step_30W_start_sec)
    end_idx   = np.searchsorted(time, step_30W_end_sec)
    return float(np.mean(smo2_smooth[start_idx:end_idx]))


def normalize_smo2(smo2_smooth: np.ndarray, baseline: float) -> np.ndarray:
    """
    smo2_drop = baseline - smo2 → положительный при десатурации.
    Нормированное падение сопоставимо между испытуемыми.
    """
    return baseline - smo2_smooth
```

---

## БЛОК 3 — Признаки в скользящем окне

```python
def nirs_features(smo2_smooth: np.ndarray,
                  smo2_drop: np.ndarray,
                  time: np.ndarray,
                  thb: np.ndarray,
                  window_start_idx: int,
                  window_end_idx: int,
                  include_thb: bool = False) -> list:
    """
    5 основных NIRS-признаков + опциональный THb.
    """
    sl     = slice(window_start_idx, window_end_idx)
    smo2_w = smo2_smooth[sl]
    drop_w = smo2_drop[sl]
    time_w = time[sl]

    if len(smo2_w) < 3:
        n = 6 if include_thb else 5
        return [np.nan] * n

    # 1. Абсолютное значение SmO₂ (среднее по окну)
    smo2_mean = np.mean(smo2_w)

    # 2. Накопленное падение от baseline
    drop_mean = np.mean(drop_w)

    # 3. Наклон SmO₂ в окне (%/сек) — ключевой breakpoint-маркер
    #    Отрицательный при десатурации, ускоряется после LT1
    slope, *_ = linregress(time_w - time_w[0], smo2_w)

    # 4. Мгновенная скорость изменения (производная)
    dsmo2_dt = np.mean(np.gradient(smo2_w, time_w))

    # 5. Стандартное отклонение в окне
    smo2_std = np.std(smo2_w)

    features = [smo2_mean, drop_mean, slope, dsmo2_dt, smo2_std]

    if include_thb:
        thb_w = thb[sl]
        features.append(float(np.mean(thb_w)) if len(thb_w) > 0 else np.nan)

    return features


NIRS_FEATURE_NAMES = [
    'smo2_mean',    # абсолютный уровень
    'smo2_drop',    # падение от baseline — нормализованное
    'smo2_slope',   # наклон (%/сек) — ключевой маркер
    'dsmo2_dt',     # мгновенная скорость
    'smo2_std',     # вариабельность
    # 'thb_mean',   # раскомментировать если include_thb=True
]
```

---

## Итог

| Признак | Приоритет | Физиологический смысл |
|---|---|---|
| `smo2_mean` | ⚠️ Средний | Зависит от ATT, межсубъектная вариация |
| `smo2_drop` | ✅ Высокий | Нормализует offset, отражает накопленную десатурацию |
| `smo2_slope` | ✅ Высокий | Ускорение падения = breakpoint = LT-маркер |
| `dsmo2_dt` | ✅ Высокий | Чувствительнее к переходу чем slope |
| `smo2_std` | ⚠️ Средний | Нестабильность перфузии / артефакты |
| `thb_mean` | ❌ Низкий | Не меняется, cwNIRS-артефакт при нагрузке |

Итого: **5 основных признаков** (+ 1 опциональный THb)

---

## Параметры для М/М

| Параметр | Значение | Обоснование |
|---|---|---|
| Порог артефакта | 5% за 1 пакет | Физиологически невозможный скачок SmO₂ |
| Целевая частота | 4 Гц | Соответствует Moxy, без апсемплинга |
| Сглаживание | Savitzky-Golay, окно 10 с, порядок 2 | Сохраняет форму кривой |
| Базовая линия | Вся первая минута теста (ступень 30 Вт) | Унификация с EMG и единый ранний baseline для мультимодального пайплайна |
| Нормализация | baseline − smo2 | Убирает межсубъектный offset по ATT |

---

## Связанные файлы

- `preprocessing/gyroscope_pca.md` — временна́я синхронизация
- `preprocessing/emg_pipeline.md` — ЭМГ пайплайн (параллельный блок)
- `domain_knowledge.md` раздел 3 — физиология NIRS
