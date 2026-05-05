# HRV: пайплайн предобработки и извлечения признаков

> Файл: `preprocessing/hrv_pipeline.md`  
> Канал: Zephyr BioHarness 3, BLE, событийный ряд RR-интервалов (мс)  
> Ключевой признак: DFA-α1 (Gronwald, Rogers 2019–2022)

---

## Специфика канала

RR — это НЕ равноинтервальный сигнал. Это последовательность длин интервалов:
```
[823, 819, 815, 831, 820, ...]  мс — один RR на один удар сердца
```

Временна́я ось накапливается суммированием: t[i] = t[i-1] + RR[i].  
При разрыве BLE-соединения время синхронизируется с шиной Sensync.

Это определяет весь пайплайн: сначала детекция артефактов,  
потом интерполяция на равномерную сетку — только для тех признаков где нужна.

---

## БЛОК 0 — Загрузка и восстановление временно́й оси

```python
import numpy as np

def load_rr(hdf5_path: str, channel: str = 'zephyr') -> dict:
    """
    Загружает RR-интервалы из HDF5.
    Zephyr пишет: rr_ms (длина интервала), time_sec (накопленное время).
    """
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        rr_ms   = f[f'{channel}/rr_ms'][:]     # длины интервалов в мс
        time_sec = f[f'{channel}/time'][:]      # время события в сек от старта
    return {'rr_ms': rr_ms, 'time_sec': time_sec}
```

---

## БЛОК 1 — Детекция и замена артефактов

Артефакт = интервал физиологически невозможный или аномально отличающийся  
от соседних. Критерий ±20% от скользящей медианы — стандарт в HRV-анализе.

```python
def detect_rr_artifacts(rr_ms: np.ndarray,
                         time_sec: np.ndarray,
                         window_beats: int = 5,
                         threshold: float = 0.20) -> dict:
    """
    Детектирует артефакты в RR-ряду.
    
    Args:
        rr_ms: последовательность RR-интервалов (мс)
        time_sec: временна́я ось (сек)
        window_beats: размер окна скользящей медианы (ударов)
        threshold: порог отклонения от медианы (доли, 0.20 = 20%)
    Returns:
        dict с artifact_mask и очищенным rr_ms
    """
    from scipy.ndimage import median_filter
    
    # Скользящая медиана
    rr_median = median_filter(rr_ms.astype(float), size=window_beats,
                               mode='reflect')
    
    # Артефакт = отклонение > threshold от медианы
    deviation = np.abs(rr_ms - rr_median) / (rr_median + 1e-6)
    artifact_mask = deviation > threshold
    
    # Абсолютные границы: RR < 300 мс (>200 уд/мин) или > 2000 мс (<30 уд/мин)
    artifact_mask |= (rr_ms < 300) | (rr_ms > 2000)
    
    # Замена артефактных точек кубической интерполяцией по соседним валидным
    valid_idx = np.where(~artifact_mask)[0]
    rr_clean = rr_ms.copy().astype(float)
    
    if len(valid_idx) > 3:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(valid_idx, rr_ms[valid_idx])
        artifact_idx = np.where(artifact_mask)[0]
        rr_clean[artifact_idx] = cs(artifact_idx)
    
    return {
        'rr_clean': rr_clean,
        'artifact_mask': artifact_mask,
        'artifact_fraction': artifact_mask.mean(),
    }
```

---

## БЛОК 2 — Интерполяция на равномерную сетку

Нужна только для признаков требующих равноинтервального ряда  
(RMSSD считается из сырого ряда, DFA-α1 — тоже, спектр — требует).

```python
def interpolate_rr(rr_clean: np.ndarray, time_sec: np.ndarray,
                   fs: float = 4.0) -> dict:
    """
    Интерполирует RR-ряд на равномерную временну́ю сетку.
    fs=4 Гц — стандарт для HRV-спектрального анализа.
    
    Используется только для LF/HF (не для DFA-α1 и RMSSD).
    """
    from scipy.interpolate import CubicSpline
    
    # Временна́я ось RR: начало каждого интервала
    rr_time = time_sec  # уже накопленное время из Zephyr
    
    time_uniform = np.arange(rr_time[0], rr_time[-1], 1.0 / fs)
    cs = CubicSpline(rr_time, rr_clean)
    rr_uniform = cs(time_uniform)
    
    return {'time_uniform': time_uniform, 'rr_uniform': rr_uniform}
```

---

## БЛОК 3 — Признаки в скользящем окне

### Вспомогательная функция: сбор RR-интервалов окна

```python
def get_rr_window(rr_clean: np.ndarray, time_sec: np.ndarray,
                   window_start_sec: float,
                   window_end_sec: float) -> np.ndarray:
    """
    Возвращает RR-интервалы попавшие в временно́е окно.
    """
    mask = (time_sec >= window_start_sec) & (time_sec < window_end_sec)
    return rr_clean[mask]
```

### A. Временна́я область

```python
def time_domain_hrv(rr_window: np.ndarray) -> list:
    """
    RMSSD, SDNN, mean_rr — быстро, надёжно, не требует интерполяции.
    
    RMSSD — главный парасимпатический маркер.
    Падает при нарастании нагрузки, особенно при переходе VT1/LT1.
    """
    if len(rr_window) < 5:
        return [np.nan, np.nan, np.nan]
    
    mean_rr = np.mean(rr_window)                           # мс
    sdnn    = np.std(rr_window, ddof=1)                    # мс
    rmssd   = np.sqrt(np.mean(np.diff(rr_window)**2))      # мс
    
    return [mean_rr, sdnn, rmssd]

TIME_HRV_NAMES = ['mean_rr', 'sdnn', 'rmssd']
```

### B. Нелинейные: DFA-α1

DFA-α1 — ключевой признак. Значение 0.75 соответствует VT1/LT1.

```python
def dfa_alpha1(rr_window: np.ndarray,
               n_min: int = 4,
               n_max: int = 16) -> float:
    """
    Детрендированный флуктуационный анализ, краткосрочный показатель α1.
    Окно масштабов n=4–16 кардиоциклов (Gronwald & Rogers, 2019–2022).
    
    Физиологическая интерпретация:
        α1 ≈ 1.0  — покой (фрактальные свойства, хаотичность)
        α1 ≈ 0.75 — VT1/LT1 (потеря фрактальности — маркер перехода)
        α1 < 0.75 — нагрузка выше VT1
    
    Args:
        rr_window: очищенные RR-интервалы (мс), минимум n_max*2 точек
        n_min, n_max: диапазон масштабов (рекомендации Gronwald)
    Returns:
        alpha1: скаляр или np.nan если данных недостаточно
    """
    N = len(rr_window)
    if N < n_max * 2:
        return np.nan
    
    # Интегрированный ряд (cumsum от отклонений от среднего)
    rr_centered = rr_window - np.mean(rr_window)
    y = np.cumsum(rr_centered)
    
    scales = range(n_min, n_max + 1)
    fluctuations = []
    
    for n in scales:
        # Разбиваем на непересекающиеся окна длиной n
        n_windows = N // n
        if n_windows < 2:
            continue
        
        rms_list = []
        for w in range(n_windows):
            segment = y[w*n:(w+1)*n]
            # Линейный тренд в сегменте
            x_seg = np.arange(n)
            coeffs = np.polyfit(x_seg, segment, 1)
            trend = np.polyval(coeffs, x_seg)
            rms_list.append(np.sqrt(np.mean((segment - trend)**2)))
        
        if rms_list:
            fluctuations.append(np.mean(rms_list))
    
    if len(fluctuations) < 3:
        return np.nan
    
    # Наклон log-log → α1
    log_scales = np.log(list(scales[:len(fluctuations)]))
    log_fluct  = np.log(np.array(fluctuations) + 1e-12)
    
    alpha1, _ = np.polyfit(log_scales, log_fluct, 1)
    return float(alpha1)
```

### C. Нелинейные: диаграмма Пуанкаре

```python
def poincare_features(rr_window: np.ndarray) -> list:
    """
    SD1, SD2, SD1/SD2 из диаграммы Пуанкаре (RRn vs RRn+1).
    
    SD1 ≈ RMSSD/√2 — краткосрочная вариабельность (парасимпатика)
    SD2 — долгосрочная вариабельность
    SD1/SD2 — вегетативный баланс (падает при нарастании нагрузки)
    """
    if len(rr_window) < 4:
        return [np.nan, np.nan, np.nan]
    
    rr1 = rr_window[:-1]   # RRn
    rr2 = rr_window[1:]    # RRn+1
    
    # Оси эллипса Пуанкаре
    sd1 = np.sqrt(0.5 * np.mean((rr2 - rr1)**2))
    sd2 = np.sqrt(2 * np.std(rr_window, ddof=1)**2 - 0.5 * np.mean((rr2 - rr1)**2))
    sd2 = np.sqrt(max(sd2, 0))   # защита от отрицательного подкоренного
    
    ratio = sd1 / (sd2 + 1e-10)
    
    return [sd1, sd2, ratio]

POINCARE_NAMES = ['sd1', 'sd2', 'sd1_sd2_ratio']
```

---

## Сборка полного вектора HRV-признаков

```python
def extract_hrv_features(rr_clean: np.ndarray,
                          time_sec: np.ndarray,
                          window_start_sec: float,
                          window_end_sec: float) -> np.ndarray:
    """
    Полный вектор HRV-признаков для одного окна.
    Не требует интерполяции — все признаки считаются из сырого RR-ряда.
    
    Returns:
        np.ndarray shape (7,)
    """
    rr_w = get_rr_window(rr_clean, time_sec,
                          window_start_sec, window_end_sec)
    
    if len(rr_w) < 8:   # минимум для DFA-α1 (n_max=16 нужно ≥32, но 8 = fallback)
        return np.full(7, np.nan)
    
    time_feats    = time_domain_hrv(rr_w)       # 3 признака
    alpha1        = dfa_alpha1(rr_w)             # 1 признак
    poincare      = poincare_features(rr_w)      # 3 признака
    
    return np.array(time_feats + [alpha1] + poincare)


HRV_FEATURE_NAMES = [
    'mean_rr',      # средний RR (мс) — грубый маркер ЧСС
    'sdnn',         # общая вариабельность (мс)
    'rmssd',        # парасимпатика — падает до VT1
    'dfa_alpha1',   # ключевой маркер: 0.75 = VT1/LT1 (Gronwald 2019)
    'sd1',          # краткосрочная вариабельность ≈ RMSSD/√2
    'sd2',          # долгосрочная вариабельность
    'sd1_sd2_ratio',# вегетативный баланс
]
# Итого: 7 признаков
```

---

## Итоговая таблица признаков

| Признак | Приоритет | Физиологический смысл |
|---|---|---|
| `mean_rr` | ✅ Высокий | = ЧСС в другом виде, маркер интенсивности |
| `rmssd` | ✅ Высокий | Парасимпатика, падает при нарастании нагрузки |
| `dfa_alpha1` | ✅ Высокий | Порог 0.75 = VT1, ключевой нелинейный маркер |
| `sd1` | ✅ Высокий | ≈ RMSSD, краткосрочная вариабельность Пуанкаре |
| `sdnn` | ⚠️ Средний | Общая вариабельность, менее специфичен |
| `sd2` | ⚠️ Средний | Долгосрочная вариабельность |
| `sd1_sd2_ratio` | ⚠️ Средний | Вегетативный баланс |
| LF/HF | ❌ Не берём | Нестабилен при нагрузке, требует длинного окна |
| SampEn | ❌ Опционально | Медленно считается, добавить после baseline |

**Baseline:** `mean_rr`, `rmssd`, `dfa_alpha1`, `sd1` — 4 признака.

---

## Требования к длине окна

| Признак | Минимум RR в окне | При ЧСС 120 = сек |
|---|---|---|
| mean_rr, rmssd | 5 | ~2.5 сек |
| DFA-α1 (n=4–16) | 32 (n_max × 2) | ~16 сек |
| sd1, sd2 | 4 | ~2 сек |
| LF/HF спектр | ~240 (60 сек × 4 Гц) | 60 сек |

Вывод: окно 30 сек при ЧСС ≥ 60 уд/мин даёт ≥ 30 RR → DFA-α1 считается.  
При ЧСС < 60 (маловероятно на нагрузочном тесте) — DFA-α1 может быть nan.

---

## Параметры для М/М

| Параметр | Значение | Обоснование |
|---|---|---|
| Порог артефакта | ±20% от скользящей медианы (окно 5 ударов) | Стандарт HRV-анализа |
| Абсолютные границы | 300–2000 мс | Физиологически допустимый диапазон |
| Замена артефактов | Кубическая интерполяция по валидным | Сохраняет непрерывность ряда |
| DFA масштабы | n = 4–16 кардиоциклов | Рекомендации Gronwald & Rogers (2019–2022) |
| Окно признаков | 30 сек, шаг 5 сек | Компромисс онлайн/точность |
| Интерполяция | Только для LF/HF (не используется в baseline) | DFA и RMSSD — из сырого ряда |

---

## Связанные файлы

- `preprocessing/nirs_pipeline.md` — параллельный канал
- `preprocessing/emg_pipeline.md` — параллельный канал
- `domain_knowledge.md` раздел 4 — физиология HRV и DFA-α1
