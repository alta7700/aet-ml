# ЭМГ: полный пайплайн предобработки и извлечения признаков

> Файл: `preprocessing/emg_pipeline.md`  
> Конфигурация датчиков (снизу вверх по ноге):
>   Delsys Avanti #1 (VL дистальная) → Train.Red NIRS → Delsys Avanti #2 (VL проксимальная)
> Каналы: 2 × VL (1 биполярный канал на датчик), 1926 Гц  
> Зависимость: требует `phases` из `preprocessing/gyroscope_pca.md`

---

## Конфигурация — методологическая новизна

Два датчика на одной мышце (VL) в дистальной и проксимальной зонах:
- RF не записывается
- Оба датчика дают 1 биполярный ЭМГ-канал каждый
- NIRS (Train.Red) находится между датчиками → локальная оксигенация
  совпадает пространственно с ЭМГ-зонами

**Что это даёт:**
При нарастании усталости и рекрутировании дополнительных МЕ паттерн
активации по длине VL меняется. Проксимальная зона рекрутируется позже
и сильнее реагирует на высокую нагрузку. Разностные признаки (prox − dist)
в литературе по детекции порога не исследовались → это новизна работы.

---

## Архитектура пайплайна

```
VL_distal (1926 Гц)    VL_proximal (1926 Гц)
        │                       │
        └──────┬────────────────┘
               │
        БЛОК 0: Препроцессинг
          notch 50 Гц → [bandpass 20–450 Гц] → нормализация на первую минуту (30 Вт)
               │
        БЛОК 1: Фазовая сегментация (из gyroscope_pca)
          → 4 потока: dist_push, dist_pull, prox_push, prox_pull
               │
        БЛОК 2: Признаки в скользящем окне (30 с, шаг 5 с)
          Временная область + Частотная + Вейвлеты
               │
        БЛОК 3: Производные признаки (prox − dist)
               │
        ВЫХОД: 64 (ЭМГ) + 8 (производные) + 5 (кинематика) = 77 признаков
```

---

## БЛОК 0 — Препроцессинг

По фильтрации: Delsys Trigno Avanti имеет аппаратный аналоговый bandpass ~20–450 Гц.
Программный bandpass дублирует его с точными цифровыми характеристиками.
Notch 50 Гц аппаратно отсутствует — обязателен программно.

```python
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np

def preprocess_emg(raw: np.ndarray, fs: int = 1926,
                   apply_bandpass: bool = True) -> np.ndarray:
    """
    Шаг 1: notch 50 Гц
    Шаг 2: bandpass 20–450 Гц (если apply_bandpass=True)
    """
    b_n, a_n = iirnotch(50.0, Q=30.0, fs=fs)
    filtered = filtfilt(b_n, a_n, raw)

    if apply_bandpass:
        b_bp, a_bp = butter(4, [20.0, 450.0], btype='bandpass', fs=fs)
        filtered = filtfilt(b_bp, a_bp, filtered)

    return filtered


def compute_baseline_rms(emg_filtered: np.ndarray,
                          step_30W_start_sec: float,
                          step_30W_end_sec: float,
                          fs: int = 1926) -> float:
    """
    Baseline = RMS первой минуты теста, то есть ступени 30 Вт.

    В этом проекте baseline фиксируется по протоколу жёстко:
    вся первая ступень 30 Вт используется как опорный участок
    для межсессионной и межсубъектной сопоставимости.

    Важно: нормализация делается отдельно для каждого датчика
    (distal и proximal имеют разные абсолютные уровни активации).
    """
    start_idx = int(step_30W_start_sec * fs)
    end_idx   = int(step_30W_end_sec * fs)
    return float(np.sqrt(np.mean(emg_filtered[start_idx:end_idx]**2)))


def normalize_emg(emg_filtered: np.ndarray, baseline_rms: float) -> np.ndarray:
    """Нормировка на baseline RMS — безразмерный сигнал."""
    return emg_filtered / (baseline_rms + 1e-12)
```

---

## БЛОК 1 — Фазовая сегментация

```python
def segment_by_phase(emg_norm: np.ndarray,
                     phases: dict,
                     emg_fs: int = 1926,
                     gyro_fs: int = 148) -> dict:
    """
    Разбивает нормированный ЭМГ на push/pull сегменты.
    Конвертирует индексы гироскопа (148 Гц) → индексы ЭМГ (1926 Гц).

    ratio = 1926 / 148 ≈ 13.0
    """
    ratio = emg_fs / gyro_fs
    push_segments, pull_segments = [], []

    for push_sl, pull_sl in zip(phases['push_slices'], phases['pull_slices']):
        ep_s, ep_e = int(push_sl.start * ratio), int(push_sl.stop * ratio)
        el_s, el_e = int(pull_sl.start * ratio), int(pull_sl.stop * ratio)
        if ep_e > ep_s and el_e > el_s:
            push_segments.append(emg_norm[ep_s:ep_e])
            pull_segments.append(emg_norm[el_s:el_e])

    return {'push': push_segments, 'pull': pull_segments}
```

---

## БЛОК 2 — Признаки в скользящем окне (16 на поток)

### A. Временна́я область (4 признака)

```python
def time_domain_features(segment: np.ndarray) -> list:
    rms = np.sqrt(np.mean(segment**2))
    mav = np.mean(np.abs(segment))
    wl  = np.sum(np.abs(np.diff(segment)))
    zcr = np.sum(np.diff(np.sign(segment)) != 0)
    return [rms, mav, wl, zcr]

TIME_FEATURE_NAMES = ['rms', 'mav', 'wl', 'zcr']
```

### B. Частотная область (6 признаков)

```python
from scipy.signal import welch

def freq_domain_features(segment: np.ndarray, fs: int = 1926) -> list:
    nperseg = min(len(segment), 512)
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
    cumpower = np.cumsum(psd)
    total    = cumpower[-1] + 1e-12

    mdf = freqs[np.searchsorted(cumpower, total / 2)]
    mnf = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)
    p_low  = np.sum(psd[(freqs >= 20)  & (freqs < 50)])
    p_mid  = np.sum(psd[(freqs >= 50)  & (freqs < 150)])
    p_high = np.sum(psd[(freqs >= 150) & (freqs <= 450)])
    ratio_mid_high = p_mid / (p_high + 1e-12)

    return [mdf, mnf, p_low, p_mid, p_high, ratio_mid_high]

FREQ_FEATURE_NAMES = ['mdf', 'mnf', 'p_low', 'p_mid', 'p_high', 'ratio_mid_high']
```

### C. Вейвлеты db4, уровни d2–d5 (6 признаков)

```python
import pywt

# При fs=1926 Гц, db4, level=6:
# coeffs = [a6, d6, d5, d4, d3, d2, d1]  (pywt порядок)
# d1: 481–963 Гц → шум           → НЕ ИСПОЛЬЗУЕМ (индекс 6)
# d2: 240–481 Гц → высокочаст.   → индекс 5
# d3: 120–240 Гц → осн. актив.   → индекс 4
# d4:  60–120 Гц → осн. актив.   → индекс 3
# d5:  30– 60 Гц → низкочаст.    → индекс 2
# d6:  15– 30 Гц → артефакты     → НЕ ИСПОЛЬЗУЕМ (индекс 1)
# a6:   0– 15 Гц → DC/тренд      → НЕ ИСПОЛЬЗУЕМ (индекс 0)

def wavelet_features(segment: np.ndarray,
                     wavelet: str = 'db4', level: int = 6) -> list:
    coeffs = pywt.wavedec(segment, wavelet, level=level)
    energies_all = np.array([np.sum(c**2) for c in coeffs])
    total = energies_all.sum() + 1e-12

    # d2–d5 → индексы 2,3,4,5 в массиве coeffs
    e_d5, e_d4, e_d3, e_d2 = [np.sum(coeffs[i]**2) / total
                                for i in [2, 3, 4, 5]]

    e_useful = np.array([e_d2, e_d3, e_d4, e_d5]) + 1e-12
    we = -np.sum(e_useful * np.log(e_useful))          # вейвлет-энтропия
    ratio_low_high = (e_d4 + e_d5) / (e_d2 + e_d3 + 1e-12)

    return [e_d2, e_d3, e_d4, e_d5, we, ratio_low_high]

WAVELET_FEATURE_NAMES = ['e_d2', 'e_d3', 'e_d4', 'e_d5',
                          'wavelet_entropy', 'ratio_low_high']
```

---

## БЛОК 3 — Производные признаки (prox − dist)

```python
def differential_features(feats_prox: np.ndarray,
                            feats_dist: np.ndarray) -> list:
    """
    Разностные и относительные признаки между проксимальной и дистальной зонами.
    Отражают изменение паттерна активации по длине мышцы при нарастании нагрузки.

    Литература: в контексте детекции порога не исследовалось → новизна.
    """
    # Индексы в 16-признаковом векторе потока:
    RMS_IDX = 0   # из time_domain
    MDF_IDX = 4   # из freq_domain (после 4 time features)
    ED4_IDX = 14  # из wavelet (4 time + 6 freq + 2 wavelet уровней до d4)

    delta_rms  = feats_prox[RMS_IDX] - feats_dist[RMS_IDX]   # prox - dist
    ratio_rms  = feats_prox[RMS_IDX] / (feats_dist[RMS_IDX] + 1e-12)
    delta_mdf  = feats_prox[MDF_IDX] - feats_dist[MDF_IDX]
    delta_we   = feats_prox[14] - feats_dist[14]  # wavelet_entropy: индекс 4+6+4=14

    return [delta_rms, ratio_rms, delta_mdf, delta_we]

DIFFERENTIAL_FEATURE_NAMES = [
    'delta_rms_prox_dist',    # rms_prox - rms_dist: растёт при неравномерном рекрутировании
    'ratio_rms_prox_dist',    # rms_prox / rms_dist: нормированный вариант
    'delta_mdf_prox_dist',    # mdf_prox - mdf_dist: частотный градиент по мышце
    'delta_we_prox_dist',     # entropy_prox - entropy_dist: сложность активации
]
# × 2 фазы (push/pull) = 8 производных признаков
```

---

## Сборка полного вектора

```python
def build_feature_names() -> list:
    """Генерирует полный список имён признаков."""
    names = []
    per_stream = TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES + WAVELET_FEATURE_NAMES
    # 4 потока: dist_push, dist_pull, prox_push, prox_pull
    for zone in ['dist', 'prox']:
        for phase in ['push', 'pull']:
            prefix = f'VL_{zone}_{phase}'
            names += [f'{prefix}_{n}' for n in per_stream]
    # Производные × 2 фазы
    for phase in ['push', 'pull']:
        names += [f'{n}_{phase}' for n in DIFFERENTIAL_FEATURE_NAMES]
    # Кинематика
    from preprocessing.gyroscope_pca import KINEMATICS_FEATURE_NAMES
    names += KINEMATICS_FEATURE_NAMES
    return names

ALL_FEATURE_NAMES = build_feature_names()
# 4 потока × 16 + 4 diff × 2 фазы + 5 кинематика = 64 + 8 + 5 = 77 признаков


# Baseline — первый запуск
BASELINE_FEATURES = [
    'VL_dist_push_rms', 'VL_dist_pull_rms',
    'VL_prox_push_rms', 'VL_prox_pull_rms',
    'VL_dist_push_mdf', 'VL_prox_push_mdf',
    'delta_rms_prox_dist_push',
    'delta_mdf_prox_dist_push',
    'cadence_rpm', 'phase_ratio',
]
```

---

## Итоговый вектор ЭМГ + ИМУ (77 признаков)

| Группа | Потоки / источник | Признаков |
|---|---|---|
| Временна́я область | dist_push, dist_pull, prox_push, prox_pull | 4×4 = 16 |
| Частотная область | то же | 4×6 = 24 |
| Вейвлеты d2–d5 | то же | 4×6 = 24 |
| Производные prox−dist | push + pull | 4×2 = 8 |
| Кинематика (гироскоп) | — | 5 |
| **Итого** | | **77** |

---

## Параметры для М/М

| Параметр | Значение | Обоснование |
|---|---|---|
| Датчики ЭМГ | 2× Delsys Trigno Avanti на VL (1 канал каждый) | Дистальная + проксимальная зоны |
| Расположение | Снизу: Avanti → Train.Red → Avanti | NIRS между ЭМГ-зонами |
| Notch | 50 Гц, Q=30 | Сетевая помеха |
| Bandpass | 20–450 Гц, Butterworth 4-й пор. | Рабочий диапазон sEMG |
| Нормализация | RMS первой минуты теста, то есть ступени 30 Вт, отдельно для каждого датчика | Протокольно фиксированный baseline для всех сессий |
| Вейвлет | db4, level=6, уровни d2–d5 | Стандарт для sEMG [Wang et al., 2018] |
| Окно | 30 с, шаг 5 с | Компромисс: HRV требует ≥16 с для DFA-α1 |
| Welch nperseg | 512 | Частотное разрешение ~3.8 Гц |

---

## Связанные файлы

- `preprocessing/gyroscope_pca.md` — детекция фаз (зависимость)
- `preprocessing/nirs_pipeline.md` — NIRS канал (Train.Red между датчиками)
- `preprocessing/hrv_pipeline.md` — HRV канал
