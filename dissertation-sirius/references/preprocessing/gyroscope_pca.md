# Гироскоп: PCA-детекция фаз педалирования

> Файл: `preprocessing/gyroscope_pca.md`  
> Канал: Delsys Trigno Avanti, гироскоп 3 оси, 148 Гц  
> Назначение: определить ведущую ось вращения бедра и выделить push/pull фазы

---

## Проблема

Датчик приклеен на мышцу, не на педаль. При каждой наклейке ориентация осей X/Y/Z разная.
Нельзя заранее знать какая ось будет «основной» для конкретного испытуемого.

**Три варианта решения:**

| Вариант | Метод | Проблема |
|---|---|---|
| A | Выбирать ось вручную | Не воспроизводимо, ошибки оператора |
| B | `argmax([var_x, var_y, var_z])` | Ломается при диагональной ориентации |
| C | PCA (рекомендуется) | Всегда даёт ось максимальной вариации |

---

## Решение: PCA один раз в начале сессии

Физическое обоснование: ориентация датчика **не меняется** в ходе сессии →
матрица проекции W постоянна → достаточно вычислить один раз.

### Почему НЕ пересчитывать в каждом окне
- Пересчёт добавляет «дрейф» проекции из-за шума без физического смысла
- Результат детерминированный и воспроизводимый только при фиксированной W
- Один раз + фиксация = стандартная практика для калибровки IMU

---

## Код

```python
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import find_peaks

def calibrate_gyro_pca(gyro_xyz_calib: np.ndarray) -> np.ndarray:
    """
    Вычисляет вектор проекции W по калибровочному сегменту.
    
    Args:
        gyro_xyz_calib: сегмент гироскопа во время стационарного педалирования,
                        shape (N, 3), рекомендуется последние 60 сек ступени 60 Вт
    Returns:
        W: вектор проекции shape (3,) — фиксировать на всю сессию
    """
    pca = PCA(n_components=1)
    pca.fit(gyro_xyz_calib)
    W = pca.components_[0]   # shape (3,)
    return W


def apply_gyro_projection(gyro_xyz: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Проецирует трёхосевой сигнал на ведущую ось.
    
    Args:
        gyro_xyz: полная запись гироскопа, shape (N, 3)
        W: вектор проекции из calibrate_gyro_pca
    Returns:
        gyro_main: одномерный сигнал, shape (N,)
    """
    gyro_main = gyro_xyz @ W
    return gyro_main


def fix_sign(gyro_main: np.ndarray, fs: int = 148) -> np.ndarray:
    """
    Гарантирует что push-фаза (активная) = положительный пик.
    PCA не гарантирует знак — исправляем по первому значимому пику.
    
    Args:
        gyro_main: проецированный сигнал
        fs: частота дискретизации гироскопа (148 Гц для Delsys Trigno Avanti)
    Returns:
        gyro_main_corrected: сигнал с правильным знаком
    """
    # Смотрим на первые 3 секунды (достаточно для пары оборотов при 60–100 rpm)
    window = gyro_main[:3 * fs]
    first_peak_idx = np.argmax(np.abs(window))
    if gyro_main[first_peak_idx] < 0:
        gyro_main = -gyro_main
    return gyro_main


def validate_pca(gyro_main: np.ndarray, fs: int = 148,
                 cadence_rpm_expected: float = 90.0) -> bool:
    """
    Проверяет качество PCA через автокорреляцию.
    Если каданс ~90 rpm → период ~0.67 сек → лаг ~99 сэмплов при 148 Гц.
    
    Returns:
        True если автокорреляция показывает чёткий периодический паттерн
    """
    segment = gyro_main[:10 * fs]   # первые 10 сек
    autocorr = np.correlate(segment - segment.mean(),
                            segment - segment.mean(), mode='full')
    autocorr = autocorr[len(autocorr)//2:]   # берём только положительные лаги
    
    # Ищем пик в диапазоне возможных каденсов 50–120 rpm
    lag_min = int(60 / 120 * fs)   # 120 rpm → 37 сэмплов
    lag_max = int(60 / 50  * fs)   # 50 rpm  → 178 сэмплов
    
    peak_region = autocorr[lag_min:lag_max]
    peak_val = np.max(peak_region)
    baseline = np.max(autocorr[lag_max:lag_max*2]) if lag_max*2 < len(autocorr) else 0
    
    snr = peak_val / (baseline + 1e-10)
    return snr > 2.0   # пик в 2× выше фона — достаточно


def detect_pedal_phases(gyro_main: np.ndarray, fs: int = 148,
                        min_cadence_rpm: float = 50.0,
                        prominence: float = 0.3) -> dict:
    """
    Детектирует пики push-фазы и возвращает временные метки фаз.
    
    Args:
        gyro_main: одномерный сигнал с правильным знаком
        fs: частота дискретизации
        min_cadence_rpm: минимальный ожидаемый каданс
        prominence: минимальная высота пика (в единицах сигнала после нормировки)
    Returns:
        dict с ключами:
            'peaks': индексы пиков push-фазы
            'push_slices': список slice объектов для push-сегментов
            'pull_slices': список slice объектов для pull-сегментов
            'cadence_rpm': каданс на каждый оборот
    """
    min_distance = int(60 / min_cadence_rpm * fs)   # минимум сэмплов между пиками
    
    peaks, props = find_peaks(gyro_main,
                              distance=min_distance,
                              prominence=prominence)
    
    push_slices, pull_slices, cadences = [], [], []
    
    for i in range(len(peaks) - 1):
        cycle_len = peaks[i+1] - peaks[i]
        half = cycle_len // 4   # push-фаза ≈ половина оборота ≈ ±quarter от пика
        
        push_start = max(0, peaks[i] - half)
        push_end   = min(len(gyro_main), peaks[i] + half)
        pull_start = push_end
        pull_end   = peaks[i+1] - half
        
        if pull_end > pull_start:
            push_slices.append(slice(push_start, push_end))
            pull_slices.append(slice(pull_start, pull_end))
            cadences.append(60 / (cycle_len / fs))
    
    return {
        'peaks': peaks,
        'push_slices': push_slices,
        'pull_slices': pull_slices,
        'cadence_rpm': np.array(cadences),
    }


# ─── Полный пайплайн одной сессии ──────────────────────────────────────────

def process_gyroscope(gyro_xyz: np.ndarray,
                      calib_start_sec: float,
                      calib_end_sec: float,
                      fs: int = 148) -> dict:
    """
    Полный пайплайн от сырого гироскопа до фаз педалирования.

    Args:
        gyro_xyz: полная запись, shape (N, 3)
        calib_start_sec: начало калибровочного сегмента в секундах
                         (рекомендуется: начало ступени 60 Вт + 60 сек)
        calib_end_sec:   конец калибровочного сегмента
                         (рекомендуется: конец ступени 60 Вт - 15 сек)
        fs: частота дискретизации
    Returns:
        dict с W, gyro_main, phases
    """
    calib_idx = slice(int(calib_start_sec * fs), int(calib_end_sec * fs))

    # 1. Калибровка PCA
    W = calibrate_gyro_pca(gyro_xyz[calib_idx])

    # 2. Проекция всей сессии
    gyro_main = apply_gyro_projection(gyro_xyz, W)

    # 3. Коррекция знака
    gyro_main = fix_sign(gyro_main, fs=fs)

    # 4. Валидация
    is_valid = validate_pca(gyro_main, fs=fs)
    if not is_valid:
        raise ValueError("PCA не нашёл чёткого периодического паттерна. "
                         "Проверить качество сигнала гироскопа.")

    # 5. Детекция фаз
    phases = detect_pedal_phases(gyro_main, fs=fs)

    return {'W': W, 'gyro_main': gyro_main, 'phases': phases,
            'pca_valid': is_valid}
```

---

## Параметры калибровки

| Параметр | Значение | Обоснование |
|---|---|---|
| Калибровочный сегмент | Последние 60–90 сек ступени 60 Вт | Стационарное педалирование, мышца уже активна |
| Почему не 30 Вт | На 30 Вт нагрузка почти пассивная, паттерн нестабильный | — |
| Минимальный каданс | 50 rpm | Нижняя граница для тренированных триатлонистов |
| Prominence | 0.3 (после нормировки) | Подобрать на пилотных данных |

---

## Кинематические признаки из фаз (в окне 30 сек)

```python
def kinematics_features(phases: dict, window_slice: slice,
                         fs: int = 148) -> list:
    """
    Извлекает кинематические признаки для скользящего окна.
    """
    # Находим обороты попавшие в окно
    w_start = window_slice.start
    w_end   = window_slice.stop
    
    in_window = [i for i, sl in enumerate(phases['push_slices'])
                 if sl.start >= w_start and sl.stop <= w_end]
    
    if len(in_window) < 3:
        return [np.nan] * 5
    
    cads = phases['cadence_rpm'][in_window]
    push_durs = np.array([(phases['push_slices'][i].stop -
                           phases['push_slices'][i].start) / fs * 1000
                          for i in in_window])
    pull_durs = np.array([(phases['pull_slices'][i].stop -
                           phases['pull_slices'][i].start) / fs * 1000
                          for i in in_window])
    
    cadence       = np.mean(cads)
    push_dur_ms   = np.mean(push_durs)
    pull_dur_ms   = np.mean(pull_durs)
    phase_ratio   = push_dur_ms / (pull_dur_ms + 1e-10)
    cadence_cv    = np.std(cads) / (np.mean(cads) + 1e-10)
    
    return [cadence, push_dur_ms, pull_dur_ms, phase_ratio, cadence_cv]

# Имена признаков:
KINEMATICS_FEATURE_NAMES = [
    'cadence_rpm',       # об/мин — падает при усталости у любителей
    'push_dur_ms',       # длительность push-фазы
    'pull_dur_ms',       # длительность pull-фазы
    'phase_ratio',       # push/pull — меняется при изменении техники
    'cadence_cv',        # коэффициент вариации каданса — маркер нестабильности
]
```

---

## Связанные файлы

- `preprocessing/emg_pipeline.md` — полный пайплайн ЭМГ, использует `phases` из этого файла
- `domain_knowledge.md` раздел 6 — архитектура модели
