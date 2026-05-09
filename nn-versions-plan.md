# План нейросетевых версий для v0101+

Цель: улучшить предсказание времени до лактатного порога через deep learning.

**Текущий baseline:**
- v0011 (ElasticNet/GBM): MAE LT2 = 1.859 мин, MAE LT1 = 2.277 мин
- v0101 (LSTM ensemble, неоптимизированный): MAE LT2 = 6.997 мин ⚠️

---

## v0101 — Оптимизированный LSTM (переделка) ⚠️ ОТМЕНЁН

**Вывод:** LSTM с нахлестующимися окнами принципиально неправилен.
Окна с шагом 5 сек дают 83% нахлест — LSTM видит почти одинаковые
данные на каждом шаге, нечему учиться. MAE ~7 мин (хуже naive baseline).

**Что делать вместо:**
→ TCN (v0102) решает это архитектурно через dilated convolutions.
→ Log-spaced sampling (v0104) решает через выбор независимых точек.

---

## v0101 — Оптимизированный LSTM (переделка) — ИСТОРИЯ

**Проблемы текущего v0101:**
- seq_length=5 окон (55 сек) → слишком мало контекста
- hidden_size=32 → слишком маленькая сеть
- epochs=30 → недообучение
- batch_size=32 → слишком большой батч для точности

**Оптимизация:**
```python
class Config:
    seq_length = 15          # 75 сек информации (в 3 раза больше)
    hidden_size = 128        # в 4 раза больше
    num_layers = 3           # глубже
    epochs = 100             # больше обучения
    batch_size = 16          # меньше батч → точнее градиент
    learning_rate = 0.0005   # медленнее, стабильнее
    dropout = 0.4            # больше регуляризации
    patience = 20            # больше терпения
```

**Архитектура:** LSTM 1/2/3 слоя + GRU ensemble
**Ensemble:** усреднение предсказаний от 3+ моделей

**Ожидание:** MAE улучшится на 1-2 мин (зависит от seq_length значимости)

**Файл:** `scripts/v0101_lstm_temporal.py`

---

## v0102 — TCN (Temporal Convolutional Network)

**Идея:** Вместо рекуррентности → параллельные дилатированные свертки

**Архитектура:**
```python
class TCNBlock(nn.Module):
    # Дилатированные свертки (kernel_size=3)
    # Dilation rates: [1, 2, 4, 8, 16, 32]
    # ReLU activation + Dropout между слоями
    # Residual connections
    # Output: Linear layer для регрессии

class TemporalCNN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        # 5-7 TCN блоков
        # Growing dilation для захвата multi-scale паттернов
```

**Параметры:**
- seq_length = 15-20 (те же 75-100 сек)
- hidden_size = 128
- num_layers = 5-7
- kernel_size = 3
- dropout = 0.4
- epochs = 100

**Преимущества TCN:**
- ✅ Параллельная обработка (не рекуррентная) → быстро на CPU
- ✅ Дилатированные свертки могут ловить долгие зависимости
- ✅ Меньше параметров чем LSTM с той же памятью
- ✅ Стабильнее на коротких последовательностях

**Ожидание:** Может быть лучше LSTM благодаря параллелизму

**Файл:** `scripts/v0102_tcn_temporal.py`

---

## v0103 — Wavelet-CNN-LSTM гибрид 🌊

**Идея:** Деноизирование + Feature extraction + Prediction

**Pipeline:**
```
Входные признаки (X_train: n_windows × n_features)
    ↓
Continuous Wavelet Transform (CWT)
    → Каждый признак → Scalogram (матрица time × scales)
    → Shape: (n_windows, n_features, n_scales_time, n_scales_freq)
    ↓
CNN (2D convolutions на scalograms)
    → kernel_size = 3×3
    → Filters: [64, 128, 256]
    → Batch norm + ReLU + Max pooling
    → Output: (n_windows, 256)
    ↓
LSTM (2-3 слоя)
    → hidden_size = 128
    → Output: (n_windows, hidden)
    ↓
Linear layer → y_pred (скалярное предсказание)
```

**Параметры:**
```python
class WaveletConfig:
    wavelet = 'morlet'                    # или 'mexhat', 'ricker'
    scales = np.arange(1, 100)            # 99 масштабов вейвлета
    cnn_filters = [64, 128, 256]
    lstm_hidden = 128
    lstm_layers = 2
    seq_length = 15
    epochs = 100
    learning_rate = 0.0005
```

**Почему Wavelet-CNN?**
- ✅ Вейвлеты хороши для физиологических сигналов (EMG, NIRS, HRV)
- ✅ CWT дает time-frequency representation → выявляет скрытые паттерны
- ✅ CNN извлекает признаки из scalograms (как из изображений)
- ✅ LSTM на top фичей → финальное предсказание
- ✅ Гибридный подход часто бьет pure LSTM/CNN на time series

**Сложность:** Средняя (новая архитектура, но joblib уже настроен)

**Ожидание:** MAE может улучшиться на 0.5-1.5 мин если вейвлеты поймут физиологию

**Файл:** `scripts/v0103_wavelet_cnn_lstm.py`

---

## v0104 — Attention-Enhanced LSTM

**Идея:** Более длинный контекст + Selective attention на важные окна

**Архитектура:**
```python
class AttentionLSTM(nn.Module):
    # Positional encoding (transformer-style)
    # LSTM: seq_length × features → hidden_size
    # Multi-head self-attention: какие окна важны?
    # Linear layer на attention output
```

**Параметры:**
- seq_length = 30-40 окон (150-200 сек — почти вся тренировка!)
- hidden_size = 128
- num_attention_heads = 4
- lstm_layers = 2
- epochs = 100

**Почему Attention?**
- ✅ Выявляет какие моменты в истории критичны
- ✅ Может решить проблему vanishing gradient в длинных последовательностях
- ✅ Визуализируемо: видим what the model is attending to

**Ожидание:** Если длинные зависимости важны → MAE улучшится

**Файл:** `scripts/v0104_attention_lstm.py`

---

## v0105 — Монотонность как constraint

**Идея:** Добавить знание физики в функцию потерь

**Математика:**
```
Loss = MSE(y_true, y_pred) + λ * monotonic_penalty

где:
monotonic_penalty = sum(max(0, y_pred[t] - y_pred[t+1])**2 for t in range(T-1))
                  = штраф если предсказание растет (должно только падать)

λ (weight) = 1.0 или 10.0 (гиперпараметр)
```

**Архитектура:** Любая (LSTM/TCN/Wavelet), но с модифицированной Loss

**Параметры:**
- seq_length = 15
- Выбираем лучшую архитектуру из v0101-v0104
- monotonic_weight = 1.0 или 10.0 (тестируем)
- epochs = 100

**Почему монотонность?**
- ✅ Мы точно знаем что время до порога только уменьшается
- ✅ Эта информация не используется в v0101-v0104
- ✅ Может убрать "скачки вверх" в предсказаниях (артефакты)
- ✅ Приближает модель к физической реальности

**Ожидание:** MAE может улучшиться на 0.2-0.5 мин благодаря регуляризации

**Файл:** `scripts/v0105_monotonic_lstm.py`

---

## v0106 — Ensemble из всех best моделей

**Идея:** Комбинировать лучшие из v0101-v0105

**Pipeline:**
```
v0101 (opt LSTM) → pred1
v0102 (TCN)      → pred2
v0103 (Wave-CNN) → pred3
v0104 (Attention)→ pred4
v0105 (Monotonic)→ pred5

Ensemble prediction:
  y_pred_ensemble = w1*pred1 + w2*pred2 + ... + w5*pred5
  
где weights тюнируются на validation fold
```

**Параметры:**
- Веса: можно равные (1/5 каждый) или тюнировать
- Калман постпроцессинг как обычно

**Ожидание:** Ensemble часто бьет любую single модель на 0.3-1.0 мин

**Файл:** `scripts/v0106_ensemble.py`

---

## 📊 Сводная таблица

| Версия | Архитектура | seq_len | hidden | complexity | ожидание |
|---|---|---|---|---|---|
| v0101 | LSTM (opt) | 15 | 128 | ⭐⭐ | baseline улучшен |
| v0102 | TCN | 15-20 | 128 | ⭐⭐ | быстро на CPU |
| v0103 | Wavelet-CNN-LSTM | 15 | 128 | ⭐⭐⭐ | физиология |
| v0104 | Attention LSTM | 30-40 | 128 | ⭐⭐⭐ | длинные зависимости |
| v0105 | Monotonic Loss | 15 | 128 | ⭐⭐ | регуляризация |
| v0106 | Ensemble | - | - | ⭐⭐⭐⭐ | best result |

---

## 🔄 Очередность запусков

**Фаза 1 (базовое улучшение):**
1. v0101 (переделка с opt params) ← СЕЙЧАС
2. v0102 (TCN) и v0103 (Wavelet) параллельно
3. Выбираем лучшую из v0101/v0102/v0103

**Фаза 2 (refinement):**
4. v0104 (Attention) если seq_length зависимость важна
5. v0105 (Monotonic) на лучшую модель из фазы 1

**Фаза 3 (final):**
6. v0106 (Ensemble) комбинирует все best models

---

## 📝 Метрики для каждой версии

Для каждой версии сохраняем:
- ✅ `results/v{N}/summary.csv` — все конфиги, MAE raw/kalman
- ✅ `results/v{N}/best_per_set.csv` — лучший per (набор признаков, таргет)
- ✅ `results/v{N}/report.md` — финальный отчет vs v0011
- ✅ `results/v{N}/honest_baselines.csv` — gap анализ
- ✅ Attention weights visualization (для v0104)
- ✅ Wavelet scalograms sample (для v0103)

---

## 🎯 Goal

**v0011 (current best):** MAE LT2 = 1.859 мин

**Target:** MAE LT2 < 2.5 мин (если нейросети не смогут бить линейные модели, хотя бы не хуже)

Если ни одна нейросеть не улучшит:
- Может быть нужна feature engineering
- Или задача чисто линейная и глубокое обучение не поможет
- Или нужны совсем другие архитектуры (Transformer, GraphNN)

---

## 💡 Notes

- joblib уже настроен → все версии будут быстрыми параллельно
- Калман постпроцессинг (σ_obs=150) остается одинаковым
- Честные baselines запускаются для всех версий автоматически
- Если версия медленная → уменьшить epochs/batch_size, но сохранить seq_length
