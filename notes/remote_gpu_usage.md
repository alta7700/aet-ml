# Работа с удалённым GPU-сервером

## Подключение

```bash
ssh ml   # алиас из ~/.ssh/config → 185.182.108.135, user=root
```

Структура на сервере:
```
~/scripts/          # все скрипты моделей (rsync из локального scripts/)
~/logs/             # логи запусков: v0011_lt1.log, v0101_lt2.log, ...
~/results/          # результаты: results/vXXXX/ypred_lt1_EMG.npy, ...
~/диссер/dataset/   # датасет (parquet-файлы) — не изменяется на сервере
```

---

## Синхронизация скриптов на сервер

```bash
# Один файл
rsync -av scripts/v0011_modality_ablation.py ml:~/scripts/

# Все скрипты разом
rsync -av scripts/v0*.py ml:~/scripts/
```

**Важно:** все v0101–v0107 импортируют `prepare_data` и `get_feature_cols` из `v0011_modality_ablation.py`. После любых правок в v0011 — синхронизировать его первым.

---

## Запуск моделей

### Общий шаблон

```bash
nohup python3 -u /root/scripts/<script>.py --target <lt1|lt2> \
    > ~/logs/<version>_<target>.log 2>&1 &
```

Флаг `-u` — unbuffered stdout, чтобы логи писались сразу, а не буферизовались.

### Порядок запуска (проверено)

**Запускать сначала все lt1, потом все lt2** — так GPU не делит память между двумя разными целевыми переменными.

**Нейросети запускать с паузой 45 секунд** между скриптами — иначе несколько CUDA-контекстов инициализируются одновременно и возникает deadlock:

```bash
# v0011 — CPU-only (sklearn), запускается сразу
nohup python3 -u /root/scripts/v0011_modality_ablation.py --target lt1 \
    > ~/logs/v0011_lt1.log 2>&1 &

# Нейросети — с паузой 45с
for script in v0101_lstm_temporal v0102_tcn_temporal v0103_wavelet_cnn \
              v0104_attention_lstm v0105_monotonic_tcn \
              v0106a_wavelet_attention v0106b_wavelet_tcn \
              v0106c_wavelet_attention_mono v0107_ensemble; do
  sleep 45
  ver=$(echo $script | grep -oP "v[0-9]+[a-z]*")
  nohup python3 -u /root/scripts/${script}.py --target lt1 \
      > ~/logs/${ver}_lt1.log 2>&1 &
  echo "${ver} lt1 PID=$!"
done
```

### Таблица скриптов

| Версия  | Скрипт                              | Тип       |
|---------|-------------------------------------|-----------|
| v0011   | v0011_modality_ablation.py          | CPU (sklearn zoo) |
| v0101   | v0101_lstm_temporal.py              | GPU (LSTM ensemble) |
| v0102   | v0102_tcn_temporal.py               | GPU (TCN) |
| v0103   | v0103_wavelet_cnn.py                | GPU (Wavelet CNN) |
| v0104   | v0104_attention_lstm.py             | GPU (Attention LSTM) |
| v0105   | v0105_monotonic_tcn.py              | GPU (Monotonic TCN) |
| v0106a  | v0106a_wavelet_attention.py         | GPU |
| v0106b  | v0106b_wavelet_tcn.py               | GPU |
| v0106c  | v0106c_wavelet_attention_mono.py    | GPU |
| v0107   | v0107_ensemble.py                   | GPU (ensemble) |

---

## GPU contention — важное ограничение

**Проблема:** при запуске 18–20 нейросетевых процессов одновременно (10 версий × 2 таргета) они конкурируют за GPU. CUDA сериализует операции разных процессов → каждый работает в 10–15× медленнее, суммарное время хуже, чем последовательно.

**Решение:** запускать не более 5–6 нейросетевых процессов одновременно. Эмпирически:
- 5–6 процессов → GPU ~90–95%, VRAM ~2.2GB → приемлемо
- 10 процессов → GPU 98–99%, VRAM ~3.9GB → всё ещё работает, но медленнее
- 18–20 процессов → GPU 99%, VRAM ~7.8GB → прогресса почти нет

**Рабочая стратегия:** запустить первую волну (v0011 + 4–5 нейросетей), мониторить, добавлять по одной модели по мере завершения.

---

## Мониторинг

### Сводная таблица (запускать с локальной машины)

```bash
ssh ml '
for model in v0011 v0101 v0102 v0103 v0104 v0105 v0106a v0106b v0106c v0107; do
  for tgt in lt1 lt2; do
    logfile=~/logs/${model}_${tgt}.log
    if [ -f "$logfile" ]; then
      last=$(tail -1 "$logfile")
      size=$(wc -l < "$logfile")
      echo "${model}|${tgt}|${size}|${last}"
    else
      echo "${model}|${tgt}|—|НЕТ ЛОГА"
    fi
  done
done
echo "==="
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used \
    --format=csv,noheader,nounits
echo "==="
find ~/results -name "ypred_*.npy" | wc -l
echo "==="
ps aux | grep -E "v0[01][01][0-9]" | grep -v grep | wc -l
'
```

### Только GPU

```bash
ssh ml 'nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used \
    --format=csv,noheader,nounits'
```

### Хвост лога конкретной модели

```bash
ssh ml 'tail -20 ~/logs/v0101_lt1.log'
```

### Проверка завершения

Каждый скрипт при завершении пишет строку `✅ Сохранено:` и сохраняет `ypred_*.npy` / `ytrue_*.npy`. Признак завершения версии:

```bash
ssh ml 'grep -l "Сохранено" ~/logs/v01*_lt1.log 2>/dev/null'
```

### Подсчёт npy-файлов

Ожидаемое число при полном завершении всех lt1 (или lt2):
- 4 фичесета × 2 варианта (with_abs + noabs) × 10 версий = **80 файлов** (ypred) + 80 (ytrue)

```bash
ssh ml 'find ~/results -name "ypred_*.npy" | wc -l'
```

---

## Управление процессами

### Убить все модели

```bash
ssh ml 'pkill -9 -f "v0[01][01][0-9]"; sleep 1; \
    echo "Осталось: $(ps aux | grep -E "v0[01][01][0-9]" | grep -v grep | wc -l)"'
```

### Убить конкретную версию

```bash
ssh ml 'pkill -f "v0104_attention_lstm"'
```

### Очистить результаты и логи (полный сброс)

```bash
ssh ml 'rm -rf ~/results/v0* && rm -f ~/logs/v0*.log && echo "Очищено"'
```

---

## Структура результатов

```
~/results/
  v0011/
    ypred_lt1_EMG.npy          # предсказания, фичесет EMG, with_abs
    ytrue_lt1_EMG.npy
    ypred_lt1_NIRS.npy
    ytrue_lt1_NIRS.npy
    ypred_lt1_EMG_NIRS.npy
    ypred_lt1_EMG_NIRS_HRV.npy
    noabs/
      ypred_lt1_EMG.npy        # вариант без абсолютных признаков
      ...
    summary.csv                # сводная таблица метрик версии
  v0101/
    ...
```

Соглашение об именах: `ypred_{target}_{fset_tag}.npy`, где `fset_tag = fset.replace("+", "_")`.

---

## Агрегация результатов после завершения

```bash
# Синхронизировать результаты на локальную машину (атомарно, только завершённые версии)
rsync -av --no-partial ml:~/results/v0011/ /Users/tascan/Desktop/диссер/results/v0011/

# Рассчитать сводный CSV по всем версиям
python3 scripts/aggregate_results.py
# → results/summary_all_versions.csv
```

---

## Типичные проблемы

| Проблема | Диагноз | Решение |
|----------|---------|---------|
| SSH exit 255 | pkill вернул ненулевой код | Использовать `kill $(pgrep ...)` |
| Логи не обновляются | Буферизация stdout | Флаг `-u` в python3 |
| GPU 99%, прогресса нет | Слишком много процессов | Уменьшить до 5–6 нейросетей |
| CUDA deadlock при старте | Одновременная инициализация | Пауза 45с между запусками |
| Сервер перезагрузился | Все процессы убиты | Проверить npy-файлы, перезапустить незавершённые |
