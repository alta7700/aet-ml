# new_arch — унифицированный training pipeline

Самодостаточный модуль для обучения моделей регрессии `time_to_lt_sec` (порог LT1/LT2)
на смешанных модальностях (EMG + NIRS + HRV). Три семейства моделей,
единая schema артефактов, LOSO-кросс-валидация.

## Состав

- **Lin** — 29 классических архитектур (Ridge, Huber, ElasticNet, GBM, SVR).
- **LSTM** — 16 архитектур: stateless / stateful / attention × {6×30, 6×15, 6×5, 12×30, 12×15} × {none, CWT}.
- **TCN** — 4 архитектуры: PureTCN, MediumTCN, DwtTCN, WaveNetTCN, все **causal** (без утечки будущего).

Каждый запуск даёт стандартизированный набор артефактов:
- `models.csv` (общий для архитектуры) — метадата всех конкретных model configurations.
- `predictions_{model_id}.parquet` — long-format: одна строка = одно предсказание.
- `history.csv` — train loss/MAE per (fold, epoch). Только для NN.
- `model_{model_id}_epoch-{NN}.{pt|joblib}` — checkpoint всех LOSO-folds на одной эпохе.

Подробности schema — в [common_lib.py](common_lib.py).

## Структура

```
new_arch/
├── pyproject.toml             # uv-проект, изолированное окружение
├── README.md
├── common_lib.py              # schemas, save helpers, naming, validation
├── features.py                # prepare_data, get_feature_cols, EXCLUDE_ABS
├── kalman.py                  # kalman_smooth (постфильтр)
├── training_utils.py          # CwtCache, prepare_X_for_fold, get_device
├── architectures.py           # LINEAR_ARCHS, LSTM_ARCHS, TCN_ARCHS
├── linear_runner.py           # sklearn + LOSO + --grid-all batch mode
├── lstm_runner.py             # stateless + attention LSTM
├── lstm_stateful_runner.py    # stateful LSTM с TBPTT (только CPU)
├── tcn_runner.py              # causal TCN (Pure / Medium / DWT / WaveNet)
├── _smoke_test.py             # юнит-тесты common_lib
├── models/                    # nn.Module: lstm.py, tcn.py
├── dataset_pipeline/          # самодостаточная копия пайплайна сборки датасета
├── dataset/                   # parquet'ы + CWT cache (не в git)
├── orchestrator/              # GPU-aware orchestrator для massive запусков
└── results/                   # артефакты экспериментов (не в git)
    └── {architecture_id}/
        ├── models.csv
        └── {model_id}/
            ├── history.csv                  # только NN
            ├── model_{...}_epoch-{NN}.pt    # все folds в одном файле
            └── predictions_{model_id}.parquet
```

## Установка окружения

Текущий каталог — отдельный uv-проект. Зависимости устанавливаются изолированно.

```bash
# Только Lin (без torch)
uv sync

# С torch (Linux + CUDA / macOS + MPS — один общий wheel из PyPI, последний)
uv sync --extra torch
```

На сервере с CUDA: PyTorch ≥ 2.7 backward-совместим с CUDA 13 через рантайм-обёртки,
дополнительный `--index-url` не требуется. На macOS тот же `--extra torch` ставит
билд с поддержкой MPS.

## Подготовка датасета

Финальные parquet'ы лежат в [dataset/](dataset/) (не в git — слишком крупные):

- `merged_features_ml.parquet` — основной (windows × features + targets, ~4000 строк, 18 subjects)
- `session_params.parquet` — per-subject baseline'ы (NIRS, HRV baseline)
- `cwt_cache.npz` — precomputed CWT для EMG (для `wavelet_mode=cwt` LSTM-моделей)
- остальное — промежуточные таблицы pipeline'а

Регенерация датасета (если изменились исходные субъекты):
```bash
uv run python dataset_pipeline/orchestrator.py
```

## Запуск отдельных раннеров

```bash
# Linear: одна модель
uv run python linear_runner.py --architecture Lin1 --target lt1 --feature-set EMG

# Linear: ВЕСЬ декартов набор (29 архитектур × 2 target × 5 fset × 2 abs = 580) в одном процессе
uv run python linear_runner.py --grid-all --n-jobs 8

# LSTM (stateless / attention)
uv run python lstm_runner.py --architecture LSTM1 --target lt1 --feature-set EMG+NIRS+HRV
uv run python lstm_runner.py --architecture LSTM13 --target lt2 --feature-set EMG+NIRS+HRV --no-abs

# LSTM stateful (только CPU — MPS не поддерживает LSTM с пробросом state)
uv run python lstm_stateful_runner.py --architecture LSTM7 --target lt1 --feature-set EMG+NIRS+HRV

# TCN (Causal Pure / DWT / WaveNet)
uv run python tcn_runner.py --architecture TCN1 --target lt1 --feature-set EMG+NIRS+HRV
uv run python tcn_runner.py --architecture TCN3 --target lt1 --feature-set EMG+NIRS+HRV  # forced_wavelet=dwt
```

Параметры runner'ов:
- `--target {lt1,lt2}` — какой порог предсказываем.
- `--feature-set {EMG,NIRS,HRV,EMG+NIRS,EMG+NIRS+HRV}` — набор признаков.
- `--with-abs` / `--no-abs` — включать ли абсолютные NIRS/HRV (по умолчанию `--with-abs`).
- `--wavelet-mode {none,cwt,dwt,...}` — для NN, по умолчанию берётся из `arch.forced_wavelet_mode`.
- `--max-epochs N` — переопределить число эпох (smoke-режим).
- `--checkpoint-every N` — частота промежуточных чекпоинтов.

## Запуск через orchestrator (на сервере с CUDA)

Для массового перебора всех архитектур × конфигов есть GPU-aware orchestrator.
Использует NVML напрямую (`nvidia-ml-py`), управляет очередью задач с двумя порогами:
GPU utilization < 85% **и** GPU memory < 80%.

```bash
# 1. Сгенерировать список задач (jobs.csv)
uv run python orchestrator/gen_jobs.py            # все: 1 batch Lin + 240 NN = 241 jobs
uv run python orchestrator/gen_jobs.py --gpu-only # только GPU-задачи (без Lin и CPU-only LSTM) = 168 jobs

# 2. Запустить orchestrator
uv run python orchestrator/run.py
```

Полный набор параметров orchestrator'а:
```
--jobs PATH                  путь к jobs.csv  (по умолч. orchestrator/jobs.csv)
--state PATH                 путь к process.csv  (state-file для restart)
--log-dir PATH               куда писать логи каждой job
--gpu-util-threshold INT     порог GPU util %, по умолч. 85
--gpu-mem-threshold INT      порог GPU mem %, по умолч. 75
--host-mem-threshold INT     порог host RAM %, по умолч. 50
--max-running INT            максимум одновременно running job'ов, по умолч. 5
--poll-interval INT          секунд между опросами GPU, по умолч. 10
--gpu-index INT              номер GPU (по умолч. 0)
```

Свойства orchestrator'а:
- Запускает задачи **в отдельных process group** (`start_new_session=True`) — orchestrator
  можно убить Ctrl-C, запущенные процессы продолжат работать.
- На рестарте подхватывает `process.csv`: для running записей проверяет `os.kill(pid, 0)`,
  мёртвые помечает `done`/`failed`/`died` по содержимому `.exit`-файла.
- Запускает по одной задаче за тик — позволяет метрикам GPU обновиться перед следующим запуском.
- `process.csv` пишется атомарно (через .tmp + rename).
- Для concurrent-safe записи в общий `models.csv` используется `fcntl.flock`.

## Артефакты и schema

### `results/{architecture_id}/models.csv`
Общая таблица для архитектуры (одна строка на model_id).
Поля: `architecture_id, model_id, family, target, feature_set, with_abs,
wavelet_mode, window_size_sec, sequence_length, stride_sec, sample_stride_sec,
model_name, full_model_name, hyperparams_json`.

### `results/{architecture_id}/{model_id}/predictions_{model_id}.parquet`
Long-format: одна строка = одно предсказание модели.
Поля: `model_id, fold_id, subject_id, epoch, window_size_sec, sequence_length,
stride_sec, sample_stride_sec, sample_index, sample_start_sec, sample_end_sec,
y_true, y_pred`.

Для NN модели — N эпох × M samples per fold. Для Lin — `epoch=0`.

### `results/{architecture_id}/{model_id}/history.csv` (только NN)
Train history per (fold, epoch). Поля: `model_id, architecture_id, fold_id,
epoch, train_loss, val_loss, train_mae, val_mae, lr`.

### Checkpoints
- `model_{model_id}_epoch-{NN}.pt` (NN) — `torch.save(dict{fold_id: state_dict})`.
- `model_{model_id}_epoch-000.joblib` (Lin) — `joblib.dump(dict{fold_id: pipeline})`.

Загрузка:
```python
import torch
states = torch.load("model_LSTM1_xxx_epoch-040.pt", weights_only=True, map_location="cpu")
# states["loso_subject_S001"] → state_dict

import joblib
pipelines = joblib.load("model_Lin1_xxx_epoch-000.joblib")
# pipelines["loso_subject_S001"] → {"imputer": ..., "scaler": ..., "model": ...}
```

## Дизайн-решения

- **LOSO как единственный механизм оценки** — нет внутри-fold val-split. Раннеры
  обучают N эпох до конца, на каждых `checkpoint_every_epochs` сохраняют snapshot
  (state_dict + predictions на test). Выбор "лучшей эпохи" — на этапе analysis pipeline.
- **`model_id = {architecture_id}_{hash8}`**, где `hash8` детерминирован от
  `(hyperparams, target, feature_set, with_abs, wavelet_mode)`. Один и тот же конфиг
  всегда даёт одинаковый `model_id`.
- **`forced_wavelet_mode`** в `ArchitectureSpec` — если архитектура конструктивно
  использует вейвлет (например, DwtTCN), wavelet_mode фиксируется на уровне арх.
- **Causal TCN** — переписаны на причинные свёртки (`F.pad` слева), нет утечки
  будущих значений внутри окна.
- **Stateful LSTM на CPU** — MPS-бэкенд не поддерживает LSTM с пробросом `(h, c)`.

## Smoke-тесты

```bash
uv run python _smoke_test.py
```

12 проверок: детерминизм `model_id`, валидация predictions schema, upsert `models.csv`
с fcntl-lock, корректность checkpoint'ов для torch/sklearn.
