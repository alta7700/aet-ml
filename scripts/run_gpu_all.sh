#!/usr/bin/env bash
# Запуск всех нейросетевых скриптов на GPU-сервере.
# Каждый скрипт пишет лог в ~/logs/v01XX.log
# Запускать: bash ~/scripts/run_gpu_all.sh

set -e
mkdir -p ~/logs ~/results

export PYTHONUNBUFFERED=1
SCRIPTS=~/scripts

run() {
    local name=$1
    local script=$2
    shift 2
    echo "[$(date '+%H:%M:%S')] Запуск $name..."
    python3 -u "$SCRIPTS/$script" "$@" > ~/logs/${name}.log 2>&1
    echo "[$(date '+%H:%M:%S')] $name ГОТОВО"
}

echo "=============================="
echo " GPU-прогон нейросетей v0102–v0107"
echo " $(date)"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

run v0102 v0102_tcn_temporal.py
run v0103 v0103_wavelet_cnn.py
run v0104 v0104_attention_lstm.py
run v0105 v0105_monotonic_tcn.py
run v0106a v0106a_wavelet_attention.py
run v0106b v0106b_wavelet_tcn.py
run v0106c v0106c_wavelet_attention_mono.py
run v0107 v0107_ensemble.py

echo ""
echo "=============================="
echo " Все скрипты завершены: $(date)"
echo "=============================="
