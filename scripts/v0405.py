"""v0405 — DWT-TCN, seq=30 (150 сек), with_abs.

Haar-DWT вдоль временной оси → две параллельные TCN-ветви (approx + detail).
dilations=[1,2,4], kernel=3, RF=15 на ветвь, n_ch=32 (16 на ветвь).
Признаки: EMG / EMG+NIRS / EMG+NIRS+HRV (with_abs).

Запуск:
    PYTHONPATH=. uv run python scripts/v0405.py --target both
"""

from __future__ import annotations

from scripts.v0400_lib import TcnExperimentCfg, run_tcn_experiment


CFG = TcnExperimentCfg(
    name="v0405",
    arch="dwt",
    seq_len=30,
    with_abs=True,
    n_channels=32,
)

if __name__ == "__main__":
    run_tcn_experiment(CFG)
