"""v0407 — WaveNet-TCN, seq=30 (150 сек), with_abs.

WaveNet-стиль: kernel=2, gated activation (tanh·σ), skip-connections.
dilations=[1,2,4,8,16], residual=32, skip=64, RF=32.
Признаки: EMG / EMG+NIRS / EMG+NIRS+HRV (with_abs).

Запуск:
    PYTHONPATH=. uv run python scripts/v0407.py --target both
"""

from __future__ import annotations

from scripts.v0400_lib import TcnExperimentCfg, run_tcn_experiment


CFG = TcnExperimentCfg(
    name="v0407",
    arch="wavenet",
    seq_len=30,
    with_abs=True,
    n_channels=32,
)

if __name__ == "__main__":
    run_tcn_experiment(CFG)
