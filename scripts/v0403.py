"""v0403 — Medium TCN, seq=60 (5 мин), with_abs.

Расширенный контекст: dilations=[1,2,4,8,16], kernel=3, RF=63, n_ch=32.
Признаки: EMG / EMG+NIRS / EMG+NIRS+HRV (with_abs).

Запуск:
    PYTHONPATH=. uv run python scripts/v0403.py --target both
"""

from __future__ import annotations

from scripts.v0400_lib import TcnExperimentCfg, run_tcn_experiment


CFG = TcnExperimentCfg(
    name="v0403",
    arch="medium",
    seq_len=60,
    with_abs=True,
    n_channels=32,
)

if __name__ == "__main__":
    run_tcn_experiment(CFG)
