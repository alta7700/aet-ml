"""v0401 — Pure TCN, seq=30 (150 сек), with_abs.

Чистый dilated TCN: dilations=[1,2,4,8], kernel=3, RF=31, n_ch=32.
Признаки: EMG / EMG+NIRS / EMG+NIRS+HRV (with_abs).

Запуск:
    PYTHONPATH=. uv run python scripts/v0401.py --target both
"""

from __future__ import annotations

from scripts.v0400_lib import TcnExperimentCfg, run_tcn_experiment


CFG = TcnExperimentCfg(
    name="v0401",
    arch="pure",
    seq_len=30,
    with_abs=True,
    n_channels=32,
)

if __name__ == "__main__":
    run_tcn_experiment(CFG)
