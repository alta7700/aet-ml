"""v0406 — DWT-TCN, seq=30 (150 сек), noabs.

Зеркало v0405 без абсолютных признаков NIRS/HRV.

Запуск:
    PYTHONPATH=. uv run python scripts/v0406.py --target both
"""

from __future__ import annotations

from scripts.v0400_lib import TcnExperimentCfg, run_tcn_experiment


CFG = TcnExperimentCfg(
    name="v0406",
    arch="dwt",
    seq_len=30,
    with_abs=False,
    n_channels=32,
)

if __name__ == "__main__":
    run_tcn_experiment(CFG)
