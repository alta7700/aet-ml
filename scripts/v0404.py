"""v0404 — Medium TCN, seq=60 (5 мин), noabs.

Зеркало v0403 без абсолютных признаков NIRS/HRV.

Запуск:
    PYTHONPATH=. uv run python scripts/v0404.py --target both
"""

from __future__ import annotations

from scripts.v0400_lib import TcnExperimentCfg, run_tcn_experiment


CFG = TcnExperimentCfg(
    name="v0404",
    arch="medium",
    seq_len=60,
    with_abs=False,
    n_channels=32,
)

if __name__ == "__main__":
    run_tcn_experiment(CFG)
