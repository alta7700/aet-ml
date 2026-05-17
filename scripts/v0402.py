"""v0402 — Pure TCN, seq=30 (150 сек), noabs.

Зеркало v0401 без абсолютных признаков NIRS/HRV.

Запуск:
    PYTHONPATH=. uv run python scripts/v0402.py --target both
"""

from __future__ import annotations

from scripts.v0400_lib import TcnExperimentCfg, run_tcn_experiment


CFG = TcnExperimentCfg(
    name="v0402",
    arch="pure",
    seq_len=30,
    with_abs=False,
    n_channels=32,
)

if __name__ == "__main__":
    run_tcn_experiment(CFG)
