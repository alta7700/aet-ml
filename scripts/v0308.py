"""vv0308 — Stateful LSTM, окно 30 сек, шаг 30 сек + CWT.

Запуск:
    PYTHONPATH=. uv run python scripts/v0308.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0308",
    family="stateful",
    seq_len=None,
    internal_stride_sec=30,
    outer_stride_sec=5,
    use_wavelet=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
