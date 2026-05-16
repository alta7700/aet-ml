"""vv0316 — Attention LSTM, 20×30 сек, шаг 30 сек + CWT.

Запуск:
    PYTHONPATH=. uv run python scripts/v0316.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0316",
    family="attention",
    seq_len=20,
    internal_stride_sec=30,
    outer_stride_sec=5,
    use_wavelet=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
