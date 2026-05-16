"""vv0304 — Stateless LSTM, 6×30 сек, шаг 15 сек + CWT.

Запуск:
    PYTHONPATH=. uv run python scripts/v0304.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0304",
    family="stateless",
    seq_len=6,
    internal_stride_sec=15,
    outer_stride_sec=5,
    use_wavelet=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
