"""vv0313 — Attention LSTM, 12×30 сек, шаг 30 сек (контекст 360 сек).

Запуск:
    PYTHONPATH=. uv run python scripts/v0313.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0313",
    family="attention",
    seq_len=12,
    internal_stride_sec=30,
    outer_stride_sec=5,
    use_wavelet=False,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
