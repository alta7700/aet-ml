"""vv0315 — Attention LSTM, 20×30 сек, шаг 30 сек (контекст 600 сек).

Запуск:
    PYTHONPATH=. uv run python scripts/v0315.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0315",
    family="attention",
    seq_len=20,
    internal_stride_sec=30,
    outer_stride_sec=5,
    use_wavelet=False,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
