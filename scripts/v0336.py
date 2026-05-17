"""v0336 — Attention LSTM, 12×30 сек, шаг 15 сек + CWT (with_abs).

Зеркало v0318 с with_abs=True (318+18=336).

Запуск:
    PYTHONPATH=. uv run python scripts/v0336.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0336",
    family="attention",
    seq_len=12,
    internal_stride_sec=15,
    outer_stride_sec=5,
    use_wavelet=True,
    with_abs=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
