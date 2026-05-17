"""v0333 — Attention LSTM, 20×30 сек, шаг 30 сек, контекст 600 сек (with_abs).

Зеркало v0315 с with_abs=True (315+18=333).
Внимание: seq_len=20 может давать пустые результаты из-за слишком длинного контекста.

Запуск:
    PYTHONPATH=. uv run python scripts/v0333.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0333",
    family="attention",
    seq_len=20,
    internal_stride_sec=30,
    outer_stride_sec=5,
    use_wavelet=False,
    with_abs=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
