"""v0331 — Attention LSTM, 12x30 сек, шаг 30 сек (with_abs).

Зеркало v0313 с with_abs=True (включены абсолютные признаки NIRS/HRV).

Запуск:
    PYTHONPATH=. uv run python scripts/v0331.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0331",
    family="attention",
    seq_len=12,
    internal_stride_sec=30,
    outer_stride_sec=5,
    use_wavelet=False,
    with_abs=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
