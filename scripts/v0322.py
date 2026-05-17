"""v0322 — Stateless LSTM, 6x15 сек, шаг 15 сек + CWT (with_abs).

Зеркало v0304 с with_abs=True (включены абсолютные признаки NIRS/HRV).

Запуск:
    PYTHONPATH=. uv run python scripts/v0322.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0322",
    family="stateless",
    seq_len=6,
    internal_stride_sec=15,
    outer_stride_sec=5,
    use_wavelet=True,
    with_abs=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
