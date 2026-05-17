"""v0321 — Stateless LSTM, 6x15 сек, шаг 15 сек (with_abs).

Зеркало v0303 с with_abs=True (включены абсолютные признаки NIRS/HRV).

Запуск:
    PYTHONPATH=. uv run python scripts/v0321.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0321",
    family="stateless",
    seq_len=6,
    internal_stride_sec=15,
    outer_stride_sec=5,
    use_wavelet=False,
    with_abs=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
