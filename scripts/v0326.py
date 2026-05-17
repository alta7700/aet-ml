"""v0326 — Stateful LSTM, шаг 30 сек + CWT (with_abs).

Зеркало v0308 с with_abs=True (включены абсолютные признаки NIRS/HRV).

Запуск:
    PYTHONPATH=. uv run python scripts/v0326.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0326",
    family="stateful",
    seq_len=None,
    internal_stride_sec=30,
    outer_stride_sec=5,
    use_wavelet=True,
    with_abs=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
