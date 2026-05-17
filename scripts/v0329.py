"""v0329 — Stateful LSTM, шаг 5 сек (with_abs).

Зеркало v0311 с with_abs=True (включены абсолютные признаки NIRS/HRV).

Запуск:
    PYTHONPATH=. uv run python scripts/v0329.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0329",
    family="stateful",
    seq_len=None,
    internal_stride_sec=5,
    outer_stride_sec=5,
    use_wavelet=False,
    with_abs=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
