"""v0327 — Stateful LSTM, шаг 15 сек (with_abs).

Зеркало v0309 с with_abs=True (включены абсолютные признаки NIRS/HRV).

Запуск:
    PYTHONPATH=. uv run python scripts/v0327.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0327",
    family="stateful",
    seq_len=None,
    internal_stride_sec=15,
    outer_stride_sec=5,
    use_wavelet=False,
    with_abs=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
