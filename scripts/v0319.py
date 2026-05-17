"""v0319 — Stateless LSTM, 6x30 сек, шаг 30 сек (без перекрытия) (with_abs).

Зеркало v0301 с with_abs=True (включены абсолютные признаки NIRS/HRV).

Запуск:
    PYTHONPATH=. uv run python scripts/v0319.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0319",
    family="stateless",
    seq_len=6,
    internal_stride_sec=30,
    outer_stride_sec=5,
    use_wavelet=False,
    with_abs=True,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
