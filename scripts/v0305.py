"""vv0305 — Stateless LSTM, 6×30 сек, шаг 5 сек (перекрытие 25).

Запуск:
    PYTHONPATH=. uv run python scripts/v0305.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0305",
    family="stateless",
    seq_len=6,
    internal_stride_sec=5,
    outer_stride_sec=5,
    use_wavelet=False,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
