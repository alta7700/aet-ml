"""vv0301 — Stateless LSTM, 6×30 сек, шаг 30 сек (без перекрытия).

Запуск:
    PYTHONPATH=. uv run python scripts/v0301.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0301",
    family="stateless",
    seq_len=6,
    internal_stride_sec=30,
    outer_stride_sec=5,
    use_wavelet=False,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
