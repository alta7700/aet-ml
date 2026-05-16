"""vv0303 — Stateless LSTM, 6×30 сек, шаг 15 сек (перекрытие 15).

Запуск:
    PYTHONPATH=. uv run python scripts/v0303.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0303",
    family="stateless",
    seq_len=6,
    internal_stride_sec=15,
    outer_stride_sec=5,
    use_wavelet=False,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
