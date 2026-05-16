"""vv0309 — Stateful LSTM, окно 30 сек, шаг 15 сек.

Запуск:
    PYTHONPATH=. uv run python scripts/v0309.py --target both
"""

from __future__ import annotations

from scripts.v0300_lib import ExperimentCfg, run_experiment


CFG = ExperimentCfg(
    name="v0309",
    family="stateful",
    seq_len=None,
    internal_stride_sec=15,
    outer_stride_sec=5,
    use_wavelet=False,
    chunk_size=10,
)


if __name__ == "__main__":
    run_experiment(CFG)
