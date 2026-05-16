"""v0106a_short — абляция v0106a (Wavelet-Attention LSTM): короткий контекст.

Изменение от v0106a: seq_length 12 → 6 (3 мин вместо 6 мин).
Цель: та же, что у v0103b_short — проверить чувствительность wavelet-моделей
к длине контекста на N=18 LOSO.

Wavelet-Attention имеет на голове LSTM, который по LSTM-абляции лучше всего
работает на 75 сек / 3 мин контексте. Длинный контекст 6 мин для него
заведомо избыточен.

Все остальные параметры (window_step=6, scales=[1,2,4,8,16], wavelet=morl)
оставлены исходными. CLI у v0106a не принимает --seq-len, поэтому
seq_length меняется через monkey-patch Config.

Запуск:
    PYTHONPATH=. uv run python scripts/v0106a_short.py \\
        --target both --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import v0106a_wavelet_attention as m  # noqa: E402

m.OUT_DIR = ROOT / "results" / "v0106a_short"
m.Config.seq_length = 6


if __name__ == "__main__":
    m.main()
