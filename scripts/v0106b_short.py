"""v0106b_short — абляция v0106b (Wavelet-TCN): короткий контекст + RF↓.

Изменения от v0106b:
  seq_length: 12 → 6                       (3 мин вместо 6 мин)
  dilations:  [1,2,4,8,16] → [1,2,4]       (RF 63 → 15, влезает в seq=6)

Цель: применить к Wavelet-TCN ту же связку, что сработала на чистом TCN
(v0102b_short): короткий контекст + малые дилатации. Без обеих правок
свёртки смотрели в zero-padding, как в v0102.

Запуск:
    PYTHONPATH=. uv run python scripts/v0106b_short.py \\
        --target both --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import v0106b_wavelet_tcn as m  # noqa: E402

m.OUT_DIR = ROOT / "results" / "v0106b_short"
m.Config.seq_length = 6
m.Config.dilations  = [1, 2, 4]


if __name__ == "__main__":
    m.main()
