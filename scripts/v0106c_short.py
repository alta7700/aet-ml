"""v0106c_short — абляция v0106c (Wavelet-Attention с моно-штрафом):
короткий контекст.

Изменение от v0106c: seq_length 12 → 6 (3 мин вместо 6 мин).
mono_weight оставлен дефолтным; если после правки контекста std-ratio
по-прежнему сильно проседает (раньше было 0,32 — заметно ниже остальных
wavelet) — это значит, что моно-штраф нужно отдельно подкрутить.

Запуск:
    PYTHONPATH=. uv run python scripts/v0106c_short.py \\
        --target both --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import v0106c_wavelet_attention_mono as m  # noqa: E402

m.OUT_DIR = ROOT / "results" / "v0106c_short"
m.Config.seq_length = 6


if __name__ == "__main__":
    m.main()
