"""v0102b_both — абляция v0102: оба фикса сразу.

Изменения от v0102:
  dilations:  [1, 2, 4, 8, 16] → [1, 2, 4]   (RF 63 → 15)
  seq_length: 24 → 64                         (контекст 12 → 32 мин)
  → RF комфортно укладывается в окно, и контекст шире, чем у dil-only.

Цель: проверить, даёт ли совместный фикс что-то сверх dil-only или
seq-only. Если результат лучше обоих — оба фактора независимо помогают;
если совпадает с лучшим из них — достаточно одного фактора.

Запуск:
    PYTHONPATH=. uv run python scripts/v0102b_both.py \\
        --target both --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _force_cli_arg(name: str, value: str) -> None:
    if name in sys.argv:
        i = sys.argv.index(name)
        sys.argv[i + 1 : i + 2] = [value]
    else:
        sys.argv += [name, value]


_force_cli_arg("--seq-len", "64")

from scripts import v0102_tcn_temporal as m  # noqa: E402

m.OUT_DIR = ROOT / "results" / "v0102b_both"
m.Config.dilations = [1, 2, 4]


if __name__ == "__main__":
    m.main()
