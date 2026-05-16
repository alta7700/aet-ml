"""v0102b_short — абляция v0102: короткие дилатации + короткий контекст.

Изменения от v0102:
  dilations:  [1, 2, 4, 8, 16] → [1, 2, 4]   (RF 63 → 15)
  seq_length: 24 → 15                         (контекст 12 → 7,5 мин)
  → RF (15) ровно совпадает с длиной окна; контекст сопоставим с v0101
     (75 сек) масштабирован в 6 раз через прореживание (15 × 30 сек).

Цель: отделить эффект «уменьшение RF» от эффекта «уменьшение контекста».
Если v0102b_short даст результат лучше v0102b_dil (тот же RF, но контекст
12 мин) — значит длинный контекст сам по себе вредил. Если хуже или ≈
— дело именно в архитектурном RF, а длина окна второстепенна.

Запуск:
    PYTHONPATH=. uv run python scripts/v0102b_short.py \\
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


_force_cli_arg("--seq-len", "15")

from scripts import v0102_tcn_temporal as m  # noqa: E402

m.OUT_DIR = ROOT / "results" / "v0102b_short"
m.Config.dilations = [1, 2, 4]


if __name__ == "__main__":
    m.main()
