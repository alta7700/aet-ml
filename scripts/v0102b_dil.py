"""v0102b_dil — абляция v0102: только уменьшение дилатаций.

Изменения от v0102:
  dilations: [1, 2, 4, 8, 16] → [1, 2, 4]
  → receptive field 63 → 15, теперь укладывается в seq_length=24.
  seq_length и window_step остаются дефолтными (24 × 30 сек = 12 мин).

Цель: проверить, был ли источник коллапса v0102 в том, что RF (63) сильно
превышал длину последовательности (24), из-за чего свёртки в основном
смотрели в zero-padding.

Все аргументы CLI совпадают с v0102 (см. оригинал). Запуск:
    PYTHONPATH=. uv run python scripts/v0102b_dil.py \\
        --target both --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import v0102_tcn_temporal as m  # noqa: E402

# Папка вывода — отдельная, чтобы не затирать v0102.
m.OUT_DIR = ROOT / "results" / "v0102b_dil"
# Урезаем дилатации; в Config.dilations типовая ссылка — TCNRegressor
# читает их через config.dilations, никакого хардкода в forward нет.
m.Config.dilations = [1, 2, 4]


if __name__ == "__main__":
    m.main()
