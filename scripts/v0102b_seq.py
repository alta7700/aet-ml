"""v0102b_seq — абляция v0102: только удлинение последовательности.

Изменения от v0102:
  seq_length: 24 → 64 шагов × 30 сек = 32 мин контекста
  → теперь RF (63) с дилатациями [1,2,4,8,16] укладывается в окно (64).
  dilations остаются исходными.

Цель: проверить, лечит ли проблему RF/seq именно расширение окна (без
изменения архитектуры).

Все остальные аргументы CLI идентичны v0102 (см. оригинал); --seq-len
жёстко устанавливается этой обёрткой в 64. Запуск:
    PYTHONPATH=. uv run python scripts/v0102b_seq.py \\
        --target both --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _force_cli_arg(name: str, value: str) -> None:
    """Перетирает значение --name в sys.argv (или добавляет, если нет)."""
    if name in sys.argv:
        i = sys.argv.index(name)
        sys.argv[i + 1 : i + 2] = [value]
    else:
        sys.argv += [name, value]


# main() в v0102 явно перезаписывает Config из CLI (config.seq_length =
# args.seq_len), поэтому подсовываем seq_length через sys.argv.
_force_cli_arg("--seq-len", "64")

from scripts import v0102_tcn_temporal as m  # noqa: E402

m.OUT_DIR = ROOT / "results" / "v0102b_seq"


if __name__ == "__main__":
    m.main()
