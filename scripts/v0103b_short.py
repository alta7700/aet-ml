"""v0103b_short — абляция v0103 (Wavelet-CNN): короткий контекст.

Изменение от v0103: seq_length 12 → 6 (3 мин вместо 6 мин).
Цель: проверить, повторит ли Wavelet-CNN ту же закономерность, что
наблюдалась у LSTM и TCN — длинный контекст вреден на N=18 LOSO.

Все остальные параметры (window_step=6, scales=[1,2,4,8,16], wavelet=morl)
оставлены исходными.

Запуск:
    PYTHONPATH=. uv run python scripts/v0103b_short.py \\
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


# v0103 явно перезаписывает Config из CLI; подсовываем seq_length=6.
_force_cli_arg("--seq-len", "6")

from scripts import v0103_wavelet_cnn as m  # noqa: E402

m.OUT_DIR = ROOT / "results" / "v0103b_short"


if __name__ == "__main__":
    m.main()
