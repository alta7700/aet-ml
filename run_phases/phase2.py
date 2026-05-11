"""Фаза 2 — обучение всех версий моделей.

Топология запуска:

  merged_features_ml.parquet
          │
          ├─── ПАРАЛЛЕЛЬНО ─────────────────────────────────────────────────────┐
          │  v0001  ElasticNet                                                   │
          │  v0002  ElasticNet + Kalman-пост                                     │
          │  v0004  Ridge / Huber (baseline)                                     │
          │  v0005  SHAP + Conformal (Ridge/Huber)                               │
          │  v0006  Stage-0 нормировка                                           │
          │  v0007  Зоопарк × 7 стратегий вариабельности                        │
          │  v0008  Зоопарк × 8 комбинаций SampEn                               │
          └─────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
          v0009  Kalman-постобработка поверх v0008-best
                  │
                  ▼
          v0010  Расширенная сетка Kalman (граница применимости)

          v0102  TCN (базовый)
          v0103  Wavelet-CNN
          v0104  Attention-LSTM
          v0105  TCN + Monotonic loss
          v0106a Wavelet + Attention-LSTM (гибрид)
          v0106b Wavelet → TCN (лучший результат)
          v0106c Wavelet + Attention + Monotonic
          v0107  Ensemble: WavTCN + ElasticNet

Каждый скрипт самодостаточен и пишет в results/v{NNNN}/.
ML-скрипты запускаются с --no-plots; NN-скрипты без флагов (параллелят через joblib внутри).

Запуск:
  uv run python run_phases/phase2.py
  uv run python run_phases/phase2.py --only-versions v0004 v0009
  uv run python run_phases/phase2.py --only-versions v0106b v0107
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"

# Явное указание скрипта для версий, у которых несколько файлов с одним префиксом
_SCRIPT_OVERRIDES: dict[str, str] = {
    "v0011": "v0011_modality_ablation.py",
}

# Версии, которые могут выполняться параллельно (независимы друг от друга)
PARALLEL_VERSIONS = ["v0001", "v0002", "v0004", "v0005", "v0006", "v0007", "v0008"]

# Версии, которые должны выполняться последовательно после параллельного блока
SEQUENTIAL_VERSIONS = ["v0009", "v0010", "v0011", "v0011a", "v0011b", "v0011c"]

# Нейросетевые версии — каждая сама параллелит через joblib (n_jobs=-1),
# поэтому запускаем последовательно друг за другом
NN_VERSIONS = [
    "v0102",   # TCN (базовый)
    "v0103",   # Wavelet-CNN
    "v0104",   # Attention-LSTM
    "v0105",   # TCN + Monotonic loss
    "v0106a",  # Wavelet + Attention-LSTM (гибрид)
    "v0106b",  # Wavelet → TCN (лучший результат)
    "v0106c",  # Wavelet + Attention + Monotonic
    "v0107",   # Ensemble: WavTCN + ElasticNet
]

ALL_VERSIONS = PARALLEL_VERSIONS + SEQUENTIAL_VERSIONS + NN_VERSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Фаза 2: обучение всех версий моделей.")
    parser.add_argument(
        "--only-versions",
        nargs="+",
        metavar="vNNNN",
        help="Запустить только указанные версии (например: v0004 v0009).",
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="Передавать графики (по умолчанию --no-plots для headless-запуска).",
    )
    return parser.parse_args()


def _script(version: str) -> Path:
    """Ищет файл скрипта по префиксу версии."""
    if version in _SCRIPT_OVERRIDES:
        path = _SCRIPTS / _SCRIPT_OVERRIDES[version]
        if not path.exists():
            raise FileNotFoundError(f"Override-скрипт для {version} не найден: {path}")
        return path
    matches = sorted(_SCRIPTS.glob(f"{version}_*.py"))
    if not matches:
        raise FileNotFoundError(f"Скрипт для {version} не найден в {_SCRIPTS}")
    return matches[0]


def run_version(version: str, plots: bool) -> tuple[str, int, float]:
    """Запускает один версионированный скрипт. Возвращает (version, returncode, elapsed)."""
    script = _script(version)
    cmd = ["uv", "run", "python", str(script)]
    # Версии без поддержки --no-plots: NN-скрипты и v0011-серия
    _NO_PLOTS_UNSUPPORTED = set(NN_VERSIONS) | {"v0011", "v0011a", "v0011b", "v0011c"}
    if not plots and version not in _NO_PLOTS_UNSUPPORTED:
        cmd.append("--no-plots")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return version, result.returncode, elapsed


def run_seq(version: str, plots: bool) -> None:
    """Запускает версию синхронно; прерывает фазу при ошибке."""
    print(f"\n{'─' * 60}")
    print(f"  {version}: {_script(version).name}")
    print(f"{'─' * 60}")
    ver, code, elapsed = run_version(version, plots)
    if code != 0:
        print(f"\n[ОШИБКА] {ver} завершилась с кодом {code}. Останавливаемся.")
        sys.exit(code)
    print(f"  ✓ {ver} — {elapsed:.1f} с")


def run_parallel_block(versions: list[str], plots: bool) -> None:
    """Запускает список версий параллельно; прерывает при любой ошибке."""
    if not versions:
        return
    print(f"\n{'─' * 60}")
    print(f"  Параллельно: {', '.join(versions)}")
    print(f"{'─' * 60}")

    t0 = time.perf_counter()
    failed = []

    with ThreadPoolExecutor(max_workers=len(versions)) as pool:
        futures = {pool.submit(run_version, v, plots): v for v in versions}
        for future in as_completed(futures):
            ver, code, elapsed = future.result()
            if code != 0:
                print(f"  ✗ {ver} — код {code}")
                failed.append(ver)
            else:
                print(f"  ✓ {ver} — {elapsed:.1f} с")

    if failed:
        print(f"\n[ОШИБКА] Версии завершились с ошибкой: {failed}. Останавливаемся.")
        sys.exit(1)

    print(f"  Параллельный блок завершён — {time.perf_counter() - t0:.1f} с")


def main() -> None:
    args = parse_args()
    plots = args.with_plots

    # Фильтрация по --only-versions
    if args.only_versions:
        requested = set(args.only_versions)
        unknown = requested - set(ALL_VERSIONS)
        if unknown:
            print(f"[ОШИБКА] Неизвестные версии: {unknown}. Доступны: {ALL_VERSIONS}")
            sys.exit(1)
        parallel = [v for v in PARALLEL_VERSIONS if v in requested]
        sequential = [v for v in SEQUENTIAL_VERSIONS if v in requested]
        nn_versions = [v for v in NN_VERSIONS if v in requested]
    else:
        parallel = PARALLEL_VERSIONS
        sequential = SEQUENTIAL_VERSIONS
        nn_versions = NN_VERSIONS

    # Пропускаем версии, для которых нет скрипта
    def exists(v: str) -> bool:
        try:
            _script(v)
            return True
        except FileNotFoundError:
            print(f"  [SKIP] {v}: скрипт не найден")
            return False

    parallel = [v for v in parallel if exists(v)]
    sequential = [v for v in sequential if exists(v)]
    nn_versions = [v for v in nn_versions if exists(v)]

    print(f"\n{'#' * 60}")
    print(f"  ФАЗА 2: обучение моделей")
    print(f"  Параллельно: {parallel}")
    print(f"  Последовательно (ML): {sequential}")
    print(f"  Последовательно (NN): {nn_versions}")
    print(f"{'#' * 60}")

    t_start = time.perf_counter()

    run_parallel_block(parallel, plots)

    for version in sequential:
        run_seq(version, plots)

    # NN-версии запускаются строго последовательно: каждая занимает все ядра через joblib
    for version in nn_versions:
        run_seq(version, plots)

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'#' * 60}")
    print(f"  ФАЗА 2 ЗАВЕРШЕНА — {elapsed_total:.1f} с")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
