"""Генератор jobs.csv для orchestrator'а.

Создаёт декартов набор:
  архитектура × target × feature_set × with_abs.

  Lin    — одна batch-задача (linear_runner --grid-all), считает все 580 моделей.
  LSTM1..LSTM16 × {lt1, lt2} × {EMG, EMG+NIRS, EMG+NIRS+HRV} × {abs, no_abs}
  TCN1..TCN4    × {lt1, lt2} × {EMG, EMG+NIRS, EMG+NIRS+HRV} × {abs, no_abs}

Для stateful LSTM (LSTM7..LSTM12) runner отличается — берётся lstm_stateful_runner.py.

Опции:
  --gpu-only         только NN-задачи без линейных моделей. Порядок при этом
                     сохраняет первые 168 job_id совместимыми с уже
                     сгенерированным GPU-only планом: сначала non-stateful
                     LSTM, затем TCN, затем stateful LSTM.
  --families Lin ...  ограничить генерацию выбранными семействами. Допустимые
                     значения: Lin, LSTM, TCN. Можно передать несколько
                     значений через пробел или через запятую.
  --tcn-max-epochs N  переопределить max_epochs только для TCN-архитектур.

Выход: orchestrator/jobs.csv.
  job_id, runner, architecture_id, target, feature_set, with_abs, wavelet_mode, cmd
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from architectures import LINEAR_ARCHS, LSTM_ARCHS, TCN_ARCHS


OUT_PATH = Path(__file__).resolve().parent / "jobs.csv"

TARGETS = ["lt1", "lt2"]
FEATURE_SETS_LIN = ["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"]
FEATURE_SETS_NN = ["EMG", "EMG+NIRS", "EMG+NIRS+HRV"]
WITH_ABS_VARIANTS = [True, False]


def _runner_for(arch) -> str:
    """Какой runner запускать для архитектуры."""
    if arch.family == "Lin":
        return "linear_runner.py"
    if arch.family == "TCN":
        return "tcn_runner.py"
    if arch.family == "LSTM":
        if arch.model_class_name == "LSTMStatefulRegressor":
            return "lstm_stateful_runner.py"
        return "lstm_runner.py"
    raise ValueError(f"Неизвестная family={arch.family!r}")


def _is_stateful_lstm(arch) -> bool:
    """Является ли архитектура stateful LSTM."""
    return arch.family == "LSTM" and arch.model_class_name == "LSTMStatefulRegressor"


def _build_cmd(runner: str, arch, target: str, fset: str, with_abs: bool) -> str:
    flag = "--with-abs" if with_abs else "--no-abs"
    return (
        f"PYTHONPATH=. uv run python {runner} "
        f"--architecture {arch.architecture_id} "
        f"--target {target} "
        f"--feature-set {fset} {flag}"
    )


def _tcn_epochs_arg(arch, tcn_max_epochs: int | None) -> str:
    """Возвращает CLI-аргумент для TCN-эпох, если он нужен."""
    if arch.family != "TCN" or tcn_max_epochs is None:
        return ""
    return f" --max-epochs {tcn_max_epochs}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="генератор jobs.csv для orchestrator'а")
    p.add_argument("--gpu-only", action="store_true",
                   help="только NN-задачи без Lin; порядок сохраняет совместимость job_id")
    p.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="какие семейства генерировать: Lin, LSTM, TCN; можно через пробел или запятую",
    )
    p.add_argument(
        "--tcn-max-epochs",
        type=int,
        default=None,
        help="сколько эпох запускать только для TCN-архитектур",
    )
    p.add_argument(
        "--phase-split-ablation",
        action="store_true",
        help="вместо полного грида сгенерировать ablation на phase_split: "
             "3 батч-задачи Lin × feature_set × target × abs × "
             "phase_split ∈ {split, full_cycles, full_window}.",
    )
    p.add_argument(
        "--phase-split-feature-sets",
        nargs="+",
        default=["EMG"],
        help="какие feature_set включать в phase_split-ablation. "
             "Допустимо: EMG, EMG+NIRS, EMG+NIRS+HRV. "
             "По умолчанию: EMG.",
    )
    p.add_argument(
        "--shift-sensitivity",
        action="store_true",
        help="вместо полного грида сгенерировать sensitivity-анализ на сдвиг "
             "целевой переменной для 12 финалистов (вариант B): "
             "LT2 × {45,90,135,180}c для 6 моделей + "
             "LT1 × {45,90}c для 6 моделей = 36 LOSO-прогонов. "
             "См. SHIFT_SENSITIVITY_FINALISTS ниже.",
    )
    return p.parse_args()


# ─── Sensitivity-анализ задержки капиллярного лактата (вариант B) ──────────
#
# 12 финалистов, отобранных по топ-2 на семейство для каждой target.
# Шортлист — строго топ-2 на семейство по lt_mae_median_policy_mean в
# baseline-конфигурации (phase_split="split"), без принудительного разделения
# на ElasticNet/SVR.
#
# Базовая точка (shift_sec=0) уже есть в существующих результатах — для
# каждой отобранной модели она НЕ перегенерируется здесь.
SHIFT_SENSITIVITY_FINALISTS: list[dict] = [
    # ── LT2: топ-2 baseline (оба ElasticNet на HRV+abs) ──
    {"target": "lt2", "architecture_id": "Lin15", "feature_set": "HRV",
     "with_abs": True,  "wavelet_mode": "none", "note": "Lin top-1 (ElasticNet)"},
    {"target": "lt2", "architecture_id": "Lin17", "feature_set": "HRV",
     "with_abs": True,  "wavelet_mode": "none", "note": "Lin top-2 (ElasticNet)"},
    {"target": "lt2", "architecture_id": "LSTM5", "feature_set": "EMG+NIRS+HRV",
     "with_abs": True,  "wavelet_mode": "none", "note": "LSTM top-1"},
    {"target": "lt2", "architecture_id": "LSTM7", "feature_set": "EMG+NIRS+HRV",
     "with_abs": True,  "wavelet_mode": "none", "note": "LSTM top-2 (stateful)"},
    {"target": "lt2", "architecture_id": "TCN1",  "feature_set": "EMG+NIRS+HRV",
     "with_abs": True,  "wavelet_mode": "none", "note": "TCN top-1"},
    {"target": "lt2", "architecture_id": "TCN3",  "feature_set": "EMG+NIRS+HRV",
     "with_abs": True,  "wavelet_mode": "dwt",  "note": "TCN top-2 (DWT)"},
    # ── LT1: топ-2 baseline (оба SVR на NIRS+abs) ──
    {"target": "lt1", "architecture_id": "Lin25", "feature_set": "NIRS",
     "with_abs": True,  "wavelet_mode": "none", "note": "Lin top-1 (SVR)"},
    {"target": "lt1", "architecture_id": "Lin24", "feature_set": "NIRS",
     "with_abs": True,  "wavelet_mode": "none", "note": "Lin top-2 (SVR)"},
    {"target": "lt1", "architecture_id": "LSTM3", "feature_set": "EMG+NIRS+HRV",
     "with_abs": False, "wavelet_mode": "none", "note": "LSTM top-1"},
    {"target": "lt1", "architecture_id": "LSTM5", "feature_set": "EMG+NIRS+HRV",
     "with_abs": False, "wavelet_mode": "none", "note": "LSTM top-2"},
    {"target": "lt1", "architecture_id": "TCN3",  "feature_set": "EMG+NIRS+HRV",
     "with_abs": False, "wavelet_mode": "dwt",  "note": "TCN top-1 (DWT)"},
    {"target": "lt1", "architecture_id": "TCN1",  "feature_set": "EMG+NIRS+HRV",
     "with_abs": False, "wavelet_mode": "none", "note": "TCN top-2"},
]

# Сетка сдвигов: LT2 — полный диапазон (¼, ½, ¾, полная ступень 180c);
# LT1 — короткая (¼ и ½) как отрицательный контроль (физиологически
# задержка для LT1 нестабильна, и мы ожидаем рост MAE уже при малых
# сдвигах).
SHIFT_GRID: dict[str, list[int]] = {
    "lt2": [45, 90, 135, 180],
    "lt1": [45, 90],
}


def _parse_families(raw: list[str] | None) -> set[str] | None:
    """Нормализует список семейств из CLI."""
    if not raw:
        return None
    out: set[str] = set()
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if part not in {"Lin", "LSTM", "TCN"}:
                raise SystemExit(f"Неизвестное семейство: {part!r}. Допустимо: Lin, LSTM, TCN")
            out.add(part)
    return out


def _arch_by_id(architecture_id: str):
    """Поиск ArchitectureSpec по architecture_id в LINEAR_ARCHS/LSTM_ARCHS/TCN_ARCHS."""
    for arch in LINEAR_ARCHS + LSTM_ARCHS + TCN_ARCHS:
        if arch.architecture_id == architecture_id:
            return arch
    raise SystemExit(f"Не найдена архитектура: {architecture_id!r}")


def _build_shift_cmd(runner: str, arch, target: str, fset: str,
                     with_abs: bool, wavelet_mode: str, shift_sec: int) -> str:
    """CLI для одного sensitivity-прогона.

    Для Lin вызывается single-arch режим (БЕЗ --grid-all), чтобы посчитать
    ровно одну модель. Для NN — обычный single-arch вызов.
    """
    abs_flag = "--with-abs" if with_abs else "--no-abs"
    wave_flag = f" --wavelet-mode {wavelet_mode}" if wavelet_mode != "none" else ""
    return (
        f"PYTHONPATH=. uv run python {runner} "
        f"--architecture {arch.architecture_id} "
        f"--target {target} "
        f"--feature-set {fset} {abs_flag}"
        f"{wave_flag} "
        f"--target-shift-sec {shift_sec}"
    )


def main() -> None:
    args = parse_args()
    families = _parse_families(args.families)
    rows: list[dict] = []
    n = 0

    # ── Sensitivity-анализ задержки капиллярного лактата ───────────────────
    # 12 финалистов × своя сетка сдвигов на target.
    # Базовая точка (shift=0) НЕ генерируется — берётся из существующих
    # baseline-прогонов в results/.
    if args.shift_sensitivity:
        for fin in SHIFT_SENSITIVITY_FINALISTS:
            arch = _arch_by_id(fin["architecture_id"])
            runner = _runner_for(arch)
            target = fin["target"]
            shifts = SHIFT_GRID[target]
            for shift_sec in shifts:
                n += 1
                rows.append({
                    "job_id": f"J{n:04d}",
                    "runner": Path(runner).name,
                    "architecture_id": arch.architecture_id,
                    "target": target,
                    "feature_set": fin["feature_set"],
                    "with_abs": str(fin["with_abs"]).lower(),
                    "wavelet_mode": fin["wavelet_mode"],
                    "cmd": _build_shift_cmd(
                        runner, arch, target,
                        fin["feature_set"], fin["with_abs"],
                        fin["wavelet_mode"], shift_sec,
                    ),
                })
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        # Сводка по target/family.
        from collections import Counter
        c = Counter()
        for r in rows:
            c[r["target"]] += 1
        print(f"[shift-sensitivity] Сгенерировано {len(rows)} задач:")
        for t, count in sorted(c.items()):
            print(f"  {t}: {count}")
        print(f"\n→ {OUT_PATH.resolve()}")
        return

    # ── Ablation-режим: phase_split на feature_set с EMG-блоком ────────────
    # Эмитит 3 батч-задачи linear_runner --grid-all с разными --phase-split
    # значениями. Каждая батч-job обрабатывает
    # Lin × feature_set × {lt1,lt2} × {abs,no_abs}.
    if args.phase_split_ablation:
        allowed_phase_sets = {"EMG", "EMG+NIRS", "EMG+NIRS+HRV"}
        phase_feature_sets: list[str] = []
        for raw in args.phase_split_feature_sets:
            if raw not in allowed_phase_sets:
                raise SystemExit(
                    f"--phase-split-feature-sets: недопустимое значение {raw!r}. "
                    f"Разрешены: {sorted(allowed_phase_sets)}"
                )
            if raw not in phase_feature_sets:
                phase_feature_sets.append(raw)
        if not phase_feature_sets:
            raise SystemExit("--phase-split-feature-sets: список пуст")
        feature_sets_arg = " ".join(phase_feature_sets)
        for ps in ("split", "full_cycles", "full_window"):
            n += 1
            rows.append({
                "job_id": f"J{n:04d}",
                "runner": "linear_runner.py",
                "architecture_id": "ALL_LIN",
                "target": "both",
                "feature_set": ",".join(phase_feature_sets),
                "with_abs": "both",
                "wavelet_mode": "none",
                "cmd": (
                    "PYTHONPATH=. uv run python linear_runner.py --grid-all "
                    f"--feature-sets {feature_sets_arg} "
                    f"--phase-split {ps}"
                ),
            })
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[ablation] Сгенерировано {len(rows)} batch-задач "
              f"(phase_split ∈ split / full_cycles / full_window; "
              f"feature_set = {phase_feature_sets}).")
        print(f"→ {OUT_PATH.resolve()}")
        return

    def add(arch, fset_list):
        nonlocal n
        runner = _runner_for(arch)
        for target, fset, with_abs in itertools.product(TARGETS, fset_list, WITH_ABS_VARIANTS):
            n += 1
            rows.append({
                "job_id": f"J{n:04d}",
                "runner": Path(runner).name,
                "architecture_id": arch.architecture_id,
                "target": target,
                "feature_set": fset,
                "with_abs": str(with_abs).lower(),
                "wavelet_mode": arch.forced_wavelet_mode or "none",
                "cmd": _build_cmd(runner, arch, target, fset, with_abs)
                       + _tcn_epochs_arg(arch, args.tcn_max_epochs),
            })

    if not args.gpu_only:
        # Лин — батч-режим: одна job на весь декартов набор Lin × target × fset × abs.
        # Работает на CPU (joblib parallel), не нагружает GPU.
        if families is None or "Lin" in families:
            n += 1
            rows.append({
                "job_id": f"J{n:04d}",
                "runner": "linear_runner.py",
                "architecture_id": "ALL_LIN",
                "target": "both",
                "feature_set": "all",
                "with_abs": "both",
                "wavelet_mode": "none",
                "cmd": "PYTHONPATH=. uv run python linear_runner.py --grid-all",
            })

    selected_families = families or {"Lin", "LSTM", "TCN"}

    if "LSTM" in selected_families:
        if args.gpu_only:
            for arch in LSTM_ARCHS:
                if _is_stateful_lstm(arch):
                    continue
                add(arch, FEATURE_SETS_NN)
            for arch in LSTM_ARCHS:
                if not _is_stateful_lstm(arch):
                    continue
                add(arch, FEATURE_SETS_NN)
        else:
            for arch in LSTM_ARCHS:
                add(arch, FEATURE_SETS_NN)

    if "TCN" in selected_families:
        for arch in TCN_ARCHS:
            add(arch, FEATURE_SETS_NN)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Сводка по семействам.
    from collections import Counter
    c = Counter()
    for r in rows:
        family = r["architecture_id"][:3]  # Lin, LST, TCN
        c[family] += 1
    print(f"Сгенерировано {len(rows)} задач:")
    for fam, count in sorted(c.items()):
        print(f"  {fam}*  {count}")
    print(f"\n→ {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
