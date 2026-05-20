"""Sensitivity-анализ задержки капиллярного лактата с ПЕРЕОБУЧЕНИЕМ.

Принципиальное отличие от ``label_shift_sensitivity.py``: тот модуль считает
post-hoc метрики на сохранённых predictions baseline-моделей (без
переобучения). Это математически корректно для линейной регрессии, но для
нелинейных моделей (SVR, LSTM, TCN) даёт лишь верхнюю оценку устойчивости.

Здесь же читаются predictions моделей, ПЕРЕОБУЧЕННЫХ с явным сдвигом target
(`linear_runner.py / lstm_runner.py / tcn_runner.py --target-shift-sec δ`).
Это и есть честный sensitivity-эксперимент.

Шортлист 12 финалистов (вариант B): топ-1 ElasticNet + топ-1 SVR на каждое
семейство Lin, плюс топ-2 на LSTM и TCN. ElasticNet строки служат
контролем вычислений (математическая инвариантность к сдвигу — по
построению), SVR/LSTM/TCN — реальный sensitivity-сигнал.

Сетка сдвигов:
  LT2 — {0, 45, 90, 135, 180} с (¼, ½, ¾, полная ступень).
  LT1 — {0, 45, 90} с (отрицательный контроль).

Использование:
    cd new_arch
    uv run python -m analysis.shift_sensitivity_retrain \
        [--results-root results] \
        [--out-dir analysis_out/figures_thesis]

Выход:
    analysis_out/figures_thesis/shift_lt2.csv
    analysis_out/figures_thesis/shift_lt1.csv
    analysis_out/figures_thesis/shift_summary.md

Зависит только от results/ — НЕ требует actualizированного analysis_out/cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from architectures import LINEAR_ARCHS, LSTM_ARCHS, TCN_ARCHS
from common_lib import build_model_id


# ── Список финалистов — синхронизирован с orchestrator/gen_jobs.py ─────────
#
# При изменении этого списка ОБЯЗАТЕЛЬНО синхронизируй
# SHIFT_SENSITIVITY_FINALISTS в orchestrator/gen_jobs.py.
SHIFT_SENSITIVITY_FINALISTS: list[dict] = [
    # ── LT2: топ-2 baseline (оба ElasticNet на HRV+abs) ──
    {"target": "lt2", "architecture_id": "Lin15", "feature_set": "HRV",
     "with_abs": True,  "wavelet_mode": "none",
     "family_label": "Lin top-1",        "role": "baseline"},
    {"target": "lt2", "architecture_id": "Lin17", "feature_set": "HRV",
     "with_abs": True,  "wavelet_mode": "none",
     "family_label": "Lin top-2",        "role": "baseline"},
    {"target": "lt2", "architecture_id": "LSTM5", "feature_set": "EMG+NIRS+HRV",
     "with_abs": True,  "wavelet_mode": "none",
     "family_label": "LSTM top-1",       "role": "sensitivity"},
    {"target": "lt2", "architecture_id": "LSTM7", "feature_set": "EMG+NIRS+HRV",
     "with_abs": True,  "wavelet_mode": "none",
     "family_label": "LSTM top-2",       "role": "sensitivity"},
    {"target": "lt2", "architecture_id": "TCN1",  "feature_set": "EMG+NIRS+HRV",
     "with_abs": True,  "wavelet_mode": "none",
     "family_label": "TCN top-1",        "role": "sensitivity"},
    {"target": "lt2", "architecture_id": "TCN3",  "feature_set": "EMG+NIRS+HRV",
     "with_abs": True,  "wavelet_mode": "dwt",
     "family_label": "TCN top-2 (DWT)",  "role": "sensitivity"},
    # ── LT1: топ-2 baseline (оба SVR на NIRS+abs) ──
    {"target": "lt1", "architecture_id": "Lin25", "feature_set": "NIRS",
     "with_abs": True,  "wavelet_mode": "none",
     "family_label": "Lin top-1",        "role": "baseline"},
    {"target": "lt1", "architecture_id": "Lin24", "feature_set": "NIRS",
     "with_abs": True,  "wavelet_mode": "none",
     "family_label": "Lin top-2",        "role": "baseline"},
    {"target": "lt1", "architecture_id": "LSTM3", "feature_set": "EMG+NIRS+HRV",
     "with_abs": False, "wavelet_mode": "none",
     "family_label": "LSTM top-1",       "role": "sensitivity"},
    {"target": "lt1", "architecture_id": "LSTM5", "feature_set": "EMG+NIRS+HRV",
     "with_abs": False, "wavelet_mode": "none",
     "family_label": "LSTM top-2",       "role": "sensitivity"},
    {"target": "lt1", "architecture_id": "TCN3",  "feature_set": "EMG+NIRS+HRV",
     "with_abs": False, "wavelet_mode": "dwt",
     "family_label": "TCN top-1 (DWT)",  "role": "sensitivity"},
    {"target": "lt1", "architecture_id": "TCN1",  "feature_set": "EMG+NIRS+HRV",
     "with_abs": False, "wavelet_mode": "none",
     "family_label": "TCN top-2",        "role": "sensitivity"},
]

SHIFT_GRID: dict[str, list[int]] = {
    "lt2": [0, 45, 90, 135, 180],
    "lt1": [0, 45, 90],
}


def _arch_by_id(architecture_id: str):
    """Поиск ArchitectureSpec по architecture_id."""
    for arch in LINEAR_ARCHS + LSTM_ARCHS + TCN_ARCHS:
        if arch.architecture_id == architecture_id:
            return arch
    raise KeyError(f"architecture {architecture_id!r} не найдена")


def _resolve_model_id(fin: dict, shift_sec: int) -> str:
    """Восстанавливает model_id для финалиста на заданном shift_sec.

    При shift=0 — возвращает baseline model_id (хэш без target_shift в payload,
    совместимый с существующими baseline-результатами).
    """
    arch = _arch_by_id(fin["architecture_id"])
    return build_model_id(
        arch,
        target=fin["target"],
        feature_set=fin["feature_set"],
        with_abs=bool(fin["with_abs"]),
        wavelet_mode=fin["wavelet_mode"],
        target_shift_sec=int(shift_sec),
    )


def _lt_mae_median_policy_at_epoch(preds: pd.DataFrame) -> pd.Series:
    """Считает lt_mae_median_policy для каждой эпохи.

    Policy «median»: для каждой пары (epoch, subject_id) предсказанное
    LT-время — медиана `sample_end_sec + y_pred` по всем окнам субъекта;
    истинное LT-время — медиана `sample_end_sec + y_true` (≈ константа
    по субъекту). Ошибка — модуль разности этих двух чисел. Финальная
    метрика — среднее по испытуемым.

    Возвращает Series, индексированный по epoch, со значениями MAE в секундах.
    Метрика совпадает с `lt_mae_median_policy_mean` основного рейтинга.
    """
    df = preds.assign(
        lt_hat_sec=preds["sample_end_sec"] + preds["y_pred"],
        lt_true_sec=preds["sample_end_sec"] + preds["y_true"],
    )
    per_subject = (
        df.groupby(["epoch", "subject_id"])
          .agg(lt_hat_median=("lt_hat_sec", "median"),
               lt_true_median=("lt_true_sec", "median"))
          .reset_index()
    )
    per_subject["abs_err"] = (
        per_subject["lt_hat_median"] - per_subject["lt_true_median"]
    ).abs()
    return per_subject.groupby("epoch")["abs_err"].mean()


def _best_epoch_lt_mae(preds: pd.DataFrame) -> tuple[int, float]:
    """Выбирает лучшую эпоху по lt_mae_median_policy и возвращает (epoch, mae_sec)."""
    per_epoch = _lt_mae_median_policy_at_epoch(preds)
    best_epoch = int(per_epoch.idxmin())
    return best_epoch, float(per_epoch.loc[best_epoch])


def _load_subject_mean_mae(results_root: Path, architecture_id: str,
                           model_id: str) -> float | None:
    """Читает predictions parquet и возвращает lt_mae_median_policy_mean
    на лучшей эпохе.

    Метрика та же, что используется в основном рейтинге диссертации
    (`lt_mae_median_policy_mean` в `model_summary.parquet`), что
    обеспечивает совпадение значений Δ=0 в sensitivity-таблице с baseline
    из главы 3.

    Возвращает None, если файл не найден (модель ещё не обучена).
    """
    p = results_root / architecture_id / model_id / f"predictions_{model_id}.parquet"
    if not p.exists():
        return None
    preds = pd.read_parquet(p)
    if preds.empty:
        return None
    _, mae = _best_epoch_lt_mae(preds)
    return mae


def build_shift_table(results_root: Path) -> dict[str, pd.DataFrame]:
    """Собирает таблицы для LT1 и LT2.

    Возвращает {"lt2": df, "lt1": df}, где df:
      index: (architecture_id, family_label, role)
      columns: shift_sec ∈ grid (int)
      values: subject-mean MAE в секундах (NaN если model_id отсутствует)
    """
    out: dict[str, pd.DataFrame] = {}
    for target in ("lt2", "lt1"):
        grid = SHIFT_GRID[target]
        fins = [f for f in SHIFT_SENSITIVITY_FINALISTS if f["target"] == target]
        rows: list[dict] = []
        for fin in fins:
            row: dict = {
                "architecture_id": fin["architecture_id"],
                "family_label":    fin["family_label"],
                "role":            fin["role"],
            }
            for shift in grid:
                mid = _resolve_model_id(fin, shift)
                mae = _load_subject_mean_mae(
                    results_root, fin["architecture_id"], mid
                )
                row[f"shift_{shift}s_mae_sec"] = mae
                row[f"shift_{shift}s_model_id"] = mid
            rows.append(row)
        out[target] = pd.DataFrame(rows)
    return out


def render_markdown_summary(tables: dict[str, pd.DataFrame]) -> str:
    """Markdown-сводка с относительными изменениями MAE."""
    lines: list[str] = []
    lines.append("# Sensitivity к сдвигу target (переобучение)")
    lines.append("")
    lines.append("Финалисты: топ-2 на семейство на LT2 и LT1. В семействе Lin "
                 "сознательно отобраны топ-1 ElasticNet (control: математическая "
                 "инвариантность) и топ-1 SVR (sensitivity: нелинейная модель).")
    lines.append("")
    for target in ("lt2", "lt1"):
        grid = SHIFT_GRID[target]
        df = tables[target]
        lines.append(f"## {target.upper()}")
        lines.append("")
        # Заголовок таблицы
        header = ["model", "role"] + [f"Δ={s}s" for s in grid] + [f"Δ=180/Δ=0" if target == "lt2" else f"Δ=90/Δ=0"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        max_shift = grid[-1]
        for _, r in df.iterrows():
            base = r[f"shift_0s_mae_sec"]
            cells = [r["family_label"], r["role"]]
            for s in grid:
                mae = r[f"shift_{s}s_mae_sec"]
                cells.append("—" if pd.isna(mae) else f"{mae:.1f}")
            # Relative change
            top = r[f"shift_{max_shift}s_mae_sec"]
            if pd.isna(base) or pd.isna(top) or base == 0:
                cells.append("—")
            else:
                cells.append(f"{top/base:.2f}×")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("**Интерпретация.** Строки `Lin (ElasticNet)` должны давать "
                 "плато по построению (линейная регрессия инвариантна к "
                 "константному сдвигу target). Строки `Lin (SVR)`, `LSTM`, `TCN` "
                 "несут содержательный sensitivity-сигнал. Для LT2 ожидается "
                 "умеренное изменение MAE в пределах физиологически правдоподобного "
                 "диапазона задержки; для LT1 — рост MAE даже при малых сдвигах, "
                 "подтверждающий нестабильность задержки при низкой интенсивности.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="sensitivity-анализ сдвига target с переобучением"
    )
    ap.add_argument("--results-root", type=Path,
                    default=_ROOT / "results",
                    help="директория с results/{arch}/{model_id}/")
    ap.add_argument("--out-dir", type=Path,
                    default=_ROOT / "analysis_out" / "figures_thesis",
                    help="директория для CSV/MD выходов")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_shift_table(args.results_root)

    for target, df in tables.items():
        out_csv = args.out_dir / f"shift_{target}.csv"
        df.to_csv(out_csv, index=False)
        # Краткая консольная сводка
        n_total = sum(
            df.filter(like="_mae_sec").notna().sum()
        )
        n_expected = len(df) * len(SHIFT_GRID[target])
        print(f"  [{target}] {n_total}/{n_expected} ячеек MAE заполнено → {out_csv}")

    md_path = args.out_dir / "shift_summary.md"
    md_path.write_text(render_markdown_summary(tables), encoding="utf-8")
    print(f"  → {md_path}")


if __name__ == "__main__":
    main()
