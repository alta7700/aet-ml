"""Шаг 07: упаковка results/final/ для диссертации.

Собирает:
- results/final/dissertation_tables/ — готовые таблицы для текста «Результаты»;
- results/final/ENV.md — версии пакетов, сиды, дата;
- обновляет results/final/README.md — полная навигация по всем артефактам.

Запуск:
    PYTHONPATH=. uv run python scripts/final_package.py
"""

from __future__ import annotations

import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
TABLES = FINAL / "dissertation_tables"


# ─────────────────────── Главная таблица результатов ───────────────────────

def build_main_results_table() -> pd.DataFrame:
    """Главная таблица «Результаты» — финалисты по обоим таргетам."""
    wide = pd.read_csv(FINAL / "01_ranking_wide.csv")

    # Финалисты: для каждого target — победитель + мультимодальный + лучший NN.
    picks = [
        ("lt2", "v0011", "with_abs", "HRV",          "победитель LT2"),
        ("lt2", "v0011", "with_abs", "EMG+NIRS+HRV", "мультимодальный LT2"),
        ("lt2", "v0107", "with_abs", "EMG+NIRS+HRV", "лучший NN — LT2"),
        ("lt1", "v0011", "with_abs", "EMG+NIRS",     "победитель LT1"),
        ("lt1", "v0011", "with_abs", "EMG+NIRS+HRV", "мультимодальный LT1"),
        ("lt1", "v0107", "noabs",    "EMG+NIRS+HRV", "лучший NN — LT1"),
    ]
    rows = []
    for target, version, variant, fset, note in picks:
        m = wide[(wide["target"] == target) & (wide["version"] == version)
                 & (wide["variant"] == variant) & (wide["feature_set"] == fset)]
        if m.empty:
            continue
        r = m.iloc[0]
        rows.append({
            "target": target,
            "model": version,
            "family": r["family"],
            "feature_set": fset,
            "variant": variant,
            "note": note,
            "loso_mae_median_min": r["loso_mae_median"],
            "loso_mae_iqr_min": r["loso_mae_iqr"],
            "raw_mae_min": r["raw_mae_min"],
            "kalman_mae_min": r["kalman_mae_min"],
            "rho": r["rho"],
            "r2": r["r2"],
            "loso_n_subjects": r["loso_n_subjects"],
        })
    return pd.DataFrame(rows)


def build_modality_table() -> pd.DataFrame:
    """Лучшая модель на каждой модальности (по loso_mae_median, with_abs)."""
    wide = pd.read_csv(FINAL / "01_ranking_wide.csv")
    rows = []
    for target in ("lt1", "lt2"):
        for fset in ["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"]:
            sub = wide[(wide["target"] == target) & (wide["feature_set"] == fset)
                       & (wide["variant"] == "with_abs")]
            sub = sub[sub["loso_mae_median"].notna()]
            if sub.empty:
                continue
            best = sub.loc[sub["loso_mae_median"].idxmin()]
            rows.append({
                "target": target, "feature_set": fset,
                "best_model": best["version"], "family": best["family"],
                "loso_mae_median_min": best["loso_mae_median"],
                "rho": best["rho"], "r2": best["r2"],
            })
    return pd.DataFrame(rows)


def build_significance_table() -> pd.DataFrame:
    """Сводка значимости: ablation + champion cross-modality."""
    rows = []

    # Ablation with_abs vs noabs
    ab = pd.read_csv(FINAL / "11_abs_vs_noabs.csv")
    n_equiv = int((ab["verdict"] == "эквивалентно").sum())
    rows.append({
        "test": "ablation with_abs vs noabs (NN; 72 конфигурации)",
        "result": f"{n_equiv}/72 эквивалентны (86%); медиана |Δ|=0.02 мин",
        "conclusion": "абсолютные признаки не нужны для NN",
    })

    # Champion cross-modality
    cm = pd.read_csv(FINAL / "14_champion_cross_modality.csv")
    for _, r in cm.iterrows():
        rows.append({
            "test": f"{r['target']}: {r['fset_a']} vs {r['fset_b']} (v0011)",
            "result": f"Δmedian={r['median_diff']:+.2f} мин; p={r['p']:.3f}",
            "conclusion": r["verdict"],
        })

    # Pairwise: v0011 доминирование
    wc = pd.read_csv(FINAL / "13_pairwise_win_counts.csv")
    v11 = wc[wc["version"] == "v0011"]
    total_wins = int(v11["sig_wins"].sum())
    total_losses = int(v11["sig_losses"].sum())
    rows.append({
        "test": "pairwise Wilcoxon+Holm: v0011 vs все NN (with_abs)",
        "result": f"{total_wins} значимых побед, {total_losses} поражений",
        "conclusion": "линейная модель статистически доминирует",
    })
    return pd.DataFrame(rows)


def build_two_experts_table() -> pd.DataFrame:
    """Сводка по гипотезе двух экспертов."""
    comp = pd.read_csv(FINAL / "two_experts" / "16_system_comparison.csv")
    comp = comp[comp["threshold_tag"] == "thr13"].copy()
    return comp[["system", "mae_median", "mae_mean",
                 "vs_generalist_delta", "vs_generalist_p"]]


def build_conformal_table() -> pd.DataFrame:
    """Сводка conformal-покрытия."""
    return pd.read_csv(FINAL / "conformal" / "coverage_summary.csv")


# ─────────────────────── ENV.md ───────────────────────

def write_env() -> None:
    pkgs = {}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "shap",
                 "matplotlib", "pyarrow", "h5py"]:
        try:
            mod = __import__(name if name != "scikit-learn" else "sklearn")
            pkgs[name] = getattr(mod, "__version__", "?")
        except Exception:
            pkgs[name] = "не установлен"

    lines = [
        "# ENV — окружение расчётов results/final/",
        "",
        f"- Дата сборки: {date.today().isoformat()}",
        f"- Python: {platform.python_version()}",
        f"- Платформа: {platform.platform()}",
        "",
        "## Версии пакетов",
        "",
        "| Пакет | Версия |",
        "|-------|--------|",
    ]
    for k, v in pkgs.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Сиды и детерминизм",
        "",
        "- LOSO детерминирован (нет случайности в Ridge/sklearn-зоопарке).",
        "- Маршрутизатор (LogisticRegression) — детерминирован при фиксированных данных.",
        "- Нейросетевые версии (v01xx) обучались на GPU-сервере; их предсказания "
        "берутся из сохранённых npy, не пересчитываются здесь.",
        "- Калибровка conformal — LOSO split-conformal, без случайности.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        "# порядок запуска скриптов шагов 01–07",
        "uv run python scripts/patch_session_params_hrv_baseline.py   # шаг 01",
        "PYTHONPATH=. uv run python scripts/final_per_subject_from_npy.py  # шаг 02 (данные)",
        "PYTHONPATH=. uv run python scripts/final_build_ranking.py        # шаг 02",
        "PYTHONPATH=. uv run python scripts/final_topk_views.py           # шаг 02",
        "PYTHONPATH=. uv run python scripts/final_per_subject_dist.py     # шаг 02a",
        "PYTHONPATH=. uv run python scripts/final_time_resolved.py        # шаг 02b",
        "PYTHONPATH=. uv run python scripts/final_abs_vs_noabs.py         # шаг 03",
        "PYTHONPATH=. uv run python scripts/final_v0011_hrv_coefs.py      # шаг 03 (аддендум)",
        "PYTHONPATH=. uv run python scripts/final_pairwise_wilcoxon.py    # шаг 04",
        "PYTHONPATH=. uv run python scripts/final_two_experts.py          # шаг 05",
        "PYTHONPATH=. uv run python scripts/final_shap_conformal.py       # шаг 06",
        "PYTHONPATH=. uv run python scripts/final_package.py              # шаг 07",
        "```",
    ]
    (FINAL / "ENV.md").write_text("\n".join(lines), encoding="utf-8")
    print("  → ENV.md")


# ─────────────────────── README.md ───────────────────────

README = """# results/final/ — финальные артефакты для диссертации

Собрано скриптами `scripts/final_*.py`. План работ — в `final-plan/`.
Окружение и порядок воспроизведения — в `ENV.md`.

## Краткие выводы (для раздела «Результаты»)

1. **Минимальный достаточный набор сенсоров зависит от порога.**
   LT2 — достаточно ВСР (HRV ≈ EMG+NIRS+HRV, p=0.97). LT1 — достаточно
   ЭМГ+NIRS (≈ EMG+NIRS+HRV, p=0.44). Мультимодальность не даёт значимого
   выигрыша.
2. **Линейная регрессия (v0011) статистически доминирует** над всеми NN
   (paired Wilcoxon + Holm): победитель в 6/8 конфигураций, не уступает нигде.
3. **Лучшие модели**: LT2 — v0011/HRV (loso MAE 1.86 мин, ρ=0.91, R²=0.82);
   LT1 — v0011/EMG+NIRS (loso MAE 2.11 мин, ρ=0.74, R²=0.52).
4. **Абсолютные признаки не нужны для NN** (86% конфигураций эквивалентны),
   но **нужны для линейной модели** (hrv_mean_rr_ms — 44.6% веса).
5. **«Два эксперта»**: симметричная гипотеза отвергнута (trained-спец на n=6
   недообучен), асимметричная подтверждена (спец для нетренированных +
   generalist для тренированных: oracle p=0.034).
6. **Conformal**: LT2 интервалы практичны (±3.5 мин при 80%), LT1 — широки
   (±9 мин при 90%).
7. **Слабость LT1 структурирована**: 79% окон с ошибкой >7 мин — тренированные
   субъекты. Это один и тот же феномен в шагах 02a/02b/05/06.

## dissertation_tables/ — готовые таблицы для текста

| Файл | Содержание |
|------|------------|
| `table_main_results.csv`   | Главная таблица «Результаты» — финалисты обоих таргетов |
| `table_by_modality.csv`    | Лучшая модель на каждой модальности |
| `table_significance.csv`   | Сводка статистических тестов (ablation, cross-modality, pairwise) |
| `table_two_experts.csv`    | Сравнение систем «два эксперта» vs generalist |
| `table_conformal.csv`      | Покрытие и ширина conformal-интервалов |

## Артефакты по шагам

### Шаг 02 — ранжирование
- `01_ranking_wide.csv` — широкая таблица (164×25) с производными LOSO-метриками
- `02_topk_by_target_fset.md`, `03_topk_by_target_version.md`,
  `04_topk_by_target_fset_family.md` — Top-10 по трём уровням группировки
- `05_candidates_intersection.csv` — пересечение топов
- `06_top5_by_target_variant.md`, `07_top5_by_target_modality.md` — узкие Top-5
- `per_subject_all.csv` — длинная сводка per-subject (5212 строк) + сверка со старыми
- `per_subject_rebuild_report.csv` — статусы пересборки per_subject из npy
- `plots/stability_vs_mae.png`, `plots/kalman_gain_by_version.png`

### Шаг 02a — распределение per-subject ошибок
- `eda/per_subject_dist/08_per_subject_table.csv`
- `eda/per_subject_dist/09_trained_vs_untrained.csv` — Mann-Whitney trained vs untrained
- `eda/per_subject_dist/plots/` — гистограммы, scatter с ковариатами

### Шаг 02b — анализ ошибки по времени до порога
- `eda/time_resolved/10_error_by_time_bins.csv`
- `eda/time_resolved/plots/` — кривые ошибки по бинам

### Шаг 03 — ablation with_abs vs noabs
- `11_abs_vs_noabs.csv` — paired Wilcoxon, 72 конфигурации
- `11b_v0011_hrv_coefs.csv` — коэффициенты линейной модели (абсолют vs тренд)
- `plots/abs_vs_noabs_volcano.png`

### Шаг 04 — парные сравнения
- `12_pairwise_wilcoxon.csv` — все пары версий, Wilcoxon + Holm
- `13_pairwise_win_counts.csv` — счёт значимых побед/поражений
- `14_champion_cross_modality.csv` — кросс-модальные сравнения чемпионов
- `plots/pairwise/` — heatmap p_holm по группам

### Шаг 05 — два эксперта (LT1)
- `two_experts/subject_meta.csv` — разметка trained/untrained + ковариаты
- `two_experts/15_per_subject_predictions.csv` — предсказания всех систем
- `two_experts/16_system_comparison.csv` — сравнение систем + Wilcoxon
- `two_experts/17_router_metrics.csv` — важность признаков маршрутизатора
- `two_experts/plots/systems_comparison.png`

### Шаг 06 — SHAP + conformal
- `shap/<tag>/` — beeswarm, bar, shap_importance.csv, shap_by_group.csv
- `conformal/<tag>/intervals.csv` — per-subject интервалы
- `conformal/coverage_summary.csv`, `conformal/conformal_summary.png`

## Важные оговорки
- Все модели обучены LOSO, n=18 (некоторые NN — n=16–17, см. `loso_n_subjects`).
- `loso_mae_median` (по субъектам) систематически ниже window-weighted MAE —
  при цитировании указывать, какая метрика.
- v0011 per_subject не различает variant; для него `noabs` приравнен к `with_abs`.
- npy для lt2 у части NN короче ожидаемого на 17–98 окон (внутренние dropna) —
  это учтено через список субъектов из per_subject.csv.
"""


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    print("Главная таблица результатов…")
    build_main_results_table().to_csv(TABLES / "table_main_results.csv", index=False)
    print("  → table_main_results.csv")

    print("Таблица по модальностям…")
    build_modality_table().to_csv(TABLES / "table_by_modality.csv", index=False)
    print("  → table_by_modality.csv")

    print("Таблица значимости…")
    build_significance_table().to_csv(TABLES / "table_significance.csv", index=False)
    print("  → table_significance.csv")

    print("Таблица «два эксперта»…")
    build_two_experts_table().to_csv(TABLES / "table_two_experts.csv", index=False)
    print("  → table_two_experts.csv")

    print("Таблица conformal…")
    build_conformal_table().to_csv(TABLES / "table_conformal.csv", index=False)
    print("  → table_conformal.csv")

    print("ENV.md…")
    write_env()

    print("README.md…")
    (FINAL / "README.md").write_text(README, encoding="utf-8")
    print("  → README.md")

    print(f"\nГотово. Папка {FINAL} упакована.")


if __name__ == "__main__":
    main()
