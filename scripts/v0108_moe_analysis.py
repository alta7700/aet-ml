"""v0108 — Анализ per-subject метрик для проектирования MoE (Mixture of Experts)

Шаги:
  1. Разметка тренированности субъектов (LT2 > 13 мин → trained=1)
  2. Сбор всех per_subject.csv, нормализация в единый формат
  3. MAE по группам (trained vs untrained) для каждой модели
  4. Матрица «лучший эксперт» per subject
  5. Вывод рекомендаций по выбору экспертов для v0108
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

_ROOT    = Path(__file__).resolve().parents[1]
RESULTS  = _ROOT / "results"
DATASET  = _ROOT / "dataset"
OUT_DIR  = RESULTS / "v0108_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Конфиг анализа
TARGET_FOCUS   = ["lt2"]            # "lt2" / "lt1" / оба → ["lt1", "lt2"]
FSET_FOCUS     = "EMG+NIRS+HRV"    # лучший набор признаков
VARIANT_FOCUS  = "noabs"            # noabs убирает монотонные абсолютные признаки
TRAINED_THRESH = 13.0               # мин — порог LT2 для тренированных
EXCLUDE        = {"v0106c"}         # слабые/сломанные скрипты


# ─── Шаг 1: разметка тренированности ─────────────────────────────────────────

def make_trained_map() -> pd.Series:
    """subject_id → 1 (trained) / 0 (untrained) по порогу LT2."""
    tgt = pd.read_parquet(DATASET / "targets.parquet")
    lt2_min = tgt.groupby("subject_id")["target_time_to_lt2_center_sec"].max() / 60
    trained = (lt2_min > TRAINED_THRESH).astype(int)
    print(f"[разметка] trained={trained.sum()}, untrained={(trained==0).sum()}")
    print(f"  trained:   {sorted(trained[trained==1].index.tolist())}")
    print(f"  untrained: {sorted(trained[trained==0].index.tolist())}")
    return trained.rename("trained")


# ─── Шаг 2: загрузка и нормализация per_subject ───────────────────────────────

def load_all_per_subject(trained_map: pd.Series) -> pd.DataFrame:
    """Загружает все per_subject.csv, возвращает единый DataFrame."""
    frames = []

    # v0011: колонка model вместо variant, берём лучшую модель per subject
    p = RESULTS / "v0011" / "per_subject.csv"
    if p.exists():
        df = pd.read_csv(p)
        # лучший Kalman-результат на субъект × fset × target
        df = df.loc[df.groupby(["feature_set", "target", "subject_id"])["mae_min"].idxmin()]
        df["variant"] = "best_model"
        df["script"]  = "v0011"
        frames.append(df[["script", "variant", "feature_set", "target", "subject_id", "mae_min"]])

    # v0102–v0106: одинаковая структура
    for ver in ["v0102", "v0103", "v0104", "v0105", "v0106a", "v0106b", "v0106c"]:
        if ver in EXCLUDE:
            print(f"  [пропуск] {ver} (в EXCLUDE)")
            continue
        p = RESULTS / ver / "per_subject.csv"
        if not p.exists():
            print(f"  [пропуск] {ver} (нет файла)")
            continue
        df = pd.read_csv(p)
        df["script"] = ver
        frames.append(df[["script", "variant", "feature_set", "target", "subject_id", "mae_min"]])

    # v0107: mae_ens_min → mae_min
    p = RESULTS / "v0107" / "per_subject.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df.rename(columns={"mae_ens_min": "mae_min"})
        df["script"] = "v0107"
        frames.append(df[["script", "variant", "feature_set", "target", "subject_id", "mae_min"]])

    all_ps = pd.concat(frames, ignore_index=True)
    all_ps = all_ps.join(trained_map, on="subject_id")

    print(f"\n[загрузка] итого строк: {len(all_ps)}, скриптов: {all_ps['script'].nunique()}")
    return all_ps


# ─── Шаг 3: MAE по группам trained vs untrained ────────────────────────────────

def analyze_group_mae(all_ps: pd.DataFrame) -> pd.DataFrame:
    """Для каждой модели считает средний MAE на trained/untrained группах."""
    focus = all_ps[
        (all_ps["variant"].isin([VARIANT_FOCUS, "best_model"])) &
        (all_ps["feature_set"] == FSET_FOCUS) &
        (all_ps["target"].isin(TARGET_FOCUS))
    ]

    group_mae = (
        focus
        .groupby(["script", "target", "trained"])["mae_min"]
        .mean()
        .unstack("trained")
        .rename(columns={0: "mae_untrained", 1: "mae_trained"})
        .round(3)
    )
    group_mae["delta"] = (group_mae["mae_trained"] - group_mae["mae_untrained"]).round(3)
    # delta < 0 → модель лучше на тренированных
    # delta > 0 → модель хуже на тренированных
    group_mae["ratio"] = (group_mae["mae_trained"] / group_mae["mae_untrained"]).round(2)

    print("\n" + "="*65)
    print(f"ШАГ 3: MAE по группам ({FSET_FOCUS}, {VARIANT_FOCUS}, target={TARGET_FOCUS})")
    print("="*65)
    print(group_mae.sort_values(["target", "mae_untrained"]).to_string())
    return group_mae


# ─── Шаг 4: матрица «лучший эксперт» per subject ──────────────────────────────

def best_expert_matrix(all_ps: pd.DataFrame) -> pd.DataFrame:
    """Для каждого субъекта × target находит лучшую модель."""
    focus = all_ps[
        (all_ps["variant"].isin([VARIANT_FOCUS, "best_model"])) &
        (all_ps["feature_set"] == FSET_FOCUS) &
        (all_ps["target"].isin(TARGET_FOCUS))
    ]

    # лучшая модель на субъект × target
    idx = focus.groupby(["subject_id", "target"])["mae_min"].idxmin()
    best = focus.loc[idx][["subject_id", "target", "script", "mae_min", "trained"]].copy()
    best = best.rename(columns={"script": "best_model", "mae_min": "best_mae"})

    # сводная таблица по субъектам
    pivot = best.pivot_table(
        index=["subject_id", "trained"],
        columns="target",
        values=["best_model", "best_mae"],
        aggfunc="first"
    )
    pivot.columns = ["_".join(c) for c in pivot.columns]
    pivot = pivot.reset_index().sort_values("trained")

    print("\n" + "="*65)
    print("ШАГ 4: Лучший эксперт per subject")
    print("="*65)
    print(pivot.to_string(index=False))

    # частота выигрышей по моделям
    print("\n--- Частота «побед» по моделям ---")
    wins = best.groupby(["trained", "best_model"]).size().unstack(fill_value=0)
    print(wins.to_string())

    return best


# ─── Шаг 5: рекомендации для v0108 ───────────────────────────────────────────

def print_recommendations(group_mae: pd.DataFrame, best: pd.DataFrame):
    print("\n" + "="*65)
    print("ШАГ 5: Рекомендации для v0108 MoE")
    print("="*65)

    for tgt in TARGET_FOCUS:
        sub = group_mae.xs(tgt, level="target") if "target" in group_mae.index.names else group_mae
        print(f"\n  TARGET: {tgt.upper()}")

        # лучшая модель на trained
        best_tr  = sub["mae_trained"].idxmin()
        best_utr = sub["mae_untrained"].idxmin()
        print(f"  Лучший на TRAINED:    {best_tr}  (MAE={sub.loc[best_tr,'mae_trained']:.3f})")
        print(f"  Лучший на UNTRAINED:  {best_utr}  (MAE={sub.loc[best_utr,'mae_untrained']:.3f})")

        # модели с наибольшей специализацией (|delta| > порога)
        specialized = sub[sub["delta"].abs() > 0.3].sort_values("delta")
        if not specialized.empty:
            print(f"  Специализированные (|Δ| > 0.3 мин):")
            for mdl, row in specialized.iterrows():
                direction = "лучше на TRAINED" if row["delta"] < 0 else "лучше на UNTRAINED"
                print(f"    {mdl}: Δ={row['delta']:+.3f} → {direction}")
        else:
            print("  Нет явно специализированных моделей")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("v0108 MoE Analysis")
    print(f"  target={TARGET_FOCUS}, fset={FSET_FOCUS}, variant={VARIANT_FOCUS}")
    print(f"  trained порог LT2 > {TRAINED_THRESH} мин\n")

    trained_map = make_trained_map()
    all_ps      = load_all_per_subject(trained_map)

    # Сохраняем объединённый датасет
    all_ps.to_csv(OUT_DIR / "all_per_subject.csv", index=False)
    print(f"\n  → сохранено: {OUT_DIR / 'all_per_subject.csv'}")

    group_mae = analyze_group_mae(all_ps)
    group_mae.to_csv(OUT_DIR / "group_mae.csv")

    best = best_expert_matrix(all_ps)
    best.to_csv(OUT_DIR / "best_expert_per_subject.csv", index=False)

    print_recommendations(group_mae, best)

    print(f"\n✅ Готово: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
