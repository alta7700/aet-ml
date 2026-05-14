"""Единый скрипт: пересборка per_subject метрик из ypred/ytrue .npy.

Заменяет `final_rebuild_per_subject_lt1.py` (тот покрывал только lt1).
Здесь и lt1, и lt2 пересобираются из npy для ВСЕХ интересующих нас комбинаций
(version × variant × target × feature_set). На выходе сравниваем с записанным
в существующем per_subject.csv и репортим расхождения — на случай, если
старая выгрузка протухла.

Алгоритм
--------
1. Загрузить prepare_data → df_<target> (4130 окон/18 субъектов для lt1,
   3651/18 для lt2). Это исходное «полное множество» окон.
2. Для каждой комбинации `(version, variant, target, fset_tag)`:
   - Найти npy ypred/ytrue.
   - Из старого per_subject.csv этой комбинации взять список subject_id
     (это «кто реально вошёл в LOSO» для этого скрипта). Если файла нет
     либо колонка subject_id неполна — пробуем все 18.
   - df_subset = df[df.subject_id.isin(subjects)].sort_values(["subject_id",
     "window_start_sec"]) — порядок строго sorted, как в LOSO.
   - Проверяем len(df_subset) == len(npy). Если не сходится — пробуем
     дроп субъектов по одному (на случай, если per_subject.csv тоже неточен).
   - Считаем per-subject MAE_min и R² → пишем в общий long-CSV.
3. Сравниваем с существующими MAE_per_subject — пишем колонку diff.

Артефакты
---------
- results/<version>/per_subject_full.csv — обновлённые per-subject метрики
  по версии (схема: variant, feature_set, target, subject_id, mae_min, r2).
- results/final/per_subject_all.csv — длинная сводная таблица + колонки
  `mae_min_old`, `mae_min_diff` для сверки с per_subject.csv.

Запуск
------
    PYTHONPATH=. uv run python scripts/final_per_subject_from_npy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v0011_modality_ablation import prepare_data  # noqa: E402

DATASET_DIR = ROOT / "dataset"
RESULTS = ROOT / "results"
FINAL = RESULTS / "final"

VERSIONS_NN = ["v0101", "v0102", "v0103", "v0104", "v0105",
               "v0106a", "v0106b", "v0106c", "v0107"]
VERSION_V0011 = "v0011"

TARGET_COL = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}

FSET_TAGS_NN = ["EMG", "NIRS", "EMG_NIRS", "EMG_NIRS_HRV"]
FSET_TAGS_V0011 = FSET_TAGS_NN + ["HRV"]


# ─────────────────────── Базовый индекс ───────────────────────

def build_df_target(df_raw: pd.DataFrame,
                    session_params: pd.DataFrame,
                    target: str) -> pd.DataFrame:
    """Возвращает df после prepare_data + dropna(target_col), с колонками
    subject_id, window_id, window_start_sec, target_col.
    """
    df = prepare_data(df_raw, session_params, target)
    target_col = TARGET_COL[target]
    if target_col not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=[target_col]).copy()
    return df[["subject_id", "window_id", "window_start_sec", target_col]]


# ─────────────────────── Чтение old per_subject ───────────────────────

def load_old_per_subject(version: str) -> pd.DataFrame:
    """Возвращает старый per_subject.csv в нормализованной схеме
    (variant, feature_set, target, subject_id, mae_min)."""
    path = RESULTS / version / "per_subject.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if version == VERSION_V0011:
        # feature_set, target, model, subject_id, mae_min, r2
        df["variant"] = "with_abs"
        return df[["variant", "feature_set", "target", "subject_id", "mae_min"]]
    if version == "v0107":
        df = df.rename(columns={"mae_ens_min": "mae_min"})
    cols = ["variant", "feature_set", "target", "subject_id", "mae_min"]
    return df[[c for c in cols if c in df.columns]]


# ─────────────────────── Восстановление одного npy ───────────────────────

def slice_per_subject(df_target: pd.DataFrame,
                      subjects: list[str],
                      y_pred: np.ndarray,
                      y_true: np.ndarray) -> pd.DataFrame:
    """Режет npy по границам субъектов и считает MAE_min, R² per-subject."""
    parts = []
    cur = 0
    rows: list[dict] = []
    for s in subjects:
        sub = df_target[df_target["subject_id"] == s].sort_values("window_start_sec")
        n = len(sub)
        if n == 0:
            continue
        yp = y_pred[cur:cur + n]
        yt = y_true[cur:cur + n]
        cur += n
        if len(yp) < 2 or not np.isfinite(yp).all() or not np.isfinite(yt).all():
            rows.append({"subject_id": s, "mae_min": float("nan"), "r2": float("nan")})
            continue
        mae = mean_absolute_error(yt, yp) / 60.0
        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = float("nan")
        rows.append({"subject_id": s,
                     "mae_min": round(float(mae), 4),
                     "r2": round(float(r2), 3)})
    if cur != len(y_pred):
        # Защита: суммарная длина не совпадает.
        return pd.DataFrame()
    return pd.DataFrame(rows)


def determine_subjects(df_target: pd.DataFrame,
                        npy_len: int,
                        candidate_subjects: list[str]) -> list[str] | None:
    """Подбирает подсписок субъектов так, чтобы суммарное число окон совпало
    с длиной npy. Сначала пробует полный список, потом удаляет «маленьких»
    субъектов до совпадения. Возвращает None если не получилось.
    """
    counts = df_target.groupby("subject_id").size()
    counts = counts.reindex(candidate_subjects).dropna().astype(int)
    sorted_subjects = sorted(counts.index)
    total = int(counts.sum())
    if total == npy_len:
        return sorted_subjects

    # Перебираем удаление подмножеств размером 1..3 (маленькие subject_id вначале не дропаем)
    n_diff = total - npy_len
    if n_diff <= 0:
        return None

    # Кандидаты на удаление: субъекты, чей размер равен n_diff (часто 1 субъект)
    matches = counts[counts == n_diff].index.tolist()
    if matches:
        keep = [s for s in sorted_subjects if s not in matches[:1]]
        return keep

    # Иначе ищем пару, тройку с такой суммой
    from itertools import combinations
    for r in (2, 3):
        for combo in combinations(sorted_subjects, r):
            if int(counts[list(combo)].sum()) == n_diff:
                keep = [s for s in sorted_subjects if s not in combo]
                return keep
    return None


def rebuild_one(df_target: pd.DataFrame,
                yp_path: Path,
                yt_path: Path,
                expected_subjects: list[str]) -> pd.DataFrame | None:
    yp = np.load(yp_path)
    yt = np.load(yt_path)
    if len(yp) != len(yt):
        return None

    subjects = determine_subjects(df_target, len(yp), expected_subjects)
    if subjects is None:
        return None
    df_subset = df_target[df_target["subject_id"].isin(subjects)]
    if len(df_subset) != len(yp):
        return None
    return slice_per_subject(df_target, subjects, yp, yt)


# ─────────────────────── Сборка по версии ───────────────────────

def process_version(version: str,
                    df_by_target: dict[str, pd.DataFrame],
                    old_per_subject: pd.DataFrame,
                    is_v0011: bool) -> tuple[pd.DataFrame, list[dict]]:
    """Возвращает (per_subject_full_df, report_rows)."""
    vdir = RESULTS / version
    rows: list[dict] = []
    report: list[dict] = []

    fsets = FSET_TAGS_V0011 if is_v0011 else FSET_TAGS_NN
    variants = [("with_abs", vdir)]
    if not is_v0011:
        variants.append(("noabs", vdir / "noabs"))

    for variant, vroot in variants:
        if not vroot.exists():
            continue
        for target in ("lt1", "lt2"):
            df_t = df_by_target[target]
            if df_t.empty:
                continue
            all_subjects = sorted(df_t["subject_id"].unique())
            # Если есть старый per_subject — берём список оттуда.
            mask = (
                (old_per_subject["target"] == target)
                & (old_per_subject["variant"] == variant)
            )
            # для v0011 variant=with_abs искусственно, fset matching ниже
            for fset_tag in fsets:
                yp_path = vroot / f"ypred_{target}_{fset_tag}.npy"
                yt_path = vroot / f"ytrue_{target}_{fset_tag}.npy"
                if not (yp_path.exists() and yt_path.exists()):
                    continue
                fset_label = fset_tag.replace("_", "+")
                # Список ожидаемых субъектов из старого per_subject
                expected = all_subjects
                if not old_per_subject.empty:
                    sub_mask = mask & (old_per_subject["feature_set"] == fset_label)
                    expected_from_old = sorted(
                        old_per_subject.loc[sub_mask, "subject_id"].unique().tolist()
                    )
                    if expected_from_old:
                        expected = expected_from_old

                ps = rebuild_one(df_t, yp_path, yt_path, expected)
                if ps is None or ps.empty:
                    report.append({"version": version, "variant": variant,
                                   "target": target, "feature_set": fset_label,
                                   "status": "FAILED size match"})
                    continue
                ps["variant"] = variant
                ps["feature_set"] = fset_label
                ps["target"] = target
                rows.append(ps)
                report.append({"version": version, "variant": variant,
                               "target": target, "feature_set": fset_label,
                               "status": f"ok (n={len(ps)})"})

    if not rows:
        return pd.DataFrame(), report
    out = pd.concat(rows, ignore_index=True)
    return out[["variant", "feature_set", "target",
                "subject_id", "mae_min", "r2"]], report


# ─────────────────────── Сверка new vs old ───────────────────────

def compare_with_old(new_df: pd.DataFrame,
                     old_df: pd.DataFrame,
                     version: str) -> pd.DataFrame:
    """К new_df приклеивает mae_min_old и mae_min_diff."""
    if old_df.empty:
        new_df = new_df.copy()
        new_df["mae_min_old"] = float("nan")
        new_df["mae_min_diff"] = float("nan")
        new_df["version"] = version
        return new_df

    # Для v0011 в old.variant всё with_abs (синтетически). У NN — обе variants.
    key = ["variant", "feature_set", "target", "subject_id"]
    merged = new_df.merge(
        old_df.rename(columns={"mae_min": "mae_min_old"}),
        on=key, how="left",
    )
    merged["mae_min_diff"] = (merged["mae_min"] - merged["mae_min_old"]).round(4)
    merged["version"] = version
    return merged


# ─────────────────────── Main ───────────────────────

def main() -> None:
    print("Загрузка merged_features_ml + session_params…")
    df_raw = pd.read_parquet(DATASET_DIR / "merged_features_ml.parquet")
    session_params = pd.read_parquet(DATASET_DIR / "session_params.parquet")

    print("Подготовка df_target…")
    df_by_target = {tgt: build_df_target(df_raw, session_params, tgt)
                    for tgt in ("lt1", "lt2")}
    for tgt, df in df_by_target.items():
        print(f"  {tgt}: {len(df)} окон, {df['subject_id'].nunique()} субъектов")

    all_report: list[dict] = []
    all_long_rows: list[pd.DataFrame] = []

    for version in VERSIONS_NN + [VERSION_V0011]:
        print(f"\n[{version}]")
        old = load_old_per_subject(version)
        full, report = process_version(
            version, df_by_target, old, is_v0011=(version == VERSION_V0011),
        )
        all_report.extend(report)
        for r in report:
            print(f"  {r['variant']}/{r['target']}/{r['feature_set']}: {r['status']}")
        if not full.empty:
            out_path = RESULTS / version / "per_subject_full.csv"
            full.to_csv(out_path, index=False)
            print(f"  → {out_path.name}")

            merged = compare_with_old(full, old, version)
            all_long_rows.append(merged)

    FINAL.mkdir(parents=True, exist_ok=True)
    if all_long_rows:
        long_df = pd.concat(all_long_rows, ignore_index=True)
        long_df = long_df[["version", "variant", "feature_set", "target",
                           "subject_id", "mae_min", "mae_min_old", "mae_min_diff",
                           "r2"]]
        long_path = FINAL / "per_subject_all.csv"
        long_df.to_csv(long_path, index=False)
        print(f"\n→ {long_path} ({long_df.shape})")

        # Сводка расхождений
        big_diff = long_df[long_df["mae_min_diff"].abs() > 0.01]
        if not big_diff.empty:
            print(f"\nОбнаружены расхождения с per_subject.csv: {len(big_diff)} строк (|diff|>0.01 мин)")
            print(big_diff.groupby(["version", "target"]).size().to_string())
        else:
            print("\nВсе значения per_subject совпадают со старым per_subject.csv (|diff|≤0.01).")

    rep_df = pd.DataFrame(all_report)
    rep_path = FINAL / "per_subject_rebuild_report.csv"
    rep_df.to_csv(rep_path, index=False)
    print(f"→ {rep_path}")


if __name__ == "__main__":
    main()
