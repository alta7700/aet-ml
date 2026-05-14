"""Восстановление per_subject метрик для lt1 (и lt2 как проверка) из npy.

Контекст: в results/<version>/per_subject.csv во ВСЕХ 10 версиях
сохранены только записи target=lt2. Для lt1 per-subject отсутствует,
хотя ypred_lt1_*.npy и ytrue_lt1_*.npy сохранены.

В npy предсказания идут в порядке `sorted(subject_id)` × `sort_values(window_start_sec)`
после prepare_data + dropna(target_col). Воспроизводим этот же порядок и
режем npy по границам субъектов → получаем per-subject MAE и R².

Результат: results/<version>/per_subject_full.csv с колонками
    variant, feature_set, target, subject_id, mae_min, r2

Запуск:
    PYTHONPATH=. uv run python scripts/final_rebuild_per_subject_lt1.py
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

# Импортируем prepare_data из эталона v0011. Все NN-версии используют тот же
# препроцессинг (см. notes/remote_gpu_usage.md).
from v0011_modality_ablation import prepare_data  # noqa: E402

DATASET_DIR = ROOT / "dataset"
RESULTS = ROOT / "results"

VERSIONS_NN = ["v0101", "v0102", "v0103", "v0104", "v0105",
               "v0106a", "v0106b", "v0106c", "v0107"]
VERSION_V0011 = "v0011"

# Целевая колонка по таргету (как в v0011_modality_ablation).
TARGET_COL = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}

# Возможные fset_tag (имя в npy). У v0011 есть HRV отдельно.
FSET_TAGS = ["EMG", "NIRS", "EMG_NIRS", "EMG_NIRS_HRV"]
FSET_TAGS_V0011 = FSET_TAGS + ["HRV"]


def build_subject_index(df_raw: pd.DataFrame,
                        session_params: pd.DataFrame,
                        target: str) -> pd.DataFrame:
    """Возвращает df с колонками subject_id, window_id в порядке npy."""
    df_prep = prepare_data(df_raw, session_params, target)
    target_col = TARGET_COL[target]
    if target_col not in df_prep.columns:
        return pd.DataFrame()
    df = df_prep.dropna(subset=[target_col]).copy()

    # Имитация loso_predict: subjects = sorted(unique); для каждого
    # subject_id порядок sort_values("window_start_sec").
    subjects = sorted(df["subject_id"].unique())
    parts = []
    for s in subjects:
        sub = df[df["subject_id"] == s].sort_values("window_start_sec")
        parts.append(sub[["subject_id", "window_id", "window_start_sec",
                          target_col]])
    return pd.concat(parts, ignore_index=True)


def per_subject_metrics(subj_idx: pd.DataFrame,
                        y_pred: np.ndarray,
                        y_true: np.ndarray) -> pd.DataFrame:
    """Возвращает MAE_min и R² по каждому subject_id."""
    if len(subj_idx) != len(y_pred):
        raise ValueError(
            f"Length mismatch: subj_idx={len(subj_idx)}, npy={len(y_pred)}"
        )
    rows = []
    for sid, sub in subj_idx.groupby("subject_id"):
        idx = sub.index.values
        yp = y_pred[idx]
        yt = y_true[idx]
        if len(yp) < 2 or not np.isfinite(yp).all() or not np.isfinite(yt).all():
            rows.append({"subject_id": sid, "mae_min": float("nan"), "r2": float("nan")})
            continue
        mae = mean_absolute_error(yt, yp) / 60.0
        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = float("nan")
        rows.append({"subject_id": sid, "mae_min": round(float(mae), 4),
                     "r2": round(float(r2), 3)})
    return pd.DataFrame(rows)


def rebuild_from_npy(version: str,
                     subj_idx_by_target: dict[str, pd.DataFrame],
                     fset_tags: list[str],
                     has_noabs: bool) -> pd.DataFrame:
    """Восстанавливает per-subject из npy одной версии."""
    vdir = RESULTS / version
    rows: list[dict] = []

    variants = [("with_abs", vdir)]
    if has_noabs:
        variants.append(("noabs", vdir / "noabs"))

    for variant, vroot in variants:
        if not vroot.exists():
            continue
        for target in ("lt1", "lt2"):
            subj_idx = subj_idx_by_target[target]
            if subj_idx.empty:
                continue
            for fset_tag in fset_tags:
                yp_path = vroot / f"ypred_{target}_{fset_tag}.npy"
                yt_path = vroot / f"ytrue_{target}_{fset_tag}.npy"
                if not (yp_path.exists() and yt_path.exists()):
                    continue
                yp = np.load(yp_path)
                yt = np.load(yt_path)
                if len(yp) != len(subj_idx):
                    print(f"    [skip] {version}/{variant}/{target}/{fset_tag}: "
                          f"npy={len(yp)}, expected={len(subj_idx)}")
                    continue
                m = per_subject_metrics(subj_idx, yp, yt)
                m["variant"] = variant
                m["feature_set"] = fset_tag.replace("_", "+")
                m["target"] = target
                rows.append(m)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["variant", "feature_set", "target", "subject_id", "mae_min", "r2"]]


def merge_with_old(version: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """Сливает новый (lt1, NN-схема) с существующим per_subject.csv (lt2).

    Для NN-версий — берём из старого все lt2 строки.
    Для v0011 — старый имеет схему (feature_set, target, model, ...);
    сворачиваем его в (variant, feature_set, target, subject_id, mae_min, r2),
    выбирая для каждой (target, fset) лучшую внутреннюю модель по медиане MAE.
    """
    old_path = RESULTS / version / "per_subject.csv"
    if not old_path.exists():
        return new_df

    old = pd.read_csv(old_path)
    if version == VERSION_V0011:
        # Схема: feature_set, target, model, subject_id, mae_min, r2.
        med = (old.groupby(["feature_set", "target", "model"])["mae_min"]
                  .median().reset_index())
        winners = med.loc[med.groupby(["feature_set", "target"])["mae_min"]
                             .idxmin()][["feature_set", "target", "model"]]
        old_norm = old.merge(winners, on=["feature_set", "target", "model"])
        # variant в v0011 не различается; ставим with_abs.
        old_norm["variant"] = "with_abs"
        old_norm = old_norm[["variant", "feature_set", "target",
                             "subject_id", "mae_min", "r2"]]
    elif version == "v0107":
        # v0107: variant, feature_set, target, subject_id, mae_ens_min, mae_tcn_min, ...
        old_norm = old.rename(columns={"mae_ens_min": "mae_min"}).copy()
        old_norm["r2"] = float("nan")
        old_norm = old_norm[["variant", "feature_set", "target",
                             "subject_id", "mae_min", "r2"]]
    elif "variant" in old.columns:
        old_norm = old[["variant", "feature_set", "target",
                        "subject_id", "mae_min", "r2"]].copy()
    else:
        return new_df

    # Берём из старого только те (variant, fset, target), которых нет в new.
    new_keys = set(zip(new_df["variant"], new_df["feature_set"], new_df["target"]))
    mask = ~old_norm.apply(
        lambda r: (r["variant"], r["feature_set"], r["target"]) in new_keys,
        axis=1,
    )
    combined = pd.concat([new_df, old_norm[mask]], ignore_index=True)
    return combined


def main() -> None:
    print("Загрузка merged_features_ml + session_params...")
    df_raw = pd.read_parquet(DATASET_DIR / "merged_features_ml.parquet")
    session_params = pd.read_parquet(DATASET_DIR / "session_params.parquet")

    print("Построение индекса субъектов по таргету...")
    subj_idx_by_target = {}
    for tgt in ("lt1", "lt2"):
        idx = build_subject_index(df_raw, session_params, tgt)
        print(f"  {tgt}: {len(idx)} окон, {idx['subject_id'].nunique()} субъектов")
        subj_idx_by_target[tgt] = idx

    print()
    # NN-версии: lt1 из npy, lt2 — из старого per_subject.csv.
    for v in VERSIONS_NN:
        print(f"[{v}]")
        new_df = rebuild_from_npy(v, subj_idx_by_target,
                                   fset_tags=FSET_TAGS, has_noabs=True)
        full = merge_with_old(v, new_df)
        if full.empty:
            print("  нет данных, пропуск")
            continue
        out_path = RESULTS / v / "per_subject_full.csv"
        full.to_csv(out_path, index=False)
        n_lt1 = int((full["target"] == "lt1").sum())
        n_lt2 = int((full["target"] == "lt2").sum())
        print(f"  → {out_path.name} (rows: lt1={n_lt1}, lt2={n_lt2})")

    # v0011: npy в корне для лучшей внутренней модели зоопарка на каждую (target, fset).
    # noabs/ для v0011 пуст (npy не сохранён). Variant=with_abs.
    print(f"[{VERSION_V0011}]")
    new_v11 = rebuild_from_npy(VERSION_V0011, subj_idx_by_target,
                                fset_tags=FSET_TAGS_V0011, has_noabs=False)
    full_v11 = merge_with_old(VERSION_V0011, new_v11)
    out_path = RESULTS / VERSION_V0011 / "per_subject_full.csv"
    full_v11.to_csv(out_path, index=False)
    n_lt1 = int((full_v11["target"] == "lt1").sum())
    n_lt2 = int((full_v11["target"] == "lt2").sum())
    print(f"  → {out_path.name} (rows: lt1={n_lt1}, lt2={n_lt2})")

    print("\nГотово.")


if __name__ == "__main__":
    main()
