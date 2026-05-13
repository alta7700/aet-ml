"""v0011 — Анализ коэффициентов ElasticNet по модальностям.

Для каждой (feature_set, target) комбинации:
  - обучает лучшую EN-модель на всех данных (полный fit, не LOSO)
  - считает усреднённые |coef| по LOSO-фолдам
  - показывает топ-20 признаков и отдельно: какие NIRS-фичи выжили

Запуск:
  uv run python scripts/v0011_coef_analysis.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
# Импортируем утилиты из v0011 напрямую
sys.path.insert(0, str(_ROOT / "scripts"))
from v0011_modality_ablation import (
    prepare_data,
    get_feature_cols,
    NIRS_FEATURES,
    RUNNING_NIRS_FEATURES,
    HRV_FEATURES,
    INTERACTION_FEATURES,
)

# ─── Лучшие EN-конфиги из summary.csv ────────────────────────────────────────
# (alpha, l1_ratio) для каждой (feature_set, target) пары
BEST_EN = {
    ("EMG",          "lt2"): (1.0, 0.2),
    ("EMG",          "lt1"): (1.0, 0.2),
    ("NIRS",         "lt2"): (1.0, 0.2),   # SVR был лучше, берём лучший EN
    ("NIRS",         "lt1"): (1.0, 0.2),
    ("EMG+NIRS",     "lt2"): (1.0, 0.2),
    ("EMG+NIRS",     "lt1"): (1.0, 0.2),
    ("EMG+NIRS+HRV", "lt2"): (0.1, 0.5),
    ("EMG+NIRS+HRV", "lt1"): (1.0, 0.2),
}

TARGET_COL = {
    "lt2": "target_time_to_lt2_center_sec",
    "lt1": "target_time_to_lt1_pchip_sec",
}


def loso_coefs(df: pd.DataFrame, feat_cols: list[str],
               target_col: str, alpha: float, l1: float) -> np.ndarray:
    """Усреднённые |coef| по LOSO-фолдам (в пространстве стандартизованных признаков)."""
    subjects = sorted(df["subject_id"].unique())
    coef_sum = np.zeros(len(feat_cols))

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        mdl = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, random_state=42)

        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        mdl.fit(X_tr, train[target_col].values)
        coef_sum += np.abs(mdl.coef_)

    return coef_sum / len(subjects)


def nirs_group(feat: str) -> str:
    """Определяет группу NIRS-признака для читаемого вывода."""
    all_nirs = set(NIRS_FEATURES + RUNNING_NIRS_FEATURES)
    all_hrv  = set(HRV_FEATURES + INTERACTION_FEATURES)
    if feat in all_nirs:
        return "NIRS"
    if feat in all_hrv:
        return "HRV"
    return "EMG/kin"


def analyse(df_raw: pd.DataFrame, session_params: pd.DataFrame,
            feature_set: str, target: str) -> None:
    target_col = TARGET_COL[target]

    if target not in ("lt1", "lt2"):
        return
    if target == "lt2" and "target_time_to_lt2_center_sec" not in df_raw.columns:
        print(f"  [skip] нет таргета {target_col}")
        return

    df = prepare_data(df_raw, session_params, target)
    if target_col not in df.columns:
        print(f"  [skip] нет колонки {target_col} после prepare_data")
        return

    df = df.dropna(subset=[target_col])
    feat_cols = get_feature_cols(df, feature_set)

    if not feat_cols:
        print(f"  [skip] нет признаков для {feature_set}")
        return

    alpha, l1 = BEST_EN[(feature_set, target)]
    print(f"\n{'='*60}")
    print(f"  {feature_set} / {target.upper()}  |  EN(α={alpha}, l1={l1})")
    print(f"  Субъектов: {df['subject_id'].nunique()}, признаков: {len(feat_cols)}")
    print(f"{'='*60}")

    avg_coef = loso_coefs(df, feat_cols, target_col, alpha, l1)

    coef_df = pd.DataFrame({
        "feature": feat_cols,
        "avg_abs_coef": avg_coef,
        "group": [nirs_group(f) for f in feat_cols],
    }).sort_values("avg_abs_coef", ascending=False)

    # Топ-20 признаков
    top20 = coef_df.head(20)
    print("\nТоп-20 признаков по |coef| (усредн. по LOSO):")
    print(f"{'Признак':<45} {'|coef|':>8}  {'Группа'}")
    print("-" * 65)
    for _, row in top20.iterrows():
        print(f"  {row['feature']:<43} {row['avg_abs_coef']:>8.4f}  {row['group']}")

    # Сколько признаков ненулевые (EN зануляет)
    nonzero = (avg_coef > 1e-6).sum()
    print(f"\nНенулевых признаков: {nonzero} / {len(feat_cols)}")

    # Отдельно NIRS-фичи
    nirs_part = set(NIRS_FEATURES + RUNNING_NIRS_FEATURES)
    nirs_df = coef_df[coef_df["feature"].isin(nirs_part)].copy()
    if not nirs_df.empty:
        nirs_nonzero = (nirs_df["avg_abs_coef"] > 1e-6).sum()
        print(f"\nNIRS-признаки ({len(nirs_df)} из {len(feat_cols)}):"
              f"  {nirs_nonzero} ненулевых")
        print(f"{'Признак':<45} {'|coef|':>8}")
        print("-" * 55)
        for _, row in nirs_df.iterrows():
            marker = "✓" if row["avg_abs_coef"] > 1e-6 else "✗"
            print(f"  {marker} {row['feature']:<42} {row['avg_abs_coef']:>8.4f}")

    # HRV + interaction (если есть)
    hrv_part = set(HRV_FEATURES + INTERACTION_FEATURES)
    hrv_df = coef_df[coef_df["feature"].isin(hrv_part)].copy()
    if not hrv_df.empty:
        hrv_nonzero = (hrv_df["avg_abs_coef"] > 1e-6).sum()
        print(f"\nHRV-признаки ({len(hrv_df)} из {len(feat_cols)}):"
              f"  {hrv_nonzero} ненулевых")
        print(f"{'Признак':<45} {'|coef|':>8}")
        print("-" * 55)
        for _, row in hrv_df.iterrows():
            marker = "✓" if row["avg_abs_coef"] > 1e-6 else "✗"
            print(f"  {marker} {row['feature']:<42} {row['avg_abs_coef']:>8.4f}")


def main() -> None:
    data_dir = DEFAULT_DATASET_DIR
    merged_path = data_dir / "merged_features_ml.parquet"
    session_params_path = data_dir / "session_params.parquet"

    print(f"Загружаю данные из {merged_path}")
    df_raw = pd.read_parquet(merged_path)
    session_params = pd.read_parquet(session_params_path) if session_params_path.exists() else pd.DataFrame()

    # Анализируем только наборы с >1 модальностью (интересует вклад NIRS/HRV)
    configs = [
        ("EMG+NIRS",     "lt2"),
        ("EMG+NIRS",     "lt1"),
        ("EMG+NIRS+HRV", "lt2"),
        ("EMG+NIRS+HRV", "lt1"),
    ]

    for feature_set, target in configs:
        analyse(df_raw, session_params, feature_set, target)

    print("\n\nДополнительно — только EMG и только NIRS:")
    for feature_set, target in [("EMG", "lt2"), ("NIRS", "lt2")]:
        analyse(df_raw, session_params, feature_set, target)


if __name__ == "__main__":
    main()
