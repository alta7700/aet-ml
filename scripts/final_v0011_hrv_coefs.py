"""Мини-проверка: коэффициенты линейной модели v0011 на HRV / lt2.

Цель — подтвердить гипотезу, что линейная модель опирается одновременно на
абсолютный якорь (hrv_mean_rr_ms) и на трендово-вариабельностные признаки
(DFA-α1, RMSSD, SDNN). Обучаем модель на всех данных (HRV / lt2),
стандартизованные коэффициенты → вклад каждого признака.

Запуск:
    PYTHONPATH=. uv run python scripts/final_v0011_hrv_coefs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v0011_modality_ablation import prepare_data, get_feature_cols  # noqa: E402

DATASET_DIR = ROOT / "dataset"
FINAL = ROOT / "results" / "final"

TARGET_COL = "target_time_to_lt2_center_sec"


def main() -> None:
    df_raw = pd.read_parquet(DATASET_DIR / "merged_features_ml.parquet")
    sp = pd.read_parquet(DATASET_DIR / "session_params.parquet")

    df = prepare_data(df_raw, sp, "lt2")
    df = df.dropna(subset=[TARGET_COL])
    feat_cols = get_feature_cols(df, "HRV")
    print(f"HRV feature_cols ({len(feat_cols)}): {feat_cols}")

    X = df[feat_cols].values
    y = df[TARGET_COL].values

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_proc = sc.fit_transform(imp.fit_transform(X))

    # Ridge(α=1000) — лучшая по loso_mae_median в шаге 02.
    model = Ridge(alpha=1000.0)
    model.fit(X_proc, y)

    coefs = pd.Series(model.coef_, index=feat_cols).sort_values(key=np.abs, ascending=False)
    coef_df = pd.DataFrame({
        "feature": coefs.index,
        "std_coef": coefs.values.round(2),
        "abs_coef": np.abs(coefs.values).round(2),
        "share_pct": (np.abs(coefs.values) / np.abs(coefs.values).sum() * 100).round(1),
    })
    out_path = FINAL / "11b_v0011_hrv_coefs.csv"
    coef_df.to_csv(out_path, index=False)

    print("\nСтандартизованные коэффициенты Ridge(α=1000), HRV / lt2:")
    print(coef_df.to_string(index=False))
    print(f"\n→ {out_path}")

    # Группировка: абсолютный уровень vs вариабельность/тренд
    abs_feats = {"hrv_mean_rr_ms"}
    abs_share = coef_df[coef_df["feature"].isin(abs_feats)]["share_pct"].sum()
    var_share = coef_df[~coef_df["feature"].isin(abs_feats)]["share_pct"].sum()
    print(f"\nВклад в |коэффициенты|:")
    print(f"  Абсолютный уровень (hrv_mean_rr_ms): {abs_share:.1f}%")
    print(f"  Вариабельность/тренд (SDNN, RMSSD, SD1/SD2, DFA-α1): {var_share:.1f}%")


if __name__ == "__main__":
    main()
