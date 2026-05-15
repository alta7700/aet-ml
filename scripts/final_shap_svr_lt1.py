"""SHAP-интерпретация для фактических LT1-победителей: SVR с RBF-ядром.

Существующий `final_shap_conformal.py` использует LinearExplainer на Ridge α=1000,
что для LT2 (где победитель — Ridge / ElasticNet) корректно, но для LT1 (где
победитель — SVR с RBF) даёт лишь линейную проекцию.

Этот скрипт обучает фактическую SVR на полной выборке LT1 и считает SHAP через
PermutationExplainer (model-agnostic) на подвыборке окон.

Артефакты — в results/final/shap/lt1_EN_svr/ и results/final/shap/lt1_ENH_svr/.

Запуск:
    PYTHONPATH=. uv run python scripts/final_shap_svr_lt1.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v0011_modality_ablation import prepare_data, get_feature_cols  # noqa: E402

DATASET_DIR = ROOT / "dataset"
SHAP_DIR = ROOT / "results" / "final" / "shap"
TARGET_COL = "target_time_to_lt1_pchip_sec"

# SVR-гиперпараметры — победитель LT1 из v0011/best_per_set.csv
SVR_C = 100.0
SVR_EPSILON = 1.0
SVR_KERNEL = "rbf"

# Параметры PermutationExplainer
N_BACKGROUND = 100   # фон для baseline
N_TEST = 300         # подвыборка окон для оценки SHAP
SEED = 42

# Число процессов joblib для параллельного расчёта SHAP по чанкам X_te.
# Bottleneck — SVR.predict (RBF), однопоточный → выигрыш почти линейный.
N_JOBS = 10

CONFIGS = [
    ("EMG+NIRS",     "lt1_EN_svr"),
    ("EMG+NIRS+HRV", "lt1_ENH_svr"),
]


def _shap_chunk(X_chunk: np.ndarray, predict_fn, X_bg: np.ndarray,
                max_evals: int):
    """Считает SHAP на одном чанке тестовых окон в отдельном процессе.

    Каждый воркер создаёт собственный PermutationExplainer (он не разделяет
    состояние между чанками, поэтому конкатенация результатов корректна).
    """
    explainer = shap.PermutationExplainer(predict_fn, X_bg, max_evals=max_evals)
    return explainer(X_chunk)


def feature_group(name: str) -> str:
    if name.startswith("z_vl_") or name.startswith("vl_"):
        return "EMG"
    if name.startswith("z_") and not name.startswith("z_vl_"):
        return "kinematics"
    if name.startswith("hrv_"):
        return "HRV"
    if (name.startswith("trainred_") or name.startswith("smo2_")
            or name.startswith("hhb_")):
        return "NIRS"
    if name.startswith("feat_"):
        return "interaction"
    return "other"


def run_one(feature_set: str, tag: str, df: pd.DataFrame) -> None:
    out_dir = SHAP_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = get_feature_cols(df, feature_set)
    print(f"\n[{tag}] feature_set={feature_set}, {len(feat_cols)} признаков, "
          f"{len(df)} окон")

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X = sc.fit_transform(imp.fit_transform(df[feat_cols].values))
    y = df[TARGET_COL].values

    print(f"  Обучение SVR(C={SVR_C}, ε={SVR_EPSILON}, kernel={SVR_KERNEL})…")
    t0 = time.perf_counter()
    model = SVR(C=SVR_C, epsilon=SVR_EPSILON, kernel=SVR_KERNEL)
    model.fit(X, y)
    print(f"  ✓ обучено за {time.perf_counter() - t0:.1f} с")

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X), size=min(N_BACKGROUND, len(X)), replace=False)
    te_idx = rng.choice(len(X), size=min(N_TEST, len(X)), replace=False)
    X_bg = X[bg_idx]
    X_te = X[te_idx]

    max_evals = 2 * len(feat_cols) + 1
    # Разбиваем X_te на N_JOBS чанков и считаем SHAP параллельно процессами.
    # Чанков ровно N_JOBS — это минимизирует накладные расходы на сериализацию
    # модели и фона по сравнению с большим числом мелких задач.
    n_jobs = min(N_JOBS, len(X_te))
    chunks = np.array_split(X_te, n_jobs)
    print(f"  PermutationExplainer: фон {len(X_bg)}, тест {len(X_te)}, "
          f"параллельно на {n_jobs} процессах (по ~{len(chunks[0])} окон в чанке)…")
    t0 = time.perf_counter()
    parts = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_shap_chunk)(chunk, model.predict, X_bg, max_evals)
        for chunk in chunks
    )
    # Склейка частей обратно в один shap.Explanation. Порядок сохраняется,
    # так как np.array_split и Parallel(...) возвращают результаты в порядке
    # подачи задач.
    shap_values = shap.Explanation(
        values=np.concatenate([p.values for p in parts], axis=0),
        base_values=np.concatenate([np.atleast_1d(p.base_values) for p in parts], axis=0),
        data=np.concatenate([p.data for p in parts], axis=0),
        feature_names=feat_cols,
    )
    print(f"  ✓ SHAP посчитан за {time.perf_counter() - t0:.1f} с")

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": mean_abs.round(4),
        "group": [feature_group(c) for c in feat_cols],
    }).sort_values("mean_abs_shap", ascending=False)
    total = imp_df["mean_abs_shap"].sum()
    imp_df["share_pct"] = (imp_df["mean_abs_shap"] / total * 100).round(1) if total > 0 else 0
    imp_df.to_csv(out_dir / "shap_importance.csv", index=False)

    group_imp = (imp_df.groupby("group")["mean_abs_shap"].sum()
                 .sort_values(ascending=False))
    group_imp_pct = (group_imp / group_imp.sum() * 100).round(1) if group_imp.sum() > 0 else group_imp * 0
    group_df = pd.DataFrame({"group": group_imp.index,
                             "mean_abs_shap": group_imp.values.round(4),
                             "share_pct": group_imp_pct.values})
    group_df.to_csv(out_dir / "shap_by_group.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, features=X_te, feature_names=feat_cols,
                      show=False, max_display=20)
    plt.title(f"SHAP beeswarm — {tag} (SVR RBF)")
    plt.tight_layout()
    plt.savefig(out_dir / "beeswarm.png", dpi=120)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, features=X_te, feature_names=feat_cols,
                      plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP mean|value| — {tag} (SVR RBF)")
    plt.tight_layout()
    plt.savefig(out_dir / "bar.png", dpi=120)
    plt.close()

    print(f"  Top-5: " + ", ".join(
        f"{r.feature}({r.share_pct}%)" for r in imp_df.head(5).itertuples()
    ))
    print(f"  По группам: " + ", ".join(
        f"{g}:{p}%" for g, p in zip(group_df["group"], group_df["share_pct"])
    ))


def main() -> None:
    print("Загрузка датасета…")
    df_raw = pd.read_parquet(DATASET_DIR / "merged_features_ml.parquet")
    sp = pd.read_parquet(DATASET_DIR / "session_params.parquet")
    df = prepare_data(df_raw, sp, "lt1").dropna(subset=[TARGET_COL])
    print(f"  LT1: {len(df)} окон, {df['subject_id'].nunique()} субъектов")

    for fset, tag in CONFIGS:
        run_one(fset, tag, df)

    print("\nГотово.")


if __name__ == "__main__":
    main()
