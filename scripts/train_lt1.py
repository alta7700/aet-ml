"""Обучение и оценка модели предсказания LT1.

Использует те же признаки и подход, что и LT2:
  - ElasticNet на nirs_hrv-признаках
  - LOSO CV (Leave-One-Subject-Out)
  - Сравнение с наивными baseline (predict-mean, elapsed_sec)
  - Визуализация: predicted vs actual, траектории по участникам,
    сравнение с LT2 (MAE), коэффициенты модели

Таргет: target_time_to_lt1_sec (секунды до LT1 в момент конца окна).
Usable: участники с lt1_time_label_quality in {high, medium} и lt1_equals_lt2 = 0.

Использование:
  python scripts/train_lt1.py
  python scripts/train_lt1.py --output-dir results/lt1 --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR

# ─────────────────────── Наборы признаков ──────────────────────────────────

NIRS_FEATURES = [
    "trainred_smo2_mean", "trainred_smo2_drop", "trainred_smo2_slope",
    "trainred_smo2_std",
    "trainred_hhb_mean", "trainred_hhb_slope", "trainred_hhb_std",
    "trainred_hbdiff_mean", "trainred_hbdiff_slope", "trainred_hbdiff_std",
    "trainred_thb_mean", "trainred_thb_slope", "trainred_thb_std",
]

HRV_FEATURES = [
    "hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms",
    "hrv_sd1_ms", "hrv_sd2_ms", "hrv_sd1_sd2_ratio",
    "hrv_dfa_alpha1",
]

CONTEXT_FEATURES = ["elapsed_sec", "current_power_w"]

FEATURE_SETS = {
    "nirs_hrv": NIRS_FEATURES + HRV_FEATURES,
    "nirs_only": NIRS_FEATURES,
    "hrv_only": HRV_FEATURES,
    "nirs_hrv_ctx": NIRS_FEATURES + HRV_FEATURES + CONTEXT_FEATURES,
}


# ─────────────────────── LOSO CV ──────────────────────────────────────────

def run_loso(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str = "elasticnet",
    alpha: float = 0.5,
    l1_ratio: float = 0.9,
) -> dict[str, object]:
    """LOSO CV на указанных признаках. Возвращает метрики и предсказания."""
    subjects = sorted(df["subject_id"].unique())
    preds_all: list[np.ndarray] = []
    trues_all: list[np.ndarray] = []
    subj_all: list[np.ndarray] = []

    for test_subj in subjects:
        train_mask = df["subject_id"] != test_subj
        test_mask = df["subject_id"] == test_subj

        X_tr = df.loc[train_mask, feature_cols].values
        y_tr = df.loc[train_mask, target_col].values
        X_te = df.loc[test_mask, feature_cols].values
        y_te = df.loc[test_mask, target_col].values

        # Пайплайн: импьютер → скейлер → модель
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        if model_name == "elasticnet":
            estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        elif model_name == "ridge":
            estimator = Ridge(alpha=alpha)
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")

        X_tr_imp = imputer.fit_transform(X_tr)
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        estimator.fit(X_tr_sc, y_tr)

        X_te_imp = imputer.transform(X_te)
        X_te_sc = scaler.transform(X_te_imp)
        y_pred = estimator.predict(X_te_sc)

        preds_all.append(y_pred)
        trues_all.append(y_te)
        subj_all.append(np.full(len(y_te), test_subj))

    y_pred_all = np.concatenate(preds_all)
    y_true_all = np.concatenate(trues_all)
    subj_labels = np.concatenate(subj_all)

    # Метрики в минутах
    mae_sec = float(mean_absolute_error(y_true_all, y_pred_all))
    r2 = float(r2_score(y_true_all, y_pred_all))
    rho = float(spearmanr(y_true_all, y_pred_all).statistic)

    # MAE по каждому участнику
    per_subj_mae: dict[str, float] = {}
    for s in subjects:
        mask = subj_labels == s
        per_subj_mae[s] = float(mean_absolute_error(y_true_all[mask], y_pred_all[mask]))

    return {
        "mae_min": mae_sec / 60.0,
        "mae_sec": mae_sec,
        "r2": r2,
        "rho": rho,
        "per_subj_mae_min": {k: v / 60.0 for k, v in per_subj_mae.items()},
        "y_pred": y_pred_all,
        "y_true": y_true_all,
        "subjects": subj_labels,
    }


def predict_mean_baseline(df: pd.DataFrame, target_col: str) -> float:
    """LOSO baseline: предсказываем среднее по обучающей выборке."""
    subjects = sorted(df["subject_id"].unique())
    errors: list[float] = []
    for test_subj in subjects:
        train_mask = df["subject_id"] != test_subj
        test_mask = df["subject_id"] == test_subj
        y_tr = df.loc[train_mask, target_col].values
        y_te = df.loc[test_mask, target_col].values
        y_pred = np.full(len(y_te), np.mean(y_tr))
        errors.append(float(mean_absolute_error(y_te, y_pred)))
    return float(np.mean(errors)) / 60.0


def predict_elapsed_baseline(df: pd.DataFrame, target_col: str) -> float:
    """LOSO baseline: линейная регрессия по elapsed_sec."""
    subjects = sorted(df["subject_id"].unique())
    errors: list[float] = []
    for test_subj in subjects:
        train_mask = df["subject_id"] != test_subj
        test_mask = df["subject_id"] == test_subj
        X_tr = df.loc[train_mask, ["elapsed_sec"]].values
        y_tr = df.loc[train_mask, target_col].values
        X_te = df.loc[test_mask, ["elapsed_sec"]].values
        y_te = df.loc[test_mask, target_col].values
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        X_te_imp = imp.transform(X_te)
        coeffs = np.polyfit(X_tr_imp.ravel(), y_tr, 1)
        y_pred = np.polyval(coeffs, X_te_imp.ravel())
        errors.append(float(mean_absolute_error(y_te, y_pred)))
    return float(np.mean(errors)) / 60.0


# ─────────────────────── Сравнение с LT2 ─────────────────────────────────

def get_lt2_results_for_comparison(df_lt2: pd.DataFrame) -> dict[str, float]:
    """Запускает LOSO ElasticNet(nirs_hrv) на LT2 для тех же участников."""
    target = "target_time_to_lt2_center_sec"
    features = NIRS_FEATURES + HRV_FEATURES
    available = [c for c in features if c in df_lt2.columns]
    result = run_loso(df_lt2, available, target)
    return {"mae_min": result["mae_min"], "r2": result["r2"], "rho": result["rho"]}


# ─────────────────────── Визуализация ─────────────────────────────────────

def plot_results(
    lt1_result: dict,
    df_lt1: pd.DataFrame,
    df_lt2_subj9: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Строит 4 панели: predicted vs actual, residuals, trajectории, коэффициенты."""

    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(df_lt1["subject_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}

    y_true = lt1_result["y_true"] / 60.0   # → минуты
    y_pred = lt1_result["y_pred"] / 60.0
    subj_labels = lt1_result["subjects"]
    mae_min = lt1_result["mae_min"]
    r2 = lt1_result["r2"]
    rho = lt1_result["rho"]

    # ── 1. Predicted vs Actual ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in subjects:
        mask = subj_labels == s
        ax.scatter(y_true[mask], y_pred[mask], color=subj_color[s],
                   alpha=0.25, s=8, label=s)
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="идеал")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Истинное время до LT1, мин")
    ax.set_ylabel("Предсказанное время до LT1, мин")
    ax.set_title(f"LT1: Predicted vs Actual\nMAE={mae_min:.2f} мин, R²={r2:.3f}, ρ={rho:.3f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend(markerscale=2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "lt1_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. MAE по участникам: LT1 vs LT2 ──────────────────────────────────
    lt2_result = get_lt2_results_for_comparison(df_lt2_subj9)
    per_subj_lt1 = lt1_result["per_subj_mae_min"]
    # LT2 per-subject для тех же 9 участников
    target_lt2 = "target_time_to_lt2_center_sec"
    features = [c for c in NIRS_FEATURES + HRV_FEATURES if c in df_lt2_subj9.columns]
    lt2_full = run_loso(df_lt2_subj9, features, target_lt2)
    per_subj_lt2 = lt2_full["per_subj_mae_min"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(subjects))
    w = 0.35
    lt1_vals = [per_subj_lt1.get(s, np.nan) for s in subjects]
    lt2_vals = [per_subj_lt2.get(s, np.nan) for s in subjects]
    bars1 = ax.bar(x - w/2, lt1_vals, w, label="LT1 MAE", color="#1f77b4", alpha=0.8)
    bars2 = ax.bar(x + w/2, lt2_vals, w, label="LT2 MAE", color="#ff7f0e", alpha=0.8)
    ax.axhline(mae_min, color="#1f77b4", linestyle="--", linewidth=1.2,
               label=f"LT1 среднее {mae_min:.2f} мин")
    ax.axhline(lt2_full["mae_min"], color="#ff7f0e", linestyle="--", linewidth=1.2,
               label=f"LT2 среднее {lt2_full['mae_min']:.2f} мин")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel("MAE, мин")
    ax.set_title("MAE по участникам: LT1 vs LT2 (LOSO, ElasticNet, nirs_hrv)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "lt1_vs_lt2_mae.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. Траектории по участникам ────────────────────────────────────────
    n_subj = len(subjects)
    ncols = 3
    nrows = int(np.ceil(n_subj / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, s in enumerate(subjects):
        ax = axes_flat[i]
        subj_df = df_lt1[df_lt1["subject_id"] == s].sort_values("elapsed_sec")
        mask = subj_labels == s
        t_elapsed = subj_df["elapsed_sec"].values / 60.0
        y_t = (subj_df["target_time_to_lt1_sec"].values) / 60.0
        y_p_sorted = y_pred[mask]
        lt1_quality = subj_df["lt1_time_label_quality"].iloc[0]
        mae_s = per_subj_lt1.get(s, np.nan)

        ax.plot(t_elapsed, y_t, "k-", linewidth=1.5, label="Истина")
        ax.plot(t_elapsed, y_p_sorted, color=subj_color[s], linewidth=1.2,
                alpha=0.85, label="Предсказание")
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="LT1")
        ax.set_title(f"{s} | MAE={mae_s:.1f} мин | q={lt1_quality}", fontsize=9)
        ax.set_xlabel("Прошло, мин", fontsize=8)
        ax.set_ylabel("До LT1, мин", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("LT1: Траектории предсказания по участникам (LOSO, ElasticNet, nirs_hrv)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "lt1_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. Сравнение наборов признаков ─────────────────────────────────────
    target = "target_time_to_lt1_sec"
    comparisons = {}
    for fs_name, fs_cols in FEATURE_SETS.items():
        available = [c for c in fs_cols if c in df_lt1.columns]
        if len(available) < 2:
            continue
        r = run_loso(df_lt1, available, target)
        comparisons[fs_name] = r["mae_min"]

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(comparisons.keys())
    vals = [comparisons[n] for n in names]
    colors = ["#1f77b4" if n == "nirs_hrv" else "#aec7e8" for n in names]
    bars = ax.barh(names, vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlabel("MAE, мин (LOSO)")
    ax.set_title("LT1: Сравнение наборов признаков (ElasticNet)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "lt1_feature_sets.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Графики сохранены в {output_dir}")


# ─────────────────────── Главная функция ──────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="LOSO обучение модели LT1.")
    parser.add_argument(
        "--merged-file",
        type=Path,
        default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/lt1"),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не строить графики.",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа."""
    args = parse_args()

    merged = pd.read_parquet(args.merged_file)
    print(f"Загружен датасет: {merged.shape}")

    # ── Фильтрация: только usable LT1 окна ──
    df_lt1 = merged[merged["target_time_to_lt1_usable"] == 1].copy()
    print(f"LT1 usable окна: {len(df_lt1)} ({df_lt1['subject_id'].nunique()} участников)")
    print(f"Участники: {sorted(df_lt1['subject_id'].unique())}")

    target = "target_time_to_lt1_sec"
    main_features = [c for c in NIRS_FEATURES + HRV_FEATURES if c in df_lt1.columns]
    print(f"Признаков nirs_hrv: {len(main_features)}")

    # ── Основная модель: ElasticNet(nirs_hrv) ──
    print("\n═══ ElasticNet(nirs_hrv) ═══")
    result = run_loso(df_lt1, main_features, target, model_name="elasticnet",
                      alpha=0.5, l1_ratio=0.9)
    mae = result["mae_min"]
    r2 = result["r2"]
    rho = result["rho"]
    print(f"MAE  = {mae:.3f} ± {np.std([v for v in result['per_subj_mae_min'].values()]):.3f} мин")
    print(f"R²   = {r2:.3f}")
    print(f"ρ    = {rho:.3f}")
    print("MAE по участникам:")
    for s, m in sorted(result["per_subj_mae_min"].items()):
        print(f"  {s}: {m:.2f} мин")

    # ── Наивные baseline ──
    print("\n═══ Baseline ═══")
    mae_mean = predict_mean_baseline(df_lt1, target)
    mae_elapsed = predict_elapsed_baseline(df_lt1, target)
    print(f"Predict-mean:     {mae_mean:.3f} мин")
    print(f"Elapsed linear:   {mae_elapsed:.3f} мин")
    print(f"ElasticNet / mean = {mae/mae_mean:.3f}")

    # ── Все наборы признаков ──
    print("\n═══ Наборы признаков ═══")
    for fs_name, fs_cols in FEATURE_SETS.items():
        available = [c for c in fs_cols if c in df_lt1.columns]
        if len(available) < 2:
            print(f"  {fs_name}: пропущено (мало признаков)")
            continue
        r = run_loso(df_lt1, available, target)
        print(f"  {fs_name:20s}: MAE={r['mae_min']:.3f} мин, R²={r['r2']:.3f}, ρ={r['rho']:.3f}")

    # ── Сравнение с LT2 (те же 9 участников) ──
    subj_lt1 = df_lt1["subject_id"].unique()
    df_lt2_9 = merged[
        merged["subject_id"].isin(subj_lt1)
        & merged["nirs_valid"].eq(1)
        & merged["hrv_valid"].eq(1)
    ].copy()
    print(f"\n═══ LT2 (те же 9 участников, {len(df_lt2_9)} окон) ═══")
    lt2_features = [c for c in NIRS_FEATURES + HRV_FEATURES if c in df_lt2_9.columns]
    lt2_r = run_loso(df_lt2_9, lt2_features, "target_time_to_lt2_center_sec")
    print(f"LT2 MAE = {lt2_r['mae_min']:.3f} мин, R²={lt2_r['r2']:.3f}, ρ={lt2_r['rho']:.3f}")
    print(f"LT1 MAE = {mae:.3f} мин, R²={r2:.3f}, ρ={rho:.3f}")
    print(f"Улучшение LT1 vs LT2: {(lt2_r['mae_min'] - mae):.3f} мин")

    # ── Графики ──
    if not args.no_plots:
        plot_results(
            lt1_result=result,
            df_lt1=df_lt1,
            df_lt2_subj9=df_lt2_9,
            output_dir=args.output_dir,
        )

    print(f"\n✅ Готово. Результаты в {args.output_dir}")


if __name__ == "__main__":
    main()
