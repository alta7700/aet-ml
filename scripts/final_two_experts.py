"""Шаг 05: гипотеза «два эксперта» для lt1 — specialists + router.

Из шагов 02a/02b: тренированные (time_to_lt2 > 13 мин) предсказываются для lt1
значимо хуже. Проверяем, помогает ли специализация:
- generalist: одна Ridge-модель на всех 18, LOSO;
- два специалиста (trained / untrained), каждый — LOSO внутри своей группы;
- oracle routing: субъект → специалист истинной группы (верхняя граница);
- router: логрег на baseline-признаках → предсказанная группа → real routing;
- paired Wilcoxon: generalist vs oracle, generalist vs real-router.

Модальность — EMG+NIRS (победитель lt1 в шагах 02/04). Модель — Ridge,
α выбирается по LOSO на полной выборке.

Запуск:
    PYTHONPATH=. uv run python scripts/final_two_experts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v0011_modality_ablation import prepare_data, get_feature_cols  # noqa: E402

DATASET_DIR = ROOT / "dataset"
OUT = ROOT / "results" / "final" / "two_experts"
PLOTS = OUT / "plots"

TARGET_COL_LT1 = "target_time_to_lt1_pchip_sec"
TARGET_COL_LT2 = "target_time_to_lt2_center_sec"
FEATURE_SET = "EMG+NIRS"
TRAINED_THRESHOLD_MIN = 13.0
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]

# Признаки маршрутизатора (НЕ используют таргет lt1).
ROUTING_FEATURES = [
    "age", "bmi", "body_fat_pct", "phase_angle", "weight", "height",
    "hrv_hr_baseline_bpm", "hrv_rmssd_baseline_ms",
    "emg_vl_dist_baseline_rms", "emg_vl_prox_baseline_rms",
    "nirs_smo2_baseline_mean",
]


# ─────────────────────── Разметка субъектов ───────────────────────

def build_subject_meta() -> pd.DataFrame:
    sp = pd.read_parquet(DATASET_DIR / "session_params.parquet")
    subs = pd.read_parquet(DATASET_DIR / "subjects.parquet")
    wn = pd.read_parquet(DATASET_DIR / "windows.parquet")
    tg = pd.read_parquet(DATASET_DIR / "targets.parquet")

    m = wn[["window_id", "subject_id", "window_end_sec", "elapsed_sec"]].merge(
        tg[["window_id", "target_time_to_lt2_center_sec"]], on="window_id")
    first = m.sort_values(["subject_id", "window_end_sec"]).groupby("subject_id").first().reset_index()
    first["time_to_lt2_min"] = (first["target_time_to_lt2_center_sec"]
                                + first["elapsed_sec"]) / 60.0

    meta = (subs[["subject_id", "age", "sex", "bmi", "body_fat_pct",
                  "phase_angle", "weight", "height"]]
            .merge(sp[["subject_id", "hrv_hr_baseline_bpm", "hrv_rmssd_baseline_ms",
                       "emg_vl_dist_baseline_rms", "emg_vl_prox_baseline_rms",
                       "nirs_smo2_baseline_mean"]], on="subject_id")
            .merge(first[["subject_id", "time_to_lt2_min"]], on="subject_id"))
    meta["trained"] = (meta["time_to_lt2_min"] > TRAINED_THRESHOLD_MIN).astype(int)
    return meta


# ─────────────────────── LOSO для регрессора ───────────────────────

def loso_predict(df: pd.DataFrame, feat_cols: list[str], target_col: str,
                 subjects: list[str], alpha: float) -> dict[str, dict]:
    """LOSO внутри заданного списка субъектов. Возвращает per-subject dict."""
    result = {}
    for test_s in subjects:
        train = df[(df["subject_id"].isin(subjects)) & (df["subject_id"] != test_s)]
        test = df[df["subject_id"] == test_s].sort_values("window_start_sec")
        if train.empty or test.empty:
            continue
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        mdl = Ridge(alpha=alpha)
        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        mdl.fit(X_tr, train[target_col].values)
        y_pred = mdl.predict(X_te)
        y_true = test[target_col].values
        result[test_s] = {
            "y_pred": y_pred, "y_true": y_true,
            "mae_min": mean_absolute_error(y_true, y_pred) / 60.0,
        }
    return result


def fit_full_predict(df: pd.DataFrame, feat_cols: list[str], target_col: str,
                     train_subjects: list[str], test_subjects: list[str],
                     alpha: float) -> dict[str, dict]:
    """Обучает на train_subjects, предсказывает test_subjects (без LOSO)."""
    train = df[df["subject_id"].isin(train_subjects)]
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    mdl = Ridge(alpha=alpha)
    X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
    mdl.fit(X_tr, train[target_col].values)

    result = {}
    for s in test_subjects:
        test = df[df["subject_id"] == s].sort_values("window_start_sec")
        if test.empty:
            continue
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        y_pred = mdl.predict(X_te)
        y_true = test[target_col].values
        result[s] = {"y_pred": y_pred, "y_true": y_true,
                     "mae_min": mean_absolute_error(y_true, y_pred) / 60.0}
    return result


def select_alpha(df: pd.DataFrame, feat_cols: list[str], target_col: str,
                 subjects: list[str]) -> float:
    """Выбор α по медиане LOSO-MAE на полной выборке."""
    best_alpha, best_med = RIDGE_ALPHAS[0], np.inf
    for a in RIDGE_ALPHAS:
        res = loso_predict(df, feat_cols, target_col, subjects, a)
        med = float(np.median([r["mae_min"] for r in res.values()]))
        print(f"  α={a}: median LOSO MAE = {med:.4f}")
        if med < best_med:
            best_med, best_alpha = med, a
    return best_alpha


# ─────────────────────── Router ───────────────────────

def loso_router(meta: pd.DataFrame) -> pd.DataFrame:
    """LOSO логрег: routing_features → trained. Возвращает per-subject prob+pred."""
    rows = []
    feat = [c for c in ROUTING_FEATURES if c in meta.columns]
    subjects = sorted(meta["subject_id"])
    for test_s in subjects:
        train = meta[meta["subject_id"] != test_s]
        test = meta[meta["subject_id"] == test_s]
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        X_tr = sc.fit_transform(imp.fit_transform(train[feat].values))
        X_te = sc.transform(imp.transform(test[feat].values))
        clf.fit(X_tr, train["trained"].values)
        prob = float(clf.predict_proba(X_te)[0, 1])
        pred = int(prob >= 0.5)
        rows.append({"subject_id": test_s, "trained_true": int(test["trained"].iloc[0]),
                     "trained_prob": round(prob, 4), "trained_pred": pred})
    return pd.DataFrame(rows)


def router_feature_importance(meta: pd.DataFrame) -> pd.DataFrame:
    """Коэффициенты логрега на всей выборке (для интерпретации)."""
    feat = [c for c in ROUTING_FEATURES if c in meta.columns]
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    X = sc.fit_transform(imp.fit_transform(meta[feat].values))
    clf.fit(X, meta["trained"].values)
    coef = pd.Series(clf.coef_[0], index=feat).sort_values(key=np.abs, ascending=False)
    return pd.DataFrame({"feature": coef.index,
                         "std_coef": coef.values.round(3),
                         "abs_coef": np.abs(coef.values).round(3)})


# ─────────────────────── Сборка системы ───────────────────────

def run_pipeline(threshold_min: float, tag: str) -> dict:
    print(f"\n{'='*60}\nПОРОГ trained = time_to_lt2 > {threshold_min} мин  [{tag}]\n{'='*60}")

    df_raw = pd.read_parquet(DATASET_DIR / "merged_features_ml.parquet")
    sp = pd.read_parquet(DATASET_DIR / "session_params.parquet")
    df = prepare_data(df_raw, sp, "lt1").dropna(subset=[TARGET_COL_LT1])
    feat_cols = get_feature_cols(df, FEATURE_SET)

    meta = build_subject_meta()
    meta["trained"] = (meta["time_to_lt2_min"] > threshold_min).astype(int)
    all_subjects = sorted(df["subject_id"].unique())
    meta = meta[meta["subject_id"].isin(all_subjects)].reset_index(drop=True)

    trained = sorted(meta[meta["trained"] == 1]["subject_id"])
    untrained = sorted(meta[meta["trained"] == 0]["subject_id"])
    print(f"Тренированных: {len(trained)}, нетренированных: {len(untrained)}")

    print("Выбор α (Ridge) по LOSO на полной выборке:")
    alpha = select_alpha(df, feat_cols, TARGET_COL_LT1, all_subjects)
    print(f"  → α = {alpha}")

    # 1. Generalist
    gen = loso_predict(df, feat_cols, TARGET_COL_LT1, all_subjects, alpha)

    # 2. Specialists
    # untrained-спец: LOSO внутри untrained
    spec_unt_loso = loso_predict(df, feat_cols, TARGET_COL_LT1, untrained, alpha)
    # untrained-спец на всех 12 → предсказание trained
    spec_unt_full = fit_full_predict(df, feat_cols, TARGET_COL_LT1,
                                     untrained, trained, alpha)
    # trained-спец: LOSO внутри trained
    spec_tr_loso = loso_predict(df, feat_cols, TARGET_COL_LT1, trained, alpha)
    # trained-спец на всех 6 → предсказание untrained
    spec_tr_full = fit_full_predict(df, feat_cols, TARGET_COL_LT1,
                                    trained, untrained, alpha)

    # Объединяем: у каждого субъекта MAE от обоих специалистов
    untrained_spec_mae = {**{s: spec_unt_loso[s]["mae_min"] for s in spec_unt_loso},
                          **{s: spec_unt_full[s]["mae_min"] for s in spec_unt_full}}
    trained_spec_mae = {**{s: spec_tr_loso[s]["mae_min"] for s in spec_tr_loso},
                        **{s: spec_tr_full[s]["mae_min"] for s in spec_tr_full}}

    # 3. Oracle routing (симметричный: trained→trained-spec, untrained→untrained-spec)
    oracle_mae = {}
    for s in all_subjects:
        is_tr = int(meta.loc[meta["subject_id"] == s, "trained"].iloc[0])
        oracle_mae[s] = trained_spec_mae[s] if is_tr else untrained_spec_mae[s]

    # 3b. Асимметричная система: untrained → untrained-spec, trained → generalist.
    # Мотивация: trained-спец обучается на ≤6 субъектах и недообучен; generalist
    # для тренированных надёжнее. Untrained-спец (на 12) лучше generalist.
    asym_oracle_mae = {}
    for s in all_subjects:
        is_tr = int(meta.loc[meta["subject_id"] == s, "trained"].iloc[0])
        asym_oracle_mae[s] = gen[s]["mae_min"] if is_tr else untrained_spec_mae[s]

    # 4. Router
    router = loso_router(meta)
    router = router.set_index("subject_id")
    real_mae = {}
    asym_real_mae = {}
    for s in all_subjects:
        pred_tr = int(router.loc[s, "trained_pred"])
        real_mae[s] = trained_spec_mae[s] if pred_tr else untrained_spec_mae[s]
        # Асимметричный с реальным гейтом: pred trained → generalist, pred untrained → spec
        asym_real_mae[s] = gen[s]["mae_min"] if pred_tr else untrained_spec_mae[s]

    # Сводная per-subject таблица
    rows = []
    for s in all_subjects:
        is_tr = int(meta.loc[meta["subject_id"] == s, "trained"].iloc[0])
        rows.append({
            "subject_id": s, "trained": is_tr,
            "mae_generalist": round(gen[s]["mae_min"], 4),
            "mae_untrained_spec": round(untrained_spec_mae[s], 4),
            "mae_trained_spec": round(trained_spec_mae[s], 4),
            "mae_oracle": round(oracle_mae[s], 4),
            "mae_asym_oracle": round(asym_oracle_mae[s], 4),
            "mae_router": round(real_mae[s], 4),
            "mae_asym_router": round(asym_real_mae[s], 4),
            "router_trained_prob": round(float(router.loc[s, "trained_prob"]), 4),
            "router_pred": int(router.loc[s, "trained_pred"]),
            "router_correct": int(router.loc[s, "trained_pred"] == is_tr),
        })
    per_subj = pd.DataFrame(rows)

    # 5. Paired Wilcoxon
    def paired(a_col, b_col):
        a = per_subj[a_col].values
        b = per_subj[b_col].values
        try:
            p = float(wilcoxon(a, b, alternative="two-sided").pvalue)
        except Exception:
            p = float("nan")
        return round(float(np.median(a - b)), 4), round(p, 4)

    d_oracle, p_oracle = paired("mae_generalist", "mae_oracle")
    d_router, p_router = paired("mae_generalist", "mae_router")
    d_asym_o, p_asym_o = paired("mae_generalist", "mae_asym_oracle")
    d_asym_r, p_asym_r = paired("mae_generalist", "mae_asym_router")

    def srow(name, col, d, p):
        return {"system": name,
                "mae_median": round(per_subj[col].median(), 4),
                "mae_mean": round(per_subj[col].mean(), 4),
                "vs_generalist_delta": d, "vs_generalist_p": p}

    comparison = pd.DataFrame([
        srow("generalist", "mae_generalist", 0.0, np.nan),
        srow("oracle_sym", "mae_oracle", d_oracle, p_oracle),
        srow("real_router_sym", "mae_router", d_router, p_router),
        srow("oracle_asym", "mae_asym_oracle", d_asym_o, p_asym_o),
        srow("real_router_asym", "mae_asym_router", d_asym_r, p_asym_r),
    ])

    router_acc = float(per_subj["router_correct"].mean())
    print(f"\nТочность маршрутизатора (LOSO): {router_acc:.2%}")
    print(comparison.to_string(index=False))

    return {
        "tag": tag, "threshold": threshold_min, "alpha": alpha,
        "n_trained": len(trained), "n_untrained": len(untrained),
        "per_subject": per_subj, "comparison": comparison,
        "router_accuracy": router_acc, "meta": meta,
    }


def plot_systems(res_main: dict) -> None:
    per = res_main["per_subject"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Boxplot по системам
    ax = axes[0]
    cols = ["mae_generalist", "mae_oracle", "mae_router",
            "mae_asym_oracle", "mae_asym_router"]
    labels = ["generalist", "oracle_sym", "router_sym",
              "oracle_asym", "router_asym"]
    ax.boxplot([per[c].values for c in cols], tick_labels=labels, patch_artist=True,
               boxprops=dict(facecolor="#cce5ff"))
    for i, c in enumerate(cols, 1):
        for is_tr, color in [(1, "C3"), (0, "C0")]:
            sub = per[per["trained"] == is_tr]
            ax.scatter(np.full(len(sub), i) + np.random.uniform(-0.1, 0.1, len(sub)),
                       sub[c], color=color, s=25, alpha=0.7, zorder=3)
    ax.set_ylabel("per-subject MAE, мин")
    ax.set_title(f"lt1 — системы (порог {res_main['threshold']} мин)\n"
                 f"красные = trained, синие = untrained")
    ax.grid(alpha=0.3, axis="y")

    # Generalist vs asym-oracle по субъектам
    ax = axes[1]
    for is_tr, color, lbl in [(1, "C3", "trained"), (0, "C0", "untrained")]:
        sub = per[per["trained"] == is_tr]
        ax.scatter(sub["mae_generalist"], sub["mae_asym_oracle"], color=color, s=50,
                   alpha=0.7, label=lbl)
        for _, r in sub.iterrows():
            ax.annotate(r["subject_id"], (r["mae_generalist"], r["mae_asym_oracle"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = [0, max(per[["mae_generalist", "mae_asym_oracle"]].max()) * 1.05]
    ax.plot(lim, lim, "k--", alpha=0.4)
    ax.set_xlabel("MAE generalist, мин")
    ax.set_ylabel("MAE asym-oracle, мин")
    ax.set_title("Асимметричная система (untrained→spec, trained→generalist)\n"
                 "ниже диагонали = система лучше")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "systems_comparison.png", dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    # Основной порог 13 мин
    res_main = run_pipeline(TRAINED_THRESHOLD_MIN, "thr13")

    # Сохраняем основные артефакты
    res_main["meta"].to_csv(OUT / "subject_meta.csv", index=False)
    res_main["per_subject"].to_csv(OUT / "15_per_subject_predictions.csv", index=False)

    # Чувствительность: медианный порог
    median_thr = float(res_main["meta"]["time_to_lt2_min"].median())
    res_sens = run_pipeline(round(median_thr, 1), "median_split")

    # Сводка по обоим порогам
    comp_main = res_main["comparison"].copy()
    comp_main["threshold_tag"] = "thr13"
    comp_sens = res_sens["comparison"].copy()
    comp_sens["threshold_tag"] = "median_split"
    comparison_all = pd.concat([comp_main, comp_sens], ignore_index=True)
    comparison_all.to_csv(OUT / "16_system_comparison.csv", index=False)

    # Router метрики + важность признаков
    fi = router_feature_importance(res_main["meta"])
    fi.to_csv(OUT / "17_router_metrics.csv", index=False)
    print("\nВажность признаков маршрутизатора (|std_coef|):")
    print(fi.to_string(index=False))

    plot_systems(res_main)

    print(f"\nАртефакты в {OUT}")


if __name__ == "__main__":
    main()
