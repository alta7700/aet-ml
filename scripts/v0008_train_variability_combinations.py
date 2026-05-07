"""v0008 — Комбинации лучших признаков вариабельности × зоопарк моделей

Версия:    v0008
Дата:      2026-05-07
Предыдущая версия: v0007_train_variability_features.py
Результаты: results/v0008/

Итоги v0007 (зоопарк):
  LT2: GBM n=200 — лучший класс моделей (2.043–2.096).
       SampEn 30s + GBM = 2.043 (Δ=-0.055 vs v0004 Ridge).
  LT1: Линейные (Huber/Ridge/ElasticNet) лучше деревьев (RF/GBM переобучаются, 9 участников).
       SampEn 30s + Huber = 1.998 (Δ=-0.023 vs v0004 Huber).

Что тестируем в v0008 — комбинации лучших:
  1. v0004-репликация
  2. SampEn 30s
  3. SampEn 60s
  4. timing_cv            (load/rest_duration_cv)
  5. SampEn 30s+60s
  6. SampEn 30s+timing_cv
  7. SampEn 60s+timing_cv
  8. SampEn 30s+60s+timing_cv

  Для каждой комбинации — весь зоопарк моделей.

Зоопарк: Ridge α=100, Huber ε=1.35, ElasticNet, SVR rbf, RF n=200, GBM n=200

Воспроизведение:
  uv run python scripts/v0008_train_variability_combinations.py
  uv run python scripts/v0008_train_variability_combinations.py --target lt1
  uv run python scripts/v0008_train_variability_combinations.py --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR

OUT_DIR = _ROOT / "results" / "v0008"

# ─── Зоопарк моделей ──────────────────────────────────────────────────────────

ZOO: dict[str, callable] = {
    "Ridge α=100":  lambda: Ridge(alpha=100.0, fit_intercept=True, solver="auto"),
    "Huber ε=1.35": lambda: HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500, fit_intercept=True),
    "ElasticNet":   lambda: ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=5000, random_state=0, fit_intercept=True),
    "SVR rbf":      lambda: SVR(kernel="rbf", C=10.0, epsilon=0.1),
    "RF n=200":     lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GBM n=200":    lambda: GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=3),
}

# ─── Базовые признаки (из v0004) ─────────────────────────────────────────────

NIRS_FEATURES = [
    "trainred_smo2_mean", "trainred_smo2_drop", "trainred_smo2_slope",
    "trainred_smo2_std", "trainred_dsmo2_dt",
    "trainred_hhb_mean", "trainred_hhb_slope", "trainred_hhb_std", "trainred_dhhb_dt",
    "trainred_hbdiff_mean", "trainred_hbdiff_slope", "trainred_hbdiff_std",
    "trainred_thb_mean", "trainred_thb_slope", "trainred_thb_std",
]

HRV_FEATURES = [
    "hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms",
    "hrv_sd1_ms", "hrv_sd2_ms", "hrv_sd1_sd2_ratio",
    "hrv_dfa_alpha1",
]

SAMPEN_30 = ["load_sampen_30s", "rest_sampen_30s"]
SAMPEN_60 = ["load_sampen_60s", "rest_sampen_60s"]
TIMING_CV = ["load_duration_cv", "rest_duration_cv"]

EMG_RAW_PREFIX = "vl_"


# ─── Feature engineering ──────────────────────────────────────────────────────

def add_session_z_emg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [c for c in df.columns if c.startswith(EMG_RAW_PREFIX)]:
        m = df.groupby("subject_id")[col].transform("mean")
        s = df.groupby("subject_id")[col].transform("std")
        df[f"z_{col}"] = (df[col] - m) / (s + 1e-8)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)
    df["feat_smo2_x_rr"] = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
    df["feat_smo2_x_dfa"] = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    df["feat_rr_per_watt"] = df["hrv_mean_rr_ms"] / pw
    return df


def add_running_nirs_features(df: pd.DataFrame, session_params: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sp = session_params.set_index("subject_id")
    for col in ["smo2_from_running_max", "hhb_from_running_min", "smo2_rel_drop_pct"]:
        df[col] = np.nan
    for subj, g in df.groupby("subject_id"):
        g = g.sort_values("window_center_sec")
        idx = g.index
        smo2 = g["trainred_smo2_mean"].values
        hhb = g["trainred_hhb_mean"].values
        rmax = np.maximum.accumulate(np.where(np.isfinite(smo2), smo2, -np.inf))
        rmin = np.minimum.accumulate(np.where(np.isfinite(hhb), hhb, np.inf))
        rmax = np.where(np.isinf(rmax), np.nan, rmax)
        rmin = np.where(np.isinf(rmin), np.nan, rmin)
        df.loc[idx, "smo2_from_running_max"] = rmax - smo2
        df.loc[idx, "hhb_from_running_min"] = hhb - rmin
        if subj in sp.index:
            b = float(sp.loc[subj, "nirs_smo2_baseline_mean"])
            if np.isfinite(b) and b > 0:
                df.loc[idx, "smo2_rel_drop_pct"] = (b - smo2) / b * 100.0
    return df


# ─── LOSO CV ──────────────────────────────────────────────────────────────────

def loso(df: pd.DataFrame, features: list[str], target: str, model_factory) -> dict:
    subjects = sorted(df["subject_id"].unique())
    feat_cols = [f for f in features if f in df.columns]
    preds, trues, subjs = [], [], []
    coefs_list: list[np.ndarray] = []
    importances_list: list[np.ndarray] = []

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test = df[df["subject_id"] == test_s]
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        model = model_factory()
        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        model.fit(X_tr, train[target].values)
        y_pred = model.predict(X_te)
        preds.append(y_pred)
        trues.append(test[target].values)
        subjs.append(np.full(len(y_pred), test_s))

        if hasattr(model, "coef_"):
            coefs_list.append(model.coef_)
        elif hasattr(model, "feature_importances_"):
            importances_list.append(model.feature_importances_)

    y_pred_all = np.concatenate(preds)
    y_true_all = np.concatenate(trues)
    subj_all = np.concatenate(subjs)
    mae_min = float(mean_absolute_error(y_true_all, y_pred_all)) / 60.0
    per_subj = {
        s: float(mean_absolute_error(y_true_all[subj_all == s],
                                     y_pred_all[subj_all == s])) / 60.0
        for s in subjects
    }
    return {
        "mae_min": mae_min,
        "mae_std": float(np.std(list(per_subj.values()))),
        "r2": float(r2_score(y_true_all, y_pred_all)),
        "rho": float(spearmanr(y_true_all, y_pred_all).statistic),
        "per_subj_mae_min": per_subj,
        "y_pred": y_pred_all, "y_true": y_true_all, "subjects": subj_all,
        "features_used": feat_cols,
        "coef_mean":  np.mean(coefs_list, axis=0) if coefs_list else None,
        "coef_std":   np.std(coefs_list, axis=0) if coefs_list else None,
        "imp_mean":   np.mean(importances_list, axis=0) if importances_list else None,
    }


def print_feature_importance(result: dict, top_n: int = 15) -> None:
    feats = result.get("features_used", [])
    coef_mean = result.get("coef_mean")
    imp_mean = result.get("imp_mean")

    if coef_mean is not None:
        order = np.argsort(np.abs(coef_mean))[::-1][:top_n]
        coef_std = result.get("coef_std")
        print(f"\n  {'Признак':<48s} {'Коэф (μ±σ)':<20s} {'Направление'}")
        print("  " + "─" * 84)
        for i in order:
            mu = coef_mean[i]; sd = coef_std[i] if coef_std is not None else 0.0
            sign = "→ позже" if mu > 0 else "→ раньше"
            print(f"  {feats[i]:<48s}  {mu:+.1f} ± {sd:.1f}   {sign}")
    elif imp_mean is not None:
        order = np.argsort(imp_mean)[::-1][:top_n]
        print(f"\n  {'Признак':<48s} {'Важность (μ)':<15s}")
        print("  " + "─" * 65)
        for i in order:
            print(f"  {feats[i]:<48s}  {imp_mean[i]:.4f}")


def baseline_mae(df: pd.DataFrame, target: str) -> float:
    subjects = sorted(df["subject_id"].unique())
    errors = [
        float(mean_absolute_error(
            df[df["subject_id"] == s][target].values,
            np.full(df[df["subject_id"] == s].shape[0],
                    df[df["subject_id"] != s][target].mean()),
        ))
        for s in subjects
    ]
    return float(np.mean(errors)) / 60.0


def print_zoo_table(
    results_matrix: dict[str, dict[str, dict]],
    configs: list[str],
    ref_model: str,
) -> tuple[str, str]:
    """Печатает таблицу (конфиг × модель). ref_model — модель для расчёта Δmin."""
    model_names = list(ZOO.keys())
    col_w = 12

    # Δ reference = v0004-репликация + ref_model
    ref_mae = results_matrix["v0004-репликация"][ref_model]["mae_min"]

    header = f"  {'Конфигурация':<26s}" + "".join(f"{m:>{col_w}s}" for m in model_names)
    print(header)
    print("  " + "─" * (26 + col_w * len(model_names) + 14))

    best_mae = float("inf")
    best_cfg = best_mdl = ""

    for cfg in configs:
        row = f"  {cfg:<26s}"
        cfg_min = float("inf")
        for mdl in model_names:
            r = results_matrix[cfg][mdl]
            mae = r["mae_min"]
            if mae < best_mae:
                best_mae = mae; best_cfg = cfg; best_mdl = mdl
            cfg_min = min(cfg_min, mae)
            row += f"{mae:>{col_w}.3f}"
        delta = cfg_min - ref_mae
        marker = " ★" if delta < -0.005 else "  "
        print(row + f"   Δmin={delta:+.3f}{marker}")

    print("  " + "─" * (26 + col_w * len(model_names) + 14))
    print(f"\n  ★ ЛУЧШАЯ: {best_cfg} + {best_mdl}  MAE={best_mae:.3f}  Δ={best_mae - ref_mae:+.3f} vs v0004+{ref_model}")
    return best_cfg, best_mdl


# ─── Визуализация ─────────────────────────────────────────────────────────────

def _plot_best(result: dict, df: pd.DataFrame, title: str, output_path: Path) -> None:
    subjects = sorted(df["subject_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}
    y_true = result["y_true"] / 60.0; y_pred = result["y_pred"] / 60.0
    subj_labels = result["subjects"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax = axes[0]
    for s in subjects:
        m = subj_labels == s
        ax.scatter(y_true[m], y_pred[m], color=subj_color[s], alpha=0.3, s=8, label=s)
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Истина, мин"); ax.set_ylabel("Предсказание, мин")
    ax.set_title(f"MAE={result['mae_min']:.3f} | R²={result['r2']:.3f} | ρ={result['rho']:.3f}")
    ax.legend(markerscale=2, fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax2 = axes[1]
    per = result["per_subj_mae_min"]
    subs_sorted = sorted(per.keys(), key=lambda s: per[s])
    vals = [per[s] for s in subs_sorted]
    bars = ax2.barh(subs_sorted, vals, color=[subj_color[s] for s in subs_sorted], alpha=0.85)
    ax2.axvline(result["mae_min"], color="black", lw=1.5, ls="--")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=8)
    ax2.set_xlabel("MAE, мин"); ax2.set_title("MAE по участникам"); ax2.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path.name}")


# ─── LT2 ─────────────────────────────────────────────────────────────────────

def run_lt2(df_full: pd.DataFrame, no_plots: bool = False) -> dict:
    print("\n" + "═" * 80)
    print("LT2 — комбинации вариабельности × зоопарк моделей")
    print("═" * 80)

    df = df_full[df_full["window_valid_all_required"] == 1].copy()
    df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    df = add_interaction_features(df)
    df = add_session_z_emg(df)
    print(f"Окон: {len(df)}, участников: {df['subject_id'].nunique()}")

    target = "target_time_to_lt2_center_sec"
    z_emg = [c for c in [
        "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
        "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
        "z_vl_prox_rest_rms", "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
    ] if c in df.columns]
    inter = [c for c in ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"] if c in df.columns]
    base = NIRS_FEATURES + HRV_FEATURES + inter + z_emg

    configs: dict[str, list[str]] = {
        "v0004-репликация":         base,
        "SampEn 30s":               base + SAMPEN_30,
        "SampEn 60s":               base + SAMPEN_60,
        "timing_cv":                base + TIMING_CV,
        "SampEn 30s+60s":           base + SAMPEN_30 + SAMPEN_60,
        "SampEn 30s+timing_cv":     base + SAMPEN_30 + TIMING_CV,
        "SampEn 60s+timing_cv":     base + SAMPEN_60 + TIMING_CV,
        "SampEn 30s+60s+timing_cv": base + SAMPEN_30 + SAMPEN_60 + TIMING_CV,
    }

    b_mae = baseline_mae(df, target)
    print(f"Baseline predict-mean: {b_mae:.3f} мин\n")

    results_matrix: dict[str, dict[str, dict]] = {}
    total = len(configs) * len(ZOO)
    done = 0
    for cfg, feats in configs.items():
        results_matrix[cfg] = {}
        avail = [f for f in feats if f in df.columns]
        for mdl_name, mdl_factory in ZOO.items():
            done += 1
            print(f"  [{done:2d}/{total}] {cfg:<26s} + {mdl_name:<13s}", end="", flush=True)
            r = loso(df, avail, target, mdl_factory)
            results_matrix[cfg][mdl_name] = r
            print(f"  MAE={r['mae_min']:.3f}")

    print()
    # Δ относительно v0004 + Ridge (лучшая линейная LT2)
    best_cfg, best_mdl = print_zoo_table(results_matrix, list(configs.keys()), ref_model="Ridge α=100")
    best_result = results_matrix[best_cfg][best_mdl]
    print_feature_importance(best_result)

    if not no_plots:
        _plot_best(best_result, df, f"LT2 v0008: {best_cfg} + {best_mdl}", OUT_DIR / "lt2" / "lt2_best.png")

    return {
        "best_cfg": best_cfg, "best_mdl": best_mdl,
        "best_result": best_result,
        "results_matrix": results_matrix,
        "df": df,
        "base_mae": results_matrix["v0004-репликация"]["Ridge α=100"]["mae_min"],
    }


# ─── LT1 ─────────────────────────────────────────────────────────────────────

def run_lt1(df_full: pd.DataFrame, no_plots: bool = False) -> dict:
    print("\n" + "═" * 80)
    print("LT1 — комбинации вариабельности × зоопарк моделей")
    print("═" * 80)

    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    df = df_full[df_full["target_time_to_lt1_usable"] == 1].copy()
    df = df[df["nirs_valid"] == 1].copy()
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    df = add_session_z_emg(df)
    if not session_params.empty:
        df = add_running_nirs_features(df, session_params)
    print(f"Окон: {len(df)}, участников: {df['subject_id'].nunique()}")
    print(f"Участники: {sorted(df['subject_id'].unique())}")

    target = "target_time_to_lt1_sec"
    z_emg = [c for c in [
        "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
        "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
        "z_vl_prox_rest_rms", "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
    ] if c in df.columns]
    base = z_emg + HRV_FEATURES

    configs: dict[str, list[str]] = {
        "v0004-репликация":         base,
        "SampEn 30s":               base + SAMPEN_30,
        "SampEn 60s":               base + SAMPEN_60,
        "timing_cv":                base + TIMING_CV,
        "SampEn 30s+60s":           base + SAMPEN_30 + SAMPEN_60,
        "SampEn 30s+timing_cv":     base + SAMPEN_30 + TIMING_CV,
        "SampEn 60s+timing_cv":     base + SAMPEN_60 + TIMING_CV,
        "SampEn 30s+60s+timing_cv": base + SAMPEN_30 + SAMPEN_60 + TIMING_CV,
    }

    b_mae = baseline_mae(df, target)
    print(f"Baseline predict-mean: {b_mae:.3f} мин\n")

    results_matrix: dict[str, dict[str, dict]] = {}
    total = len(configs) * len(ZOO)
    done = 0
    for cfg, feats in configs.items():
        results_matrix[cfg] = {}
        avail = [f for f in feats if f in df.columns]
        for mdl_name, mdl_factory in ZOO.items():
            done += 1
            print(f"  [{done:2d}/{total}] {cfg:<26s} + {mdl_name:<13s}", end="", flush=True)
            r = loso(df, avail, target, mdl_factory)
            results_matrix[cfg][mdl_name] = r
            print(f"  MAE={r['mae_min']:.3f}")

    print()
    # Δ относительно v0004 + Huber (лучшая линейная LT1)
    best_cfg, best_mdl = print_zoo_table(results_matrix, list(configs.keys()), ref_model="Huber ε=1.35")
    best_result = results_matrix[best_cfg][best_mdl]
    print_feature_importance(best_result)

    if not no_plots:
        _plot_best(best_result, df, f"LT1 v0008: {best_cfg} + {best_mdl}", OUT_DIR / "lt1" / "lt1_best.png")

    return {
        "best_cfg": best_cfg, "best_mdl": best_mdl,
        "best_result": best_result,
        "results_matrix": results_matrix,
        "df": df,
        "base_mae": results_matrix["v0004-репликация"]["Huber ε=1.35"]["mae_min"],
    }


# ─── Сводка ───────────────────────────────────────────────────────────────────

def save_summary(lt2_out: dict | None, lt1_out: dict | None) -> None:
    rows = []
    for task, out in [("LT2", lt2_out), ("LT1", lt1_out)]:
        if out is None:
            continue
        for cfg, mdl_results in out["results_matrix"].items():
            for mdl_name, r in mdl_results.items():
                rows.append({
                    "task": task, "config": cfg, "model": mdl_name,
                    "mae_min": r["mae_min"], "mae_std": r["mae_std"],
                    "r2": r["r2"], "rho": r["rho"],
                    "delta_vs_base": r["mae_min"] - out["base_mae"],
                    "best": (cfg == out["best_cfg"] and mdl_name == out["best_mdl"]),
                })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["task", "mae_min"]).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\n✅ Сводка: {OUT_DIR / 'summary.csv'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0008: комбинации вариабельности × зоопарк моделей.")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("v0008 — КОМБИНАЦИИ ПРИЗНАКОВ ВАРИАБЕЛЬНОСТИ × ЗООПАРК МОДЕЛЕЙ")
    print("=" * 80)
    df_full = pd.read_parquet(args.dataset)
    print(f"Датасет: {df_full.shape[0]} окон, {df_full['subject_id'].nunique()} участников")

    lt2_out = lt1_out = None
    if args.target in ("lt2", "both"):
        lt2_out = run_lt2(df_full, no_plots=args.no_plots)
    if args.target in ("lt1", "both"):
        lt1_out = run_lt1(df_full, no_plots=args.no_plots)
    save_summary(lt2_out, lt1_out)
    print("\n✅ Готово.")


if __name__ == "__main__":
    main()
