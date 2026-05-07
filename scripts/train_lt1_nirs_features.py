"""Инжиниринг NIRS-признаков для LT1 и обучение модели.

Проблема: within-subject ρ(SmO2, LT1) = 0.968 — идеально,
но между субъектами абсолютный SmO2 не коррелирует с временем LT1.
У одного LT1 при SmO2=75%, у другого при SmO2=62%.

Решение: признаки относительного положения внутри сессии:
  1. smo2_from_running_max  — насколько упал от пика в этой сессии
  2. hhb_from_running_min   — насколько вырос от минимума
  3. smo2_rel_drop          — (baseline - current) / baseline (%)
  4. hhb_rel_rise           — (current - baseline) / |baseline| (%)
  5. smo2_slope_accel       — slope_30s - slope_120s (ускорение дропа)

Признаки 1-2 — каузальные running-статистики по сессии (смотрим только в прошлое).
Признак 5 требует отдельного расчёта slope на 120-секундном контексте.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR

OUTPUT_DIR = Path("results/lt1")

# ─────────────────────── Признаки ─────────────────────────────────────────

BASE_NIRS = [
    "trainred_smo2_mean", "trainred_smo2_drop", "trainred_smo2_slope", "trainred_smo2_std",
    "trainred_hhb_mean", "trainred_hhb_slope", "trainred_hhb_std",
    "trainred_hbdiff_mean", "trainred_hbdiff_slope", "trainred_hbdiff_std",
    "trainred_thb_mean", "trainred_thb_slope", "trainred_thb_std",
]
BASE_HRV = [
    "hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms",
    "hrv_sd1_ms", "hrv_sd2_ms", "hrv_sd1_sd2_ratio",
    "hrv_dfa_alpha1",
]
NEW_NIRS = [
    "smo2_from_running_max",
    "hhb_from_running_min",
    "smo2_rel_drop_pct",
    "hhb_rel_rise_pct",
    "hbdiff_from_running_max",
]


def add_running_nirs_features(df: pd.DataFrame, session_params: pd.DataFrame) -> pd.DataFrame:
    """Добавляет running-признаки на основе истории сессии.

    Для каждого субъекта и каждого окна считаем:
      - running max/min по всем окнам ДО и ВКЛЮЧАЯ данное (causal)
      - нормировку на baseline из session_params
    """
    df = df.copy()
    sp = session_params.set_index("subject_id")

    df["smo2_from_running_max"] = np.nan
    df["hhb_from_running_min"] = np.nan
    df["smo2_rel_drop_pct"] = np.nan
    df["hhb_rel_rise_pct"] = np.nan
    df["hbdiff_from_running_max"] = np.nan

    for subj, g in df.groupby("subject_id"):
        g = g.sort_values("window_center_sec")
        idx = g.index

        # Running max SmO2 (каузально: expэндинг максимум)
        smo2 = g["trainred_smo2_mean"].values
        hhb = g["trainred_hhb_mean"].values
        hbdiff = g["trainred_hbdiff_mean"].values

        smo2_running_max = np.maximum.accumulate(
            np.where(np.isfinite(smo2), smo2, -np.inf)
        )
        hhb_running_min = np.minimum.accumulate(
            np.where(np.isfinite(hhb), hhb, np.inf)
        )
        hbdiff_running_max = np.maximum.accumulate(
            np.where(np.isfinite(hbdiff), hbdiff, -np.inf)
        )

        # Заменяем -inf/inf обратно на nan
        smo2_running_max = np.where(np.isinf(smo2_running_max), np.nan, smo2_running_max)
        hhb_running_min = np.where(np.isinf(hhb_running_min), np.nan, hhb_running_min)
        hbdiff_running_max = np.where(np.isinf(hbdiff_running_max), np.nan, hbdiff_running_max)

        df.loc[idx, "smo2_from_running_max"] = smo2_running_max - smo2
        df.loc[idx, "hhb_from_running_min"] = hhb - hhb_running_min
        df.loc[idx, "hbdiff_from_running_max"] = hbdiff_running_max - hbdiff

        # Нормировка на baseline
        if subj in sp.index:
            baseline_smo2 = float(sp.loc[subj, "nirs_smo2_baseline_mean"])
            if np.isfinite(baseline_smo2) and baseline_smo2 > 0:
                # (baseline - current) / baseline × 100
                df.loc[idx, "smo2_rel_drop_pct"] = (
                    (baseline_smo2 - smo2) / baseline_smo2 * 100.0
                )

            # Для HHb нет baseline в session_params, используем running min как proxy
            # hhb_rel_rise = (current - min_in_session) / (max_in_session_so_far - min)
            hhb_range = smo2_running_max - hhb_running_min  # приблизительно
            # Используем первые 3 окна как «baseline» HHb
            first_hhb = g["trainred_hhb_mean"].iloc[:3].mean()
            if np.isfinite(first_hhb) and first_hhb != 0:
                df.loc[idx, "hhb_rel_rise_pct"] = (
                    (hhb - first_hhb) / abs(first_hhb) * 100.0
                )

    return df


# ─────────────────────── LOSO ─────────────────────────────────────────────

def loso(df: pd.DataFrame, features: list[str], target: str,
         alpha: float = 0.5, l1_ratio: float = 0.9) -> dict:
    """LOSO CV с ElasticNet."""
    subjects = sorted(df["subject_id"].unique())
    preds, trues, subjs = [], [], []

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test = df[df["subject_id"] == test_s]
        X_tr = train[features].values
        y_tr = train[target].values
        X_te = test[features].values
        y_te = test[target].values

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        X_tr = sc.fit_transform(imp.fit_transform(X_tr))
        X_te = sc.transform(imp.transform(X_te))
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        preds.append(y_pred)
        trues.append(y_te)
        subjs.append(np.full(len(y_te), test_s))

    y_pred_all = np.concatenate(preds)
    y_true_all = np.concatenate(trues)
    subj_all = np.concatenate(subjs)

    mae = float(mean_absolute_error(y_true_all, y_pred_all))
    r2 = float(r2_score(y_true_all, y_pred_all))
    rho = float(spearmanr(y_true_all, y_pred_all).statistic)
    per_subj = {s: float(mean_absolute_error(y_true_all[subj_all == s],
                                              y_pred_all[subj_all == s]))
                for s in subjects}

    return {"mae_min": mae/60, "r2": r2, "rho": rho,
            "per_subj_mae_min": {k: v/60 for k, v in per_subj.items()},
            "y_pred": y_pred_all, "y_true": y_true_all, "subjects": subj_all}


def print_result(name: str, r: dict) -> None:
    """Печатает результат одной конфигурации."""
    per = r["per_subj_mae_min"]
    std = np.std(list(per.values()))
    print(f"  {name:45s}  MAE={r['mae_min']:.3f}±{std:.3f} мин  R²={r['r2']:.3f}  ρ={r['rho']:.3f}")


# ─────────────────────── Визуализация ─────────────────────────────────────

def plot_comparison(results: dict[str, dict], output_dir: Path) -> None:
    """Сравнительный барчарт MAE по конфигурациям."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    vals = [results[n]["mae_min"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = []
    for n in names:
        if "new" in n.lower() or "running" in n.lower():
            colors.append("#2ca02c")
        elif "base" in n.lower():
            colors.append("#aec7e8")
        else:
            colors.append("#1f77b4")

    bars = ax.barh(names, vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)

    ax.set_xlabel("MAE, мин (LOSO)")
    ax.set_title("LT1: Сравнение признаков\n(зелёный = новые running-признаки)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "lt1_nirs_features_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {output_dir}/lt1_nirs_features_comparison.png")


def plot_running_features(df: pd.DataFrame, output_dir: Path) -> None:
    """Показывает running-признаки по участникам — интуиция за новыми фичами."""
    subjects = sorted(df["subject_id"].unique())
    ncols = 3
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))

    for i, s in enumerate(subjects):
        ax = axes_flat[i]
        g = df[df["subject_id"] == s].sort_values("window_center_sec")
        t = g["window_center_sec"].values / 60.0
        lt1_t = float(g["lt1_power_w"].iloc[0]) if "lt1_power_w" in g else np.nan

        # Нормируем running-признаки для отображения
        smo2 = g["smo2_from_running_max"].values
        hhb = g["hhb_from_running_min"].values
        target = g["target_time_to_lt1_sec"].values / 60.0

        ax2 = ax.twinx()
        ax.plot(t, smo2, color="#1f77b4", linewidth=1.2, label="SmO2↓от пика")
        ax.plot(t, hhb, color="#d62728", linewidth=1.2, linestyle="--", label="HHb↑от минимума")
        ax2.plot(t, target, color="gray", linewidth=1, alpha=0.5, label="До LT1, мин")
        ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)

        ax.set_title(f"{s}", fontsize=9)
        ax.set_xlabel("Время теста, мин", fontsize=8)
        ax.set_ylabel("Running Δ (у.е.)", fontsize=8)
        ax2.set_ylabel("До LT1, мин", fontsize=8, color="gray")
        ax.legend(fontsize=7, loc="upper left")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Running NIRS-признаки vs время до LT1", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "lt1_running_features_vis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {output_dir}/lt1_running_features_vis.png")


# ─────────────────────── Main ─────────────────────────────────────────────

def main() -> None:
    """Основная логика."""
    # Загрузка
    merged = pd.read_parquet(DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    session_params = pd.read_parquet(DEFAULT_DATASET_DIR / "session_params.parquet")

    # LT1 usable
    df_lt1 = merged[merged["target_time_to_lt1_usable"] == 1].copy()
    df_lt1 = df_lt1[df_lt1["nirs_valid"] == 1].copy()
    print(f"LT1 usable + nirs_valid: {len(df_lt1)} окон, {df_lt1['subject_id'].nunique()} участников")

    # Добавляем running-признаки
    df_lt1 = add_running_nirs_features(df_lt1, session_params)

    target = "target_time_to_lt1_sec"
    results: dict[str, dict] = {}

    print("\n─── Within-subject ρ для новых признаков ───")
    for col in NEW_NIRS:
        if col not in df_lt1.columns:
            continue
        rhos = []
        for s, g in df_lt1.groupby("subject_id"):
            v = g[col]
            if v.nunique() > 3 and v.notna().sum() > 5:
                rhos.append(g[col].corr(g[target], method="spearman"))
        if rhos:
            print(f"  {col:35s}  mean_rho={np.mean(rhos):.3f}  std={np.std(rhos):.3f}")

    print("\n─── LOSO MAE ───")

    # 0. Baseline
    from sklearn.dummy import DummyRegressor
    subjs = sorted(df_lt1["subject_id"].unique())
    mean_errors = []
    for s in subjs:
        tr = df_lt1[df_lt1["subject_id"] != s][target].values
        te = df_lt1[df_lt1["subject_id"] == s][target].values
        mean_errors.append(mean_absolute_error(te, np.full(len(te), tr.mean())))
    print(f"  {'Baseline (predict-mean)':45s}  MAE={np.mean(mean_errors)/60:.3f} мин")

    # 1. Старые NIRS (as-is)
    base_nirs = [c for c in BASE_NIRS if c in df_lt1.columns]
    r = loso(df_lt1, base_nirs, target)
    results["Base NIRS (as-is)"] = r
    print_result("Base NIRS (as-is)", r)

    # 2. Только новые running-признаки
    new_feats = [c for c in NEW_NIRS if c in df_lt1.columns]
    r = loso(df_lt1, new_feats, target)
    results["New running NIRS only"] = r
    print_result("New running NIRS only", r)

    # 3. Base NIRS + новые
    r = loso(df_lt1, base_nirs + new_feats, target)
    results["Base NIRS + running"] = r
    print_result("Base NIRS + running", r)

    # 4. Base NIRS + running + HRV
    hrv_feats = [c for c in BASE_HRV if c in df_lt1.columns]
    r = loso(df_lt1, base_nirs + new_feats + hrv_feats, target)
    results["Base NIRS + running + HRV"] = r
    print_result("Base NIRS + running + HRV", r)

    # 5. Только running + HRV (без абсолютного SmO2)
    r = loso(df_lt1, new_feats + hrv_feats, target)
    results["Running NIRS + HRV"] = r
    print_result("Running NIRS + HRV", r)

    # 6. Только HRV (контроль)
    r = loso(df_lt1, hrv_feats, target)
    results["HRV only (контроль)"] = r
    print_result("HRV only (контроль)", r)

    # 7. smo2_from_running_max + HRV — минималистичная физиологическая модель
    minimal_nirs = [c for c in ["smo2_from_running_max", "hhb_from_running_min",
                                 "smo2_rel_drop_pct"] if c in df_lt1.columns]
    r = loso(df_lt1, minimal_nirs + hrv_feats, target)
    results["SmO2_running + HRV (минималист)"] = r
    print_result("SmO2_running + HRV (минималист)", r)

    # ── Сводка ──
    best_name = min(results, key=lambda k: results[k]["mae_min"])
    best = results[best_name]
    print(f"\n★ Лучшая конфигурация: {best_name}")
    print(f"  MAE  = {best['mae_min']:.3f} мин")
    print(f"  R²   = {best['r2']:.3f}")
    print(f"  ρ    = {best['rho']:.3f}")
    print("  MAE по участникам:")
    for s, m in sorted(best["per_subj_mae_min"].items()):
        print(f"    {s}: {m:.2f} мин")

    # ── Графики ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, OUTPUT_DIR)
    plot_running_features(df_lt1, OUTPUT_DIR)

    print("\n✅ Готово.")


if __name__ == "__main__":
    main()
