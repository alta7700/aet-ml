"""v0011c — Диагностика: насколько честна модель?

Вопрос: модель реально отслеживает физиологию или просто предсказывает
полную длительность теста и отсчитывает таймер?

Три инструмента:

1. Baseline-модели
   MeanPredictor      — предсказывает медиану LT2 из обучающих субъектов,
                        одинаково для всех окон. Baseline межсубъектной вариабельности.
   FirstWindowPredictor — обучается ТОЛЬКО на первом окне каждого субъекта,
                        предсказывает одно и то же число для всех последующих окон.
                        Если ≈ полной модели — sequential tracking бесполезен.

2. Stability score
   predicted_LT2_abs[t] = window_start_sec[t] + y_pred[t]
   std(predicted_LT2_abs) per subject, усреднённое по всем субъектам.
   ≈ 0  → модель = таймер (фиксирует полную длительность с первого окна)
   большой → реальное отслеживание

3. MAE vs σ_obs при фиксированном σ_p=5
   Показывает, при каком σ_obs модель перестаёт обновляться и деградирует
   к FirstWindowPredictor.

Запуск:
  uv run python scripts/v0011c_honest_baselines.py
  uv run python scripts/v0011c_honest_baselines.py --target lt2
  uv run python scripts/v0011c_honest_baselines.py --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
sys.path.insert(0, str(_ROOT / "scripts"))
from v0011_modality_ablation import (
    prepare_data,
    get_feature_cols,
    kalman_smooth,
)

OUT_DIR = _ROOT / "results" / "v0011c"

TARGET_CFG = {
    "lt2": "target_time_to_lt2_center_sec",
    "lt1": "target_time_to_lt1_pchip_sec",
}

# σ_obs сетка при σ_p=5
SIGMA_OBS_CURVE = [30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
SIGMA_P_FIXED   = 5.0

V0011_BEST = {
    ("EMG+NIRS+HRV", "lt2"): ("EN(α=0.1,l1=0.5)",
        lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)),
    ("EMG+NIRS+HRV", "lt1"): ("GBM(n=50,d=2)",
        lambda: GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)),
    ("EMG+NIRS",     "lt2"): ("EN(α=1.0,l1=0.2)",
        lambda: ElasticNet(alpha=1.0, l1_ratio=0.2, max_iter=5000, random_state=42)),
    ("EMG+NIRS",     "lt1"): ("Ridge(α=1000)",
        lambda: Ridge(alpha=1000)),
    ("EMG",          "lt2"): ("EN(α=1.0,l1=0.2)",
        lambda: ElasticNet(alpha=1.0, l1_ratio=0.2, max_iter=5000, random_state=42)),
    ("EMG",          "lt1"): ("Ridge(α=1000)",
        lambda: Ridge(alpha=1000)),
    ("NIRS",         "lt2"): ("SVR(C=10,ε=0.1)",
        lambda: SVR(kernel="rbf", C=10, epsilon=0.1)),
    ("NIRS",         "lt1"): ("SVR(C=10,ε=1.0)",
        lambda: SVR(kernel="rbf", C=10, epsilon=1.0)),
}


# ─── LOSO → сырые предсказания per-subject ───────────────────────────────────

def loso_raw_per_subject(df: pd.DataFrame, feat_cols: list[str],
                         target_col: str, model_factory) -> dict[str, dict]:
    """LOSO без Kalman. Возвращает per-subject: t_sec, y_pred, y_true."""
    subjects = sorted(df["subject_id"].unique())
    result = {}

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        mdl = model_factory()

        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        mdl.fit(X_tr, train[target_col].values)

        result[test_s] = {
            "t_sec":  test["window_start_sec"].values,
            "y_pred": mdl.predict(X_te),
            "y_true": test[target_col].values,
        }

    return result


# ─── Baseline 1: MeanPredictor ────────────────────────────────────────────────

def mean_predictor_loso(df: pd.DataFrame, target_col: str) -> dict[str, dict]:
    """Для каждого тестового субъекта предсказывает медиану LT2 из train."""
    subjects = sorted(df["subject_id"].unique())
    result = {}

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        # Медиана LT2 по одному значению на субъект из train
        train_lt2 = train.groupby("subject_id")[target_col].first()
        pred_val  = float(train_lt2.median())

        y_true = test[target_col].values
        # Предсказание времени ДО порога: pred_val - t_sec
        y_pred = np.maximum(pred_val - test["window_start_sec"].values, 0)

        result[test_s] = {
            "t_sec":  test["window_start_sec"].values,
            "y_pred": y_pred,
            "y_true": y_true,
        }

    return result


# ─── Baseline 2: FirstWindowPredictor ────────────────────────────────────────

def first_window_predictor_loso(df: pd.DataFrame, feat_cols: list[str],
                                 target_col: str, model_factory) -> dict[str, dict]:
    """Обучается на первом окне каждого субъекта.
    Предсказывает одно число (predicted_LT2_abs) для всех окон субъекта.
    """
    subjects = sorted(df["subject_id"].unique())
    result = {}

    for test_s in subjects:
        train_full = df[df["subject_id"] != test_s]
        test_full  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        # Берём только первое окно каждого субъекта
        train_first = train_full.groupby("subject_id").first().reset_index()
        test_first  = test_full.iloc[[0]]

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        mdl = model_factory()

        X_tr = sc.fit_transform(imp.fit_transform(train_first[feat_cols].values))
        X_te = sc.transform(imp.transform(test_first[feat_cols].values))
        mdl.fit(X_tr, train_first[target_col].values)

        # Предсказание — фиксированное для всего теста
        pred_duration = float(mdl.predict(X_te)[0])

        t_sec  = test_full["window_start_sec"].values
        y_true = test_full[target_col].values
        # predicted time_to_LT = pred_duration - elapsed (но не < 0)
        y_pred = np.maximum(pred_duration - t_sec, 0)

        result[test_s] = {
            "t_sec":        t_sec,
            "y_pred":       y_pred,
            "y_true":       y_true,
            "pred_duration": pred_duration,
        }

    return result


# ─── Метрики из per-subject dict ─────────────────────────────────────────────

def aggregate_metrics(per_subject: dict,
                      sigma_p: float = 0.0,
                      sigma_obs: float = 0.0,
                      apply_kalman: bool = False) -> dict:
    """MAE (мин), R², ρ, stability_std (с) по всем субъектам."""
    all_pred, all_true = [], []
    stab_stds = []

    for subj, d in per_subject.items():
        y_pred = d["y_pred"].copy()
        y_true = d["y_true"]
        t_sec  = d["t_sec"]

        if apply_kalman:
            y_pred = kalman_smooth(y_pred, sigma_p, sigma_obs)

        all_pred.append(y_pred)
        all_true.append(y_true)

        # Stability: std предсказанного абсолютного времени LT
        pred_abs = t_sec + y_pred
        stab_stds.append(float(np.std(pred_abs)))

    y_p = np.concatenate(all_pred)
    y_t = np.concatenate(all_true)

    return {
        "mae_min":       round(mean_absolute_error(y_t, y_p) / 60.0, 4),
        "r2":            round(r2_score(y_t, y_p), 3),
        "rho":           round(float(spearmanr(y_t, y_p).statistic), 3),
        "stability_std": round(float(np.mean(stab_stds)), 1),
    }


# ─── Кривая MAE vs σ_obs ─────────────────────────────────────────────────────

def mae_vs_sigma_obs(raw_per_subject: dict,
                     sigma_obs_values: list[float],
                     sigma_p: float) -> pd.DataFrame:
    rows = []
    for so in sigma_obs_values:
        m = aggregate_metrics(raw_per_subject, sigma_p, so, apply_kalman=True)
        m["sigma_obs"] = so
        rows.append(m)
    return pd.DataFrame(rows)


# ─── Графики ─────────────────────────────────────────────────────────────────

def plot_summary(results: dict, target: str, out_dir: Path) -> None:
    """Два графика: (1) MAE vs σ_obs кривые, (2) таблица stability."""
    fsets = [f for f in ["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"]
             if f in results]
    colors = {
        "EMG": "#E07B39", "NIRS": "#5B9BD5",
        "EMG+NIRS": "#70AD47", "EMG+NIRS+HRV": "#7030A0",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Диагностика честности модели — {target.upper()}", fontsize=13)

    # ── Левый: MAE vs σ_obs ───────────────────────────────────────────────────
    ax = axes[0]
    for fset in fsets:
        curve_df = results[fset]["curve"]
        ax.plot(curve_df["sigma_obs"], curve_df["mae_min"],
                color=colors[fset], linewidth=2.0, marker="o", markersize=5,
                label=fset)

        # Горизонтальные линии baselines
        mean_mae = results[fset]["mean_pred"]["mae_min"]
        fw_mae   = results[fset]["first_win"]["mae_min"]
        ax.axhline(mean_mae, color=colors[fset], linewidth=0.8,
                   linestyle=":", alpha=0.5)
        ax.axhline(fw_mae,   color=colors[fset], linewidth=0.8,
                   linestyle="--", alpha=0.7)

    # Вертикаль на σ_obs=150 (v0011 default) и σ_obs=500 (v0011b optimal)
    ax.axvline(150, color="gray",  linewidth=1.0, linestyle="-",
               alpha=0.5, label="v0011 (σ=150)")
    ax.axvline(500, color="black", linewidth=1.0, linestyle="-",
               alpha=0.5, label="v0011b (σ=500)")

    ax.set_xscale("log")
    ax.set_xlabel("σ_obs (с)", fontsize=11)
    ax.set_ylabel("Kalman MAE (мин)", fontsize=11)
    ax.set_title(f"MAE vs σ_obs  (σ_p={SIGMA_P_FIXED})\n"
                 f"-- FirstWindow baseline   ··· Mean baseline", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Правый: stability + baseline MAE таблица ──────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    col_labels = ["Модальность", "MeanPred\nMAE", "FirstWin\nMAE",
                  "Raw MAE", "Kalman(150)\nMAE", "Kalman(500)\nMAE",
                  "Stability\nstd (с)"]
    table_data = []
    for fset in fsets:
        r = results[fset]
        # Kalman MAE при σ=150 и σ=500
        k150 = r["curve"][r["curve"]["sigma_obs"] == 150]["mae_min"].values
        k500 = r["curve"][r["curve"]["sigma_obs"] == 500]["mae_min"].values
        k150_str = f"{k150[0]:.3f}" if len(k150) else "—"
        k500_str = f"{k500[0]:.3f}" if len(k500) else "—"
        stab = r["curve"][r["curve"]["sigma_obs"] == 500]["stability_std"].values
        stab_str = f"{stab[0]:.0f}" if len(stab) else "—"
        table_data.append([
            fset,
            f"{r['mean_pred']['mae_min']:.3f}",
            f"{r['first_win']['mae_min']:.3f}",
            f"{r['raw']['mae_min']:.3f}",
            k150_str,
            k500_str,
            stab_str,
        ])

    tbl = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.8)

    # Подсветить строки цветом
    for i, fset in enumerate(fsets):
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(colors[fset] + "33")

    ax2.set_title("Сводная таблица: MAE (мин) и Stability std", fontsize=10)

    plt.tight_layout()
    path = out_dir / f"honest_summary_{target}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def plot_stability_trajectories(raw_per_subj: dict, fw_per_subj: dict,
                                fset: str, target: str, out_dir: Path,
                                sigma_p: float, sigma_obs_honest: float) -> None:
    """Для лучшего набора: full model с σ_obs=honest vs FirstWindow per subject."""
    subjects = sorted(raw_per_subj.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(subjects) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Честная модель ({fset} / {target.upper()}):\n"
        f"Full model σ_obs={sigma_obs_honest:.0f} vs FirstWindow vs Истина",
        fontsize=12, y=1.01)

    for idx, subj in enumerate(subjects):
        ax = axes[idx // n_cols][idx % n_cols]

        d_raw = raw_per_subj[subj]
        d_fw  = fw_per_subj[subj]
        t_min = d_raw["t_sec"] / 60.0

        y_true   = d_raw["y_true"] / 60.0
        y_kalman = kalman_smooth(d_raw["y_pred"], sigma_p, sigma_obs_honest) / 60.0
        y_fw     = d_fw["y_pred"] / 60.0

        ax.plot(t_min, y_true,   color="black",   linewidth=2.0, label="Истина")
        ax.plot(t_min, y_fw,     color="#999999",  linewidth=1.5,
                linestyle="--", label=f"FirstWin")
        ax.plot(t_min, y_kalman, color="#7030A0",  linewidth=1.8,
                label=f"Full (σ={sigma_obs_honest:.0f})")

        mae_full = mean_absolute_error(y_true, y_kalman)
        mae_fw   = mean_absolute_error(y_true, y_fw)
        ax.set_title(f"{subj}  MAE: full={mae_full:.2f} fw={mae_fw:.2f}",
                     fontsize=8, pad=3)
        ax.set_xlabel("Время теста (мин)", fontsize=7)
        ax.set_ylabel("До порога (мин)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    for idx in range(len(subjects), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="black",  linewidth=2.0, label="Истина"),
        plt.Line2D([0], [0], color="#999999", linewidth=1.5, linestyle="--",
                   label="FirstWindowPredictor"),
        plt.Line2D([0], [0], color="#7030A0", linewidth=1.8,
                   label=f"Full model (σ_obs={sigma_obs_honest:.0f})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = out_dir / f"honest_trajectories_{fset.replace('+','_')}_{target}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v0011c — честные baselines")
    parser.add_argument("--target",      choices=["lt1", "lt2", "both"], default="both")
    parser.add_argument("--feature-set", nargs="+",
                        choices=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"],
                        default=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"])
    parser.add_argument("--dataset",
                        default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    targets = ["lt2", "lt1"] if args.target == "both" else [args.target]
    all_summary = []

    for tgt in targets:
        target_col = TARGET_CFG[tgt]
        df_prep = prepare_data(df_raw, session_params, tgt)
        if target_col not in df_prep.columns:
            continue
        df_prep = df_prep.dropna(subset=[target_col])
        n_subj = df_prep["subject_id"].nunique()

        print(f"\n{'═'*70}")
        print(f"ТАРГЕТ: {tgt.upper()}  ({n_subj} субъектов)")
        print(f"{'═'*70}")

        results = {}

        for fset in args.feature_set:
            key = (fset, tgt)
            if key not in V0011_BEST:
                continue

            model_name, factory = V0011_BEST[key]
            feat_cols = get_feature_cols(df_prep, fset)
            if not feat_cols:
                continue

            print(f"\n  [{fset}]  модель: {model_name}")

            # Full LOSO raw
            print(f"    LOSO full...", end=" ", flush=True)
            raw = loso_raw_per_subject(df_prep, feat_cols, target_col, factory)
            raw_m = aggregate_metrics(raw)
            print(f"raw MAE={raw_m['mae_min']:.3f} мин")

            # MeanPredictor
            print(f"    MeanPredictor...", end=" ", flush=True)
            mean_pred = mean_predictor_loso(df_prep, target_col)
            mean_m = aggregate_metrics(mean_pred)
            print(f"MAE={mean_m['mae_min']:.3f} мин")

            # FirstWindowPredictor
            print(f"    FirstWindowPredictor...", end=" ", flush=True)
            fw_pred = first_window_predictor_loso(
                df_prep, feat_cols, target_col, factory)
            fw_m = aggregate_metrics(fw_pred)
            print(f"MAE={fw_m['mae_min']:.3f} мин")

            # Кривая MAE vs σ_obs
            print(f"    Kalman curve...", end=" ", flush=True)
            curve_df = mae_vs_sigma_obs(raw, SIGMA_OBS_CURVE, SIGMA_P_FIXED)
            print(f"done  (best={curve_df['mae_min'].min():.3f} @ "
                  f"σ_obs={curve_df.loc[curve_df['mae_min'].idxmin(), 'sigma_obs']:.0f})")

            results[fset] = {
                "raw":        raw_m,
                "mean_pred":  mean_m,
                "first_win":  fw_m,
                "curve":      curve_df,
                "_raw_subj":  raw,
                "_fw_subj":   fw_pred,
            }

            # Stability при σ_obs=500
            stab500 = curve_df[curve_df["sigma_obs"] == 500]["stability_std"].values
            print(f"    Stability (σ=500): std={stab500[0]:.0f} с  "
                  f"(FirstWin stability: "
                  f"{aggregate_metrics(fw_pred, apply_kalman=False)['stability_std']:.0f} с)")

            # Сводка
            k150 = curve_df[curve_df["sigma_obs"] == 150]["mae_min"].values[0]
            k500 = curve_df[curve_df["sigma_obs"] == 500]["mae_min"].values[0]

            gap_fw   = raw_m["mae_min"] - fw_m["mae_min"]
            gap_mean = raw_m["mae_min"] - mean_m["mae_min"]
            print(f"\n    Сводка:")
            print(f"      MeanPredictor:       {mean_m['mae_min']:.3f} мин")
            print(f"      FirstWindowPredict:  {fw_m['mae_min']:.3f} мин")
            print(f"      Raw (no Kalman):     {raw_m['mae_min']:.3f} мин  "
                  f"Δ vs FirstWin={gap_fw:+.3f}")
            print(f"      Kalman σ_obs=150:    {k150:.3f} мин")
            print(f"      Kalman σ_obs=500:    {k500:.3f} мин")

            if abs(gap_fw) < 0.05:
                verdict = "~  Sequential tracking не даёт ничего над FirstWindow"
            elif gap_fw > 0.2:
                verdict = "⚠️  Sequential tracking хуже FirstWindow (нет онлайн-ценности)"
            elif gap_fw < -0.2:
                verdict = "✓  Sequential tracking значимо лучше FirstWindow"
            else:
                verdict = f"~  Небольшой выигрыш sequential tracking: {gap_fw:+.3f} мин"
            print(f"    {verdict}")

            all_summary.append({
                "feature_set": fset, "target": tgt,
                "mae_mean_pred": mean_m["mae_min"],
                "mae_first_win": fw_m["mae_min"],
                "mae_raw":       raw_m["mae_min"],
                "mae_k150":      k150,
                "mae_k500":      k500,
                "stability_k500": float(stab500[0]) if len(stab500) else None,
            })

        if results:
            plot_summary(results, tgt, OUT_DIR)

            # Trajectories для лучшего набора
            best_fset = (
                "EMG+NIRS+HRV" if "EMG+NIRS+HRV" in results else
                list(results.keys())[-1]
            )
            # "Честный" σ_obs = где full model заметно лучше FirstWindow
            curve = results[best_fset]["curve"]
            fw_mae = results[best_fset]["first_win"]["mae_min"]
            # Находим наименьший σ_obs где MAE < fw_mae (реально лучше FirstWindow)
            better = curve[curve["mae_min"] < fw_mae]
            if not better.empty:
                honest_sigma = float(better["sigma_obs"].iloc[0])
            else:
                honest_sigma = 150.0  # fallback
            print(f"\n  Честный σ_obs для {best_fset}: {honest_sigma:.0f} с")

            plot_stability_trajectories(
                results[best_fset]["_raw_subj"],
                results[best_fset]["_fw_subj"],
                best_fset, tgt, OUT_DIR,
                SIGMA_P_FIXED, honest_sigma,
            )

    # ── Сохранение ────────────────────────────────────────────────────────────
    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(OUT_DIR / "honest_summary.csv", index=False)
        print(f"\n✓ Сохранено в {OUT_DIR}/")
        print("\nИтоговая таблица:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
