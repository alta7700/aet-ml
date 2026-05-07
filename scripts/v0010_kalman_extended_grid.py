"""v0010 — Kalman расширенная сетка параметров

Версия:    v0010
Дата:      2026-05-07
Предыдущая версия: v0009_kalman_postprocess.py
Результаты: results/v0010/

Мотивация:
  В v0009 лучшие параметры (σ_p=15с, σ_obs=150с) оказались на границе сетки.
  Расширяем в обе стороны:
    σ_p   ∈ {5, 10, 15, 30} с       (добавлены меньшие значения)
    σ_obs ∈ {150, 200, 250, 300, 400} с  (добавлены большие значения)
  Итого 4×5 = 20 комбинаций на конфиг.

  Конфигурации (лучшие из v0008):
    LT2: v0008★ SampEn 30s + GBM n=200    (raw MAE=2.043)
    LT1: v0008★ SampEn 60s+timing_cv+Huber (raw MAE=1.959)
  + v0004-ref для контроля.

Решение:
  Если MAE хуже v0009 → v0010 удаляется, фиксируются только результаты v0009.
  Если MAE лучше → фиксируются результаты обоих.

Воспроизведение:
  uv run python scripts/v0010_kalman_extended_grid.py
  uv run python scripts/v0010_kalman_extended_grid.py --no-plots
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR

OUT_DIR = _ROOT / "results" / "v0010"

# ─── Параметры сетки (расширенные) ────────────────────────────────────────────

SIGMA_PROCESS_GRID = [5.0, 10.0, 15.0, 30.0]
SIGMA_OBS_GRID     = [150.0, 200.0, 250.0, 300.0, 400.0]

# Результаты v0009 (граница поиска) — для сравнения в итоговой таблице
V0009_LT2_KALMAN = 1.943
V0009_LT1_KALMAN = 1.901

# ─── Признаки ─────────────────────────────────────────────────────────────────

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
Z_EMG = [
    "z_vl_dist_load_rms", "z_vl_dist_load_mdf", "z_vl_dist_load_mav",
    "z_vl_dist_rest_rms", "z_vl_prox_load_rms", "z_vl_prox_load_mdf",
    "z_vl_prox_rest_rms", "z_delta_rms_prox_dist_load", "z_ratio_rms_prox_dist_load",
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
    preds, trues, subjs, wstarts = [], [], [], []
    coefs_list: list[np.ndarray] = []
    importances_list: list[np.ndarray] = []

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        model = model_factory()
        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        model.fit(X_tr, train[target].values)
        y_pred = model.predict(X_te)

        preds.append(y_pred); trues.append(test[target].values)
        subjs.append(np.full(len(y_pred), test_s))
        wstarts.append(test["window_start_sec"].values)

        if hasattr(model, "coef_"):
            coefs_list.append(model.coef_)
        elif hasattr(model, "feature_importances_"):
            importances_list.append(model.feature_importances_)

    y_pred_all = np.concatenate(preds)
    y_true_all = np.concatenate(trues)
    subj_all   = np.concatenate(subjs)
    wstart_all = np.concatenate(wstarts)
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
        "y_pred": y_pred_all, "y_true": y_true_all,
        "subjects": subj_all, "window_starts": wstart_all,
        "features_used": feat_cols,
        "coef_mean":  np.mean(coefs_list, axis=0) if coefs_list else None,
        "imp_mean":   np.mean(importances_list, axis=0) if importances_list else None,
    }


# ─── Kalman ───────────────────────────────────────────────────────────────────

def kalman_smooth_subject(preds_sec: np.ndarray, wstarts_sec: np.ndarray,
                           sigma_process: float, sigma_obs: float) -> np.ndarray:
    n = len(preds_sec)
    if n == 0:
        return preds_sec.copy()
    Q = sigma_process ** 2; R = sigma_obs ** 2
    tau = preds_sec[0]; P = R
    out = np.empty(n); out[0] = tau
    for t in range(1, n):
        step = max(float(wstarts_sec[t] - wstarts_sec[t - 1]), 1.0)
        tau_p = tau - step; P_p = P + Q
        K = P_p / (P_p + R)
        tau = tau_p + K * (preds_sec[t] - tau_p); P = (1 - K) * P_p
        out[t] = tau
    return out


def apply_kalman(result: dict, sigma_process: float, sigma_obs: float) -> dict:
    y_raw  = result["y_pred"].copy()
    y_sm   = y_raw.copy()
    subjs  = result["subjects"]
    wstart = result["window_starts"]
    y_true = result["y_true"]

    for s in np.unique(subjs):
        mask = subjs == s
        y_sm[mask] = kalman_smooth_subject(y_raw[mask], wstart[mask],
                                           sigma_process, sigma_obs)

    mae = float(mean_absolute_error(y_true, y_sm)) / 60.0
    per_subj = {
        s: float(mean_absolute_error(y_true[subjs == s], y_sm[subjs == s])) / 60.0
        for s in np.unique(subjs)
    }
    return {
        **result,
        "mae_min": mae,
        "mae_std": float(np.std(list(per_subj.values()))),
        "r2": float(r2_score(y_true, y_sm)),
        "rho": float(spearmanr(y_true, y_sm).statistic),
        "per_subj_mae_min": per_subj,
        "y_pred": y_sm,
        "kalman_sigma_process": sigma_process,
        "kalman_sigma_obs": sigma_obs,
    }


def kalman_grid_search(result_raw: dict, label: str, v009_best: float) -> tuple[dict, float, float]:
    """Grid-search. Возвращает (best_result, best_sp, best_so)."""
    raw_mae = result_raw["mae_min"]
    print(f"\n  Kalman grid-search: «{label}»  raw={raw_mae:.3f}  v0009_best={v009_best:.3f}")

    # Заголовок таблицы
    header = f"  {'σ_p \\ σ_obs':>10s}" + "".join(f"  {int(so):>5d}с" for so in SIGMA_OBS_GRID)
    print(header)
    print("  " + "─" * (12 + 8 * len(SIGMA_OBS_GRID)))

    best_mae = float("inf"); best_result = None; best_sp = best_so = 0.0
    grid_rows: list[dict] = []

    for sp in SIGMA_PROCESS_GRID:
        row = f"  σ_p={sp:>4.0f}с   "
        for so in SIGMA_OBS_GRID:
            r_k = apply_kalman(result_raw, sigma_process=sp, sigma_obs=so)
            mae = r_k["mae_min"]
            delta = mae - raw_mae
            marker = "★" if mae < v009_best - 1e-4 else (" " if mae < v009_best + 1e-4 else " ")
            row += f"  {mae:.3f}({delta:+.3f}){marker}"
            if mae < best_mae:
                best_mae = mae; best_result = r_k; best_sp = sp; best_so = so
            grid_rows.append({"sigma_p": sp, "sigma_obs": so, "mae": mae,
                               "delta_raw": delta, "beats_v009": mae < v009_best - 1e-4})
        print(row)

    delta_v009 = best_mae - v009_best
    marker = "★ ЛУЧШЕ v0009" if delta_v009 < -1e-4 else ("= v0009" if abs(delta_v009) < 1e-4 else "хуже v0009")
    print(f"\n  Лучшая: σ_p={best_sp:.0f}с, σ_obs={best_so:.0f}с  MAE={best_mae:.3f}  "
          f"Δ_raw={best_mae - raw_mae:+.3f}  Δ_v009={delta_v009:+.3f}  [{marker}]")

    return best_result, best_sp, best_so


# ─── Визуализации ─────────────────────────────────────────────────────────────

def plot_before_after(result_raw: dict, result_smooth: dict,
                      df: pd.DataFrame, title: str, output_path: Path) -> None:
    subjects = sorted(df["subject_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, res, lbl in zip(axes[:2], [result_raw, result_smooth],
                             ["До Kalman",
                              f"После Kalman (σ_p={result_smooth['kalman_sigma_process']:.0f}с, "
                              f"σ_obs={result_smooth['kalman_sigma_obs']:.0f}с)"]):
        y_true = res["y_true"] / 60.0; y_pred = res["y_pred"] / 60.0
        for s in subjects:
            m = res["subjects"] == s
            ax.scatter(y_true[m], y_pred[m], color=subj_color[s], alpha=0.3, s=7, label=s)
        lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Истина, мин"); ax.set_ylabel("Предсказание, мин")
        ax.set_title(f"{lbl}\nMAE={res['mae_min']:.3f} | R²={res['r2']:.3f} | ρ={res['rho']:.3f}")
        ax.legend(markerscale=2, fontsize=6, ncol=2); ax.grid(alpha=0.3)

    ax = axes[2]
    x = np.arange(len(subjects)); w = 0.38
    vals_r = [result_raw["per_subj_mae_min"].get(s, np.nan) for s in subjects]
    vals_s = [result_smooth["per_subj_mae_min"].get(s, np.nan) for s in subjects]
    ax.bar(x - w/2, vals_r, w, label=f"Сырая  ({result_raw['mae_min']:.3f})",    alpha=0.8, color="#1f77b4")
    ax.bar(x + w/2, vals_s, w, label=f"Kalman ({result_smooth['mae_min']:.3f})", alpha=0.8, color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("MAE, мин"); ax.set_title("MAE по участникам")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  → {output_path.name}")


def plot_trajectories(result_raw: dict, result_smooth: dict, df: pd.DataFrame,
                      target: str, output_path: Path, n_show: int = 9) -> None:
    subjects = sorted(df["subject_id"].unique())[:n_show]
    palette = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    subj_color = {s: palette[i] for i, s in enumerate(subjects)}

    ncols = 3; nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle("Kalman v0010: траектории", fontsize=12, fontweight="bold")

    for i, s in enumerate(subjects):
        ax = axes_flat[i]
        sub = df[df["subject_id"] == s].sort_values("window_start_sec")
        t = sub["window_start_sec"].values / 60.0
        y_true_s = sub[target].values / 60.0
        mask = result_raw["subjects"] == s
        t_pred = result_raw["window_starts"][mask] / 60.0
        y_raw = result_raw["y_pred"][mask] / 60.0
        y_sm  = result_smooth["y_pred"][mask] / 60.0

        ax.plot(t, y_true_s, "k-", lw=2.0, label="Истина")
        ax.plot(t_pred, y_raw, color=subj_color[s], lw=1.2, ls="--", alpha=0.6, label="Сырое")
        ax.plot(t_pred, y_sm,  color=subj_color[s], lw=1.8, ls="-",  label="Kalman")
        ax.axhline(0, color="red", lw=0.8, ls=":", alpha=0.7)
        mae_r = result_raw["per_subj_mae_min"][s]
        mae_s = result_smooth["per_subj_mae_min"][s]
        ax.set_title(f"{s}  raw={mae_r:.2f}→kalman={mae_s:.2f} мин", fontsize=9)
        ax.set_xlabel("Прошло, мин", fontsize=8); ax.set_ylabel("До порога, мин", fontsize=8)
        ax.legend(fontsize=7); ax.grid(alpha=0.25)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  → {output_path.name}")


# ─── LT2 / LT1 ────────────────────────────────────────────────────────────────

def run_lt2(df_full: pd.DataFrame, no_plots: bool = False) -> list[dict]:
    print("\n" + "═" * 75)
    print("LT2 — расширенная сетка Kalman")
    print("═" * 75)

    df = df_full[df_full["window_valid_all_required"] == 1].copy()
    df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    df = add_interaction_features(df); df = add_session_z_emg(df)
    print(f"Окон: {len(df)}, участников: {df['subject_id'].nunique()}")

    target = "target_time_to_lt2_center_sec"
    z_emg = [c for c in Z_EMG if c in df.columns]
    inter = [c for c in ["feat_smo2_x_rr", "feat_smo2_x_dfa", "feat_rr_per_watt"] if c in df.columns]
    base  = NIRS_FEATURES + HRV_FEATURES + inter + z_emg

    configs = {
        "v0008★ SampEn30 + GBM": (
            base + SAMPEN_30,
            lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
            V0009_LT2_KALMAN,
        ),
    }

    rows = []
    for label, (feats, factory, v009_ref) in configs.items():
        avail = [f for f in feats if f in df.columns]
        print(f"\n[LT2] {label}  ({len(avail)} признаков)")
        r_raw = loso(df, avail, target, factory)
        print(f"  Raw:  MAE={r_raw['mae_min']:.3f} ± {r_raw['mae_std']:.3f}  R²={r_raw['r2']:.3f}  ρ={r_raw['rho']:.3f}")

        r_best, best_sp, best_so = kalman_grid_search(r_raw, label, v009_ref)
        rows.append({"task": "LT2", "config": label,
                     "raw_mae": r_raw["mae_min"], "kalman_mae": r_best["mae_min"],
                     "delta_raw": r_best["mae_min"] - r_raw["mae_min"],
                     "delta_v009": r_best["mae_min"] - v009_ref,
                     "sigma_p": best_sp, "sigma_obs": best_so})

        if not no_plots:
            plot_before_after(r_raw, r_best, df, f"LT2 v0010: {label}",
                              OUT_DIR / "lt2" / "scatter.png")
            plot_trajectories(r_raw, r_best, df, target,
                              OUT_DIR / "lt2" / "trajectories.png")
    return rows


def run_lt1(df_full: pd.DataFrame, no_plots: bool = False) -> list[dict]:
    print("\n" + "═" * 75)
    print("LT1 — расширенная сетка Kalman")
    print("═" * 75)

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
    z_emg = [c for c in Z_EMG if c in df.columns]
    base  = z_emg + HRV_FEATURES

    configs = {
        "v0008★ SampEn60+timing_cv+Huber": (
            base + SAMPEN_60 + TIMING_CV,
            lambda: HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500, fit_intercept=True),
            V0009_LT1_KALMAN,
        ),
    }

    rows = []
    for label, (feats, factory, v009_ref) in configs.items():
        avail = [f for f in feats if f in df.columns]
        print(f"\n[LT1] {label}  ({len(avail)} признаков)")
        r_raw = loso(df, avail, target, factory)
        print(f"  Raw:  MAE={r_raw['mae_min']:.3f} ± {r_raw['mae_std']:.3f}  R²={r_raw['r2']:.3f}  ρ={r_raw['rho']:.3f}")

        r_best, best_sp, best_so = kalman_grid_search(r_raw, label, v009_ref)
        rows.append({"task": "LT1", "config": label,
                     "raw_mae": r_raw["mae_min"], "kalman_mae": r_best["mae_min"],
                     "delta_raw": r_best["mae_min"] - r_raw["mae_min"],
                     "delta_v009": r_best["mae_min"] - v009_ref,
                     "sigma_p": best_sp, "sigma_obs": best_so})

        if not no_plots:
            plot_before_after(r_raw, r_best, df, f"LT1 v0010: {label}",
                              OUT_DIR / "lt1" / "scatter.png")
            plot_trajectories(r_raw, r_best, df, target,
                              OUT_DIR / "lt1" / "trajectories.png")
    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v0010: Kalman расширенная сетка.")
    p.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 75)
    print("v0010 — KALMAN РАСШИРЕННАЯ СЕТКА (σ_p ↓, σ_obs ↑)")
    print("=" * 75)
    df_full = pd.read_parquet(args.dataset)
    print(f"Датасет: {df_full.shape[0]} окон, {df_full['subject_id'].nunique()} участников")

    all_rows: list[dict] = []
    if args.target in ("lt2", "both"):
        all_rows.extend(run_lt2(df_full, no_plots=args.no_plots))
    if args.target in ("lt1", "both"):
        all_rows.extend(run_lt1(df_full, no_plots=args.no_plots))

    print("\n" + "═" * 75)
    print("ИТОГ v0010 vs v0009")
    print("═" * 75)
    print(f"  {'Задача':<6} {'Конфиг':<38} {'Raw':>6}  {'Kalman':>6}  "
          f"{'Δ_raw':>6}  {'Δ_v009':>7}  {'σ_p':>5}  {'σ_obs':>7}")
    print("  " + "─" * 83)
    for r in all_rows:
        verdict = "★ ЛУЧШЕ" if r["delta_v009"] < -1e-4 else ("≈ равно" if abs(r["delta_v009"]) < 1e-4 else "  хуже")
        print(f"  {r['task']:<6} {r['config']:<38} {r['raw_mae']:6.3f}  "
              f"{r['kalman_mae']:6.3f}  {r['delta_raw']:+6.3f}  "
              f"{r['delta_v009']:+7.3f}  {r['sigma_p']:5.0f}  {r['sigma_obs']:7.0f}  {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\n✅ Сводка: {OUT_DIR / 'summary.csv'}")
    print("\n✅ Готово.")


if __name__ == "__main__":
    main()
