#!/usr/bin/env python3
"""analyze_nn_results.py — Комплексный анализ нейросетевых моделей v0301–v0408.

Три измерения сравнения:
  1. with_abs vs noabs     — одинаковая архитектура, разный набор признаков.
  2. wavelet vs no-wavelet — внутри каждого abs-варианта (только v03xx LSTM).
  3. TCN vs LSTM           — обзор семейств архитектур.

Артефакты (→ results/analysis/):
  summary_all.csv              — таблица метрик всех версий
  abs_noabs_pairs.csv          — Δ MAE (with_abs − noabs) по каждой паре
  wavelet_pairs.csv            — Δ MAE (CWT − base) по каждой паре
  bar_mae_{target}.png         — все версии, лучший fset
  arch_overview_{target}.png   — обзор по семействам (LSTM/TCN) × variant
  abs_noabs_{target}.png       — heatmap Δ (with_abs − noabs) по архитектурам
  wavelet_pairs_{target}.png   — Δ CWT внутри noabs и with_abs
  scatter_{target}.png         — ypred vs ytrue топ-K
  per_subject_{target}.png     — heatmap субъект × версия
  wilcoxon_{target}.png        — Wilcoxon p-values
  trajectories_{raw,kalman}_{target}.png

Запуск:
  python scripts/analyze_nn_results.py
  python scripts/analyze_nn_results.py --target lt2
  python scripts/analyze_nn_results.py --versions v0301 v0302 --target lt1
  python scripts/analyze_nn_results.py --top-k 5
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ─── Пути ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR     = RESULTS_DIR / "analysis"

SIGMA_GRID = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 150.0]
FSETS      = ["EMG", "EMG+NIRS", "EMG+NIRS+HRV"]

V0011_REF  = {"lt1": {"fset": "EMG+NIRS+HRV", "mae": None},
              "lt2": {"fset": "EMG+NIRS+HRV", "mae": None}}


# ─── Метаданные версий ────────────────────────────────────────────────────────

# n = 1..18 (для v0301–v0318 и их with_abs-зеркал v0319–v0336)
_LSTM_N_META: dict[int, tuple[str, str]] = {
    1:  ("Stateless", "30с"), 2:  ("Stateless", "30с"),
    3:  ("Stateless", "15с"), 4:  ("Stateless", "15с"),
    5:  ("Stateless", "5с"),  6:  ("Stateless", "5с"),
    7:  ("Stateful",  "30с"), 8:  ("Stateful",  "30с"),
    9:  ("Stateful",  "15с"), 10: ("Stateful",  "15с"),
    11: ("Stateful",  "5с"),  12: ("Stateful",  "5с"),
    13: ("Attention", "30с"), 14: ("Attention", "30с"),
    15: ("Attention", "ctx600"), 16: ("Attention", "ctx600"),
    17: ("Attention", "15с"), 18: ("Attention", "15с"),
}

_TCN_N_META: dict[int, tuple[str, str]] = {
    1: ("Pure",    "seq30"), 2: ("Pure",    "seq30"),
    3: ("Medium",  "seq60"), 4: ("Medium",  "seq60"),
    5: ("DWT",     "seq30"), 6: ("DWT",     "seq30"),
    7: ("WaveNet", "seq30"), 8: ("WaveNet", "seq30"),
}


def get_version_meta(version: str) -> dict:
    """Возвращает метаданные версии по имени.

    Ключи: family, variant, has_cwt, arch_type, step_label, arch_label.
    """
    try:
        num = int(version[1:])   # "v0301" → 301
    except ValueError:
        return {"family": "other", "variant": "noabs", "has_cwt": False,
                "arch_type": "?", "step_label": "", "arch_label": version}

    if 301 <= num <= 318:                   # LSTM noabs
        n = num - 300
        has_cwt = (n % 2 == 0)
        arch_type, step = _LSTM_N_META.get(n, ("LSTM", f"n{n}"))
        label = f"{arch_type} {step}" + (" +CWT" if has_cwt else "")
        return {"family": "LSTM", "variant": "noabs", "has_cwt": has_cwt,
                "arch_type": arch_type, "step_label": step, "arch_label": label}

    elif 319 <= num <= 336:                 # LSTM with_abs (зеркало v0301–v0318)
        base_n = (num - 300) - 18          # 319→1, 336→18
        has_cwt = (base_n % 2 == 0)
        arch_type, step = _LSTM_N_META.get(base_n, ("LSTM", f"n{base_n}"))
        label = f"{arch_type} {step}" + (" +CWT" if has_cwt else "")
        return {"family": "LSTM", "variant": "with_abs", "has_cwt": has_cwt,
                "arch_type": arch_type, "step_label": step, "arch_label": label}

    elif 401 <= num <= 408:                 # TCN
        n = num - 400
        variant = "with_abs" if n % 2 == 1 else "noabs"
        arch_type, step = _TCN_N_META.get(n, ("TCN", "?"))
        label = f"TCN-{arch_type} {step}"
        return {"family": "TCN", "variant": variant, "has_cwt": False,
                "arch_type": arch_type, "step_label": step, "arch_label": label}

    return {"family": "other", "variant": "noabs", "has_cwt": False,
            "arch_type": "?", "step_label": "", "arch_label": version}


# ─── Пары with_abs / noabs ───────────────────────────────────────────────────
# (noabs_ver, with_abs_ver, family, arch_label, has_cwt)

ABS_NOABS_PAIRS: list[tuple[str, str, str, str, bool]] = [
    # LSTM (все 18 noabs × with_abs, смещение +18)
    ("v0301", "v0319", "LSTM", "Stateless 30с",       False),
    ("v0302", "v0320", "LSTM", "Stateless 30с +CWT",  True),
    ("v0303", "v0321", "LSTM", "Stateless 15с",       False),
    ("v0304", "v0322", "LSTM", "Stateless 15с +CWT",  True),
    ("v0305", "v0323", "LSTM", "Stateless 5с",        False),
    ("v0306", "v0324", "LSTM", "Stateless 5с +CWT",   True),
    ("v0307", "v0325", "LSTM", "Stateful 30с",        False),
    ("v0308", "v0326", "LSTM", "Stateful 30с +CWT",   True),
    ("v0309", "v0327", "LSTM", "Stateful 15с",        False),
    ("v0310", "v0328", "LSTM", "Stateful 15с +CWT",   True),
    ("v0311", "v0329", "LSTM", "Stateful 5с",         False),
    ("v0312", "v0330", "LSTM", "Stateful 5с +CWT",    True),
    ("v0313", "v0331", "LSTM", "Attn 30с",            False),
    ("v0314", "v0332", "LSTM", "Attn 30с +CWT",       True),
    ("v0315", "v0333", "LSTM", "Attn ctx600",         False),
    ("v0316", "v0334", "LSTM", "Attn ctx600 +CWT",    True),
    ("v0317", "v0335", "LSTM", "Attn 15с",            False),
    ("v0318", "v0336", "LSTM", "Attn 15с +CWT",       True),
    # TCN (noabs = чётные, with_abs = нечётные)
    ("v0402", "v0401", "TCN", "TCN-Pure seq30",    False),
    ("v0404", "v0403", "TCN", "TCN-Medium seq60",  False),
    ("v0406", "v0405", "TCN", "TCN-DWT seq30",     False),
    ("v0408", "v0407", "TCN", "TCN-WaveNet seq30", False),
]

# Пары wavelet: (base_nowave, base_wave, arch, step, variant)
WAVELET_PAIRS: list[tuple[str, str, str, str, str]] = [
    # noabs
    ("v0301", "v0302", "Stateless LSTM", "шаг 30с",   "noabs"),
    ("v0303", "v0304", "Stateless LSTM", "шаг 15с",   "noabs"),
    ("v0305", "v0306", "Stateless LSTM", "шаг 5с",    "noabs"),
    ("v0307", "v0308", "Stateful LSTM",  "шаг 30с",   "noabs"),
    ("v0309", "v0310", "Stateful LSTM",  "шаг 15с",   "noabs"),
    ("v0311", "v0312", "Stateful LSTM",  "шаг 5с",    "noabs"),
    ("v0313", "v0314", "Attention LSTM", "шаг 30с",   "noabs"),
    ("v0315", "v0316", "Attention LSTM", "ctx600",    "noabs"),
    ("v0317", "v0318", "Attention LSTM", "шаг 15с",   "noabs"),
    # with_abs
    ("v0319", "v0320", "Stateless LSTM", "шаг 30с",   "with_abs"),
    ("v0321", "v0322", "Stateless LSTM", "шаг 15с",   "with_abs"),
    ("v0323", "v0324", "Stateless LSTM", "шаг 5с",    "with_abs"),
    ("v0325", "v0326", "Stateful LSTM",  "шаг 30с",   "with_abs"),
    ("v0327", "v0328", "Stateful LSTM",  "шаг 15с",   "with_abs"),
    ("v0329", "v0330", "Stateful LSTM",  "шаг 5с",    "with_abs"),
    ("v0331", "v0332", "Attention LSTM", "шаг 30с",   "with_abs"),
    ("v0333", "v0334", "Attention LSTM", "ctx600",    "with_abs"),
    ("v0335", "v0336", "Attention LSTM", "шаг 15с",   "with_abs"),
]


# ─── Kalman ───────────────────────────────────────────────────────────────────

def kalman_smooth(y: np.ndarray, sigma_p: float = 5.0,
                  sigma_obs: float = 30.0) -> np.ndarray:
    """Одномерный фильтр Калмана."""
    n = len(y)
    x_est = np.zeros(n); p_est = np.zeros(n)
    x_est[0] = y[0]; p_est[0] = sigma_obs ** 2
    for i in range(1, n):
        x_pred = x_est[i - 1]
        p_pred = p_est[i - 1] + sigma_p ** 2
        k = p_pred / (p_pred + sigma_obs ** 2)
        x_est[i] = x_pred + k * (y[i] - x_pred)
        p_est[i] = (1 - k) * p_pred
    return x_est


# ─── Загрузка данных ──────────────────────────────────────────────────────────

def find_versions(results_dir: Path,
                  prefixes: tuple[str, ...] = ("v03", "v04")) -> list[str]:
    """Версии с NPY-файлами для указанных префиксов."""
    versions = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and any(d.name.startswith(p) for p in prefixes):
            if list(d.glob("ypred_*.npy")):
                versions.append(d.name)
    return versions


def load_npy(version_dir: Path, target: str,
             fset: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Загружает ypred / ytrue. None если файл отсутствует."""
    tag   = fset.replace("+", "_")
    yp    = version_dir / f"ypred_{target}_{tag}.npy"
    yt    = version_dir / f"ytrue_{target}_{tag}.npy"
    if not yp.exists() or not yt.exists():
        return None
    return np.load(yp), np.load(yt)


def load_per_subject(version_dir: Path) -> Optional[pd.DataFrame]:
    p = version_dir / "per_subject.csv"
    return pd.read_csv(p) if p.exists() else None


# ─── Метрики ──────────────────────────────────────────────────────────────────

def compute_metrics(ypred: np.ndarray, ytrue: np.ndarray,
                    n_subjects: Optional[int] = None) -> dict:
    """Subject-weighted MAE + Kalman grid, R², ρ, bias.

    Единицы определяются по ytrue: abs(mean) > 30 → данные в секундах → /60.
    """
    in_seconds = abs(float(np.mean(ytrue))) > 30
    yp = ypred / 60.0 if in_seconds else ypred.copy().astype(float)
    yt = ytrue / 60.0 if in_seconds else ytrue.copy().astype(float)

    r2   = r2_score(yt, yp)
    rho, _ = spearmanr(yt, yp)
    bias = float(np.mean(yp - yt))
    calib = float(np.std(yp) / (np.std(yt) + 1e-9))

    if n_subjects and n_subjects > 1 and len(yp) >= n_subjects:
        chunk = len(yp) // n_subjects
        boundaries = [(i * chunk,
                       (i + 1) * chunk if i < n_subjects - 1 else len(yp))
                      for i in range(n_subjects)]
    else:
        boundaries = [(0, len(yp))]

    def _subj_mae(yp_arr, yt_arr):
        maes = [mean_absolute_error(yt_arr[s:e], yp_arr[s:e])
                for s, e in boundaries if e > s]
        return float(np.mean(maes)) if maes else float("nan")

    mae_raw = _subj_mae(yp, yt)

    best_mae_k = float("inf"); best_sigma = SIGMA_GRID[0]; k_maes = {}
    for sigma in SIGMA_GRID:
        yp_k = np.empty_like(yp)
        for s, e in boundaries:
            yp_k[s:e] = kalman_smooth(yp[s:e], sigma_p=5.0, sigma_obs=sigma)
        mae_k = _subj_mae(yp_k, yt)
        k_maes[sigma] = round(mae_k, 4)
        if mae_k < best_mae_k:
            best_mae_k = mae_k; best_sigma = sigma

    return {
        "mae_raw": round(mae_raw, 4), "mae_kalman": round(best_mae_k, 4),
        "best_sigma": best_sigma,
        **{f"kalman_{int(s)}": k_maes[s] for s in SIGMA_GRID},
        "r2": round(r2, 3), "rho": round(rho, 3),
        "bias": round(bias, 4), "calib": round(calib, 3),
    }


def _get_per_subj_rows(per_subj: Optional[pd.DataFrame], fset: str,
                       target: str, variant: str) -> pd.DataFrame:
    """Фильтрует per_subject.csv по fset / target / variant."""
    if per_subj is None:
        return pd.DataFrame()
    mask = (per_subj.get("feature_set", pd.Series(dtype=str)) == fset) & \
           (per_subj.get("target",      pd.Series(dtype=str)) == target)
    if "variant" in per_subj.columns:
        mask &= (per_subj["variant"] == variant)
    return per_subj[mask]


def compute_per_subject_mae(per_subj: Optional[pd.DataFrame],
                             fset: str, target: str,
                             variant: str) -> tuple[float, float]:
    """Возвращает (mean, std) MAE по субъектам."""
    sub = _get_per_subj_rows(per_subj, fset, target, variant)
    if sub.empty:
        return float("nan"), float("nan")
    maes = sub["mae_min"].values
    return round(float(np.mean(maes)), 4), round(float(np.std(maes)), 4)


# ─── Построение сводной таблицы ───────────────────────────────────────────────

def build_summary(versions: list[str], targets: list[str],
                  results_dir: Path) -> pd.DataFrame:
    """Единая таблица метрик с метаданными версии."""
    records = []

    # ── v0011 noabs baseline ──────────────────────────────────────────────────
    v0011_dir        = results_dir / "v0011"
    v0011_noabs_dir  = v0011_dir / "noabs"
    v0011_per_subj   = load_per_subject(v0011_dir)
    noabs_summ_path  = v0011_noabs_dir / "summary.csv"
    noabs_summ       = (pd.read_csv(noabs_summ_path)
                        if noabs_summ_path.exists() else pd.DataFrame())

    v0011_noabs_ref: dict[tuple[str, str], float] = {}

    for target in targets:
        for fset in FSETS:
            best_mae: Optional[float] = None
            if not noabs_summ.empty:
                row = noabs_summ[(noabs_summ["feature_set"] == fset) &
                                 (noabs_summ["target"] == target)]
                if not row.empty:
                    best_mae = float(row["kalman_mae_min"].min())

            data = load_npy(v0011_noabs_dir, target, fset)
            if data is None:
                data = load_npy(v0011_dir, target, fset)

            n_s = 18
            if data is not None:
                ypred, ytrue = data
                m = compute_metrics(ypred, ytrue, n_subjects=n_s)
                if best_mae is not None:
                    m["mae_raw"] = best_mae; m["mae_kalman"] = best_mae
            else:
                if best_mae is None:
                    continue
                m = {"mae_raw": best_mae, "mae_kalman": best_mae,
                     "best_sigma": None,
                     **{f"kalman_{int(s)}": None for s in SIGMA_GRID},
                     "r2": None, "rho": None, "bias": None, "calib": None}

            if best_mae is None and data is not None:
                best_mae = m["mae_kalman"]
            if best_mae is not None:
                v0011_noabs_ref[(fset, target)] = best_mae

            ps_mean, ps_std = compute_per_subject_mae(
                v0011_per_subj, fset, target, "noabs")
            records.append({
                "version": "v0011", "family": "Linear", "variant": "noabs",
                "has_cwt": False, "arch_type": "SVR/GBM", "step_label": "",
                "arch_label": "v0011 noabs",
                "feature_set": fset, "target": target,
                **m, "ps_mae_mean": ps_mean, "ps_mae_std": ps_std,
            })

    # ── Нейросетевые версии ───────────────────────────────────────────────────
    for version in versions:
        ver_dir = results_dir / version
        per_subj = load_per_subject(ver_dir)
        meta = get_version_meta(version)
        variant = meta["variant"]

        for target in targets:
            for fset in FSETS:
                data = load_npy(ver_dir, target, fset)
                if data is None:
                    continue
                ypred, ytrue = data

                sub_rows = _get_per_subj_rows(per_subj, fset, target, variant)
                n_s = int(sub_rows["subject_id"].nunique()) if not sub_rows.empty else None

                m = compute_metrics(ypred, ytrue, n_subjects=n_s)
                ps_mean, ps_std = compute_per_subject_mae(
                    per_subj, fset, target, variant)

                records.append({
                    "version": version, **meta,
                    "feature_set": fset, "target": target,
                    **m, "ps_mae_mean": ps_mean, "ps_mae_std": ps_std,
                })

    df = pd.DataFrame(records)

    # Δ vs v0011 noabs — per modality
    df["delta_v0011"] = float("nan")
    for target in targets:
        for fset in FSETS:
            ref = v0011_noabs_ref.get((fset, target))
            if ref is None:
                ref_rows = df[(df["version"] == "v0011") & (df["target"] == target)]
                ref = float(ref_rows["mae_kalman"].min()) if not ref_rows.empty else None
            if ref is not None:
                mask = (df["target"] == target) & (df["feature_set"] == fset)
                df.loc[mask, "delta_v0011"] = df.loc[mask, "mae_kalman"] - ref
        best_ref = min(
            (v0011_noabs_ref.get((f, target), float("inf")) for f in FSETS),
            default=float("nan"))
        V0011_REF[target]["mae"] = best_ref

    # Δ vs лучшей нейросети
    nn_vers = [v for v in df["version"].unique() if v != "v0011"]
    for target in targets:
        nn_best = df[(df["version"].isin(nn_vers)) &
                     (df["target"] == target)]["mae_kalman"].min()
        df.loc[df["target"] == target, "best_nn_mae"]   = nn_best
        df.loc[df["target"] == target, "delta_best_nn"] = \
            df.loc[df["target"] == target, "mae_kalman"] - nn_best

    return df.round(4)


# ─── Wilcoxon ─────────────────────────────────────────────────────────────────

def compute_wilcoxon_matrix(versions: list[str], target: str,
                             fset: str, results_dir: Path) -> pd.DataFrame:
    subj_maes: dict[str, pd.Series] = {}
    for version in ["v0011"] + versions:
        ver_dir  = results_dir / version
        per_subj = load_per_subject(ver_dir)
        if per_subj is None:
            continue
        meta    = get_version_meta(version)
        variant = meta["variant"] if version != "v0011" else "noabs"
        sub = _get_per_subj_rows(per_subj, fset, target, variant)
        if not sub.empty:
            subj_maes[version] = sub.set_index("subject_id")["mae_min"]

    keys = list(subj_maes.keys()); n = len(keys)
    pmat = pd.DataFrame(np.ones((n, n)), index=keys, columns=keys)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = subj_maes[keys[i]], subj_maes[keys[j]]
            common = a.index.intersection(b.index)
            if len(common) < 5:
                continue
            try:
                _, p = wilcoxon(a[common].values, b[common].values)
                pmat.loc[keys[i], keys[j]] = p
                pmat.loc[keys[j], keys[i]] = p
            except Exception:
                pass
    return pmat


# ─── Графики: общие ───────────────────────────────────────────────────────────

def plot_bar_mae(df: pd.DataFrame, target: str, out_dir: Path):
    """Bar chart MAE по версиям (лучший feature set)."""
    sub  = df[df["target"] == target].copy()
    best = sub.loc[sub.groupby("version")["mae_kalman"].idxmin()].copy()
    best = best.sort_values("mae_kalman")

    family_colors = {"Linear": "#e74c3c", "LSTM": "#3498db",
                     "TCN": "#2ecc71", "other": "#95a5a6"}
    colors = [family_colors.get(r["family"], "#95a5a6") for _, r in best.iterrows()]

    fig, ax = plt.subplots(figsize=(max(8, len(best) * 0.55), 5))
    bars = ax.barh(best["version"], best["mae_kalman"],
                   color=colors, edgecolor="white")
    for bar, (_, row) in zip(bars, best.iterrows()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{row["mae_kalman"]:.3f} ({row["feature_set"]})',
                va="center", ha="left", fontsize=7.5)

    ref = V0011_REF[target]["mae"]
    if ref:
        ax.axvline(ref, color="#e74c3c", ls="--", alpha=0.5,
                   label=f"v0011 noabs: {ref:.3f}")
    legend_handles = [
        mpatches.Patch(color=c, label=k)
        for k, c in family_colors.items() if k != "other"
    ]
    ax.legend(handles=legend_handles + (ax.get_legend_handles_labels()[0][-1:]
                                        if ref else []), fontsize=8)
    ax.set_xlabel("MAE Kalman (мин)")
    ax.set_title(f"MAE по версиям — {target.upper()} (лучший fset)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_dir / f"bar_mae_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ bar_mae_{target}.png")


def plot_arch_overview(df: pd.DataFrame, target: str, out_dir: Path):
    """Grouped bar: LSTM vs TCN × noabs vs with_abs, лучший fset каждой версии."""
    sub  = df[(df["target"] == target) & (df["version"] != "v0011")].copy()
    if sub.empty:
        return
    best = sub.loc[sub.groupby("version")["mae_kalman"].idxmin()].copy()

    # Цвет по семейству, штриховка по варианту
    fam_colors  = {"LSTM": "#3498db", "TCN": "#e67e22", "other": "#95a5a6"}
    var_hatch   = {"noabs": "", "with_abs": "///"}
    var_alpha   = {"noabs": 0.9, "with_abs": 0.65}

    best["sort_key"] = best["family"].map({"LSTM": 0, "TCN": 1}).fillna(2)
    best = best.sort_values(["sort_key", "mae_kalman"])

    fig, ax = plt.subplots(figsize=(max(10, len(best) * 0.7), 5))
    for i, (_, row) in enumerate(best.iterrows()):
        color  = fam_colors.get(row["family"], "#95a5a6")
        hatch  = var_hatch.get(row["variant"], "")
        alpha  = var_alpha.get(row["variant"], 0.85)
        bar = ax.bar(i, row["mae_kalman"], color=color, hatch=hatch,
                     alpha=alpha, edgecolor="white", width=0.8)
        ax.text(i, row["mae_kalman"] + 0.03,
                f'{row["mae_kalman"]:.2f}', ha="center", fontsize=6, va="bottom")

    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(best["version"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE Kalman (мин)")
    ax.set_title(f"Обзор архитектур — {target.upper()}")

    ref = V0011_REF[target]["mae"]
    if ref:
        ax.axhline(ref, color="#e74c3c", ls="--", lw=1.2,
                   label=f"v0011 noabs: {ref:.3f}")

    legend_handles = [
        mpatches.Patch(color=fam_colors["LSTM"],  label="LSTM"),
        mpatches.Patch(color=fam_colors["TCN"],   label="TCN"),
        mpatches.Patch(color="grey", label="noabs", alpha=0.9),
        mpatches.Patch(color="grey", label="with_abs", hatch="///", alpha=0.65),
    ]
    ax.legend(handles=legend_handles + (ax.get_legend_handles_labels()[0][-1:]
                                        if ref else []), fontsize=8, ncol=2)

    # Разделитель LSTM / TCN
    lstm_count = int((best["family"] == "LSTM").sum())
    if 0 < lstm_count < len(best):
        ax.axvline(lstm_count - 0.5, color="black", lw=0.8, ls=":")

    plt.tight_layout()
    fig.savefig(out_dir / f"arch_overview_{target}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ arch_overview_{target}.png")


# ─── with_abs vs noabs ────────────────────────────────────────────────────────

def analyze_abs_vs_noabs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Δ MAE = with_abs − noabs для каждой пары × fset × target.

    Отрицательный Δ = with_abs лучше.
    """
    rows = []
    for noabs_v, abs_v, family, arch_label, has_cwt in ABS_NOABS_PAIRS:
        for target in ["lt1", "lt2"]:
            for fset in FSETS:
                b = df[(df["version"] == noabs_v) & (df["target"] == target) &
                       (df["feature_set"] == fset)]
                a = df[(df["version"] == abs_v)   & (df["target"] == target) &
                       (df["feature_set"] == fset)]
                if b.empty or a.empty:
                    continue
                mae_noabs = float(b.iloc[0]["mae_kalman"])
                mae_abs   = float(a.iloc[0]["mae_kalman"])
                delta     = mae_abs - mae_noabs   # < 0 → with_abs лучше
                rows.append({
                    "family": family, "arch_label": arch_label,
                    "has_cwt": has_cwt,
                    "noabs_ver": noabs_v, "abs_ver": abs_v,
                    "target": target, "feature_set": fset,
                    "mae_noabs": round(mae_noabs, 4),
                    "mae_abs":   round(mae_abs,   4),
                    "delta":     round(delta, 4),
                    "delta_pct": round(delta / mae_noabs * 100, 2),
                })

    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        return pairs_df

    pairs_df.to_csv(out_dir / "abs_noabs_pairs.csv", index=False)
    print("  ✅ abs_noabs_pairs.csv")

    # Сводная таблица в консоль
    print("\n  Δ MAE (with_abs − noabs), мин  [отрицательный = with_abs лучше]")
    for target in ["lt1", "lt2"]:
        print(f"\n  {target.upper()}:")
        sub = pairs_df[pairs_df["target"] == target]
        piv = sub.pivot_table(
            index=["family", "arch_label"],
            columns="feature_set", values="delta",
            aggfunc="first").round(4)
        print(piv.to_string())

    # График heatmap Δ по архитектурам × fset (отдельно на lt1 и lt2)
    for target in ["lt1", "lt2"]:
        for family in pairs_df["family"].unique():
            sub = pairs_df[(pairs_df["target"] == target) &
                           (pairs_df["family"] == family)]
            if sub.empty:
                continue
            piv = sub.pivot_table(
                index="arch_label", columns="feature_set",
                values="delta", aggfunc="first")
            if piv.empty:
                continue
            # Сортируем по медианному delta
            piv["_med"] = piv.median(axis=1)
            piv = piv.sort_values("_med").drop(columns="_med")

            fig, ax = plt.subplots(
                figsize=(max(5, piv.shape[1] * 1.5),
                         max(3, piv.shape[0] * 0.5)))
            vmax = piv.abs().max().max()
            sns.heatmap(piv, ax=ax, cmap="RdYlGn_r",
                        center=0, vmin=-vmax, vmax=vmax,
                        annot=True, fmt=".3f", annot_kws={"size": 8},
                        linewidths=0.4,
                        cbar_kws={"label": "Δ MAE (мин)"})
            ax.set_title(f"Δ MAE with_abs − noabs — {family} / {target.upper()}\n"
                         f"(зелёный = with_abs лучше, красный = хуже)")
            ax.set_xlabel("Feature set"); ax.set_ylabel("Архитектура")
            plt.tight_layout()
            fname = f"abs_noabs_{family.lower()}_{target}.png"
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ {fname}")

    return pairs_df


# ─── Wavelet pairs ────────────────────────────────────────────────────────────

def analyze_wavelet_pairs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Δ MAE = CWT − base для каждой пары × fset × target.

    Разбивает результаты на noabs / with_abs.
    """
    rows = []
    for base, cwt, arch, step, variant in WAVELET_PAIRS:
        for target in ["lt1", "lt2"]:
            for fset in FSETS:
                b = df[(df["version"] == base) & (df["target"] == target) &
                       (df["feature_set"] == fset)]
                c = df[(df["version"] == cwt)  & (df["target"] == target) &
                       (df["feature_set"] == fset)]
                if b.empty or c.empty:
                    continue
                mae_base = float(b.iloc[0]["mae_kalman"])
                mae_cwt  = float(c.iloc[0]["mae_kalman"])
                delta = mae_cwt - mae_base   # < 0 → CWT лучше
                rows.append({
                    "arch": arch, "step": step, "variant": variant,
                    "base": base, "cwt": cwt,
                    "target": target, "feature_set": fset,
                    "mae_base": round(mae_base, 4),
                    "mae_cwt":  round(mae_cwt,  4),
                    "delta":    round(delta, 4),
                    "delta_pct": round(delta / mae_base * 100, 2),
                })

    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        return pairs_df

    pairs_df.to_csv(out_dir / "wavelet_pairs.csv", index=False)
    print("  ✅ wavelet_pairs.csv")

    print("\n  Δ MAE (CWT − base), мин  [отрицательный = вейвлет помог]")
    for variant in ["noabs", "with_abs"]:
        for target in ["lt1", "lt2"]:
            sub = pairs_df[(pairs_df["variant"] == variant) &
                           (pairs_df["target"] == target)]
            if sub.empty:
                continue
            print(f"\n  {variant} / {target.upper()}:")
            piv = sub.pivot_table(
                index=["arch", "step", "base", "cwt"],
                columns="feature_set", values="delta",
                aggfunc="first").round(4)
            print(piv.to_string())

    # График: grouped bar по парам, Δ per fset, отдельно noabs/with_abs
    colors = {"EMG": "#3498db", "EMG+NIRS": "#2ecc71", "EMG+NIRS+HRV": "#e74c3c"}
    width = 0.25

    for variant in ["noabs", "with_abs"]:
        for target in ["lt1", "lt2"]:
            sub = pairs_df[(pairs_df["variant"] == variant) &
                           (pairs_df["target"] == target)].copy()
            if sub.empty:
                continue
            sub["pair_lbl"] = sub.apply(
                lambda r: f"{r['arch']}\n{r['step']}\n({r['base']}↔{r['cwt']})", axis=1)
            labels = sub["pair_lbl"].unique()
            x = np.arange(len(labels))

            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
            for i, fset in enumerate(FSETS):
                vals = []
                for lbl in labels:
                    row = sub[(sub["pair_lbl"] == lbl) & (sub["feature_set"] == fset)]
                    vals.append(float(row["delta"].iloc[0]) if not row.empty else 0.0)
                bars = ax.bar(x + (i - 1) * width, vals, width,
                              label=fset, color=colors[fset], alpha=0.85)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                v + (0.01 if v >= 0 else -0.03),
                                f"{v:+.2f}", ha="center",
                                va="bottom" if v >= 0 else "top", fontsize=6)

            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
            ax.set_ylabel("Δ MAE (мин)  [CWT − base]")
            ax.set_title(f"Влияние вейвлетов — {variant} / {target.upper()}\n"
                         f"ниже нуля = CWT помог")
            ax.legend(title="Feature set", fontsize=9)
            plt.tight_layout()
            fname = f"wavelet_pairs_{variant}_{target}.png"
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ {fname}")

    return pairs_df


# ─── Scatter / trajectories / per-subject / wilcoxon ─────────────────────────

def plot_scatter(df: pd.DataFrame, target: str, results_dir: Path,
                 out_dir: Path, top_k: int = 3):
    sub = df[df["target"] == target].nsmallest(top_k, "mae_kalman")
    fig, axes = plt.subplots(1, len(sub), figsize=(5 * len(sub), 5), squeeze=False)

    for ax, (_, row) in zip(axes[0], sub.iterrows()):
        ver_dir = results_dir / row["version"]
        data = load_npy(ver_dir / "noabs", target, row["feature_set"])
        if data is None:
            data = load_npy(ver_dir, target, row["feature_set"])
        if data is None:
            continue
        ypred, ytrue = data
        in_sec = abs(float(ytrue.mean())) > 30
        yp = ypred / 60.0 if in_sec else ypred.astype(float)
        yt = ytrue / 60.0 if in_sec else ytrue.astype(float)

        ax.scatter(yt, yp, alpha=0.3, s=10, color="#3498db")
        lim = [min(yt.min(), yp.min()) - 1, max(yt.max(), yp.max()) + 1]
        ax.plot(lim, lim, "r--", alpha=0.7, lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Истина (мин)"); ax.set_ylabel("Предсказание (мин)")
        ax.set_title(f"{row['version']} / {row['feature_set']}\n"
                     f"MAE={row['mae_kalman']:.3f}, R²={row['r2']:.3f}")
        ax.set_aspect("equal")

    fig.suptitle(f"Scatter ypred vs ytrue — {target.upper()}", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / f"scatter_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ scatter_{target}.png")


def plot_per_subject(df: pd.DataFrame, target: str, results_dir: Path,
                     out_dir: Path):
    sub  = df[df["target"] == target].copy()
    best = sub.loc[sub.groupby("version")["mae_kalman"].idxmin()]

    rows = []
    for _, row in best.iterrows():
        ver_dir  = results_dir / row["version"]
        per_subj = load_per_subject(ver_dir)
        if per_subj is None:
            continue
        variant  = row.get("variant", "noabs")
        s = _get_per_subj_rows(per_subj, row["feature_set"], target, variant)
        for subj, val in s.set_index("subject_id")["mae_min"].items():
            rows.append({"version": row["version"], "subject_id": subj, "mae_min": val})

    if not rows:
        return
    tmp = pd.DataFrame(rows)
    tmp = tmp.groupby(["version", "subject_id"], as_index=False)["mae_min"].mean()
    mat_df = tmp.pivot(index="version", columns="subject_id",
                       values="mae_min").sort_index()

    fig, ax = plt.subplots(figsize=(max(10, mat_df.shape[1] * 0.6),
                                    max(4, mat_df.shape[0] * 0.4)))
    sns.heatmap(mat_df, ax=ax, cmap="RdYlGn_r", fmt=".2f",
                annot=True, annot_kws={"size": 7}, linewidths=0.3,
                cbar_kws={"label": "MAE (мин)"})
    ax.set_title(f"Per-subject MAE (лучший fset) — {target.upper()}")
    plt.tight_layout()
    fig.savefig(out_dir / f"per_subject_{target}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ per_subject_{target}.png")


def plot_wilcoxon(versions: list[str], target: str,
                   results_dir: Path, out_dir: Path):
    fset = "EMG+NIRS+HRV"
    pmat = compute_wilcoxon_matrix(versions, target, fset, results_dir)
    if pmat.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pmat) * 0.5),
                                    max(5, len(pmat) * 0.45)))
    mask = np.eye(len(pmat), dtype=bool)
    sns.heatmap(pmat, ax=ax, cmap="RdYlGn", vmin=0, vmax=0.1,
                annot=True, fmt=".3f", annot_kws={"size": 7},
                linewidths=0.3, mask=mask,
                cbar_kws={"label": "p-value (Wilcoxon)"})
    ax.set_title(f"Wilcoxon signed-rank — {target.upper()} / {fset}\n"
                 f"(зелёный = значимо, красный = нет)")
    plt.tight_layout()
    fig.savefig(out_dir / f"wilcoxon_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ wilcoxon_{target}.png")


def plot_trajectories(df: pd.DataFrame, target: str, results_dir: Path,
                      out_dir: Path, top_k: int = 5):
    sub = df[df["target"] == target].nsmallest(top_k, "mae_kalman")

    for smooth in [False, True]:
        suffix = "kalman" if smooth else "raw"
        fig, axes = plt.subplots(len(sub), 1, figsize=(14, 4 * len(sub)),
                                 squeeze=False)
        for ax, (_, row) in zip(axes[:, 0], sub.iterrows()):
            ver_dir = results_dir / row["version"]
            data = load_npy(ver_dir / "noabs", target, row["feature_set"])
            if data is None:
                data = load_npy(ver_dir, target, row["feature_set"])
            if data is None:
                continue
            ypred, ytrue = data
            in_sec = abs(float(ytrue.mean())) > 30
            yp = ypred / 60.0 if in_sec else ypred.astype(float)
            yt = ytrue / 60.0 if in_sec else ytrue.astype(float)

            if smooth:
                yp = kalman_smooth(yp, sigma_p=5.0, sigma_obs=row["best_sigma"])

            ax.plot(yt, color="#2c3e50", alpha=0.6, lw=0.8, label="Истина")
            ax.plot(yp, color="#e74c3c", alpha=0.6, lw=0.8, label="Предсказание")
            ax.set_title(f"{row['version']} / {row['feature_set']} — "
                         f"MAE={row['mae_kalman']:.3f} мин, R²={row['r2']:.3f}")
            ax.set_ylabel("Время (мин)")
            ax.legend(fontsize=8, loc="upper right")

        axes[-1, 0].set_xlabel("Временной шаг (все субъекты конкатенированы)")
        fig.suptitle(f"Траектории — {target.upper()} "
                     f"({'Kalman' if smooth else 'Raw'})", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(out_dir / f"trajectories_{suffix}_{target}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ trajectories_{suffix}_{target}.png")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Анализ нейросетевых моделей v0301–v0408")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir",     default=str(OUT_DIR))
    parser.add_argument("--target", choices=["lt1", "lt2", "both"], default="both")
    parser.add_argument("--versions", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets  = ["lt1", "lt2"] if args.target == "both" else [args.target]
    versions = args.versions or find_versions(results_dir)

    print(f"\n{'='*60}")
    print(f"Версий найдено: {len(versions)}: {versions}")
    print(f"Таргеты: {targets}")
    print(f"Результаты → {out_dir}")
    print(f"{'='*60}\n")

    # ── 1. Сводная таблица ──
    print("📊 Вычисляю метрики...")
    df = build_summary(versions, targets, results_dir)
    df.to_csv(out_dir / "summary_all.csv", index=False)
    print(f"  ✅ summary_all.csv ({len(df)} строк)\n")

    for target in targets:
        print(f"  ТОП-5 ({target.upper()}):")
        top = df[df["target"] == target].nsmallest(5, "mae_kalman")[
            ["version", "family", "variant", "has_cwt", "feature_set",
             "mae_kalman", "r2", "rho", "delta_v0011"]]
        print(top.to_string(index=False))
        print()

    # ── 2. Основные графики ──
    print("🎨 Строю графики...")
    for target in targets:
        plot_bar_mae(df, target, out_dir)
        plot_arch_overview(df, target, out_dir)
        plot_scatter(df, target, results_dir, out_dir, top_k=args.top_k)
        plot_per_subject(df, target, results_dir, out_dir)
        plot_wilcoxon(versions, target, results_dir, out_dir)
        plot_trajectories(df, target, results_dir, out_dir, top_k=args.top_k)
        print()

    # ── 3. with_abs vs noabs ──
    print("🔍 Анализ with_abs vs noabs...")
    analyze_abs_vs_noabs(df, out_dir)

    # ── 4. Wavelet pairs (только LSTM, обе abs-вариации) ──
    print("\n⚡ Анализ пар с/без вейвлетов...")
    analyze_wavelet_pairs(df, out_dir)

    print(f"\n✅ Готово. Все файлы в {out_dir}")


if __name__ == "__main__":
    main()
