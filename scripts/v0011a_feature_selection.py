"""v0011a — Ablation по модальностям с отбором признаков внутри LOSO.

Версия:    v0011a
Дата:      2026-05-08
Предыдущая версия: v0011_modality_ablation.py

Что делает:
  Для каждого (feature_set, target):
  1. EN-selector(α_sel, l1=0.9) обучается на train-фолде → отбирает фичи с |coef|>0
  2. Финальная модель обучается ТОЛЬКО на отобранных фичах (без утечки)
  3. Kalman-постобработка (те же σ_p=15, σ_obs=150)

  Перебирает несколько α для selector-а:
    α_sel ∈ [0.01, 0.05, 0.1, 0.3, 1.0]

  Финальный зоопарк (сокращённый, т.к. selection уже регуляризует):
    Ridge(α ∈ [1, 10, 100, 1000])
    EN(α ∈ [0.01, 0.1, 1.0], l1 ∈ [0.2, 0.5])   # без l1=0.9 — дублирует selector
    GBM(n=100, d=2)
    GBM(n=200, d=2)

  Результат сравнивается с v0011 baseline.

  Дополнительно выводит:
    - Среднее число отобранных фич на фолд
    - Stability score: % фолдов, в которых фича отобрана (≥50% → стабильная)

Наборы признаков (те же, что v0011):
  EMG, NIRS, EMG+NIRS, EMG+NIRS+HRV

Запуск:
  uv run python scripts/v0011a_feature_selection.py
  uv run python scripts/v0011a_feature_selection.py --target lt2
  uv run python scripts/v0011a_feature_selection.py --feature-set EMG+NIRS EMG+NIRS+HRV
  uv run python scripts/v0011a_feature_selection.py --alpha-sel 0.1 0.3
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from collections import defaultdict
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
sys.path.insert(0, str(_ROOT / "scripts"))
from v0011_modality_ablation import (
    prepare_data,
    get_feature_cols,
    kalman_smooth,
    NIRS_FEATURES,
    RUNNING_NIRS_FEATURES,
    HRV_FEATURES,
    INTERACTION_FEATURES,
)

OUT_DIR = _ROOT / "results" / "v0011a"

# ─── Baseline из v0011 (Kalman MAE, лучший per set/target) ───────────────────
V0011_BASELINE = {
    ("EMG",          "lt2"): 3.1979,
    ("EMG",          "lt1"): 2.8957,
    ("NIRS",         "lt2"): 4.2508,
    ("NIRS",         "lt1"): 4.0629,
    ("EMG+NIRS",     "lt2"): 3.1172,
    ("EMG+NIRS",     "lt1"): 2.7499,
    ("EMG+NIRS+HRV", "lt2"): 1.8586,
    ("EMG+NIRS+HRV", "lt1"): 2.2774,
}

TARGET_CFG = {
    "lt2": "target_time_to_lt2_center_sec",
    "lt1": "target_time_to_lt1_pchip_sec",
}


# ─── Финальный зоопарк (без дублирования selector'а) ─────────────────────────

def _build_zoo() -> list[dict]:
    """Сокращённый зоопарк финальных моделей."""
    configs = []

    for alpha in [1, 10, 100, 1000]:
        configs.append({
            "name": f"Ridge(α={alpha})",
            "factory": lambda a=alpha: Ridge(alpha=a),
        })

    for alpha, l1 in itertools.product([0.01, 0.1, 1.0], [0.2, 0.5]):
        configs.append({
            "name": f"EN(α={alpha},l1={l1})",
            "factory": lambda a=alpha, l=l1: ElasticNet(
                alpha=a, l1_ratio=l, max_iter=5000, random_state=42),
        })

    for n_est, depth in [(100, 2), (200, 2)]:
        configs.append({
            "name": f"GBM(n={n_est},d={depth})",
            "factory": lambda n=n_est, d=depth: GradientBoostingRegressor(
                n_estimators=n, max_depth=d, random_state=42),
        })

    return configs


# ─── LOSO с отбором внутри фолда ─────────────────────────────────────────────

def loso_with_selection(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    model_factory,
    alpha_sel: float,
    sigma_p: float = 15.0,
    sigma_obs: float = 150.0,
) -> dict:
    """LOSO: EN-selector на train → final model на отобранных фичах → Kalman.

    Возвращает:
      kalman_mae_min, raw_mae_min, r2, rho,
      selected_counts     — число отобранных фич на каждом фолде
      feature_selection   — dict[feat] → сколько фолдов отобрано
    """
    subjects = sorted(df["subject_id"].unique())
    selector_cfg = ElasticNet(
        alpha=alpha_sel, l1_ratio=0.9, max_iter=5000, random_state=42)

    preds_raw, trues_raw, subjs_raw = [], [], []
    preds_k,   trues_k              = [], []
    selected_counts: list[int] = []
    feat_selection_hits: dict[str, int] = defaultdict(int)

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        imp_sel = SimpleImputer(strategy="median")
        sc_sel  = StandardScaler()

        # ── Selector fit на train ─────────────────────────────────────────────
        X_tr_all = sc_sel.fit_transform(imp_sel.fit_transform(
            train[feat_cols].values))
        y_tr = train[target_col].values

        sel = ElasticNet(alpha=alpha_sel, l1_ratio=0.9, max_iter=5000,
                         random_state=42)
        sel.fit(X_tr_all, y_tr)

        # Индексы отобранных фич (|coef| > 0)
        sel_mask = np.abs(sel.coef_) > 1e-9
        sel_idx  = np.where(sel_mask)[0]

        # Fallback: если всё занулено — берём топ-10 по |coef|
        if sel_idx.size == 0:
            sel_idx = np.argsort(np.abs(sel.coef_))[-10:]

        selected_counts.append(len(sel_idx))
        for i in sel_idx:
            feat_selection_hits[feat_cols[i]] += 1

        # ── Final model на отобранных фичах ───────────────────────────────────
        imp_fin = SimpleImputer(strategy="median")
        sc_fin  = StandardScaler()

        X_tr_sel = sc_fin.fit_transform(imp_fin.fit_transform(
            train[feat_cols].iloc[:, sel_idx].values))
        X_te_sel = sc_fin.transform(imp_fin.transform(
            test[feat_cols].iloc[:, sel_idx].values))

        mdl = model_factory()
        mdl.fit(X_tr_sel, y_tr)
        y_pred = mdl.predict(X_te_sel)
        y_true = test[target_col].values

        preds_raw.append(y_pred)
        trues_raw.append(y_true)
        subjs_raw.append(np.full(len(y_pred), test_s))

        # Kalman per-subject
        preds_k.append(kalman_smooth(y_pred, sigma_p, sigma_obs))
        trues_k.append(y_true)

    y_pred_all = np.concatenate(preds_raw)
    y_true_all = np.concatenate(trues_raw)
    y_pred_k   = np.concatenate(preds_k)
    y_true_k   = np.concatenate(trues_k)

    return {
        "kalman_mae_min":  round(mean_absolute_error(y_true_k,   y_pred_k)   / 60.0, 4),
        "raw_mae_min":     round(mean_absolute_error(y_true_all, y_pred_all) / 60.0, 4),
        "r2":              round(r2_score(y_true_all, y_pred_all), 3),
        "rho":             round(float(spearmanr(y_true_all, y_pred_all).statistic), 3),
        "selected_counts": selected_counts,
        "feat_hits":       dict(feat_selection_hits),
        "n_folds":         len(subjects),
    }


# ─── Один конфиг: одна (alpha_sel × final_model) пара ────────────────────────

def _run_one(cfg: dict, alpha_sel: float,
             df: pd.DataFrame, feat_cols: list[str],
             target_col: str, feature_set: str, target_name: str,
             n_subj: int,
             sigma_p: float, sigma_obs: float) -> dict:
    """Запускается параллельно через joblib."""
    t0 = time.perf_counter()
    res = loso_with_selection(
        df, feat_cols, target_col, cfg["factory"],
        alpha_sel, sigma_p, sigma_obs)
    elapsed = time.perf_counter() - t0

    avg_sel = float(np.mean(res["selected_counts"]))
    return {
        "feature_set":     feature_set,
        "target":          target_name,
        "alpha_sel":       alpha_sel,
        "model":           cfg["name"],
        "n_subjects":      n_subj,
        "n_features_in":   len(feat_cols),
        "avg_selected":    round(avg_sel, 1),
        "raw_mae_min":     res["raw_mae_min"],
        "kalman_mae_min":  res["kalman_mae_min"],
        "r2":              res["r2"],
        "rho":             res["rho"],
        "sec":             round(elapsed, 1),
        "_feat_hits":      res["feat_hits"],
        "_n_folds":        res["n_folds"],
    }


# ─── Stability report ─────────────────────────────────────────────────────────

def stability_report(records: list[dict], feat_cols: list[str],
                     feature_set: str, target: str,
                     alpha_sel: float, n_folds: int,
                     out_dir: Path) -> None:
    """Сохраняет CSV с stability score для каждого признака."""
    hits: dict[str, int] = defaultdict(int)
    count = 0
    for r in records:
        if (r["feature_set"] == feature_set and r["target"] == target
                and r["alpha_sel"] == alpha_sel):
            for feat, h in r["_feat_hits"].items():
                hits[feat] = max(hits[feat], h)  # берём максимум по моделям
            count += 1

    if count == 0:
        return

    nirs_set = set(NIRS_FEATURES + RUNNING_NIRS_FEATURES)
    hrv_set  = set(HRV_FEATURES + INTERACTION_FEATURES)

    rows = []
    for feat in feat_cols:
        h = hits.get(feat, 0)
        group = ("NIRS" if feat in nirs_set
                 else "HRV" if feat in hrv_set
                 else "EMG/kin")
        rows.append({
            "feature": feat,
            "hits": h,
            "stability_pct": round(h / n_folds * 100, 1),
            "group": group,
        })

    stab_df = (pd.DataFrame(rows)
               .sort_values("hits", ascending=False)
               .reset_index(drop=True))

    fname = f"stability_{feature_set.replace('+', '_')}_{target}_asel{alpha_sel}.csv"
    stab_df.to_csv(out_dir / fname, index=False)

    # Вывод топ стабильных
    print(f"\n  Stability ({feature_set}/{target}/α_sel={alpha_sel}):"
          f"  топ-15 по % фолдов")
    print(f"  {'Признак':<43} {'Hits':>5}  {'%':>6}  Группа")
    print("  " + "-" * 60)
    for _, row in stab_df.head(15).iterrows():
        print(f"  {row['feature']:<43} {row['hits']:>5}  "
              f"{row['stability_pct']:>5.1f}%  {row['group']}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="v0011a — отбор признаков внутри LOSO")
    parser.add_argument("--target", choices=["lt1", "lt2", "both"],
                        default="both")
    parser.add_argument("--feature-set", nargs="+",
                        choices=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"],
                        default=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"])
    parser.add_argument("--alpha-sel", nargs="+", type=float,
                        default=[0.01, 0.05, 0.1, 0.3, 1.0],
                        help="α для EN-selector (l1=0.9 фиксировано)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--sigma-p",   type=float, default=15.0)
    parser.add_argument("--sigma-obs", type=float, default=150.0)
    parser.add_argument("--dataset",
                        default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Загружаю {args.dataset}")
    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    targets = ["lt2", "lt1"] if args.target == "both" else [args.target]
    zoo = _build_zoo()

    all_records: list[dict] = []

    for tgt_name in targets:
        target_col = TARGET_CFG[tgt_name]
        df_prep = prepare_data(df_raw, session_params, tgt_name)
        if target_col not in df_prep.columns:
            print(f"[skip] нет колонки {target_col}")
            continue
        df_prep = df_prep.dropna(subset=[target_col])
        n_subj = df_prep["subject_id"].nunique()

        print(f"\n{'═'*70}")
        print(f"ТАРГЕТ: {tgt_name.upper()}  ({n_subj} субъектов)")
        print(f"{'═'*70}")

        for fset in args.feature_set:
            feat_cols = get_feature_cols(df_prep, fset)
            if not feat_cols:
                print(f"  [{fset}] — нет признаков, пропуск")
                continue

            baseline = V0011_BASELINE.get((fset, tgt_name), None)
            print(f"\n  [{fset}]  {len(feat_cols)} признаков"
                  + (f"  | v0011 baseline: {baseline:.4f} мин" if baseline else ""))

            # Все (alpha_sel × model) пары
            tasks = [
                (cfg, alpha_sel)
                for alpha_sel in args.alpha_sel
                for cfg in zoo
            ]

            results: list[dict] = Parallel(
                n_jobs=args.n_jobs, backend="loky", verbose=0)(
                delayed(_run_one)(
                    cfg, alpha_sel,
                    df_prep, feat_cols, target_col,
                    fset, tgt_name, n_subj,
                    args.sigma_p, args.sigma_obs,
                )
                for cfg, alpha_sel in tasks
            )

            all_records.extend(results)

            # Лучший результат для этого fset/target
            best = min(results, key=lambda r: r["kalman_mae_min"])
            delta = (best["kalman_mae_min"] - baseline) if baseline else None
            delta_str = (f"  Δ vs v0011: {delta:+.4f} мин"
                         if delta is not None else "")
            print(f"  Лучший: α_sel={best['alpha_sel']}  {best['model']}"
                  f"  kalman={best['kalman_mae_min']:.4f} мин"
                  f"  avg_sel={best['avg_selected']:.1f}{delta_str}")

            # Stability для лучшего alpha_sel
            best_alpha = best["alpha_sel"]
            stability_report(results, feat_cols, fset, tgt_name,
                             best_alpha, n_subj, OUT_DIR)

    # ── Сохранение ────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in all_records
    ])
    summary_df = summary_df.sort_values(
        ["feature_set", "target", "alpha_sel", "kalman_mae_min"]
    ).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    # Лучший per (feature_set, target)
    best_per = summary_df.loc[
        summary_df.groupby(["feature_set", "target"])["kalman_mae_min"].idxmin()
    ].reset_index(drop=True)
    best_per.to_csv(OUT_DIR / "best_per_set.csv", index=False)

    # ── Итоговый отчёт ────────────────────────────────────────────────────────
    _write_report(best_per, OUT_DIR)

    print(f"\n✓ Результаты сохранены в {OUT_DIR}/")
    print(f"  summary.csv: {len(summary_df)} строк")
    print(f"  best_per_set.csv: {len(best_per)} строк")


def _write_report(best_per: pd.DataFrame, out_dir: Path) -> None:
    """Генерирует report.md с таблицей сравнения v0011 vs v0011a."""
    lines = [
        "# v0011a — Ablation с отбором признаков (EN-selector, l1=0.9)",
        f"Дата: {pd.Timestamp.now().date()}",
        "Метрика: MAE (мин) после фильтра Калмана, LOSO CV.",
        "Selector: EN(l1=0.9), α_sel перебирается. Final model: Ridge/EN/GBM.",
        "",
    ]

    for tgt in ["lt2", "lt1"]:
        tgt_rows = best_per[best_per["target"] == tgt]
        if tgt_rows.empty:
            continue
        lines.append(f"## {tgt.upper()}")
        lines.append("")
        lines.append("| Набор | α_sel | Модель | avg_sel | Kalman MAE | "
                     "v0011 baseline | Δ | R² | ρ |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, row in tgt_rows.sort_values("kalman_mae_min").iterrows():
            baseline = V0011_BASELINE.get((row["feature_set"], tgt))
            delta = (f"{row['kalman_mae_min'] - baseline:+.4f}"
                     if baseline else "—")
            baseline_str = f"{baseline:.4f}" if baseline else "—"
            lines.append(
                f"| **{row['feature_set']}** "
                f"| {row['alpha_sel']} "
                f"| {row['model']} "
                f"| {row['avg_selected']:.1f} "
                f"| **{row['kalman_mae_min']:.4f}** "
                f"| {baseline_str} "
                f"| {delta} "
                f"| {row['r2']:.3f} "
                f"| {row['rho']:.3f} |"
            )
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
