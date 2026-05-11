"""v0013 — Наивный временной baseline: протокол без физиологии

Версия:    v0013
Дата:      2026-05-11
Предыдущая версия: v0011_modality_ablation.py
Результаты: results/v0013/

Что делает:
  Проверяет гипотезу «протокольной памяти»: могут ли модели предсказывать
  LT1/LT2 только из позиции окна в тесте, без каких-либо физиологических сигналов.

  Два уровня наивности:
    1. MeanPredictor  — всегда предсказывает среднее LT по обучающим субъектам
    2. ProtocolRidge  — Ridge по трём протокольным признакам:
         elapsed_sec        (время от начала теста до центра окна)
         current_power_w    (мощность на ступени = косвенное время)
         stage_index        (номер ступени: 0, 1, 2, ...)

  Если ProtocolRidge даёт MAE, сопоставимый с лучшей физиологической моделью
  (v0011, MAE ≈ 2.1 мин), это означает, что модели используют протокольную
  временну́ю информацию, а не физиологический излом.

Ожидаемые результаты:
  MeanPredictor:  MAE ≈ 3.5–5.0 мин (не знает ничего о конкретном субъекте)
  ProtocolRidge:  MAE ≈ ? мин — ключевой вопрос эксперимента
  v0011 ref:      LT2 ≈ 2.125 мин, LT1 ≈ 2.043 мин

  Если ProtocolRidge ≥ 3.0 мин → физиологические признаки добавляют
  существенную ценность сверх протокольной позиции.

Воспроизведение:
  uv run python scripts/v0013_time_baseline.py
  uv run python scripts/v0013_time_baseline.py --no-plots
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "results" / "v0013"

# Протокольные признаки — не содержат физиологической информации
PROTOCOL_FEATURES = ["elapsed_sec", "current_power_w", "stage_index"]

# Kalman-параметры (те же что в v0009/v0011)
KALMAN_SIGMA_P   = 5.0
KALMAN_SIGMA_OBS = 150.0


# ─── Kalman (идентичен v0011) ─────────────────────────────────────────────────

def kalman_smooth(y_pred: np.ndarray,
                  sigma_p: float = KALMAN_SIGMA_P,
                  sigma_obs: float = KALMAN_SIGMA_OBS) -> np.ndarray:
    n = len(y_pred)
    x = y_pred[0]
    p = sigma_obs ** 2
    smoothed = np.empty(n)
    dt = 5.0  # шаг окна, с

    for i in range(n):
        x -= dt
        p += sigma_p ** 2
        k = p / (p + sigma_obs ** 2)
        x = x + k * (y_pred[i] - x)
        p = (1 - k) * p
        smoothed[i] = x

    return smoothed


# ─── Предикторы ───────────────────────────────────────────────────────────────

class MeanPredictor:
    """Предсказывает среднее LT по обучающим субъектам (нижняя граница)."""

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


# ─── LOSO ─────────────────────────────────────────────────────────────────────

def loso_predict(df: pd.DataFrame,
                 feat_cols: list[str],
                 target_col: str,
                 model_factory) -> dict:
    subjects = sorted(df["subject_id"].unique())
    preds, trues, subjs = [], [], []

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test  = df[df["subject_id"] == test_s].sort_values("elapsed_sec")

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        mdl = model_factory()

        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values.astype(float)))
        X_te = sc.transform(imp.transform(test[feat_cols].values.astype(float)))
        mdl.fit(X_tr, train[target_col].values)
        y_pred = mdl.predict(X_te)

        preds.append(y_pred)
        trues.append(test[target_col].values)
        subjs.append(np.full(len(y_pred), test_s))

    y_pred_all = np.concatenate(preds)
    y_true_all = np.concatenate(trues)
    subj_all   = np.concatenate(subjs)

    return {
        "y_pred": y_pred_all,
        "y_true": y_true_all,
        "subjects": subj_all,
        "raw_mae_min": mean_absolute_error(y_true_all, y_pred_all) / 60.0,
        "r2":  r2_score(y_true_all, y_pred_all),
        "rho": float(spearmanr(y_true_all, y_pred_all).statistic),
    }


def apply_kalman_loso(loso_result: dict) -> float:
    y_pred_all = loso_result["y_pred"]
    y_true_all = loso_result["y_true"]
    subj_all   = loso_result["subjects"]

    preds_k, trues_k = [], []
    for s in np.unique(subj_all):
        mask = subj_all == s
        if mask.sum() == 0:
            continue
        preds_k.append(kalman_smooth(y_pred_all[mask]))
        trues_k.append(y_true_all[mask])

    return mean_absolute_error(np.concatenate(trues_k),
                               np.concatenate(preds_k)) / 60.0


# ─── Вычисление per-subject MAE ──────────────────────────────────────────────

def per_subject_mae(loso_result: dict) -> pd.DataFrame:
    rows = []
    for s in np.unique(loso_result["subjects"]):
        mask = loso_result["subjects"] == s
        mae = mean_absolute_error(
            loso_result["y_true"][mask],
            loso_result["y_pred"][mask]
        ) / 60.0
        rows.append({"subject_id": s, "mae_min": round(mae, 3)})
    return pd.DataFrame(rows).sort_values("subject_id")


# ─── Подготовка данных ────────────────────────────────────────────────────────

def prepare(df_raw: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "lt2":
        df = df_raw[df_raw["window_valid_all_required"] == 1].copy()
        df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    else:
        df = df_raw[df_raw["target_time_to_lt1_usable"] == 1].copy()
        df = df.dropna(subset=["target_time_to_lt1_sec"])

    return df.sort_values(["subject_id", "elapsed_sec"]).reset_index(drop=True)


# ─── Отчёт ───────────────────────────────────────────────────────────────────

def _delta(val: float, ref: float) -> str:
    d = val - ref
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def build_report(results: dict, ref_lt2: float, ref_lt1: float) -> str:
    lines = [
        "# v0013 — Наивный временной baseline\n",
        "## Гипотеза",
        "Если модель (Ridge по протокольным признакам) даёт MAE, сопоставимый",
        "с лучшей физиологической моделью (v0011), это свидетельствует о том,",
        "что модели опираются на временну́ю позицию окна, а не на физиологию.\n",
        "## Протокольные признаки",
        "- `elapsed_sec` — время от начала теста до центра окна",
        "- `current_power_w` — мощность текущей ступени",
        "- `stage_index` — номер ступени (0, 1, 2, …)\n",
        "## Результаты\n",
        "| Модель | LT2 raw MAE | LT2 kalman MAE | LT1 raw MAE | LT1 kalman MAE |",
        "|--------|-------------|----------------|-------------|----------------|",
    ]

    for name, r in results.items():
        lt2 = r.get("lt2", {})
        lt1 = r.get("lt1", {})

        def fmt(d, key):
            v = d.get(key, np.nan)
            return f"{v:.3f}" if np.isfinite(v) else "—"

        lines.append(
            f"| {name} "
            f"| {fmt(lt2, 'raw_mae_min')} "
            f"| {fmt(lt2, 'kalman_mae')} "
            f"| {fmt(lt1, 'raw_mae_min')} "
            f"| {fmt(lt1, 'kalman_mae')} |"
        )

    lt2_prot = results.get("ProtocolRidge", {}).get("lt2", {}).get("kalman_mae", np.nan)
    lt1_prot = results.get("ProtocolRidge", {}).get("lt1", {}).get("kalman_mae", np.nan)

    lines += [
        "",
        "## Сравнение с v0011 (лучший физиологический baseline)\n",
        f"| Таргет | v0011 (EMG+NIRS+HRV) | ProtocolRidge | Δ |",
        f"|--------|----------------------|---------------|---|",
        f"| LT2    | {ref_lt2:.3f} мин | {lt2_prot:.3f} мин | {_delta(lt2_prot, ref_lt2)} мин |",
        f"| LT1    | {ref_lt1:.3f} мин | {lt1_prot:.3f} мин | {_delta(lt1_prot, ref_lt1)} мин |",
        "",
        "## Вывод",
    ]

    # Интерпретация
    threshold = 0.3  # если разрыв > 0.3 мин — физиология добавляет ценность
    lt2_gap = lt2_prot - ref_lt2 if np.isfinite(lt2_prot) else np.nan

    if np.isfinite(lt2_gap):
        if lt2_gap > threshold:
            lines.append(
                f"ProtocolRidge хуже v0011 на **{lt2_gap:.3f} мин** (LT2): "
                f"физиологические признаки добавляют реальную ценность сверх протокольной позиции."
            )
        elif lt2_gap > 0:
            lines.append(
                f"ProtocolRidge хуже v0011 всего на **{lt2_gap:.3f} мин** (LT2): "
                f"вклад физиологии минимален — риск протокольной памяти **высокий**."
            )
        else:
            lines.append(
                f"ProtocolRidge лучше или наравне с v0011 (Δ={lt2_gap:.3f} мин, LT2): "
                f"протокольная память **подтверждена** — физиология не помогает."
            )

    return "\n".join(lines) + "\n"


# ─── График ───────────────────────────────────────────────────────────────────

def plot_scatter(loso_result: dict, title: str, path: Path) -> None:
    y_true = loso_result["y_true"] / 60.0
    y_pred = loso_result["y_pred"] / 60.0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax.set_xlabel("Истинное время до порога, мин")
    ax.set_ylabel("Предсказанное время до порога, мин")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─── Точка входа ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  v0013 — Наивный временной baseline")
    print("=" * 60)

    # Загрузка данных
    df_raw = pd.read_parquet(_ROOT / "dataset" / "merged_features_ml.parquet")
    print(f"Датасет: {df_raw['subject_id'].nunique()} субъектов, {len(df_raw)} окон")

    # Эталон v0011 (если файл доступен)
    ref_lt2 = ref_lt1 = np.nan
    ref_path = _ROOT / "results" / "v0011" / "best_per_set.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        row_lt2 = ref[(ref["feature_set"] == "EMG+NIRS+HRV") & (ref["target"] == "lt2")]
        row_lt1 = ref[(ref["feature_set"] == "EMG+NIRS+HRV") & (ref["target"] == "lt1")]
        if not row_lt2.empty:
            ref_lt2 = float(row_lt2["kalman_mae_min"].iloc[0])
        if not row_lt1.empty:
            ref_lt1 = float(row_lt1["kalman_mae_min"].iloc[0])
        print(f"Эталон v0011: LT2={ref_lt2:.3f} мин, LT1={ref_lt1:.3f} мин")

    results: dict[str, dict] = {}
    rows = []

    for target in ["lt2", "lt1"]:
        target_col = ("target_time_to_lt2_center_sec" if target == "lt2"
                      else "target_time_to_lt1_sec")
        df = prepare(df_raw, target)
        n_subj = df["subject_id"].nunique()
        n_win  = len(df)

        print(f"\n── {target.upper()} | {n_subj} субъектов, {n_win} окон ──")

        # 1. MeanPredictor
        print("  MeanPredictor...")
        res_mean = loso_predict(df, PROTOCOL_FEATURES, target_col,
                                lambda: MeanPredictor())
        k_mean = apply_kalman_loso(res_mean)
        print(f"  → raw MAE={res_mean['raw_mae_min']:.3f} мин, "
              f"kalman MAE={k_mean:.3f} мин, "
              f"R²={res_mean['r2']:.3f}, ρ={res_mean['rho']:.3f}")

        if "MeanPredictor" not in results:
            results["MeanPredictor"] = {}
        results["MeanPredictor"][target] = {
            "raw_mae_min": res_mean["raw_mae_min"],
            "kalman_mae":  k_mean,
            "r2": res_mean["r2"],
            "rho": res_mean["rho"],
        }
        rows.append({
            "model": "MeanPredictor", "target": target,
            "n_subjects": n_subj, "features": "none",
            "raw_mae_min": round(res_mean["raw_mae_min"], 4),
            "kalman_mae_min": round(k_mean, 4),
            "r2": round(res_mean["r2"], 3),
            "rho": round(res_mean["rho"], 3),
        })

        # 2. ProtocolRidge (несколько alpha)
        best_alpha = None
        best_k_mae = np.inf
        best_res   = None

        for alpha in [1, 10, 100, 1000]:
            res = loso_predict(df, PROTOCOL_FEATURES, target_col,
                               lambda a=alpha: Ridge(alpha=a))
            k_mae = apply_kalman_loso(res)
            print(f"  ProtocolRidge(α={alpha}): raw={res['raw_mae_min']:.3f}, "
                  f"kalman={k_mae:.3f}, R²={res['r2']:.3f}, ρ={res['rho']:.3f}")
            rows.append({
                "model": f"ProtocolRidge(α={alpha})", "target": target,
                "n_subjects": n_subj, "features": ",".join(PROTOCOL_FEATURES),
                "raw_mae_min": round(res["raw_mae_min"], 4),
                "kalman_mae_min": round(k_mae, 4),
                "r2": round(res["r2"], 3),
                "rho": round(res["rho"], 3),
            })
            if k_mae < best_k_mae:
                best_k_mae = k_mae
                best_alpha = alpha
                best_res   = res

        print(f"  Лучший ProtocolRidge: α={best_alpha}, kalman MAE={best_k_mae:.3f} мин")
        if "ProtocolRidge" not in results:
            results["ProtocolRidge"] = {}
        results["ProtocolRidge"][target] = {
            "raw_mae_min": best_res["raw_mae_min"],
            "kalman_mae":  best_k_mae,
            "r2": best_res["r2"],
            "rho": best_res["rho"],
            "_loso": best_res,
            "best_alpha": best_alpha,
        }

        # Per-subject MAE для лучшего ProtocolRidge
        ps = per_subject_mae(best_res)
        ps["target"] = target
        ps.to_csv(OUT_DIR / f"per_subject_{target}.csv", index=False)
        print(f"  Per-subject сохранён: {OUT_DIR / f'per_subject_{target}.csv'}")

        # График scatter
        if not args.no_plots and best_res is not None:
            plot_scatter(
                best_res,
                f"ProtocolRidge {target.upper()} (α={best_alpha})",
                OUT_DIR / f"scatter_{target}.png",
            )

    # Сохраняем сводную таблицу
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # Отчёт
    report = build_report(results, ref_lt2, ref_lt1)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    # Финальная сводка в консоль
    print("\n" + "=" * 60)
    print("  ИТОГ v0013")
    print("=" * 60)

    prot_lt2 = results.get("ProtocolRidge", {}).get("lt2", {}).get("kalman_mae", np.nan)
    prot_lt1 = results.get("ProtocolRidge", {}).get("lt1", {}).get("kalman_mae", np.nan)
    mean_lt2 = results.get("MeanPredictor", {}).get("lt2", {}).get("kalman_mae", np.nan)
    mean_lt1 = results.get("MeanPredictor", {}).get("lt1", {}).get("kalman_mae", np.nan)

    print(f"  MeanPredictor:  LT2={mean_lt2:.3f} мин,  LT1={mean_lt1:.3f} мин")
    print(f"  ProtocolRidge:  LT2={prot_lt2:.3f} мин,  LT1={prot_lt1:.3f} мин")
    if np.isfinite(ref_lt2):
        print(f"  v0011 (ref):    LT2={ref_lt2:.3f} мин,  LT1={ref_lt1:.3f} мин")
        print(f"  Δ (Prot - v011): LT2={prot_lt2 - ref_lt2:+.3f} мин, "
              f"LT1={prot_lt1 - ref_lt1:+.3f} мин")
    print(f"\n  Результаты: {OUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
