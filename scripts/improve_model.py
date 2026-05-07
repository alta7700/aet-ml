"""Три целевых улучшения поверх текущего лучшего результата.

1. Warm-up калибровка (3 мин): нормировка признаков по началу теста субъекта
2. Interaction features: SmO2×RR, SmO2×DFA, RR/W, SmO2/W
3. Huber regression: робастная к выбросу S003

Текущий baseline: ElasticNet(nirs_hrv) MAE=2.225 мин LOSO

Запуск:
  uv run python scripts/improve_model.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATASET_PATH = _ROOT / "dataset" / "merged_features_ml.parquet"
OUT_DIR = _ROOT / "results" / "improve_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_time_to_lt2_center_sec"
# Первые 3 мин протокола = 36 окон по 5 с
WARMUP_WINDOWS = 36


# ─────────────────────── Загрузка ───────────────────────

def load_base() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    df = df[df["window_valid_all_required"] == 1].copy()
    df = df.dropna(subset=[TARGET])
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    return df


# ─────────────────────── Фиче-инженерия ───────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Взаимодействия SmO2×RR, SmO2×DFA, power-normalized."""
    df = df.copy()
    pw = df["current_power_w"].clip(lower=20)

    # Взаимодействия (нормализованное произведение)
    df["feat_smo2_x_rr"]    = df["trainred_smo2_mean"] * df["hrv_mean_rr_ms"] / 1e4
    df["feat_smo2_x_dfa"]   = df["trainred_smo2_mean"] * df["hrv_dfa_alpha1"]
    df["feat_hhb_x_rr"]     = df["trainred_hhb_mean"]  * df["hrv_mean_rr_ms"] / 1e3

    # Power-normalized (физиологическая эффективность)
    df["feat_rr_per_watt"]   = df["hrv_mean_rr_ms"] / pw
    df["feat_smo2_per_watt"] = df["trainred_smo2_mean"] / pw

    return df


def add_warmup_features(df: pd.DataFrame, warmup_n: int = WARMUP_WINDOWS) -> pd.DataFrame:
    """Warm-up нормировка: первые warmup_n окон → калибровочные stats субъекта.

    Каузально: статистики вычисляются только из начала протокола (t=0..3 мин).
    В продакшне эти окна собираются ДО начала предсказания.
    """
    df = df.copy()

    warmup_feats = {
        "trainred_smo2_mean": "wu_smo2_init",
        "hrv_mean_rr_ms":     "wu_rr_init",
        "hrv_dfa_alpha1":     "wu_dfa_init",
        "trainred_hhb_mean":  "wu_hhb_init",
    }

    for subj, grp in df.groupby("subject_id"):
        idx = grp.index
        warmup_idx = idx[:warmup_n]
        for feat_src, feat_dst in warmup_feats.items():
            if feat_src not in df.columns:
                continue
            init_val = df.loc[warmup_idx, feat_src].mean()
            # Дроп = насколько текущее значение отличается от калибровки
            df.loc[idx, f"wudrop_{feat_src}"]   = init_val - df.loc[idx, feat_src]
            # Нормировка: (current - init) / init  (% изменения)
            df.loc[idx, f"wupct_{feat_src}"]    = (df.loc[idx, feat_src] - init_val) / (abs(init_val) + 1e-6)

    return df


# ─────────────────────── Наборы признаков ───────────────────────

def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    nirs  = [c for c in df.columns if c.startswith("trainred_")]
    hrv   = [c for c in df.columns if c.startswith("hrv_")
              and not c.endswith(("_valid", "_fraction", "_count"))]
    inter = [c for c in df.columns if c.startswith("feat_")]
    wu    = [c for c in df.columns if c.startswith("wudrop_") or c.startswith("wupct_")]

    return {
        "baseline":          nirs + hrv,
        "+interactions":     nirs + hrv + inter,
        "+warmup":           nirs + hrv + wu,
        "+warmup+inter":     nirs + hrv + inter + wu,
        "warmup_only":       wu,
        "warmup+inter_only": wu + inter,
    }


# ─────────────────────── Модели ───────────────────────

def make_pipelines() -> dict[str, Pipeline]:
    """Возвращает словарь {имя → Pipeline}."""
    def std(est):
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", est),
        ])

    return {
        "ElasticNet(0.5,0.9)":  std(ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=3000)),
        "ElasticNet(1.0,0.9)":  std(ElasticNet(alpha=1.0, l1_ratio=0.9, max_iter=3000)),
        "Ridge(100)":           std(Ridge(alpha=100.0)),
        "Huber(eps=1.35)":      std(HuberRegressor(epsilon=1.35, max_iter=500, alpha=0.01)),
        "Huber(eps=1.75)":      std(HuberRegressor(epsilon=1.75, max_iter=500, alpha=0.01)),
        "Huber(eps=2.5)":       std(HuberRegressor(epsilon=2.5,  max_iter=500, alpha=0.01)),
    }


# ─────────────────────── LOSO с warm-up ───────────────────────

def loso_with_warmup(df: pd.DataFrame, feat_cols: list[str],
                     pipe: Pipeline, warmup_n: int = WARMUP_WINDOWS,
                     skip_warmup_in_eval: bool = True
                     ) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """LOSO CV с warm-up калибровкой.

    Для тестового субъекта первые warmup_n окон используются только для
    вычисления warm-up статистик и исключаются из оценки.
    """
    X = df[feat_cols].values
    y = df[TARGET].values
    groups = df["subject_id"].values
    logo = LeaveOneGroupOut()

    fold_rows = []
    preds_all = np.full(len(df), np.nan)

    for tr, te in logo.split(X, y, groups=groups):
        subj = groups[te][0]

        # warm-up окна тестового субъекта (для warm-up уже вшито в фичи)
        # просто исключаем первые warmup_n из оценки (если флаг)
        if skip_warmup_in_eval:
            te_eval = te[warmup_n:]  # оцениваем только после warm-up
        else:
            te_eval = te

        if len(te_eval) == 0:
            continue

        p = clone(pipe)
        p.fit(X[tr], y[tr])
        y_pred_all_te = p.predict(X[te])
        preds_all[te] = y_pred_all_te

        y_pred_eval = y_pred_all_te[warmup_n:] if skip_warmup_in_eval else y_pred_all_te
        y_true_eval = y[te_eval]

        mae  = mean_absolute_error(y_true_eval, y_pred_eval) / 60
        rmse = np.sqrt(np.mean((y_true_eval - y_pred_eval) ** 2)) / 60
        r2   = r2_score(y_true_eval, y_pred_eval)
        rho, _ = spearmanr(y_true_eval, y_pred_eval)
        train_mae = mean_absolute_error(y[tr], p.predict(X[tr])) / 60

        fold_rows.append({
            "subject_id": subj,
            "n_test": len(te_eval),
            "mae_min": mae,
            "rmse_min": rmse,
            "r2": r2,
            "spearman": rho,
            "train_mae_min": train_mae,
        })

    fold_df = pd.DataFrame(fold_rows)
    agg = {
        "mae_min":  fold_df["mae_min"].mean(),
        "mae_std":  fold_df["mae_min"].std(),
        "r2":       fold_df["r2"].mean(),
        "spearman": fold_df["spearman"].mean(),
    }
    return agg, fold_df, preds_all


def loso_standard(df: pd.DataFrame, feat_cols: list[str],
                   pipe: Pipeline) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Стандартный LOSO без warm-up исключения."""
    return loso_with_warmup(df, feat_cols, pipe, skip_warmup_in_eval=False)


# ─────────────────────── Визуализация ───────────────────────

def plot_improvement_chart(results: list[dict]) -> None:
    """Горизонтальный bar chart всех конфигураций."""
    res_df = pd.DataFrame(results).sort_values("mae_min", ascending=False)

    def bar_color(name: str) -> str:
        if "baseline" in name:     return "#1f77b4"
        if "Huber" in name:        return "#d62728"
        if "warmup" in name and "inter" in name: return "#9467bd"
        if "warmup" in name:       return "#ff7f0e"
        if "inter" in name:        return "#2ca02c"
        return "#7f7f7f"

    colors = [bar_color(r["name"]) for _, r in res_df.iterrows()]
    baseline_mae = 2.225

    fig, ax = plt.subplots(figsize=(11, max(5, len(res_df) * 0.42)))
    bars = ax.barh(res_df["name"], res_df["mae_min"],
                   xerr=res_df["mae_std"], color=colors, alpha=0.85,
                   capsize=4, error_kw={"elinewidth": 1.5})

    for bar, (_, row) in zip(bars, res_df.iterrows()):
        delta = row["mae_min"] - baseline_mae
        sign  = "+" if delta > 0 else ""
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{row['mae_min']:.3f}  ({sign}{delta:.3f})",
                va="center", ha="left", fontsize=8.5)

    ax.axvline(baseline_mae, color="black", lw=2, ls="--", label=f"Baseline: {baseline_mae:.3f}")
    ax.set_xlabel("MAE (мин), LOSO CV — меньше лучше")
    ax.set_title("Эффект улучшений поверх baseline ElasticNet(nirs_hrv)", fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#1f77b4", label="Baseline"),
        mpatches.Patch(color="#2ca02c", label="+ Interactions"),
        mpatches.Patch(color="#ff7f0e", label="+ Warm-up"),
        mpatches.Patch(color="#9467bd", label="+ Warm-up + Inter"),
        mpatches.Patch(color="#d62728", label="Huber regression"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "improvement_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → improvement_chart.png")


def plot_per_subject_improvement(all_folds: dict[str, pd.DataFrame]) -> None:
    """Сравнение MAE по субъектам для лучших конфигураций."""
    keys = ["baseline / ElasticNet(0.5,0.9)",
            "+warmup+inter / Huber(eps=1.75)",
            "+warmup+inter / ElasticNet(0.5,0.9)"]
    available = [k for k in keys if k in all_folds][:3]
    if not available:
        available = list(all_folds.keys())[:3]

    subjects = sorted(all_folds[available[0]]["subject_id"].unique())
    x = np.arange(len(subjects))
    width = 0.25
    colors_map = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, key in enumerate(available):
        fd = all_folds[key].set_index("subject_id")
        maes = [fd.loc[s, "mae_min"] if s in fd.index else np.nan for s in subjects]
        ax.bar(x + i * width, maes, width, label=key[:40],
               color=colors_map[i], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("MAE (мин)")
    ax.set_title("MAE по субъектам: сравнение конфигураций", fontweight="bold")
    ax.axhline(2.225, color="black", ls="--", lw=1.2, alpha=0.5, label="Baseline overall")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_subject_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → per_subject_comparison.png")


def plot_warmup_effect(df: pd.DataFrame, preds_before: np.ndarray,
                       preds_after: np.ndarray) -> None:
    """Scatter: как изменились предсказания после warm-up для S003."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Эффект warm-up калибровки", fontweight="bold")

    y_true = df[TARGET].values / 60

    for ax, preds, title in zip(axes,
                                 [preds_before, preds_after],
                                 ["Без warm-up (baseline)", "С warm-up калибровкой"]):
        y_pred = preds / 60
        for i, (subj, grp) in enumerate(df.groupby("subject_id")):
            mask = df["subject_id"].values == subj
            color = "#d62728" if subj == "S003" else f"C{i}"
            alpha = 0.7 if subj == "S003" else 0.3
            s = 25 if subj == "S003" else 7
            ax.scatter(y_true[mask], y_pred[mask],
                       color=color, alpha=alpha, s=s,
                       label=subj if subj == "S003" else None, zorder=3 if subj == "S003" else 1)

        lim_min = min(y_true.min(), np.nanmin(y_pred)) - 1
        lim_max = max(y_true.max(), np.nanmax(y_pred)) + 1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.5)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("Истинное (мин)"); ax.set_ylabel("Предсказанное (мин)")

        valid = ~np.isnan(y_pred)
        mae = mean_absolute_error(y_true[valid], y_pred[valid]) / 1
        r2  = r2_score(y_true[valid], y_pred[valid])
        ax.set_title(f"{title}\nMAE={mae:.2f} мин | R²={r2:.3f}")
        if subj == "S003":
            ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "warmup_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → warmup_effect.png")


# ─────────────────────── main ───────────────────────

def main() -> None:
    print("=" * 68)
    print("УЛУЧШЕНИЯ МОДЕЛИ: warm-up + interactions + Huber")
    print("=" * 68)

    # Данные
    df = load_base()
    df = add_interaction_features(df)
    df = add_warmup_features(df, warmup_n=WARMUP_WINDOWS)

    feat_sets  = get_feature_sets(df)
    pipelines  = make_pipelines()

    print(f"\nДанных: {len(df)} окон | Субъектов: {df['subject_id'].nunique()}")
    print(f"Warm-up окон (исключ. из оценки): {WARMUP_WINDOWS} ({WARMUP_WINDOWS*5/60:.0f} мин)")
    for k, v in feat_sets.items():
        print(f"  Набор '{k}': {len(v)} признаков")

    results: list[dict] = []
    all_folds: dict[str, pd.DataFrame] = {}

    BASELINE_MAE = 2.225

    # ── Прогон всех комбинаций ──
    total = len(feat_sets) * len(pipelines)
    done  = 0

    # Сохраняем предсказания для baseline и лучшего warmup
    preds_baseline  = None
    preds_best_wu   = None

    for fs_name, feat_cols in feat_sets.items():
        # Отфильтровываем колонки которые реально есть
        feat_cols = [c for c in feat_cols if c in df.columns]
        if not feat_cols:
            continue

        for pipe_name, pipe in pipelines.items():
            done += 1
            key  = f"{fs_name} / {pipe_name}"
            use_wu = "warmup" in fs_name

            # warm-up конфиги оцениваем без первых WARMUP_WINDOWS окон в тесте
            agg, fold_df, preds = loso_with_warmup(
                df, feat_cols, pipe,
                warmup_n=WARMUP_WINDOWS,
                skip_warmup_in_eval=use_wu,
            )

            delta = agg["mae_min"] - BASELINE_MAE
            sign  = "+" if delta > 0 else ""
            print(f"  [{done:2d}/{total}] {key:<50s}  "
                  f"MAE={agg['mae_min']:.3f}±{agg['mae_std']:.3f}  "
                  f"({sign}{delta:.3f})")

            results.append({
                "name":    key,
                "fs":      fs_name,
                "model":   pipe_name,
                "mae_min": agg["mae_min"],
                "mae_std": agg["mae_std"],
                "r2":      agg["r2"],
                "spearman":agg["spearman"],
            })
            all_folds[key] = fold_df

            if fs_name == "baseline" and pipe_name == "ElasticNet(0.5,0.9)":
                preds_baseline = preds
            if fs_name == "+warmup+inter" and pipe_name == "Huber(eps=1.75)":
                preds_best_wu = preds

    # ── Сводная таблица ──
    res_df = pd.DataFrame(results).sort_values("mae_min")
    res_df.to_csv(OUT_DIR / "results.csv", index=False)

    print("\n" + "═" * 68)
    print("ИТОГ (сортировка по MAE)")
    print("═" * 68)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(res_df[["name", "mae_min", "mae_std", "r2", "spearman"]].to_string(index=False))

    best = res_df.iloc[0]
    print(f"\n🏆  Лучшая конфигурация: {best['name']}")
    print(f"    MAE = {best['mae_min']:.3f} ± {best['mae_std']:.3f} мин")
    print(f"    Δ vs baseline: {best['mae_min'] - BASELINE_MAE:+.3f} мин")
    print(f"    R² = {best['r2']:.3f} | Spearman ρ = {best['spearman']:.3f}")

    # ── Визуализации ──
    print("\nСтроим графики...")
    plot_improvement_chart(results)
    plot_per_subject_improvement(all_folds)
    if preds_baseline is not None and preds_best_wu is not None:
        plot_warmup_effect(df, preds_baseline, preds_best_wu)

    print(f"\n✅ Готово. Результаты: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
