"""Шаг 06: SHAP-интерпретация и conformal-интервалы для моделей-финалистов.

Финалисты (все — линейные v0011, Ridge α=1000):
- lt2 / HRV               — победитель lt2
- lt2 / EMG+NIRS+HRV      — мультимодальный (для сравнения)
- lt1 / EMG+NIRS          — победитель lt1
- lt1 / EMG+NIRS+HRV      — мультимодальный

SHAP: shap.LinearExplainer на модели, обученной на всей выборке. beeswarm,
bar (mean abs), агрегирование по группам признаков.

Conformal: LOSO split-conformal. Для каждого отложенного субъекта S калибровка
по остаткам LOSO-предсказаний остальных субъектов; интервал y_pred ± q(1−α).
Метрики — эмпирическое покрытие и ширина интервала, в т.ч. по группам
trained/untrained.

Запуск:
    PYTHONPATH=. uv run python scripts/final_shap_conformal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v0011_modality_ablation import prepare_data, get_feature_cols  # noqa: E402

DATASET_DIR = ROOT / "dataset"
OUT = ROOT / "results" / "final"
SHAP_DIR = OUT / "shap"
CONF_DIR = OUT / "conformal"

ALPHA_RIDGE = 1000.0
TRAINED_THRESHOLD_MIN = 13.0

# Финалисты: (target, feature_set, tag)
FINALISTS = [
    ("lt2", "HRV",          "lt2_HRV"),
    ("lt2", "EMG+NIRS+HRV", "lt2_ENH"),
    ("lt1", "EMG+NIRS",     "lt1_EN"),
    ("lt1", "EMG+NIRS+HRV", "lt1_ENH"),
]

TARGET_COL = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}

CONFORMAL_ALPHAS = [0.1, 0.2]  # 90% и 80% покрытие


def feature_group(name: str) -> str:
    """Грубая классификация признака по модальности."""
    if name.startswith("z_vl_") or name.startswith("vl_"):
        return "EMG"
    if name.startswith("z_") and not name.startswith("z_vl_"):
        return "kinematics"
    if name.startswith("hrv_"):
        return "HRV"
    if name.startswith("trainred_") or name.startswith("smo2_") or name.startswith("hhb_"):
        return "NIRS"
    if name.startswith("feat_"):
        return "interaction"
    return "other"


def load_subject_meta() -> pd.DataFrame:
    wn = pd.read_parquet(DATASET_DIR / "windows.parquet")
    tg = pd.read_parquet(DATASET_DIR / "targets.parquet")
    m = wn[["window_id", "subject_id", "window_end_sec", "elapsed_sec"]].merge(
        tg[["window_id", "target_time_to_lt2_center_sec"]], on="window_id")
    first = m.sort_values(["subject_id", "window_end_sec"]).groupby("subject_id").first().reset_index()
    first["time_to_lt2_min"] = (first["target_time_to_lt2_center_sec"]
                                + first["elapsed_sec"]) / 60.0
    first["trained"] = (first["time_to_lt2_min"] > TRAINED_THRESHOLD_MIN).astype(int)
    return first[["subject_id", "time_to_lt2_min", "trained"]]


# ─────────────────────── SHAP ───────────────────────

def run_shap(df: pd.DataFrame, feat_cols: list[str], target_col: str,
             tag: str) -> pd.DataFrame:
    """Обучает Ridge на всей выборке, считает SHAP (LinearExplainer)."""
    out_dir = SHAP_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X = sc.fit_transform(imp.fit_transform(df[feat_cols].values))
    y = df[target_col].values

    model = Ridge(alpha=ALPHA_RIDGE)
    model.fit(X, y)

    explainer = shap.LinearExplainer(model, X, feature_names=feat_cols)
    shap_values = explainer(X)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": mean_abs.round(4),
        "group": [feature_group(c) for c in feat_cols],
    }).sort_values("mean_abs_shap", ascending=False)
    imp_df["share_pct"] = (imp_df["mean_abs_shap"]
                           / imp_df["mean_abs_shap"].sum() * 100).round(1)
    imp_df.to_csv(out_dir / "shap_importance.csv", index=False)

    # Агрегирование по группам
    group_imp = (imp_df.groupby("group")["mean_abs_shap"].sum()
                 .sort_values(ascending=False))
    group_imp_pct = (group_imp / group_imp.sum() * 100).round(1)
    group_df = pd.DataFrame({"group": group_imp.index,
                             "mean_abs_shap": group_imp.values.round(4),
                             "share_pct": group_imp_pct.values})
    group_df.to_csv(out_dir / "shap_by_group.csv", index=False)

    # beeswarm
    plt.figure()
    shap.summary_plot(shap_values, features=X, feature_names=feat_cols,
                      show=False, max_display=20)
    plt.title(f"SHAP beeswarm — {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / "beeswarm.png", dpi=120)
    plt.close()

    # bar
    plt.figure()
    shap.summary_plot(shap_values, features=X, feature_names=feat_cols,
                      plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP mean|value| — {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / "bar.png", dpi=120)
    plt.close()

    print(f"  [{tag}] SHAP top-5: "
          + ", ".join(f"{r.feature}({r.share_pct}%)"
                      for r in imp_df.head(5).itertuples()))
    print(f"  [{tag}] по группам: "
          + ", ".join(f"{g}:{p}%" for g, p in zip(group_df["group"], group_df["share_pct"])))
    return imp_df


# ─────────────────────── Conformal ───────────────────────

def loso_predictions(df: pd.DataFrame, feat_cols: list[str],
                     target_col: str) -> pd.DataFrame:
    """LOSO Ridge — per-window предсказания. Возврат: subject_id, y_pred, y_true."""
    subjects = sorted(df["subject_id"].unique())
    rows = []
    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test = df[df["subject_id"] == test_s].sort_values("window_start_sec")
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        mdl = Ridge(alpha=ALPHA_RIDGE)
        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        mdl.fit(X_tr, train[target_col].values)
        y_pred = mdl.predict(X_te)
        y_true = test[target_col].values
        for yp, yt in zip(y_pred, y_true):
            rows.append({"subject_id": test_s, "y_pred": yp, "y_true": yt})
    return pd.DataFrame(rows)


def run_conformal(loso: pd.DataFrame, meta: pd.DataFrame, tag: str) -> pd.DataFrame:
    """LOSO split-conformal: калибровка по остальным субъектам."""
    out_dir = CONF_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    loso = loso.merge(meta[["subject_id", "trained"]], on="subject_id")
    loso["abs_err"] = np.abs(loso["y_pred"] - loso["y_true"])
    subjects = sorted(loso["subject_id"].unique())

    rows = []
    for alpha in CONFORMAL_ALPHAS:
        for test_s in subjects:
            calib = loso[loso["subject_id"] != test_s]["abs_err"].values
            q = float(np.quantile(calib, 1 - alpha))
            test = loso[loso["subject_id"] == test_s]
            covered = float((test["abs_err"] <= q).mean())
            rows.append({
                "tag": tag, "alpha": alpha, "target_coverage": 1 - alpha,
                "subject_id": test_s, "trained": int(test["trained"].iloc[0]),
                "q_halfwidth_sec": round(q, 2),
                "q_halfwidth_min": round(q / 60.0, 3),
                "empirical_coverage": round(covered, 4),
                "n_windows": len(test),
            })
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "intervals.csv", index=False)
    return res


def plot_conformal(all_conf: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # Покрытие по моделям/alpha
    ax = axes[0]
    summary = all_conf.groupby(["tag", "alpha"]).agg(
        cov=("empirical_coverage", "mean")).reset_index()
    tags = summary["tag"].unique()
    width = 0.35
    for i, a in enumerate(CONFORMAL_ALPHAS):
        sub = summary[summary["alpha"] == a]
        x = np.arange(len(tags)) + i * width
        ax.bar(x, [sub[sub["tag"] == t]["cov"].iloc[0] for t in tags],
               width, label=f"α={a} (target {1-a:.0%})")
        ax.axhline(1 - a, color=f"C{i}", ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(tags)) + width / 2)
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel("Эмпирическое покрытие")
    ax.set_title("Conformal: покрытие vs цель")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Ширина интервала trained vs untrained
    ax = axes[1]
    sub = all_conf[all_conf["alpha"] == 0.1]
    for is_tr, color, lbl in [(1, "C3", "trained"), (0, "C0", "untrained")]:
        g = sub[sub["trained"] == is_tr]
        for i, t in enumerate(tags):
            vals = g[g["tag"] == t]["q_halfwidth_min"]
            ax.scatter(np.full(len(vals), i) + (0.15 if is_tr else -0.15),
                       vals, color=color, s=40, alpha=0.7,
                       label=lbl if i == 0 else None)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel("Полуширина интервала, мин (α=0.1)")
    ax.set_title("Ширина conformal-интервала (90%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    CONF_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка датасета…")
    df_raw = pd.read_parquet(DATASET_DIR / "merged_features_ml.parquet")
    sp = pd.read_parquet(DATASET_DIR / "session_params.parquet")
    meta = load_subject_meta()

    df_by_target = {
        t: prepare_data(df_raw, sp, t).dropna(subset=[TARGET_COL[t]])
        for t in ("lt1", "lt2")
    }

    all_conf = []
    conf_summary_rows = []
    for target, fset, tag in FINALISTS:
        print(f"\n[{tag}] target={target}, feature_set={fset}")
        df = df_by_target[target]
        feat_cols = get_feature_cols(df, fset)
        print(f"  признаков: {len(feat_cols)}")

        # SHAP
        run_shap(df, feat_cols, TARGET_COL[target], tag)

        # Conformal
        loso = loso_predictions(df, feat_cols, TARGET_COL[target])
        mae = mean_absolute_error(loso["y_true"], loso["y_pred"]) / 60.0
        print(f"  LOSO raw MAE = {mae:.3f} мин")
        conf = run_conformal(loso, meta, tag)
        all_conf.append(conf)

        for alpha in CONFORMAL_ALPHAS:
            sub = conf[conf["alpha"] == alpha]
            conf_summary_rows.append({
                "tag": tag, "target": target, "feature_set": fset,
                "alpha": alpha, "target_coverage": 1 - alpha,
                "mean_empirical_coverage": round(sub["empirical_coverage"].mean(), 4),
                "mean_halfwidth_min": round(sub["q_halfwidth_min"].mean(), 3),
                "halfwidth_trained_min": round(
                    sub[sub["trained"] == 1]["q_halfwidth_min"].mean(), 3),
                "halfwidth_untrained_min": round(
                    sub[sub["trained"] == 0]["q_halfwidth_min"].mean(), 3),
            })

    conf_all_df = pd.concat(all_conf, ignore_index=True)
    summary_df = pd.DataFrame(conf_summary_rows)
    summary_df.to_csv(CONF_DIR / "coverage_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("CONFORMAL coverage summary:")
    print(summary_df.to_string(index=False))

    plot_conformal(conf_all_df, CONF_DIR / "conformal_summary.png")
    print(f"\nАртефакты: {SHAP_DIR} и {CONF_DIR}")


if __name__ == "__main__":
    main()
