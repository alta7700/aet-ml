"""Анализ результатов после ynorm+MSE-фикса для NN v0101–v0107.

Считает по каждой (version, target, feature_set, variant=with_abs):
- std_ratio_loso (медиана по субъектам)
- neg_r2_share (доля субъектов LOSO с R²<0)
- mae_glob, mae_med_subj, mae_best_subj
- сравнение с v0201 (= старая схема v0106b до фикса) для оценки масштаба

Сохраняет:
- results/final/postfix_summary.csv — длинная таблица
- results/final/postfix_top10_lt1.md, postfix_top10_lt2.md — топы
- results/final/plots/postfix_std_ratio_by_family.png
- results/final/plots/postfix_mae_vs_std_ratio.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.common import DEFAULT_DATASET_DIR  # noqa: E402

RES = ROOT / "results"
OUT = RES / "final"
BACKUP = RES / "_pre_ynorm_backup"

VERSIONS = ["v0101", "v0102", "v0103", "v0104", "v0106a", "v0106b",
            "v0106c", "v0107"]  # v0105 — нет результатов
FAMILY = {
    "v0101": "lstm", "v0102": "tcn", "v0103": "wavelet",
    "v0104": "lstm", "v0105": "tcn", "v0106a": "wavelet",
    "v0106b": "wavelet", "v0106c": "wavelet", "v0107": "ensemble",
    "v0011": "linear", "v0201": "wavelet (pre-fix)",
}
# Версии, которые читаем из бэкапа «до фикса» — помечаем суффиксом «_pre».
PRE_FIX = {f"{v}_pre": v for v in VERSIONS}
FSETS = ["EMG", "NIRS", "EMG_NIRS", "EMG_NIRS_HRV"]


def idx_for(target: str) -> pd.DataFrame:
    df = pd.read_parquet(DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    if target == "lt2":
        d = df[df["window_valid_all_required"] == 1] \
              .dropna(subset=["target_time_to_lt2_center_sec"])
    else:
        d = df[df["target_time_to_lt1_usable"] == 1]
    return (d.sort_values(["subject_id", "window_start_sec"])
              [["subject_id", "window_start_sec"]].reset_index(drop=True))


def calc_metrics(version: str, target: str, fset: str, idx_cache: dict) -> dict | None:
    # _pre-версии берём из бэкапа дофикс-результатов.
    if version in PRE_FIX:
        base = BACKUP / PRE_FIX[version]
        family_label = f"{FAMILY[PRE_FIX[version]]} (pre-fix)"
    else:
        base = RES / version
        family_label = FAMILY.get(version, "?")
    p_yp = base / f"ypred_{target}_{fset}.npy"
    p_yt = base / f"ytrue_{target}_{fset}.npy"
    if not p_yp.exists() or not p_yt.exists():
        return None
    yp = np.load(p_yp); yt = np.load(p_yt)
    cur = idx_cache[target]
    if len(cur) != len(yp):
        ps_path = base / "per_subject.csv"
        kept = None
        if ps_path.exists():
            ps = pd.read_csv(ps_path)
            ps = ps[(ps.target == target) & (ps.feature_set == fset.replace("_", "+"))]
            if len(ps) > 0:
                kept = sorted(ps.subject_id.unique())
        if kept is None:
            # Fallback: эвристически отбрасываем самых коротких субъектов,
            # пока размеры не сойдутся (для случаев, где per_subject.csv
            # не содержит таргета — частая ситуация при частичных прогонах).
            sizes = cur.groupby("subject_id").size().sort_values()
            kept_set = set(sizes.index)
            for s in sizes.index:
                kept_set -= {s}
                rem = sizes.loc[list(kept_set)].sum()
                if rem == len(yp):
                    kept = sorted(kept_set)
                    break
                if rem < len(yp):
                    kept = None
                    break
        if kept is None:
            return None
        cur = cur[cur.subject_id.isin(kept)].reset_index(drop=True)
        if len(cur) != len(yp):
            return None
    ratios, mae_per = [], []
    neg = 0
    n_subj = 0
    for s, _ in cur.groupby("subject_id"):
        m = cur.subject_id.values == s
        n_subj += 1
        if np.std(yt[m]) > 1e-6:
            ratios.append(np.std(yp[m]) / np.std(yt[m]))
        ss_res = np.sum((yt[m] - yp[m]) ** 2)
        ss_tot = np.sum((yt[m] - np.mean(yt[m])) ** 2)
        if ss_tot > 0 and 1 - ss_res / ss_tot < 0:
            neg += 1
        mae_per.append(np.mean(np.abs(yt[m] - yp[m])) / 60)
    return dict(
        version=version, family=family_label,
        target=target, feature_set=fset,
        std_ratio=float(np.median(ratios)) if ratios else float("nan"),
        neg_r2_share=neg / n_subj,
        mae_glob=float(np.mean(np.abs(yt - yp))) / 60,
        mae_med_subj=float(np.median(mae_per)),
        mae_best_subj=float(np.min(mae_per)),
        mae_worst_subj=float(np.max(mae_per)),
        n_subjects=n_subj,
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    idx_cache = {"lt1": idx_for("lt1"), "lt2": idx_for("lt2")}

    rows = []
    pre_versions = [f"{v}_pre" for v in VERSIONS]
    for v in VERSIONS + pre_versions + ["v0011", "v0201"]:
        for tgt in ["lt1", "lt2"]:
            for fset in FSETS:
                m = calc_metrics(v, tgt, fset, idx_cache)
                if m is not None:
                    rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "postfix_summary.csv", index=False)
    print(f"saved postfix_summary.csv: {len(df)} строк")
    print()

    # ─── Топ-10 по target (по mae_med_subj) ────────────────────────────────────
    nn = df[df.family.isin(["lstm", "tcn", "wavelet", "ensemble"])].copy()
    for tgt in ["lt1", "lt2"]:
        sub = nn[nn.target == tgt].sort_values("mae_med_subj").head(10)
        path = OUT / f"postfix_top10_{tgt}.md"
        path.write_text(
            f"# Top-10 NN-конфигураций по MAE медианного субъекта · {tgt.upper()}\n\n"
            "Сравнение: классика v0011 и снимок «до фикса» v0201 включены справочно.\n\n"
            + sub[["version", "family", "feature_set", "std_ratio", "neg_r2_share",
                   "mae_glob", "mae_med_subj", "mae_best_subj", "n_subjects"]]
                .round(2).to_markdown(index=False)
            + "\n", encoding="utf-8")
        print(f"saved {path.name}")

    # ─── Сводка: NN-семейства, std_ratio и MAE на ENH ──────────────────────────
    print()
    print("=== ENH-конфигурация (EMG+NIRS+HRV) ===")
    e = df[df.feature_set == "EMG_NIRS_HRV"].copy()
    for tgt in ["lt1", "lt2"]:
        print(f"\n{tgt.upper()}:")
        sub = e[e.target == tgt].sort_values("mae_med_subj")
        for _, r in sub.iterrows():
            print(f"  {r.version:<7} ({r.family:<19}) "
                  f"std_ratio={r.std_ratio:.2f}  "
                  f"neg_R²={r.neg_r2_share:.2f}  "
                  f"MAE_glob={r.mae_glob:.2f}  "
                  f"MAE_med_subj={r.mae_med_subj:.2f}  "
                  f"MAE_best={r.mae_best_subj:.2f}")

    # ─── График 1: std_ratio боксплот по семействам — пары «до / после» ────────
    fams_in_order = ["lstm", "tcn", "wavelet", "ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tgt in zip(axes, ["lt1", "lt2"]):
        sub = df[df.target == tgt]
        positions, data, labels, colors = [], [], [], []
        x = 1
        for fam in fams_in_order:
            d_pre   = sub[sub.family == f"{fam} (pre-fix)"]["std_ratio"].dropna().values
            d_after = sub[sub.family == fam]["std_ratio"].dropna().values
            for tag, d, color in [("до",  d_pre, "#dddddd"),
                                   ("после", d_after, "#cfe2f3")]:
                if len(d) == 0:
                    continue
                data.append(d); positions.append(x)
                labels.append(f"{fam}\n{tag}"); colors.append(color)
                x += 0.9
            x += 0.6  # промежуток между семействами
        # классика — отдельная зелёная коробка справа
        d_lin = sub[sub.version == "v0011"]["std_ratio"].dropna().values
        if len(d_lin) > 0:
            data.append(d_lin); positions.append(x)
            labels.append("linear\n(v0011)"); colors.append("#d9ead3")
        bp = ax.boxplot(data, positions=positions, widths=0.55,
                        patch_artist=True,
                        medianprops=dict(color="black", lw=1.4),
                        boxprops=dict(edgecolor="#444"))
        for box, c in zip(bp["boxes"], colors):
            box.set_facecolor(c)
        for x_pos, d in zip(positions, data):
            xs = [x_pos + (i / max(len(d) - 1, 1) - 0.5) * 0.18 for i in range(len(d))]
            ax.scatter(xs, d, s=14, color="#1f4e79", alpha=0.7, zorder=3)
        ax.axhline(1.0, color="green", lw=0.6, ls="--", alpha=0.5)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(tgt.upper())
        ax.grid(axis="y", alpha=0.3)
        if tgt == "lt1":
            ax.set_ylabel("std(ŷ)/std(y), медиана LOSO")
    fig.suptitle("Динамичность предсказаний: NN до vs после ynorm+MSE-фикса")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "postfix_std_ratio_by_family.png", dpi=130)
    plt.close(fig)
    print(f"\nsaved plot: postfix_std_ratio_by_family.png")

    # ─── График 2: scatter MAE_med_subj vs std_ratio, по target ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    color_map = {"lstm": "#d62728", "tcn": "#2ca02c", "wavelet": "#9467bd",
                 "ensemble": "#ff7f0e", "linear": "#1f77b4",
                 "wavelet (pre-fix)": "#888888"}
    for ax, tgt in zip(axes, ["lt1", "lt2"]):
        sub = df[df.target == tgt]
        for fam, color in color_map.items():
            ss = sub[sub.family == fam]
            if ss.empty:
                continue
            marker = "x" if "pre-fix" in fam else "o"
            ax.scatter(ss.std_ratio, ss.mae_med_subj,
                       color=color, label=fam, s=50, alpha=0.8, marker=marker,
                       edgecolors="black", linewidths=0.5)
        ax.set_xlabel("std(ŷ)/std(y), медиана LOSO")
        if tgt == "lt1":
            ax.set_ylabel("MAE медианного субъекта, мин")
        ax.set_title(tgt.upper())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("MAE медианного субъекта vs динамичность предсказаний")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "postfix_mae_vs_std_ratio.png", dpi=130)
    plt.close(fig)
    print("saved plot: postfix_mae_vs_std_ratio.png")


if __name__ == "__main__":
    main()
