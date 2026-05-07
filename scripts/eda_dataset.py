"""EDA датасета: корреляции, траектории, распределения.

Генерирует PNG (matplotlib/seaborn) и HTML (plotly) в папку eda/.

Запуск:
  python scripts/eda_dataset.py
  python scripts/eda_dataset.py --dataset-dir dataset --output-dir eda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без GUI — для серверного запуска
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from plotly.subplots import make_subplots
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─────────────────────── Палитра и группы признаков ───────────────────────

PALETTE = {
    "EMG":        "#1f77b4",
    "Кинематика": "#17becf",
    "NIRS":       "#2ca02c",
    "HRV":        "#9467bd",
    "Прочее":     "#7f7f7f",
}

SEABORN_STYLE = {"style": "whitegrid", "font_scale": 1.0}


def _feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Возвращает словарь {группа → список признаков}."""
    all_cols = set(df.columns)

    emg_signal = sorted(c for c in all_cols if c.startswith("vl_") and not c.startswith("vl_prox"))
    emg_prox   = sorted(c for c in all_cols if c.startswith("vl_prox"))
    emg_derived = sorted(c for c in all_cols if c.startswith("delta_") or
                         (c.startswith("ratio_rms") and "prox" in c))
    kinematic  = sorted(c for c in all_cols if c in {
        "cadence_mean_rpm", "cadence_cv",
        "load_duration_ms", "rest_duration_ms", "load_rest_ratio",
    })
    nirs = sorted(c for c in all_cols if c.startswith("trainred_"))
    hrv  = sorted(c for c in all_cols if c.startswith("hrv_") and
                  not c.endswith(("_valid", "_fraction", "_count")))

    return {
        "EMG (dist)":    emg_signal,
        "EMG (prox)":    emg_prox,
        "EMG (derived)": emg_derived,
        "Кинематика":    kinematic,
        "NIRS":          nirs,
        "HRV":           hrv,
    }


def _modality_color(col: str) -> str:
    if col.startswith(("vl_", "delta_", "ratio_rms")):
        return PALETTE["EMG"]
    if col in {"cadence_mean_rpm", "cadence_cv", "load_duration_ms",
               "rest_duration_ms", "load_rest_ratio"}:
        return PALETTE["Кинематика"]
    if col.startswith("trainred_"):
        return PALETTE["NIRS"]
    if col.startswith("hrv_"):
        return PALETTE["HRV"]
    return PALETTE["Прочее"]


# ─────────────────────── 1. Spearman vs target ───────────────────────

def plot_spearman_target(df: pd.DataFrame, out_dir: Path) -> None:
    """Топ-30 признаков по |Spearman| с target_time_to_lt2_center_sec."""

    # Используем окна с валидным таргетом и хотя бы одной валидной модальностью
    mask = df["target_binary_valid"] == 1
    subset = df[mask].copy()

    groups = _feature_groups(subset)
    all_feature_cols = [c for cols in groups.values() for c in cols]

    rows = []
    for col in all_feature_cols:
        valid = subset[[col, "target_time_to_lt2_center_sec"]].dropna()
        if len(valid) < 30:
            continue
        rho, pval = stats.spearmanr(valid[col], valid["target_time_to_lt2_center_sec"])
        rows.append({"feature": col, "rho": rho, "pval": pval,
                     "color": _modality_color(col)})

    corr_df = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
    top30 = corr_df.head(30).sort_values("rho")

    # ── PNG ──
    sns.set(**SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(top30["feature"], top30["rho"],
                   color=top30["color"], edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ  (target: время до LT2, сек)")
    ax.set_title("Топ-30 признаков по корреляции с таргетом")
    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in PALETTE.items()
                       if k != "Прочее"]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "1_spearman_target.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ 1_spearman_target.png")

    # ── HTML ──
    fig_px = px.bar(
        corr_df.head(40).sort_values("rho"),
        x="rho", y="feature", orientation="h",
        color="color",
        color_discrete_map="identity",
        title="Spearman ρ — признаки vs время до LT2",
        labels={"rho": "Spearman ρ", "feature": "Признак"},
        hover_data={"pval": ":.4f"},
    )
    fig_px.add_vline(x=0, line_color="black", line_width=1)
    fig_px.update_layout(height=900, showlegend=False)
    fig_px.write_html(out_dir / "1_spearman_target.html")
    print(f"  ✓ 1_spearman_target.html")

    # ── Также: Spearman vs binary label ──
    rows_bin = []
    for col in all_feature_cols:
        valid = subset[[col, "target_binary_label"]].dropna()
        if len(valid) < 30:
            continue
        rho, pval = stats.spearmanr(valid[col], valid["target_binary_label"])
        rows_bin.append({"feature": col, "rho": rho, "pval": pval,
                         "color": _modality_color(col)})

    corr_bin = pd.DataFrame(rows_bin).sort_values("rho", key=abs, ascending=False)
    top30_bin = corr_bin.head(30).sort_values("rho")

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.barh(top30_bin["feature"], top30_bin["rho"],
             color=top30_bin["color"], edgecolor="white", linewidth=0.4)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Spearman ρ  (target: бинарная метка 0=до/1=после LT2)")
    ax2.set_title("Топ-30 признаков по корреляции с бинарным таргетом")
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig2.savefig(out_dir / "1b_spearman_binary.png", dpi=150)
    plt.close(fig2)
    print(f"  ✓ 1b_spearman_binary.png")


# ─────────────────────── 2. Траектории выровненные по LT2 ───────────────────────

def plot_trajectories(df: pd.DataFrame, out_dir: Path) -> None:
    """Траектории ключевых признаков, выровненные по LT2 (x=0 = LT2)."""

    mask = (df["target_binary_valid"] == 1) & df["window_valid_all_required"].eq(1)
    subset = df[mask].copy()
    # x-ось: target_time_to_lt2_center_sec уже хранит «время до LT2» в секундах
    # (отрицательное = до LT2, положительное = после)
    x_col = "target_time_to_lt2_center_sec"

    key_features = [
        ("trainred_smo2_mean",   "SmO2 (%)",       PALETTE["NIRS"]),
        ("hrv_dfa_alpha1",       "DFA-α1",         PALETTE["HRV"]),
        ("vl_dist_load_rms",     "EMG RMS (VL dist load)", PALETTE["EMG"]),
        ("trainred_hhb_mean",    "HHb (отн.ед.)",  PALETTE["NIRS"]),
        ("hrv_mean_rr_ms",       "Средний RR (мс)",PALETTE["HRV"]),
        ("cadence_mean_rpm",     "Каденс (об/мин)",PALETTE["Кинематика"]),
    ]

    subjects = sorted(subset["subject_id"].unique())
    n_feats = len(key_features)

    # ── PNG ──
    sns.set(**SEABORN_STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.flatten()

    for idx, (col, label, color) in enumerate(key_features):
        ax = axes[idx]
        if col not in subset.columns:
            ax.set_visible(False)
            continue

        all_x, all_y = [], []
        for subj in subjects:
            s = subset[subset["subject_id"] == subj][[x_col, col]].dropna()
            if len(s) < 5:
                continue
            s = s.sort_values(x_col)
            ax.plot(s[x_col], s[col], alpha=0.25, linewidth=0.8,
                    color=color)
            all_x.extend(s[x_col].tolist())
            all_y.extend(s[col].tolist())

        # Медиана по сглаженным бинам
        if all_x:
            tmp = pd.DataFrame({"x": all_x, "y": all_y})
            tmp["bin"] = pd.cut(tmp["x"], bins=60)
            median_line = tmp.groupby("bin", observed=True)["y"].median()
            bin_centers = [b.mid for b in median_line.index]
            ax.plot(bin_centers, median_line.values, color="black",
                    linewidth=2.0, label="Медиана")

        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                   label="LT2")
        ax.set_xlabel("Время до LT2 (с)")
        ax.set_ylabel(label)
        ax.set_title(label)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle("Траектории признаков выровненные по LT2 (красная линия = LT2)", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "2_trajectories_lt2aligned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 2_trajectories_lt2aligned.png")

    # ── HTML (plotly, интерактивный) ──
    fig_html = make_subplots(
        rows=2, cols=3,
        subplot_titles=[label for _, label, _ in key_features],
        shared_xaxes=False,
    )
    for idx, (col, label, color) in enumerate(key_features):
        row, col_pos = divmod(idx, 3)
        row += 1
        col_pos += 1
        if col not in subset.columns:
            continue
        for subj in subjects:
            s = subset[subset["subject_id"] == subj][[x_col, col]].dropna().sort_values(x_col)
            if len(s) < 5:
                continue
            fig_html.add_trace(
                go.Scatter(x=s[x_col], y=s[col], mode="lines",
                           name=subj, line={"color": color, "width": 1},
                           opacity=0.4, showlegend=(idx == 0),
                           legendgroup=subj),
                row=row, col=col_pos,
            )
        # Вертикальная линия LT2
        fig_html.add_vline(x=0, line_color="red", line_dash="dash",
                           line_width=1.5, row=row, col=col_pos)

    fig_html.update_layout(
        height=700, title="Траектории признаков (выравнивание по LT2)",
        hovermode="x unified",
    )
    fig_html.write_html(out_dir / "2_trajectories_lt2aligned.html")
    print(f"  ✓ 2_trajectories_lt2aligned.html")


# ─────────────────────── 3. Матрицы корреляций ───────────────────────

def plot_correlation_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    """Матрицы Spearman-корреляций внутри каждой модальности."""

    mask = df["window_valid_all_required"] == 1
    subset = df[mask]

    modalities = {
        "NIRS": [c for c in df.columns if c.startswith("trainred_")],
        "HRV":  [c for c in df.columns if c.startswith("hrv_") and
                 not c.endswith(("_valid", "_fraction", "_count"))],
        # Для EMG берём только load-фазу VL_dist — репрезентативна, не перегружает
        "EMG (vl_dist_load)": [c for c in df.columns if c.startswith("vl_dist_load_")],
        "EMG (vl_prox_load)": [c for c in df.columns if c.startswith("vl_prox_load_")],
        "Кинематика + Prox-Dist derived": [
            c for c in df.columns
            if c in {"cadence_mean_rpm", "cadence_cv", "load_duration_ms",
                     "rest_duration_ms", "load_rest_ratio"}
            or c.startswith("delta_") or (c.startswith("ratio_rms") and "prox" in c)
        ],
    }

    sns.set(**SEABORN_STYLE)

    for modality_name, cols in modalities.items():
        cols = [c for c in cols if c in subset.columns]
        if len(cols) < 2:
            continue

        data = subset[cols].dropna()
        if len(data) < 10:
            continue

        corr_matrix = data.corr(method="spearman")
        safe_name = modality_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")

        # ── PNG ──
        n = len(cols)
        fig_size = max(8, n * 0.55)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
        mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, mask=mask_upper, ax=ax,
            cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            annot=(n <= 16), fmt=".2f", annot_kws={"size": 7},
            linewidths=0.3, square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"Spearman корреляции: {modality_name}")
        plt.tight_layout()
        fig.savefig(out_dir / f"3_corr_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ 3_corr_{safe_name}.png")

        # ── HTML ──
        fig_px = px.imshow(
            corr_matrix, text_auto=(n <= 16),
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"Spearman корреляции: {modality_name}",
            aspect="auto",
        )
        fig_px.update_layout(height=max(500, n * 30))
        fig_px.write_html(out_dir / f"3_corr_{safe_name}.html")
        print(f"  ✓ 3_corr_{safe_name}.html")

    # ── Мега-матрица: все модальности × ключевые признаки ──
    key_cols = [
        "vl_dist_load_rms", "vl_dist_load_mdf", "vl_dist_load_wavelet_entropy",
        "vl_prox_load_rms", "vl_prox_load_mdf",
        "delta_rms_prox_dist_load", "cadence_mean_rpm", "load_rest_ratio",
        "trainred_smo2_mean", "trainred_smo2_drop", "trainred_smo2_slope",
        "trainred_hhb_mean", "trainred_hhb_slope",
        "hrv_dfa_alpha1", "hrv_mean_rr_ms", "hrv_rmssd_ms", "hrv_sd1_sd2_ratio",
    ]
    key_cols = [c for c in key_cols if c in subset.columns]
    data_key = subset[key_cols].dropna()
    if len(data_key) > 10:
        corr_key = data_key.corr(method="spearman")
        n = len(key_cols)
        fig, ax = plt.subplots(figsize=(n * 0.7 + 1, n * 0.65 + 1))
        sns.heatmap(
            corr_key, ax=ax,
            cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.4, square=True,
        )
        ax.set_title("Ключевые признаки: межмодальные корреляции")
        plt.tight_layout()
        fig.savefig(out_dir / "3_corr_key_features.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ 3_corr_key_features.png")

        fig_px2 = px.imshow(
            corr_key, text_auto=True,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Ключевые признаки: межмодальные Spearman-корреляции",
        )
        fig_px2.update_layout(height=700)
        fig_px2.write_html(out_dir / "3_corr_key_features.html")
        print(f"  ✓ 3_corr_key_features.html")


# ─────────────────────── 4. Распределения: до vs после LT2 ───────────────────────

def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Violin-plots ключевых признаков, разбитых по binary label (0=до LT2, 1=после)."""

    mask = (df["target_binary_valid"] == 1) & (df["window_valid_all_required"] == 1)
    subset = df[mask].copy()
    subset["Зона"] = subset["target_binary_label"].map({0: "До LT2", 1: "После LT2"})

    key_features = [
        ("trainred_smo2_mean",        "SmO2 (%)",             PALETTE["NIRS"]),
        ("trainred_smo2_drop",        "SmO2 drop от baseline",PALETTE["NIRS"]),
        ("trainred_hhb_mean",         "HHb",                  PALETTE["NIRS"]),
        ("hrv_dfa_alpha1",            "DFA-α1",               PALETTE["HRV"]),
        ("hrv_mean_rr_ms",            "Средний RR (мс)",      PALETTE["HRV"]),
        ("hrv_rmssd_ms",              "RMSSD (мс)",           PALETTE["HRV"]),
        ("vl_dist_load_rms",          "EMG RMS VL dist load", PALETTE["EMG"]),
        ("vl_dist_load_mdf",          "EMG MDF VL dist load", PALETTE["EMG"]),
        ("vl_dist_load_wavelet_entropy", "Wavelet entropy VL dist", PALETTE["EMG"]),
        ("cadence_mean_rpm",          "Каденс (об/мин)",      PALETTE["Кинематика"]),
        ("load_rest_ratio",           "Load/rest ratio",      PALETTE["Кинематика"]),
        ("delta_rms_prox_dist_load",  "Δ RMS prox–dist",      PALETTE["EMG"]),
    ]

    # ── PNG ──
    sns.set(**SEABORN_STYLE)
    n_feats = len(key_features)
    ncols = 4
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for idx, (col, label, color) in enumerate(key_features):
        ax = axes[idx]
        if col not in subset.columns:
            ax.set_visible(False)
            continue
        data_plot = subset[["Зона", col]].dropna()
        if data_plot.empty:
            ax.set_visible(False)
            continue

        sns.violinplot(data=data_plot, x="Зона", y=col, ax=ax,
                       hue="Зона", legend=False,
                       palette={"До LT2": "#aec7e8", "После LT2": "#ff9896"},
                       inner="box", cut=0)
        # Добавляем медианы текстом
        for zone, grp in data_plot.groupby("Зона"):
            med = grp[col].median()
            x_pos = 0 if zone == "До LT2" else 1
            ax.text(x_pos, ax.get_ylim()[1] * 0.95, f"med={med:.2f}",
                    ha="center", fontsize=7, color="black")

        # Тест Манна-Уитни
        g0 = data_plot[data_plot["Зона"] == "До LT2"][col].dropna()
        g1 = data_plot[data_plot["Зона"] == "После LT2"][col].dropna()
        if len(g0) > 5 and len(g1) > 5:
            _, pval = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_title(f"{label}\n(Mann-Whitney {stars})", fontsize=9)
        else:
            ax.set_title(label, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for idx in range(n_feats, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Распределения признаков: до vs после LT2", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "4_distributions_lt2zones.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 4_distributions_lt2zones.png")

    # ── HTML ──
    html_cols = [col for col, _, _ in key_features if col in subset.columns]
    fig_px = make_subplots(
        rows=(len(html_cols) + 2) // 3, cols=3,
        subplot_titles=[col for col in html_cols],
    )
    zone_colors = {"До LT2": "#4c92c3", "После LT2": "#d62728"}
    for idx, col in enumerate(html_cols):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1
        for zone, color in zone_colors.items():
            vals = subset[subset["Зона"] == zone][col].dropna()
            fig_px.add_trace(
                go.Violin(y=vals, name=zone, legendgroup=zone,
                          showlegend=(idx == 0),
                          box_visible=True, meanline_visible=True,
                          line_color=color, fillcolor=color,
                          opacity=0.6),
                row=row, col=col_pos,
            )
    fig_px.update_layout(
        height=max(500, len(html_cols) // 3 * 280),
        title="Распределения признаков: До vs После LT2",
        violinmode="group",
    )
    fig_px.write_html(out_dir / "4_distributions_lt2zones.html")
    print(f"  ✓ 4_distributions_lt2zones.html")


# ─────────────────────── 5. Scatter: линейность признак → таргет ───────────────────────

def plot_linearity(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter-plots: признак vs time_to_lt2. Проверка линейности."""

    mask = (df["target_binary_valid"] == 1) & (df["window_valid_all_required"] == 1)
    subset = df[mask].copy()
    x_col = "target_time_to_lt2_center_sec"

    top_features = [
        ("trainred_smo2_mean",           "SmO2 (%)"),
        ("trainred_smo2_drop",           "SmO2 drop"),
        ("trainred_hhb_slope",           "HHb slope"),
        ("hrv_dfa_alpha1",               "DFA-α1"),
        ("hrv_mean_rr_ms",               "Средний RR (мс)"),
        ("vl_dist_load_rms",             "EMG RMS dist load"),
        ("vl_dist_load_mdf",             "EMG MDF dist load"),
        ("vl_dist_load_wavelet_entropy", "Wavelet entropy dist"),
        ("cadence_mean_rpm",             "Каденс"),
        ("delta_rms_prox_dist_load",     "Δ RMS prox-dist"),
        ("load_rest_ratio",              "Load/rest ratio"),
        ("trainred_thb_slope",           "tHb slope"),
    ]

    sns.set(**SEABORN_STYLE)
    ncols = 4
    nrows = (len(top_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    subjects = sorted(subset["subject_id"].unique())
    subj_colors = {s: cm.tab20(i / len(subjects)) for i, s in enumerate(subjects)}

    for idx, (col, label) in enumerate(top_features):
        ax = axes[idx]
        if col not in subset.columns:
            ax.set_visible(False)
            continue

        for subj in subjects:
            s = subset[subset["subject_id"] == subj][[x_col, col]].dropna()
            if len(s) < 3:
                continue
            ax.scatter(s[x_col], s[col], s=4, alpha=0.25,
                       color=subj_colors[subj])

        # Линия тренда по всем данным
        xy = subset[[x_col, col]].dropna()
        if len(xy) > 10:
            slope, intercept, r, p, _ = stats.linregress(xy[x_col], xy[col])
            xline = np.array([xy[x_col].min(), xy[x_col].max()])
            ax.plot(xline, slope * xline + intercept, "k-", linewidth=1.5,
                    label=f"r={r:.2f}")
            ax.legend(fontsize=7)

        ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Время до LT2 (с)", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9)

    for idx in range(len(top_features), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Линейность: признак vs время до LT2 (каждый цвет — участник)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "5_linearity_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 5_linearity_scatter.png")

    # ── HTML: интерактивный scatter ──
    for col, label in top_features[:4]:  # первые 4 в HTML
        if col not in subset.columns:
            continue
        xy = subset[[x_col, col, "subject_id"]].dropna()
        fig_px = px.scatter(
            xy, x=x_col, y=col, color="subject_id",
            labels={x_col: "Время до LT2 (с)", col: label},
            title=f"Линейность: {label} vs время до LT2",
            opacity=0.5,
        )
        # Добавляем линейный тренд вручную через scipy (без statsmodels)
        xy_clean = xy[[x_col, col]].dropna()
        if len(xy_clean) > 10:
            slope, intercept, r, _, _ = stats.linregress(xy_clean[x_col], xy_clean[col])
            x_line = np.array([xy_clean[x_col].min(), xy_clean[x_col].max()])
            fig_px.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept,
                mode="lines", line={"color": "black", "width": 2},
                name=f"trend r={r:.2f}", showlegend=True,
            ))
        fig_px.add_vline(x=0, line_color="red", line_dash="dash")
        safe = col.replace(".", "_")
        fig_px.write_html(out_dir / f"5_linearity_{safe}.html")
        print(f"  ✓ 5_linearity_{safe}.html")


# ─────────────────────── 6. Feature importance (Spearman within subject) ───────────────────────

def plot_within_subject_consistency(df: pd.DataFrame, out_dir: Path) -> None:
    """Box-plot: Spearman ρ каждого признака с таргетом, по каждому субъекту отдельно.

    Это честная оценка — не раздутая межиндивидуальной структурой.
    """
    mask = df["target_binary_valid"] == 1
    subset = df[mask].copy()
    subjects = sorted(subset["subject_id"].unique())

    key_cols = [
        "trainred_smo2_mean", "trainred_smo2_drop", "trainred_hhb_slope",
        "hrv_dfa_alpha1", "hrv_mean_rr_ms", "hrv_rmssd_ms",
        "vl_dist_load_rms", "vl_dist_load_mdf", "vl_dist_load_wavelet_entropy",
        "cadence_mean_rpm", "load_rest_ratio", "delta_rms_prox_dist_load",
    ]
    key_cols = [c for c in key_cols if c in subset.columns]

    x_col = "target_time_to_lt2_center_sec"
    rows = []
    for subj in subjects:
        s = subset[subset["subject_id"] == subj]
        for col in key_cols:
            xy = s[[x_col, col]].dropna()
            if len(xy) < 10:
                continue
            rho, _ = stats.spearmanr(xy[x_col], xy[col])
            rows.append({"subject": subj, "feature": col, "rho": rho,
                         "color": _modality_color(col)})

    if not rows:
        return

    ws_df = pd.DataFrame(rows)
    # Сортируем признаки по медианному |rho| среди субъектов
    feat_order = (
        ws_df.groupby("feature")["rho"]
        .apply(lambda x: x.abs().median())
        .sort_values(ascending=False)
        .index.tolist()
    )
    ws_df["feature"] = pd.Categorical(ws_df["feature"], categories=feat_order, ordered=True)

    # ── PNG ──
    sns.set(**SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(12, 6))
    feat_colors = [_modality_color(f) for f in feat_order]
    sns.boxplot(data=ws_df, x="feature", y="rho", ax=ax,
                order=feat_order, palette=feat_colors,
                flierprops={"markersize": 4})
    sns.stripplot(data=ws_df, x="feature", y="rho", ax=ax,
                  order=feat_order, color="black", size=4, alpha=0.5, jitter=True)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Spearman ρ (внутри субъекта)")
    ax.set_title("Внутрисубъектная корреляция признаков с таргетом (N=13 субъектов)")
    plt.tight_layout()
    fig.savefig(out_dir / "6_within_subject_consistency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 6_within_subject_consistency.png")

    # ── HTML ──
    fig_px = px.box(
        ws_df, x="feature", y="rho", color="feature",
        color_discrete_map={f: _modality_color(f) for f in feat_order},
        category_orders={"feature": feat_order},
        points="all", hover_data=["subject"],
        title="Внутрисубъектная Spearman ρ: признак vs время до LT2",
        labels={"rho": "Spearman ρ", "feature": "Признак"},
    )
    fig_px.add_hline(y=0, line_color="red", line_dash="dash")
    fig_px.update_layout(showlegend=False, height=500)
    fig_px.write_html(out_dir / "6_within_subject_consistency.html")
    print(f"  ✓ 6_within_subject_consistency.html")


# ─────────────────────── main ───────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA датасета — корреляции, траектории, распределения.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir",  type=Path, default=Path("eda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Читаем датасет: {args.dataset_dir / 'merged_features_ml.parquet'}")
    df = pd.read_parquet(args.dataset_dir / "merged_features_ml.parquet")
    print(f"  {len(df)} окон, {df['subject_id'].nunique()} участников")

    print("\n── 1. Spearman vs target ──")
    plot_spearman_target(df, args.output_dir)

    print("\n── 2. Траектории (LT2-aligned) ──")
    plot_trajectories(df, args.output_dir)

    print("\n── 3. Матрицы корреляций ──")
    plot_correlation_heatmaps(df, args.output_dir)

    print("\n── 4. Распределения: до vs после LT2 ──")
    plot_distributions(df, args.output_dir)

    print("\n── 5. Линейность ──")
    plot_linearity(df, args.output_dir)

    print("\n── 6. Внутрисубъектная согласованность ──")
    plot_within_subject_consistency(df, args.output_dir)

    print(f"\n✅  Все графики сохранены в: {args.output_dir.resolve()}")
    print("   PNG — для отчётов, HTML — интерактивные (открывать в браузере)")


if __name__ == "__main__":
    main()
