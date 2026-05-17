"""Схемы данных, dataclass-конфиг и константы analysis-pipeline.

Все слои кэша валидируются по этим спискам колонок, чтобы рассинхронизация
между builder'ами и потребителями ловилась моментально.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ─── Каноничные имена ──────────────────────────────────────────────────────

LtPolicy = Literal["median", "crossing", "stable_median"]
LT_POLICIES: tuple[LtPolicy, ...] = ("median", "crossing", "stable_median")

Family = Literal["Lin", "LSTM", "TCN"]
FAMILIES: tuple[Family, ...] = ("Lin", "LSTM", "TCN")

Target = Literal["lt1", "lt2"]
TARGETS: tuple[Target, ...] = ("lt1", "lt2")


# ─── Schema колонок по слоям ───────────────────────────────────────────────

PREDICTIONS_SELECTED_COLUMNS: list[str] = [
    "architecture_id", "model_id", "target", "subject_id",
    "sample_index", "sample_start_sec", "sample_end_sec",
    "y_true", "y_pred",
    "residual", "abs_err",
    "lt_true_sec", "lt_hat_sec",
    "selected_epoch",
]
# Backwards-compat alias на случай внешних импортёров.
PREDICTIONS_BEST_COLUMNS = PREDICTIONS_SELECTED_COLUMNS

# Метаколонки модели, которые денормализуются в Layer 2/3 для удобства
# группировки без джойнов с models.csv.
MODEL_META_COLUMNS: list[str] = [
    "architecture_id", "family", "target",
    "feature_set", "with_abs", "wavelet_mode",
    "window_size_sec", "sequence_length", "stride_sec", "sample_stride_sec",
    "model_name", "full_model_name", "hyperparams_json",
]

SUBJECT_METRIC_COLUMNS: list[str] = [
    "model_id", "subject_id",
    *MODEL_META_COLUMNS,
    "n_samples",
    "mae", "rmse", "bias", "std_err",
    "pearson_r", "spearman_r", "r2",
    "max_abs_err", "catastrophic_rate",
]

LT_POINT_METRIC_COLUMNS: list[str] = [
    "model_id", "subject_id",
    *MODEL_META_COLUMNS,
    "lt_true_sec",
    "lt_hat_median_sec", "lt_err_median_sec", "abs_lt_err_median_sec",
    "lt_hat_crossing_sec", "lt_err_crossing_sec", "abs_lt_err_crossing_sec",
    "zero_crossing_found",
    "lt_hat_stable_median_sec", "lt_err_stable_median_sec",
    "abs_lt_err_stable_median_sec", "stable_window_count",
    "n_samples",
]

MODEL_SUMMARY_COLUMNS: list[str] = [
    "model_id", *MODEL_META_COLUMNS,
    "selected_epoch", "epoch_policy", "n_subjects",
    # дорожка A
    "mae_mean", "mae_std", "mae_median", "mae_worst", "mae_best",
    "mae_ci_low", "mae_ci_high",
    "rmse_mean", "rmse_std",
    "bias_mean", "bias_std",
    "r2_mean", "pearson_r_mean",
    "catastrophic_rate_mean",
    # дорожка B — median policy
    "lt_mae_median_policy_mean", "lt_mae_median_policy_std",
    "lt_bias_median_policy_mean",
    "lt_mae_median_policy_ci_low", "lt_mae_median_policy_ci_high",
    # дорожка B — crossing policy
    "lt_mae_crossing_policy_mean", "lt_mae_crossing_policy_std",
    "lt_bias_crossing_policy_mean",
    "lt_mae_crossing_policy_ci_low", "lt_mae_crossing_policy_ci_high",
    "zero_crossing_coverage",
    # дорожка B — stable_median policy
    "lt_mae_stable_median_policy_mean", "lt_mae_stable_median_policy_std",
    "lt_bias_stable_median_policy_mean",
    "lt_mae_stable_median_policy_ci_low", "lt_mae_stable_median_policy_ci_high",
    "stable_window_coverage",
    # composite
    "composite_score",
    # conformal (Jackknife+) — динамически дополняется в build_model_summary
    # по списку cfg.conformal_alphas и trois LT-policy. Имена:
    #   conformal_coverage_{policy}_policy_at_alpha_{aaa}
    #   conformal_width_{policy}_policy_at_alpha_{aaa}_mean
    #   conformal_coverage_gap_{policy}_policy_at_alpha_{aaa}
    # training stability (опционально, если history доступен)
    "training_instability_mean", "converged_rate",
    "train_mae_slope_last_K_mean", "final_train_mae_mean",
]

TRAINING_DYNAMICS_COLUMNS: list[str] = [
    "model_id", "architecture_id", "family", "target",
    "fold_id", "epoch",
    "train_loss", "train_mae", "lr",
    "train_loss_delta", "train_mae_delta",
    "train_loss_rolling_mean", "train_mae_rolling_mean",
]

TRAINING_SUMMARY_COLUMNS: list[str] = [
    "model_id", "architecture_id", "family", "target", "selected_epoch",
    "n_folds", "n_epochs_mean",
    "final_train_mae_mean", "final_train_mae_std",
    "final_train_loss_mean",
    "train_mae_slope_last_K_mean", "train_loss_slope_last_K_mean",
    "training_instability_mean", "converged_rate",
    "relative_change_last_K_mean",
    "stability_window_epochs", "convergence_relative_threshold",
]

CONFORMAL_INTERVAL_COLUMNS: list[str] = [
    "model_id", "architecture_id", "target", "subject_id",
    "policy", "alpha", "nominal_coverage",
    "lt_true_sec", "lt_hat_sec", "abs_lt_err_sec",
    "qhat_sec", "qhat_quantile_used",
    "interval_low_sec", "interval_high_sec", "interval_width_sec",
    "covered",
    "calibration_n",
    "qhat_method",
]

CONFORMAL_SUMMARY_COLUMNS: list[str] = [
    "model_id", "architecture_id", "target", "policy", "alpha",
    "nominal_coverage", "empirical_coverage",
    "coverage_gap",
    "coverage_ci_low", "coverage_ci_high",
    "mean_interval_width_sec", "median_interval_width_sec",
    "interval_width_over_2mae",
    "calibration_n",
    "qhat_method",
]

ROBUSTNESS_RANKING_COLUMNS: list[str] = [
    "model_id", "architecture_id", "family", "target",
    "feature_set", "with_abs", "wavelet_mode",
    # точечные
    "lt_mae_median_policy_mean",
    "catastrophic_rate_mean",
    # uncertainty
    "conformal_coverage_at_alpha_020",
    "conformal_width_at_alpha_020",
    "conformal_undercoverage_penalty_020",
    # стабильность обучения
    "training_instability_mean",
    "converged_rate",
    # отдельные суб-скоры (z-нормированные)
    "point_score", "uncertainty_score", "stability_score",
    # общий ранг
    "robustness_rank",
]


COMPARISON_COLUMNS: list[str] = [
    "comparison_kind", "group_key", "metric",
    "condition_a", "condition_b",
    "n_subjects",
    "mean_a", "mean_b", "delta_mean",
    "median_a", "median_b", "delta_median",
    "wilcoxon_stat", "wilcoxon_pvalue",
    "ttest_stat", "ttest_pvalue",
    "cohens_d", "cliffs_delta",
    "bootstrap_ci_low", "bootstrap_ci_high",
    # Поправка на множественные сравнения (Benjamini–Hochberg по умолчанию).
    # pvalue_adj — скорректированное p-value Wilcoxon-теста, разделяется
    # внутри (comparison_kind × metric).
    "pvalue_adj", "correction_method", "reject_at_alpha",
]


# ─── Конфиг ─────────────────────────────────────────────────────────────────

@dataclass
class CompositeWeights:
    """Веса композитного скора (меньше = лучше)."""
    lt_mae_median_policy_mean: float = 1.0
    lt_mae_median_policy_std: float = 0.5
    catastrophic_rate_mean: float = 0.5
    abs_lt_bias_median_policy_mean: float = 0.25


@dataclass
class AnalysisConfig:
    """Конфиг analysis-pipeline. Сериализуется из analysis/config.toml."""

    # Пути
    results_root: Path = Path("results")
    out_root: Path = Path("analysis_out")

    # Глобальные параметры метрик
    catastrophic_threshold_sec: float = 120.0
    stable_window_threshold_sec: float = 120.0

    # Какую эпоху брать из predictions parquet:
    #   "last"  — max(epoch); честно, без peeking (рекомендуется).
    #   "best"  — argmin subject-mean MAE; это oracle/ceiling, contains test
    #             selection bias — использовать только как supporting metric.
    epoch_policy: str = "last"

    # Bootstrap CI
    bootstrap_n: int = 1000
    ci_level: float = 0.95
    bootstrap_seed: int = 42

    # Композитный скор
    composite_weights: CompositeWeights = field(default_factory=CompositeWeights)

    # multitest defaults дублируем тут чтобы reload_config их учёл (см. ниже)

    # Сравнения
    comparisons_required: tuple[str, ...] = (
        "family_overall",
        "modality_within_family",
        "abs_within_arch",
        "wavelet_within_family",
        "lt1_vs_lt2_per_model_class",
    )
    comparisons_optional: tuple[str, ...] = (
        "stateful_vs_stateless",
        "attention_vs_plain",
        "stride_within_arch",
    )
    # Метрика, на которой считаются парные сравнения по умолчанию.
    primary_metric: str = "lt_mae_median_policy_mean"
    secondary_metric: str = "mae_mean"

    # Поправка на множественные сравнения. Метод передаётся в
    # statsmodels.stats.multitest.multipletests; "fdr_bh" — Benjamini–Hochberg.
    multitest_correction: str = "fdr_bh"
    multitest_alpha: float = 0.05

    # Conformal prediction (Jackknife+ / leave-one-subject-out).
    # При N=18 субъектов 95%-интервал = max-error (тривиально), поэтому
    # репортить честно можно только умеренные alpha.
    conformal_alphas: tuple[float, ...] = (0.2, 0.3)   # 80% и 70% intervals
    conformal_bootstrap_n: int = 1000
    conformal_bootstrap_seed: int = 17

    # Training stability — параметры из history.csv.
    # convergence_relative_threshold: считаем "сошёлся" если
    # |slope_last_K| * K / mean(train_mae_last_K) < threshold,
    # т.е. относительное изменение метрики за окно меньше доли threshold.
    # Default = 0.05 → за последние K эпох train_mae меняется меньше чем на 5%.
    stability_window_epochs: int = 10
    convergence_relative_threshold: float = 0.05

    # Plotting / reporting
    top_n: int = 10
    figure_dpi: int = 150
    figure_formats: tuple[str, ...] = ("png", "svg", "pdf")

    # Прочее
    force: bool = False

    # ── derived paths ────────────────────────────────────────────────────────
    @property
    def cache_dir(self) -> Path:
        return self.out_root / "cache"

    @property
    def predictions_selected_dir(self) -> Path:
        return self.cache_dir / "predictions_selected"

    # Alias для обратной совместимости со старым кодом/тестами.
    @property
    def predictions_best_dir(self) -> Path:
        return self.predictions_selected_dir

    @property
    def epoch_selection_path(self) -> Path:
        return self.cache_dir / "epoch_selection.parquet"

    @property
    def best_epochs_path(self) -> Path:
        return self.epoch_selection_path

    @property
    def training_dynamics_path(self) -> Path:
        return self.cache_dir / "training_dynamics.parquet"

    @property
    def training_summary_path(self) -> Path:
        return self.cache_dir / "training_summary.parquet"

    @property
    def conformal_intervals_path(self) -> Path:
        return self.cache_dir / "conformal_lt_intervals.parquet"

    @property
    def conformal_summary_path(self) -> Path:
        return self.cache_dir / "conformal_summary.parquet"

    @property
    def robustness_ranking_path(self) -> Path:
        return self.cache_dir / "robustness_ranking.parquet"

    @property
    def subject_metrics_path(self) -> Path:
        return self.cache_dir / "subject_metrics.parquet"

    @property
    def lt_point_metrics_path(self) -> Path:
        return self.cache_dir / "lt_point_metrics.parquet"

    @property
    def model_summary_path(self) -> Path:
        return self.cache_dir / "model_summary.parquet"

    @property
    def comparisons_dir(self) -> Path:
        return self.out_root / "comparisons"

    @property
    def tables_dir(self) -> Path:
        return self.out_root / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.out_root / "figures"

    @property
    def conclusions_dir(self) -> Path:
        return self.out_root / "conclusions"


def load_config(path: Path | None = None) -> AnalysisConfig:
    """Загружает AnalysisConfig из TOML; если path=None — берёт дефолты."""
    if path is None:
        return AnalysisConfig()

    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]

    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    cfg = AnalysisConfig()

    # paths
    if "paths" in raw:
        p = raw["paths"]
        if "results_root" in p: cfg.results_root = Path(p["results_root"])
        if "out_root" in p:     cfg.out_root = Path(p["out_root"])

    # metrics
    if "metrics" in raw:
        m = raw["metrics"]
        cfg.catastrophic_threshold_sec = float(m.get(
            "catastrophic_threshold_sec", cfg.catastrophic_threshold_sec))
        cfg.stable_window_threshold_sec = float(m.get(
            "stable_window_threshold_sec", cfg.stable_window_threshold_sec))
        if "epoch_policy" in m:
            policy = str(m["epoch_policy"]).lower()
            if policy not in ("last", "best"):
                raise ValueError(
                    f"epoch_policy={policy!r} — допустимы 'last' | 'best'")
            cfg.epoch_policy = policy

    # bootstrap
    if "bootstrap" in raw:
        b = raw["bootstrap"]
        cfg.bootstrap_n = int(b.get("n", cfg.bootstrap_n))
        cfg.ci_level = float(b.get("ci_level", cfg.ci_level))
        cfg.bootstrap_seed = int(b.get("seed", cfg.bootstrap_seed))

    # composite
    if "composite" in raw:
        w = raw["composite"]
        cfg.composite_weights = CompositeWeights(
            lt_mae_median_policy_mean=float(w.get(
                "lt_mae_median_policy_mean",
                cfg.composite_weights.lt_mae_median_policy_mean)),
            lt_mae_median_policy_std=float(w.get(
                "lt_mae_median_policy_std",
                cfg.composite_weights.lt_mae_median_policy_std)),
            catastrophic_rate_mean=float(w.get(
                "catastrophic_rate_mean",
                cfg.composite_weights.catastrophic_rate_mean)),
            abs_lt_bias_median_policy_mean=float(w.get(
                "abs_lt_bias_median_policy_mean",
                cfg.composite_weights.abs_lt_bias_median_policy_mean)),
        )

    # comparisons / reporting
    if "comparisons" in raw:
        c = raw["comparisons"]
        if "required" in c: cfg.comparisons_required = tuple(c["required"])
        if "optional" in c: cfg.comparisons_optional = tuple(c["optional"])
        if "primary_metric" in c: cfg.primary_metric = str(c["primary_metric"])
        if "secondary_metric" in c: cfg.secondary_metric = str(c["secondary_metric"])

    if "multitest" in raw:
        mt = raw["multitest"]
        cfg.multitest_correction = str(mt.get("correction", cfg.multitest_correction))
        cfg.multitest_alpha = float(mt.get("alpha", cfg.multitest_alpha))

    if "reporting" in raw:
        r = raw["reporting"]
        cfg.top_n = int(r.get("top_n", cfg.top_n))
        cfg.figure_dpi = int(r.get("figure_dpi", cfg.figure_dpi))
        if "figure_formats" in r:
            cfg.figure_formats = tuple(r["figure_formats"])

    return cfg
