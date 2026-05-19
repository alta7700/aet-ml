"""NN-интерпретация trusted-finalist'ов через Captum.

Зеркало по API к ``analysis.shap_analysis``, но для нейросетевых моделей
(LSTM/TCN). Считает атрибуции тремя методами и аккуратно их сводит:

* **Integrated Gradients** — primary. Baseline = нулевой вектор; после
  per-fold StandardScaler это эквивалент «вход равен среднему train-fold»,
  что для z-нормированных EMG/NIRS/kinematics соответствует собственному
  отдыху субъекта (stage_index=0), а для HRV/running-NIRS — среднему по
  обучающей выборке. Семантически корректный «нейтральный» вход.
* **GradientShap** — sanity-check IG. Бэкграунд набирается случайно из
  скейленых train-данных текущего fold'а (LOSO-корректно).
* **Occlusion на уровне модальности** — модель-агностичный кросс-чек:
  обнуляются ВСЕ признаки одной модальности на всём окне последовательности,
  диффе предсказания против немаскированного forward сравнивается с
  IG/GradientShap по тем же модальностям. Расхождение видно в
  ``captum_method_consistency.parquet``.

Whitelist архитектур (см. ``CaptumConfig.allowed_model_classes``) защищает
от того, что в finalist'ы случайно попадёт незнакомая модель: stateful LSTM,
attention-варианты со скрытой памятью, что-то ещё. Если такая модель
встречается — пишется строка в ``captum_skipped.parquet`` и шаг пропускается
без падения пайплайна.

Все таблицы пишутся в ``cfg.cache_dir`` и подхватываются reporting/conclusions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from architectures import get_architecture
from common_lib import build_fold_id
from dataset_pipeline.common import DEFAULT_DATASET_DIR
from features import (
    HRV_FEATURES, INTERACTION_FEATURES, NIRS_FEATURES,
    RUNNING_NIRS_FEATURES, get_feature_cols, prepare_data,
)
from models.lstm import (
    AttentionLSTMRegressor, LSTMRegressor, StatelessSeqDataset,
)
from models.tcn import DwtTCN, PureTCN, WaveNetTCN
from training_utils import CwtCache, get_device, prepare_X_for_fold

from analysis.schemas import AnalysisConfig
from analysis.selection import build_selection_tables


TARGET_COLS = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}
BASE_STEP_SEC = 5

# ─── Восстановление модели по architecture_id ──────────────────────────────

def _build_nn_model(arch, input_size: int) -> nn.Module:
    """Строит NN-модель по ArchitectureSpec.

    Только для whitelist'нутых классов. Для всего остального бросает
    ValueError — вызывающая сторона ловит и пишет в skipped.
    """
    cls = arch.model_class_name
    hp = arch.hyperparams
    if cls == "LSTMRegressor":
        return LSTMRegressor(
            input_size=input_size,
            hidden_size=int(hp["hidden_size"]),
            num_layers=int(hp["num_layers"]),
            dropout=float(hp["dropout"]),
        )
    if cls == "AttentionLSTMRegressor":
        return AttentionLSTMRegressor(
            input_size=input_size,
            hidden_size=int(hp["hidden_size"]),
            num_layers=int(hp["num_layers"]),
            dropout=float(hp["dropout"]),
        )
    if cls == "PureTCN":
        return PureTCN(
            input_size=input_size,
            n_channels=int(hp["n_channels"]),
            kernel_size=int(hp["kernel_size"]),
            dilations=list(hp["dilations"]),
            dropout=float(hp["dropout"]),
        )
    if cls == "DwtTCN":
        return DwtTCN(
            input_size=input_size,
            n_channels=int(hp["n_channels"]),
            kernel_size=int(hp["kernel_size"]),
            dilations=list(hp["dilations"]),
            dropout=float(hp["dropout"]),
        )
    if cls == "WaveNetTCN":
        return WaveNetTCN(
            input_size=input_size,
            residual_ch=int(hp["n_channels"]),
            skip_ch=int(hp["n_channels"]) * int(hp.get("skip_channels_mult", 2)),
            dilations=list(hp["dilations"]),
            dropout=float(hp["dropout"]),
        )
    raise ValueError(f"Unsupported model_class_name={cls!r}")


# ─── Маппинг признак → модальность ──────────────────────────────────────────

def _feature_modality(name: str) -> str:
    """Та же логика, что в shap_analysis._feature_modality."""
    if name in INTERACTION_FEATURES:
        return "Interaction"
    if name in HRV_FEATURES:
        return "HRV"
    if name in NIRS_FEATURES or name in RUNNING_NIRS_FEATURES:
        return "NIRS"
    if name.startswith("z_vl_"):
        return "EMG"
    if name.startswith("z_"):
        return "Kinematics"
    return "Other"


# ─── Подготовка fold'а: вход для модели + порядок признаков ────────────────

@dataclass
class _FoldData:
    """Сырьё для атрибуции одного теста (fold = 1 субъект)."""
    feat_cols: list[str]
    n_features_in_model: int                  # учитывает CWT-конкат
    seq_len: int
    int_stride_rows: int
    out_stride_rows: int
    X_train: np.ndarray                       # для GradientShap-бэкграунда
    X_test: np.ndarray
    end_pos: np.ndarray                       # индексы последних шагов внутри test_df
    test_df: pd.DataFrame
    cwt_feat_names: list[str]                 # имена CWT-фич, добавленных в конец


def _build_fold_data(df_prep: pd.DataFrame, feat_cols: list[str],
                     target_col: str, test_subject: str,
                     seq_len: int, int_stride_rows: int, out_stride_rows: int,
                     wavelet_mode: str) -> Optional[_FoldData]:
    """Воспроизводит LOSO-fold: prepare_X_for_fold + StatelessSeqDataset."""
    train_df = df_prep[df_prep["subject_id"] != test_subject].sort_values(
        ["subject_id", "window_start_sec"])
    test_df = df_prep[df_prep["subject_id"] == test_subject].sort_values(
        "window_start_sec").reset_index()
    orig_test_index = pd.Index(test_df.pop("index").values)

    cwt: Optional[CwtCache] = CwtCache() if wavelet_mode == "cwt" else None
    X_tr, X_te = prepare_X_for_fold(
        df_prep, feat_cols, train_df.index, orig_test_index, cwt
    )
    cwt_names: list[str] = []
    if cwt is not None:
        for ch in cwt.channel_names:
            for stat in ("mean", "std"):
                for s in cwt.scales:
                    cwt_names.append(f"cwt_{ch}_{stat}_scale{int(s)}")

    test_ds = StatelessSeqDataset(
        X_te, np.zeros(len(X_te), dtype=np.float32),
        seq_len, int_stride_rows, out_stride_rows,
    )
    if len(test_ds) == 0:
        return None
    end_pos = test_ds.starts + (seq_len - 1) * int_stride_rows
    return _FoldData(
        feat_cols=feat_cols,
        n_features_in_model=X_te.shape[1],
        seq_len=seq_len,
        int_stride_rows=int_stride_rows,
        out_stride_rows=out_stride_rows,
        X_train=X_tr,
        X_test=X_te,
        end_pos=end_pos,
        test_df=test_df,
        cwt_feat_names=cwt_names,
    )


def _collect_seq_tensor(fold: _FoldData) -> torch.Tensor:
    """Собирает (N, T, F) из StatelessSeqDataset.starts."""
    starts = np.arange(0, len(fold.X_test) - (fold.seq_len - 1) * fold.int_stride_rows,
                       fold.out_stride_rows, dtype=np.int64)
    if starts.size == 0:
        return torch.empty(0, fold.seq_len, fold.n_features_in_model, dtype=torch.float32)
    arrs = np.stack([fold.X_test[st + np.arange(fold.seq_len) * fold.int_stride_rows]
                     for st in starts])
    return torch.from_numpy(arrs).float()


def _train_background(fold: _FoldData, n_baselines: int,
                      rng: np.random.Generator) -> torch.Tensor:
    """Случайные seq из train-данных fold'а для GradientShap baseline."""
    # Берём случайные допустимые старты по train (без склейки субъектов —
    # порядок train_df теряется при concat'е, но это OK: используется только
    # как distributional baseline, не как осмысленные «соседи»).
    last_start = len(fold.X_train) - (fold.seq_len - 1) * fold.int_stride_rows - 1
    if last_start <= 0:
        return torch.zeros(1, fold.seq_len, fold.n_features_in_model, dtype=torch.float32)
    n_take = min(n_baselines, last_start)
    starts = rng.choice(last_start, size=n_take, replace=False)
    arrs = np.stack([fold.X_train[st + np.arange(fold.seq_len) * fold.int_stride_rows]
                     for st in starts])
    return torch.from_numpy(arrs).float()


# ─── Атрибуции ──────────────────────────────────────────────────────────────

def _run_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
            n_steps: int, internal_bs: int, device: str) -> np.ndarray:
    """Возвращает atrr (N, T, F) numpy."""
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)
    attr = ig.attribute(
        x.to(device), baselines=baseline.to(device),
        n_steps=n_steps, internal_batch_size=internal_bs,
    )
    return attr.detach().cpu().numpy()


def _run_gradient_shap(model: nn.Module, x: torch.Tensor,
                       bg: torch.Tensor, n_samples: int,
                       stdevs: float, device: str,
                       seed: int) -> np.ndarray:
    from captum.attr import GradientShap
    gs = GradientShap(model)
    torch.manual_seed(seed)
    attr = gs.attribute(
        x.to(device), baselines=bg.to(device),
        n_samples=n_samples, stdevs=stdevs,
    )
    return attr.detach().cpu().numpy()


def _run_occlusion_modality(model: nn.Module, x: torch.Tensor,
                            feat_modalities: np.ndarray, device: str
                            ) -> dict[str, np.ndarray]:
    """Возвращает {modality: Δpred (N,)} — насколько меняется предсказание,
    если занулить ВСЕ признаки данной модальности на всех шагах.

    Δ = base_pred - pred_with_modality_masked. Положительная Δ → модель
    потеряла «полезный» сигнал из модальности. Знак — направление сдвига.
    """
    model.eval()
    x_dev = x.to(device)
    with torch.no_grad():
        base_pred = model(x_dev).detach().cpu().numpy()
    out: dict[str, np.ndarray] = {}
    for modality in sorted(set(feat_modalities.tolist())):
        mask = (feat_modalities != modality).astype(np.float32)
        # mask имеет форму (F,); broadcasting на (B, T, F)
        mask_t = torch.from_numpy(mask).to(device)
        with torch.no_grad():
            pred_masked = model(x_dev * mask_t).detach().cpu().numpy()
        out[modality] = base_pred - pred_masked
    return out


# ─── Агрегации по одной модели ─────────────────────────────────────────────

def _aggregate_attr(attr: np.ndarray, feat_modalities: np.ndarray,
                    feat_names: list[str]) -> dict[str, np.ndarray]:
    """Сворачивает (N, T, F) атрибуцию в агрегаты:
       per_feature: (F,)  — sum|attr| по (N, T)
       per_time:    (T,)  — sum|attr| по (N, F)
       per_modality:(M,)  — sum|attr| по (N, T, features_of_modality).
    """
    abs_attr = np.abs(attr)
    per_feature = abs_attr.sum(axis=(0, 1))
    per_time = abs_attr.sum(axis=(0, 2))
    mods = sorted(set(feat_modalities.tolist()))
    per_modality = np.array(
        [abs_attr[:, :, feat_modalities == m].sum() for m in mods],
        dtype=float)
    return {
        "per_feature": per_feature,
        "per_time": per_time,
        "per_modality": dict(zip(mods, per_modality)),
        "modalities": mods,
    }


# ─── Главный билдер ─────────────────────────────────────────────────────────

def _select_nn_finalists(model_summary: pd.DataFrame,
                         cfg: AnalysisConfig) -> pd.DataFrame:
    """Применяет правила trusted-отбора ОТДЕЛЬНО для каждой NN-семьи.

    Идея: при общем scope (Lin + LSTM + TCN) классика занимает верхние
    квантили и вытесняет NN из ``selected_for_shap``. Поэтому для
    NN-интерпретации мы запускаем selection повторно, ограничивая scope
    одной NN-семьёй за раз — это даёт finalist'ов внутри LSTM и внутри TCN
    как самостоятельные кандидаты.
    """
    parts: list[pd.DataFrame] = []
    for family in cfg.captum.selection_families:
        per_family_cfg = AnalysisConfig(**{**cfg.__dict__})
        per_family_cfg.selection_family = family
        tab = build_selection_tables(model_summary, per_family_cfg)
        sel = tab.get("selected_for_shap", pd.DataFrame())
        if not sel.empty:
            parts.append(sel)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _empty_outputs(cfg: AnalysisConfig) -> dict[str, pd.DataFrame]:
    empty = pd.DataFrame()
    for path in (
        cfg.captum_subject_summary_path,
        cfg.captum_global_summary_path,
        cfg.captum_modality_summary_path,
        cfg.captum_time_summary_path,
        cfg.captum_method_consistency_path,
        cfg.captum_skipped_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(path, index=False)
    return {"subject": empty, "global": empty, "modality": empty,
            "time": empty, "consistency": empty, "skipped": empty}


def build_captum_tables(model_summary: pd.DataFrame,
                        cfg: AnalysisConfig) -> dict[str, pd.DataFrame]:
    """Считает Captum-таблицы для trusted-finalist'ов NN-семейств."""
    if not cfg.captum.enabled:
        return _empty_outputs(cfg)

    finalists = _select_nn_finalists(model_summary, cfg)
    if finalists.empty:
        warnings.warn("captum: нет NN-finalist'ов (по правилам admissibility), skip")
        return _empty_outputs(cfg)

    device = get_device()
    rng = np.random.default_rng(cfg.captum.random_seed)

    # Подготовка фреймов под оба target — один раз.
    dataset_path = DEFAULT_DATASET_DIR / "merged_features_ml.parquet"
    session_params_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    raw = pd.read_parquet(dataset_path)
    session_params = (pd.read_parquet(session_params_path)
                      if session_params_path.exists() else pd.DataFrame())
    prepared: dict[str, pd.DataFrame] = {}
    for target in ("lt1", "lt2"):
        df = prepare_data(raw, session_params, target)
        df = df.dropna(subset=[TARGET_COLS[target]]).sort_values(
            ["subject_id", "window_start_sec"]).reset_index(drop=True)
        prepared[target] = df

    subject_rows: list[dict] = []
    time_rows: list[dict] = []
    modality_rows: list[dict] = []
    consistency_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for _, model_row in finalists.iterrows():
        model_id = str(model_row["model_id"])
        arch_id = str(model_row["architecture_id"])
        target = str(model_row["target"])
        try:
            arch = get_architecture(arch_id)
        except Exception as exc:
            skipped_rows.append({
                "model_id": model_id, "architecture_id": arch_id,
                "reason": f"get_architecture failed: {exc}",
            })
            continue
        if arch.model_class_name not in cfg.captum.allowed_model_classes:
            msg = (f"{model_id}: model_class={arch.model_class_name!r} "
                   f"вне whitelist {cfg.captum.allowed_model_classes} — skip")
            warnings.warn(msg)
            skipped_rows.append({
                "model_id": model_id, "architecture_id": arch_id,
                "reason": msg,
            })
            continue

        hp = arch.hyperparams
        seq_len = int(hp["seq_len"])
        int_stride_rows = int(hp["internal_stride_sec"]) // BASE_STEP_SEC
        out_stride_rows = int(hp["outer_stride_sec"]) // BASE_STEP_SEC

        df_prep = prepared[target]
        wavelet_mode = str(model_row.get("wavelet_mode", "none") or "none")
        feat_cols = get_feature_cols(
            df_prep,
            str(model_row["feature_set"]),
            with_abs=bool(model_row["with_abs"]),
            phase_split=str(model_row.get("phase_split", "split") or "split"),
        )

        # Восстанавливаем имена + модальности (после возможной CWT-конкатенации).
        # CWT-фичи разворачиваются на этапе fold (они per-fold).
        target_col = TARGET_COLS[target]

        # Путь к чекпоинтам берём НАПРЯМУЮ по model_id из model_summary
        # (это source of truth). ExperimentMetadata.from_arch не используется
        # на этом шаге специально: hyperparams архитектуры могли быть
        # отредактированы после обучения модели, и пересчёт хэша даст
        # «новый» model_id, который не совпадёт с уже сохранёнными
        # артефактами под historical-хешем.
        md = cfg.results_root / arch_id / model_id
        pt_files = sorted(md.glob(f"model_{model_id}_epoch-*.pt"))
        if not pt_files:
            skipped_rows.append({
                "model_id": model_id, "architecture_id": arch_id,
                "reason": f"no .pt checkpoint in {md}",
            })
            continue
        # selected_epoch уже посчитан в model_summary; берём конкретно его файл.
        sel_epoch = int(model_row.get("selected_epoch", 0))
        target_pt = md / f"model_{model_id}_epoch-{sel_epoch:03d}.pt"
        if not target_pt.exists():
            target_pt = pt_files[-1]
        states_by_fold = torch.load(target_pt, weights_only=True, map_location="cpu")

        subjects = sorted(df_prep["subject_id"].unique())
        for subject_id in subjects:
            fold_id = build_fold_id(subject_id)
            if fold_id not in states_by_fold:
                continue
            fold = _build_fold_data(
                df_prep, feat_cols, target_col, str(subject_id),
                seq_len, int_stride_rows, out_stride_rows, wavelet_mode,
            )
            if fold is None:
                continue

            # Имена признаков в порядке, в котором они уходят в модель.
            ordered_names = list(feat_cols) + list(fold.cwt_feat_names)
            assert len(ordered_names) == fold.n_features_in_model, (
                f"{model_id}: rebuild feature order mismatch "
                f"({len(ordered_names)} vs {fold.n_features_in_model})"
            )
            feat_modalities = np.array(
                [_feature_modality(n) for n in ordered_names])

            # Собираем тестовые последовательности.
            x = _collect_seq_tensor(fold)
            if x.shape[0] == 0:
                continue
            if (cfg.captum.max_eval_windows_per_subject is not None
                    and x.shape[0] > cfg.captum.max_eval_windows_per_subject):
                idx = rng.choice(x.shape[0],
                                 size=cfg.captum.max_eval_windows_per_subject,
                                 replace=False)
                x = x[idx]

            # Строим модель и грузим веса fold'а.
            model = _build_nn_model(arch, fold.n_features_in_model).to(device)
            model.load_state_dict(states_by_fold[fold_id])
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            baseline_zero = torch.zeros_like(x)

            # IG.
            attrs: dict[str, np.ndarray] = {}
            if cfg.captum.use_integrated_gradients:
                # IG нуждается в requires_grad на входе.
                attrs["integrated_gradients"] = _run_ig(
                    model, x, baseline_zero,
                    cfg.captum.ig_n_steps, cfg.captum.ig_internal_batch_size,
                    device,
                )

            # GradientShap.
            if cfg.captum.use_gradient_shap:
                bg = _train_background(
                    fold, cfg.captum.gradshap_n_baselines, rng)
                attrs["gradient_shap"] = _run_gradient_shap(
                    model, x, bg,
                    cfg.captum.gradshap_n_samples,
                    cfg.captum.gradshap_stdevs,
                    device, cfg.captum.random_seed,
                )

            # Occlusion на модальностях (отдельный output — не «атрибуция в
            # пространстве признак×время», а Δpred per modality).
            occlusion_delta: dict[str, np.ndarray] = {}
            if cfg.captum.use_occlusion_modality:
                occlusion_delta = _run_occlusion_modality(
                    model, x, feat_modalities, device)

            # Агрегации для каждого «настоящего» grad-метода.
            for method_name, attr in attrs.items():
                agg = _aggregate_attr(attr, feat_modalities, ordered_names)
                # per-feature → subject_summary
                for fi, fname in enumerate(ordered_names):
                    subject_rows.append({
                        "model_id": model_id,
                        "architecture_id": arch_id,
                        "family": str(model_row["family"]),
                        "target": target,
                        "subject_id": str(subject_id),
                        "method": method_name,
                        "feature_name": fname,
                        "modality": _feature_modality(fname),
                        "abs_attr_sum": float(agg["per_feature"][fi]),
                    })
                # per-time → time_summary
                for ti in range(seq_len):
                    time_rows.append({
                        "model_id": model_id,
                        "architecture_id": arch_id,
                        "target": target,
                        "subject_id": str(subject_id),
                        "method": method_name,
                        "time_step": ti,
                        "time_offset_sec": -(seq_len - 1 - ti) * int_stride_rows * BASE_STEP_SEC,
                        "abs_attr_sum": float(agg["per_time"][ti]),
                    })
                # per-modality → modality_rows (Σ|attr|)
                for mod, val in agg["per_modality"].items():
                    modality_rows.append({
                        "model_id": model_id,
                        "architecture_id": arch_id,
                        "target": target,
                        "subject_id": str(subject_id),
                        "method": method_name,
                        "modality": mod,
                        "abs_attr_sum": float(val),
                    })

            # Occlusion отдельно — модальность × |Δpred|.
            for mod, delta in occlusion_delta.items():
                modality_rows.append({
                    "model_id": model_id,
                    "architecture_id": arch_id,
                    "target": target,
                    "subject_id": str(subject_id),
                    "method": "occlusion",
                    "modality": mod,
                    "abs_attr_sum": float(np.abs(delta).sum()),
                })

    # ── DataFrames + per-(model, subject) нормализация в долю ──────────────
    subject_df = pd.DataFrame(subject_rows)
    modality_df = pd.DataFrame(modality_rows)
    time_df = pd.DataFrame(time_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    if not subject_df.empty:
        subject_df["share"] = subject_df.groupby(
            ["model_id", "subject_id", "method"])["abs_attr_sum"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else 0.0)
        # Глобальная сводка: усреднение долей по субъектам.
        global_df = (subject_df.groupby(
            ["model_id", "architecture_id", "family", "target",
             "method", "feature_name", "modality"], as_index=False)
            ["share"].mean()
            .rename(columns={"share": "share_mean_across_subjects"})
            .sort_values(["model_id", "method",
                          "share_mean_across_subjects"],
                         ascending=[True, True, False]))
    else:
        global_df = pd.DataFrame()

    if not modality_df.empty:
        modality_df["share"] = modality_df.groupby(
            ["model_id", "subject_id", "method"])["abs_attr_sum"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else 0.0)
        # Сводка по модальностям: mean share по субъектам, на метод.
        consistency_long = (modality_df.groupby(
            ["model_id", "architecture_id", "target",
             "method", "modality"], as_index=False)
            ["share"].mean()
            .rename(columns={"share": "share_mean"}))
        consistency_rows_df = consistency_long.pivot_table(
            index=["model_id", "architecture_id", "target", "modality"],
            columns="method", values="share_mean", aggfunc="first",
        ).reset_index()
        consistency_rows_df.columns.name = None
    else:
        consistency_rows_df = pd.DataFrame()

    if not time_df.empty:
        time_df["share"] = time_df.groupby(
            ["model_id", "subject_id", "method"])["abs_attr_sum"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else 0.0)

    # Запись.
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    subject_df.to_parquet(cfg.captum_subject_summary_path, index=False)
    global_df.to_parquet(cfg.captum_global_summary_path, index=False)
    modality_df.to_parquet(cfg.captum_modality_summary_path, index=False)
    time_df.to_parquet(cfg.captum_time_summary_path, index=False)
    consistency_rows_df.to_parquet(cfg.captum_method_consistency_path, index=False)
    skipped_df.to_parquet(cfg.captum_skipped_path, index=False)

    return {
        "subject": subject_df,
        "global": global_df,
        "modality": modality_df,
        "time": time_df,
        "consistency": consistency_rows_df,
        "skipped": skipped_df,
    }


# ─── Фигуры ─────────────────────────────────────────────────────────────────

_METHOD_COLORS = {
    "integrated_gradients": "#2f6db3",
    "gradient_shap": "#d97a2a",
    "occlusion": "#4d9c7d",
}
_MODALITY_ORDER = ["EMG", "NIRS", "HRV", "Interaction", "Kinematics", "Other"]


def _save_fig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_captum_figures(global_df: pd.DataFrame,
                        modality_df: pd.DataFrame,
                        time_df: pd.DataFrame,
                        consistency_df: pd.DataFrame,
                        cfg: AnalysisConfig) -> list[Path]:
    """Строит фигуры Captum по finalist-моделям LSTM/TCN.

    На каждую модель — четыре PNG:
      * top_features.png   — top-15 признаков, по методам side-by-side
      * modalities.png     — атрибуция модальностей, по методам side-by-side
      * time_axis.png      — важность шагов последовательности (IG/GradientShap)
      * consistency.png    — кросс-метод сопоставление по модальностям
    """
    out: list[Path] = []
    if global_df.empty and modality_df.empty and time_df.empty:
        return out
    base = cfg.figures_dir / "captum"
    base.mkdir(parents=True, exist_ok=True)

    methods_present: list[str] = []
    if not global_df.empty:
        methods_present = sorted(global_df["method"].unique().tolist())

    # ── 1. Top-features per (model, method) ────────────────────────────────
    if not global_df.empty:
        for model_id, g_model in global_df.groupby("model_id", sort=True):
            target = str(g_model["target"].iloc[0]).upper()
            methods = sorted(g_model["method"].unique().tolist())
            top_n = 15
            tops: dict[str, pd.DataFrame] = {}
            feat_union: list[str] = []
            for m in methods:
                t = g_model[g_model["method"] == m].sort_values(
                    "share_mean_across_subjects", ascending=False).head(top_n)
                tops[m] = t
                for f in t["feature_name"]:
                    if f not in feat_union:
                        feat_union.append(f)
            # Заполняем матрицу feature × method (доля).
            mat = np.zeros((len(feat_union), len(methods)), dtype=float)
            for j, m in enumerate(methods):
                lookup = dict(zip(tops[m]["feature_name"],
                                  tops[m]["share_mean_across_subjects"]))
                for i, f in enumerate(feat_union):
                    mat[i, j] = float(lookup.get(f, 0.0))
            fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(feat_union))))
            y = np.arange(len(feat_union))
            bar_h = 0.8 / max(1, len(methods))
            for j, m in enumerate(methods):
                ax.barh(y + (j - (len(methods) - 1) / 2) * bar_h,
                        mat[:, j], height=bar_h,
                        label=m, color=_METHOD_COLORS.get(m, "#888888"))
            ax.set_yticks(y)
            ax.set_yticklabels(feat_union)
            ax.invert_yaxis()
            ax.set_xlabel("Доля |attr| в пределах (subject, method), mean по субъектам")
            ax.set_title(f"{model_id}: top-{top_n} признаков ({target})")
            ax.legend(loc="lower right", frameon=False)
            path = base / f"{model_id}_top_features.png"
            _save_fig(fig, path, cfg.figure_dpi)
            out.append(path)

    # ── 2. Modality bars per (model) ────────────────────────────────────────
    if not modality_df.empty:
        # mean доля по субъектам внутри (model, method, modality)
        mod_mean = (modality_df.groupby(
            ["model_id", "target", "method", "modality"], as_index=False)
            ["share"].mean())
        for model_id, g_model in mod_mean.groupby("model_id", sort=True):
            target = str(g_model["target"].iloc[0]).upper()
            methods = sorted(g_model["method"].unique().tolist())
            mods_local = [m for m in _MODALITY_ORDER
                          if m in g_model["modality"].unique().tolist()]
            mat = np.zeros((len(mods_local), len(methods)), dtype=float)
            for j, mname in enumerate(methods):
                lookup = dict(zip(
                    g_model[g_model["method"] == mname]["modality"],
                    g_model[g_model["method"] == mname]["share"]))
                for i, mod in enumerate(mods_local):
                    mat[i, j] = float(lookup.get(mod, 0.0))
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(mods_local))
            bar_w = 0.8 / max(1, len(methods))
            for j, mname in enumerate(methods):
                ax.bar(x + (j - (len(methods) - 1) / 2) * bar_w,
                       mat[:, j], width=bar_w,
                       label=mname, color=_METHOD_COLORS.get(mname, "#888888"))
            ax.set_xticks(x)
            ax.set_xticklabels(mods_local)
            ax.set_ylabel("Доля |attr| (subject-mean)")
            ax.set_title(f"{model_id}: вклад модальностей ({target})")
            ax.legend(loc="upper right", frameon=False)
            path = base / f"{model_id}_modalities.png"
            _save_fig(fig, path, cfg.figure_dpi)
            out.append(path)

    # ── 3. Time-axis attribution ────────────────────────────────────────────
    if not time_df.empty:
        # Берём только грайдент-методы (Occlusion на модальности, не на шагах).
        grad_methods = [m for m in time_df["method"].unique()
                        if m in ("integrated_gradients", "gradient_shap")]
        t_mean = (time_df[time_df["method"].isin(grad_methods)]
                  .groupby(["model_id", "method", "time_step",
                            "time_offset_sec"], as_index=False)["share"].mean())
        for model_id, g_model in t_mean.groupby("model_id", sort=True):
            fig, ax = plt.subplots(figsize=(8, 4))
            for m in sorted(g_model["method"].unique().tolist()):
                sub = g_model[g_model["method"] == m].sort_values("time_step")
                ax.plot(sub["time_offset_sec"], sub["share"], marker="o",
                        label=m, color=_METHOD_COLORS.get(m, "#888888"))
            ax.set_xlabel("Сдвиг шага последовательности от точки предсказания, сек")
            ax.set_ylabel("Доля |attr| (subject-mean)")
            ax.set_title(f"{model_id}: важность шагов последовательности")
            ax.legend(loc="best", frameon=False)
            ax.grid(True, alpha=0.3)
            path = base / f"{model_id}_time_axis.png"
            _save_fig(fig, path, cfg.figure_dpi)
            out.append(path)

    # ── 4. Method consistency heatmap (modality × method) ──────────────────
    if not consistency_df.empty:
        method_cols = [c for c in consistency_df.columns
                       if c not in ("model_id", "architecture_id",
                                    "target", "modality")]
        for model_id, g_model in consistency_df.groupby("model_id", sort=True):
            mods_local = [m for m in _MODALITY_ORDER
                          if m in g_model["modality"].unique().tolist()]
            mat = np.zeros((len(mods_local), len(method_cols)), dtype=float)
            for i, mod in enumerate(mods_local):
                row = g_model[g_model["modality"] == mod].iloc[0]
                for j, c in enumerate(method_cols):
                    val = row.get(c, np.nan)
                    mat[i, j] = 0.0 if pd.isna(val) else float(val)
            fig, ax = plt.subplots(figsize=(6, max(3, 0.6 * len(mods_local))))
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(method_cols)))
            ax.set_xticklabels(method_cols, rotation=20)
            ax.set_yticks(np.arange(len(mods_local)))
            ax.set_yticklabels(mods_local)
            for i in range(len(mods_local)):
                for j in range(len(method_cols)):
                    ax.text(j, i, f"{mat[i, j]:.2f}",
                            ha="center", va="center", color="white",
                            fontsize=9)
            target = str(g_model["target"].iloc[0]).upper()
            ax.set_title(f"{model_id}: consistency методов по модальностям ({target})")
            fig.colorbar(im, ax=ax, label="share")
            path = base / f"{model_id}_consistency.png"
            _save_fig(fig, path, cfg.figure_dpi)
            out.append(path)

    return out
