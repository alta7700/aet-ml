"""common_lib — общие схемы, naming-хелперы и save-функции артефактов.

Поддерживает три семейства моделей (Lin, LSTM, TCN), сохраняет:
  • results/{architecture_id}/models.csv          — таблица всех model_id архитектуры;
  • results/{architecture_id}/{model_id}/history.csv               — только NN;
  • results/{architecture_id}/{model_id}/model_{...}_fold-{fold_id}.{pt|joblib};
  • results/{architecture_id}/{model_id}/predictions_{model_id}.parquet.

torch импортируется опционально: linear-only сценарий работает без torch.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd

# torch — опциональный (для linear-only сценария)
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# ─── Типы и константы ───────────────────────────────────────────────────────

WaveletMode = Literal["none", "dwt", "cwt", "wavelet_features", "wavelet_cnn"]
Family = Literal["Lin", "LSTM", "TCN"]

WAVELET_MODES: tuple[str, ...] = (
    "none", "dwt", "cwt", "wavelet_features", "wavelet_cnn",
)

PREDICTIONS_REQUIRED_COLUMNS: list[str] = [
    "model_id", "fold_id", "subject_id",
    "window_size_sec", "sequence_length", "stride_sec", "sample_stride_sec",
    "sample_index", "sample_start_sec", "sample_end_sec",
    "y_true", "y_pred",
]

# Поля, в которых NaN недопустимы.
PREDICTIONS_NONNULL_COLUMNS: list[str] = [
    "window_size_sec", "sequence_length", "stride_sec", "sample_stride_sec",
    "sample_index", "sample_start_sec", "sample_end_sec",
    "y_true", "y_pred",
]

MODELS_CSV_COLUMNS: list[str] = [
    "architecture_id", "model_id", "family",
    "target", "feature_set", "with_abs", "wavelet_mode",
    "window_size_sec", "sequence_length", "stride_sec", "sample_stride_sec",
    "model_name", "full_model_name", "hyperparams_json",
]

HISTORY_CSV_COLUMNS: list[str] = [
    "model_id", "architecture_id", "fold_id",
    "epoch", "train_loss", "val_loss", "train_mae", "val_mae", "lr",
]


# ─── Dataclasses ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ArchitectureSpec:
    """Описание базовой архитектуры (один на architecture_id).

    forced_wavelet_mode != None — архитектура конструктивно использует вейвлет,
    значение wavelet_mode в конкретной конфигурации обязано совпадать.
    """
    architecture_id: str
    family: Family
    architecture_name: str
    short_architecture_name: str
    model_class_name: str
    window_size_sec: int
    sequence_length: int
    stride_sec: int
    sample_stride_sec: int
    forced_wavelet_mode: WaveletMode | None
    hyperparams: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentMetadata:
    """Полная метадата одной конкретной model configuration."""
    model_id: str
    architecture_id: str
    family: Family
    target: str
    feature_set: str
    with_abs: bool
    wavelet_mode: WaveletMode
    window_size_sec: int
    sequence_length: int
    stride_sec: int
    sample_stride_sec: int
    model_name: str
    full_model_name: str
    hyperparams: dict[str, Any]

    @classmethod
    def from_arch(cls, arch: ArchitectureSpec, *,
                  target: str, feature_set: str,
                  with_abs: bool, wavelet_mode: WaveletMode) -> "ExperimentMetadata":
        """Создаёт metadata из ArchitectureSpec + конфига эксперимента.

        Валидация:
          • wavelet_mode входит в WAVELET_MODES;
          • если arch.forced_wavelet_mode задан — wavelet_mode обязан совпадать.
        """
        if wavelet_mode not in WAVELET_MODES:
            raise ValueError(
                f"wavelet_mode='{wavelet_mode}' недопустим. "
                f"Разрешены: {WAVELET_MODES}"
            )
        if arch.forced_wavelet_mode is not None and arch.forced_wavelet_mode != wavelet_mode:
            raise ValueError(
                f"Архитектура {arch.architecture_id} требует "
                f"wavelet_mode='{arch.forced_wavelet_mode}', "
                f"передано wavelet_mode='{wavelet_mode}'."
            )

        model_id = build_model_id(
            arch,
            target=target, feature_set=feature_set,
            with_abs=with_abs, wavelet_mode=wavelet_mode,
        )

        model_name = _build_model_name(arch)
        full_model_name = _build_full_model_name(
            arch, target=target, feature_set=feature_set,
            with_abs=with_abs, wavelet_mode=wavelet_mode,
        )

        return cls(
            model_id=model_id,
            architecture_id=arch.architecture_id,
            family=arch.family,
            target=target,
            feature_set=feature_set,
            with_abs=with_abs,
            wavelet_mode=wavelet_mode,
            window_size_sec=arch.window_size_sec,
            sequence_length=arch.sequence_length,
            stride_sec=arch.stride_sec,
            sample_stride_sec=arch.sample_stride_sec,
            model_name=model_name,
            full_model_name=full_model_name,
            hyperparams=dict(arch.hyperparams),
        )


# ─── Naming / paths ─────────────────────────────────────────────────────────

def build_model_id(arch: ArchitectureSpec, *,
                   target: str, feature_set: str,
                   with_abs: bool, wavelet_mode: WaveletMode) -> str:
    """Детерминированный {architecture_id}_{hash8}.

    hash8 = blake2s(digest_size=4) от JSON-сериализации
    {hyperparams, target, feature_set, with_abs, wavelet_mode}.
    """
    payload = {
        "hyperparams": arch.hyperparams,
        "target": target,
        "feature_set": feature_set,
        "with_abs": bool(with_abs),
        "wavelet_mode": wavelet_mode,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    hash8 = hashlib.blake2s(raw, digest_size=4).hexdigest()
    return f"{arch.architecture_id}_{hash8}"


def build_fold_id(subject_id: Any) -> str:
    """fold_id = loso_subject_{subject_id}.

    Если subject_id — int, форматируется с zero-pad до 2 знаков.
    Иначе используется как есть (например, 'S007').
    """
    if isinstance(subject_id, (int,)) or (
        isinstance(subject_id, str) and subject_id.isdigit()
    ):
        return f"loso_subject_{int(subject_id):02d}"
    return f"loso_subject_{subject_id}"


def build_predictions_filename(meta: ExperimentMetadata) -> str:
    """predictions_{model_id}.parquet."""
    return f"predictions_{meta.model_id}.parquet"


def build_checkpoint_filename(meta: ExperimentMetadata, fold_id: str, ext: str) -> str:
    """model_{model_id}_fold-{fold_id}.{ext}."""
    return f"model_{meta.model_id}_fold-{fold_id}.{ext}"


def arch_dir(results_root: Path, architecture_id: str) -> Path:
    """results_root/{architecture_id}/."""
    return Path(results_root) / architecture_id


def model_dir(results_root: Path, meta: ExperimentMetadata) -> Path:
    """results_root/{architecture_id}/{model_id}/."""
    return arch_dir(results_root, meta.architecture_id) / meta.model_id


# ─── Save functions ─────────────────────────────────────────────────────────

def save_models_csv(meta: ExperimentMetadata, results_root: Path) -> Path:
    """Append-with-upsert в results_root/{architecture_id}/models.csv.

    Если model_id уже есть — старая строка удаляется, добавляется новая.
    """
    csv_path = arch_dir(results_root, meta.architecture_id) / "models.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "architecture_id": meta.architecture_id,
        "model_id": meta.model_id,
        "family": meta.family,
        "target": meta.target,
        "feature_set": meta.feature_set,
        "with_abs": meta.with_abs,
        "wavelet_mode": meta.wavelet_mode,
        "window_size_sec": meta.window_size_sec,
        "sequence_length": meta.sequence_length,
        "stride_sec": meta.stride_sec,
        "sample_stride_sec": meta.sample_stride_sec,
        "model_name": meta.model_name,
        "full_model_name": meta.full_model_name,
        "hyperparams_json": json.dumps(meta.hyperparams, sort_keys=True, ensure_ascii=False),
    }

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["model_id"] != meta.model_id]
        new_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([row])

    new_df = new_df[MODELS_CSV_COLUMNS]
    new_df.to_csv(csv_path, index=False)
    return csv_path


def save_history_csv(rows: list[dict[str, Any]],
                     model_dir_path: Path,
                     meta: ExperimentMetadata) -> Path:
    """Перезаписывает history.csv (все folds в одном файле)."""
    model_dir_path = Path(model_dir_path)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    path = model_dir_path / "history.csv"

    if not rows:
        # Пустая history допустима только если случайно не передали; пишем пустой шаблон.
        pd.DataFrame(columns=HISTORY_CSV_COLUMNS).to_csv(path, index=False)
        return path

    df = pd.DataFrame(rows)
    # Гарантируем наличие model_id / architecture_id (если runner забыл).
    df["model_id"] = df.get("model_id", meta.model_id).fillna(meta.model_id)
    df["architecture_id"] = df.get("architecture_id", meta.architecture_id).fillna(meta.architecture_id)
    missing = [c for c in HISTORY_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"history.csv: отсутствуют колонки {missing}")
    df = df[HISTORY_CSV_COLUMNS]
    df.to_csv(path, index=False)
    return path


def save_model_checkpoint(model: Any,
                          model_dir_path: Path,
                          meta: ExperimentMetadata,
                          fold_id: str) -> Path:
    """Сохраняет checkpoint per-fold.

    PyTorch nn.Module → state_dict в .pt (без metadata внутри файла);
    остальное → joblib.dump в .joblib.
    """
    model_dir_path = Path(model_dir_path)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    is_torch = (torch is not None) and isinstance(model, torch.nn.Module)
    ext = "pt" if is_torch else "joblib"
    path = model_dir_path / build_checkpoint_filename(meta, fold_id, ext)

    if is_torch:
        torch.save(model.state_dict(), path)
    else:
        joblib.dump(model, path)
    return path


def validate_predictions_dataframe(df: pd.DataFrame) -> None:
    """Проверка схемы predictions.

    • missing required columns → ValueError;
    • extra columns → warnings.warn (не ошибка);
    • NaN в PREDICTIONS_NONNULL_COLUMNS → ValueError.
    """
    cols = set(df.columns)
    required = set(PREDICTIONS_REQUIRED_COLUMNS)

    missing = sorted(required - cols)
    if missing:
        raise ValueError(f"predictions: отсутствуют обязательные колонки: {missing}")

    extra = sorted(cols - required)
    if extra:
        warnings.warn(
            f"predictions: лишние колонки будут сохранены: {extra}",
            stacklevel=2,
        )

    nan_cols = [c for c in PREDICTIONS_NONNULL_COLUMNS if df[c].isna().any()]
    if nan_cols:
        raise ValueError(f"predictions: NaN в критических колонках: {nan_cols}")


def save_predictions_parquet(df: pd.DataFrame,
                             model_dir_path: Path,
                             meta: ExperimentMetadata) -> Path:
    """Валидирует и сохраняет predictions.parquet."""
    model_dir_path = Path(model_dir_path)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    validate_predictions_dataframe(df)
    path = model_dir_path / build_predictions_filename(meta)
    # Гарантируем порядок колонок: сначала required, потом extra (если есть).
    extra_cols = [c for c in df.columns if c not in PREDICTIONS_REQUIRED_COLUMNS]
    df_out = df[PREDICTIONS_REQUIRED_COLUMNS + extra_cols]
    df_out.to_parquet(path, index=False)
    return path


# ─── Internal helpers ───────────────────────────────────────────────────────

def _build_model_name(arch: ArchitectureSpec) -> str:
    """Краткое техническое имя: {model_class_name}({k=v, ...})."""
    if not arch.hyperparams:
        return arch.model_class_name
    parts = ",".join(f"{k}={v}" for k, v in sorted(arch.hyperparams.items()))
    return f"{arch.model_class_name}({parts})"


def _build_full_model_name(arch: ArchitectureSpec, *,
                           target: str, feature_set: str,
                           with_abs: bool, wavelet_mode: WaveletMode) -> str:
    """Человекочитаемое имя: {model_name} {feature_set} {abs} wavelet={wm} {TARGET}."""
    abs_tag = "with_abs" if with_abs else "no_abs"
    return (
        f"{_build_model_name(arch)} {feature_set} {abs_tag} "
        f"wavelet={wavelet_mode} {target.upper()}"
    )
