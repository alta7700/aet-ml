"""Discovery training-артефактов: models.csv и predictions parquet'ов."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from common_lib import MODELS_CSV_COLUMNS


@dataclass(frozen=True)
class DiscoveryResult:
    """Результат сканирования results/."""

    # Объединённый DataFrame всех models.csv с колонкой path к каждому файлу.
    models_df: pd.DataFrame
    # model_id → абсолютный путь к predictions_{model_id}.parquet
    predictions_paths: dict[str, Path]
    # model_id → путь к каталогу model_id (для будущей загрузки чекпоинтов и т.п.)
    model_dirs: dict[str, Path]
    # architecture_id → путь к каталогу архитектуры
    architecture_dirs: dict[str, Path]
    # Список model_id, для которых не нашлось predictions parquet.
    missing_predictions: list[str] = field(default_factory=list)


def discover_artifacts(results_root: Path) -> DiscoveryResult:
    """Обходит ``results/{architecture_id}/`` и собирает models.csv + predictions.

    Возвращает unified models_df + словари путей. Не валидирует содержимое —
    только структуру каталогов. Полная валидация — в analysis/validation.py.
    """
    results_root = Path(results_root).resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"results_root не найден: {results_root}")

    rows: list[pd.DataFrame] = []
    preds: dict[str, Path] = {}
    model_dirs: dict[str, Path] = {}
    arch_dirs: dict[str, Path] = {}
    missing: list[str] = []

    for arch_dir in sorted(results_root.iterdir()):
        if not arch_dir.is_dir():
            continue
        models_csv = arch_dir / "models.csv"
        if not models_csv.exists():
            continue

        df = pd.read_csv(models_csv)
        df["__models_csv_path"] = str(models_csv)
        rows.append(df)
        arch_dirs[arch_dir.name] = arch_dir

        for model_id in df["model_id"].astype(str):
            md = arch_dir / model_id
            if not md.is_dir():
                missing.append(model_id)
                continue
            model_dirs[model_id] = md
            pp = md / f"predictions_{model_id}.parquet"
            if pp.exists():
                preds[model_id] = pp
            else:
                missing.append(model_id)

    if not rows:
        raise FileNotFoundError(
            f"Не найдено ни одного models.csv в {results_root}")

    models_df = pd.concat(rows, ignore_index=True)
    # Приведение типов с защитой от строковых True/False.
    if "with_abs" in models_df.columns:
        models_df["with_abs"] = models_df["with_abs"].map(_to_bool)

    # Sanity: должны присутствовать обязательные колонки.
    missing_cols = [c for c in MODELS_CSV_COLUMNS if c not in models_df.columns]
    if missing_cols:
        raise ValueError(
            f"models.csv не содержит колонок: {missing_cols}. "
            f"Полученные колонки: {list(models_df.columns)}")

    return DiscoveryResult(
        models_df=models_df,
        predictions_paths=preds,
        model_dirs=model_dirs,
        architecture_dirs=arch_dirs,
        missing_predictions=missing,
    )


def _to_bool(value) -> bool:
    """Парсит True/False/'True'/'False'/1/0 в bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise ValueError(f"Не удалось распарсить bool из {value!r}")
