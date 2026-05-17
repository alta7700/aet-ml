"""lstm_runner — обучение одной LSTM*-архитектуры с LOSO.

Без val-split: LOSO — единственный механизм оценки.
На каждом fold обучаем фиксированное число эпох (max_epochs).
На каждых checkpoint_every_epochs:
  • сохраняем state_dict в RAM (для последующего grouped save);
  • делаем forward pass на test-subject и накапливаем строки в predictions
    с колонкой epoch.

После всех LOSO-fold'ов:
  • save_grouped_checkpoint per epoch — один .pt файл со state'ами всех folds;
  • predictions.parquet содержит N_epoch_checkpoints × N_samples строк.

Артефакты:
  results/{architecture_id}/models.csv
  results/{architecture_id}/{model_id}/history.csv
  results/{architecture_id}/{model_id}/model_{model_id}_epoch-{NN}.pt   (один файл на эпоху)
  results/{architecture_id}/{model_id}/predictions_{model_id}.parquet   (с колонкой epoch)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

from architectures import get_architecture
from common_lib import (
    ExperimentMetadata, arch_dir, build_fold_id, model_dir,
    save_grouped_checkpoint, save_history_csv, save_models_csv,
    save_predictions_parquet,
)
from dataset_pipeline.common import DEFAULT_DATASET_DIR
from features import get_feature_cols, prepare_data
from models.lstm import (
    AttentionLSTMRegressor, LSTMRegressor, StatelessSeqDataset,
)

_LSTM_BUILDERS = {
    "LSTMRegressor": lambda input_size, hp: LSTMRegressor(
        input_size=input_size,
        hidden_size=int(hp["hidden_size"]),
        num_layers=int(hp["num_layers"]),
        dropout=float(hp["dropout"]),
    ),
    "AttentionLSTMRegressor": lambda input_size, hp: AttentionLSTMRegressor(
        input_size=input_size,
        hidden_size=int(hp["hidden_size"]),
        num_layers=int(hp["num_layers"]),
        dropout=float(hp["dropout"]),
    ),
}
from training_utils import CwtCache, get_device, prepare_X_for_fold

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
BASE_STEP_SEC = 5

TARGET_COLS = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM runner (LOSO без val-split)")
    p.add_argument("--architecture", required=True, help="например LSTM1")
    p.add_argument("--target", choices=["lt1", "lt2"], default="lt1")
    p.add_argument("--feature-set",
                   choices=["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default="EMG+NIRS+HRV")
    p.add_argument("--with-abs", dest="with_abs", action="store_true", default=True)
    p.add_argument("--no-abs", dest="with_abs", action="store_false")
    p.add_argument("--wavelet-mode",
                   choices=["none", "dwt", "cwt", "wavelet_features", "wavelet_cnn"],
                   default=None,
                   help="по умолчанию берётся forced_wavelet_mode архитектуры")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    p.add_argument("--session-params", type=Path,
                   default=DEFAULT_DATASET_DIR / "session_params.parquet")
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return p.parse_args()


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_dataset(df_train: pd.DataFrame, X_tr: np.ndarray, y_tr: np.ndarray,
                        seq_len: int, internal_stride_rows: int,
                        outer_stride_rows: int) -> ConcatDataset | None:
    """Per-subject склейка StatelessSeqDataset."""
    groups = df_train.groupby("subject_id", sort=False).indices
    parts: list[StatelessSeqDataset] = []
    span = (seq_len - 1) * internal_stride_rows + 1
    for _, idx in groups.items():
        idx = np.asarray(idx)
        if len(idx) < span:
            continue
        ds = StatelessSeqDataset(
            X_tr[idx], y_tr[idx],
            seq_len, internal_stride_rows, outer_stride_rows,
        )
        if len(ds) > 0:
            parts.append(ds)
    if not parts:
        return None
    return ConcatDataset(parts)


def _predict_on_test(model: nn.Module, test_ds: StatelessSeqDataset,
                     batch_size: int, device: str) -> np.ndarray:
    """Forward pass на готовом StatelessSeqDataset, возвращает y_pred_norm."""
    if len(test_ds) == 0:
        return np.array([], dtype=np.float32)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for X, _ in loader:
            pred = model(X.to(device))
            out.append(pred.cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def _make_test_pred_rows(y_pred: np.ndarray, end_pos: np.ndarray,
                        test_df: pd.DataFrame, y_te_raw: np.ndarray,
                        y_sc: StandardScaler, meta: ExperimentMetadata,
                        fold_id: str, epoch: int, seq_len: int,
                        int_stride_sec: int) -> pd.DataFrame:
    """Собирает predictions DataFrame для одного (fold, epoch)."""
    if len(y_pred) == 0:
        return pd.DataFrame()
    y_pred_real = y_sc.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_true = y_te_raw[end_pos]
    win_start = test_df["window_start_sec"].astype(float).values
    sample_start_sec = win_start[end_pos] - (seq_len - 1) * int_stride_sec
    sample_end_sec = win_start[end_pos] + float(meta.window_size_sec)
    n = len(y_pred_real)
    return pd.DataFrame({
        "model_id": meta.model_id,
        "fold_id": fold_id,
        "subject_id": test_df["subject_id"].values[end_pos],
        "epoch": int(epoch),
        "window_size_sec": meta.window_size_sec,
        "sequence_length": meta.sequence_length,
        "stride_sec": meta.stride_sec,
        "sample_stride_sec": meta.sample_stride_sec,
        "sample_index": np.arange(n, dtype=np.int64),
        "sample_start_sec": sample_start_sec.astype(float),
        "sample_end_sec": sample_end_sec.astype(float),
        "y_true": y_true.astype(float),
        "y_pred": y_pred_real.astype(float),
    })


def _train_one_fold(model: nn.Module, loader: DataLoader,
                    test_ds: StatelessSeqDataset,
                    *, max_epochs: int, checkpoint_every: int,
                    lr: float, weight_decay: float, device: str,
                    batch_size: int,
                    fold_id: str, meta: ExperimentMetadata,
                    test_df: pd.DataFrame, y_te_raw: np.ndarray,
                    y_sc: StandardScaler, seq_len: int, int_stride_sec: int,
                    int_stride_rows: int):
    """LOSO fold: обучение + per-checkpoint predict + аккумулирование state'ов.

    Возвращает (history_rows, states_by_epoch, pred_frames).
      states_by_epoch: dict[epoch, state_dict_cpu]
      pred_frames: list[pd.DataFrame] — predictions для каждой checkpoint-эпохи.
    """
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    states_by_epoch: dict[int, dict] = {}
    pred_frames: list[pd.DataFrame] = []
    end_pos = test_ds.starts + (seq_len - 1) * int_stride_rows

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0; mae_sum = 0.0; n = 0
        for X, y in loader:
            X = X.to(device); y = y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = y.size(0)
            loss_sum += float(loss.item()) * bs
            mae_sum += float((pred - y).abs().mean().item()) * bs
            n += bs

        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(n, 1),
            "val_loss": float("nan"),
            "train_mae": mae_sum / max(n, 1),
            "val_mae": float("nan"),
            "lr": lr,
        })

        is_ckpt = epoch % checkpoint_every == 0 or epoch == max_epochs
        if is_ckpt:
            # State_dict на CPU (mps tensors могут не сериализоваться в .pt напрямую).
            states_by_epoch[epoch] = {
                k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()
            }
            y_pred_norm = _predict_on_test(model, test_ds, batch_size, device)
            df_frame = _make_test_pred_rows(
                y_pred_norm, end_pos, test_df, y_te_raw, y_sc,
                meta, fold_id, epoch, seq_len, int_stride_sec,
            )
            if len(df_frame) > 0:
                pred_frames.append(df_frame)

    return history, states_by_epoch, pred_frames


def run(args: argparse.Namespace) -> None:
    arch = get_architecture(args.architecture)
    if arch.family != "LSTM":
        raise SystemExit(
            f"lstm_runner работает только с family='LSTM', "
            f"получено {arch.family!r} для {arch.architecture_id}"
        )
    if arch.model_class_name == "LSTMStatefulRegressor":
        raise SystemExit(
            f"{arch.architecture_id} — stateful архитектура; "
            f"используйте lstm_stateful_runner.py"
        )

    hp = arch.hyperparams
    seq_len = int(hp["seq_len"])
    int_stride_sec = int(hp["internal_stride_sec"])
    out_stride_sec = int(hp["outer_stride_sec"])
    assert int_stride_sec % BASE_STEP_SEC == 0
    assert out_stride_sec % BASE_STEP_SEC == 0
    int_stride_rows = int_stride_sec // BASE_STEP_SEC
    out_stride_rows = out_stride_sec // BASE_STEP_SEC
    hidden = int(hp["hidden_size"])
    nlayers = int(hp["num_layers"])
    dropout = float(hp["dropout"])
    lr = float(hp["lr"])
    weight_decay = float(hp["weight_decay"])
    batch_size = int(hp["batch_size"])
    max_epochs = int(args.max_epochs or hp["max_epochs"])
    checkpoint_every = int(args.checkpoint_every or hp["checkpoint_every_epochs"])
    checkpoint_every = max(1, checkpoint_every)

    wavelet_mode = args.wavelet_mode or arch.forced_wavelet_mode or "none"
    meta = ExperimentMetadata.from_arch(
        arch,
        target=args.target, feature_set=args.feature_set,
        with_abs=args.with_abs, wavelet_mode=wavelet_mode,
    )
    device = get_device()
    _seed_all(args.seed)

    print(f"[{arch.architecture_id}] {arch.architecture_name}")
    print(f"  model_id={meta.model_id}  device={device}")
    print(f"  target={meta.target}  feature_set={meta.feature_set}  "
          f"with_abs={meta.with_abs}  wavelet_mode={meta.wavelet_mode}")
    print(f"  seq_len={seq_len}  int_stride={int_stride_sec}s  "
          f"out_stride={out_stride_sec}s  hidden={hidden}x{nlayers}")
    print(f"  max_epochs={max_epochs}  checkpoint_every={checkpoint_every}")

    md = model_dir(args.results_root, meta)
    md.mkdir(parents=True, exist_ok=True)
    save_models_csv(meta, args.results_root)
    print(f"  → {arch_dir(args.results_root, arch.architecture_id) / 'models.csv'}")

    df_raw = pd.read_parquet(args.dataset)
    session_params = pd.read_parquet(args.session_params) if args.session_params.exists() else pd.DataFrame()

    df_prep = prepare_data(df_raw, session_params, meta.target)
    target_col = TARGET_COLS[meta.target]
    df_prep = df_prep.dropna(subset=[target_col])
    feat_cols = get_feature_cols(df_prep, meta.feature_set, with_abs=meta.with_abs)
    if not feat_cols:
        raise SystemExit(f"Пустой feature_set для {meta.feature_set}")

    cwt: Optional[CwtCache] = CwtCache() if meta.wavelet_mode == "cwt" else None

    print(f"  n_subjects={df_prep['subject_id'].nunique()}  "
          f"n_features={len(feat_cols)}{f' (+CWT {cwt.n_features})' if cwt else ''}  "
          f"n_windows={len(df_prep)}")

    subjects = sorted(df_prep["subject_id"].unique())
    pred_rows: list[pd.DataFrame] = []
    history_rows: list[dict] = []
    epoch_states: dict[int, dict[str, dict]] = {}   # epoch → fold_id → state_dict

    t_total = time.perf_counter()
    for test_s in subjects:
        fold_id = build_fold_id(test_s)
        t0 = time.perf_counter()

        train_df = df_prep[df_prep["subject_id"] != test_s].sort_values(
            ["subject_id", "window_start_sec"]
        )
        test_df = df_prep[df_prep["subject_id"] == test_s].sort_values(
            "window_start_sec"
        ).reset_index()
        orig_test_index = pd.Index(test_df.pop("index").values)

        X_tr, X_te = prepare_X_for_fold(
            df_prep, feat_cols, train_df.index, orig_test_index, cwt
        )
        y_tr_raw = train_df[target_col].values.astype(np.float32)
        y_te_raw = test_df[target_col].values.astype(np.float32)
        y_sc = StandardScaler()
        y_tr = y_sc.fit_transform(y_tr_raw.reshape(-1, 1)).ravel().astype(np.float32)
        y_te_norm = y_sc.transform(y_te_raw.reshape(-1, 1)).ravel().astype(np.float32)

        train_ds = _build_train_dataset(
            train_df, X_tr, y_tr,
            seq_len, int_stride_rows, out_stride_rows,
        )
        if train_ds is None:
            print(f"  [skip {fold_id}] недостаточно train-данных")
            continue
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=False,
        )
        test_ds = StatelessSeqDataset(
            X_te, y_te_norm, seq_len, int_stride_rows, out_stride_rows,
        )
        if len(test_ds) == 0:
            print(f"  [skip {fold_id}] недостаточно test-окон")
            continue

        builder = _LSTM_BUILDERS.get(arch.model_class_name)
        if builder is None:
            raise SystemExit(
                f"lstm_runner не поддерживает {arch.model_class_name!r}. "
                f"Для stateful используйте lstm_stateful_runner.py"
            )
        model = builder(X_tr.shape[1], arch.hyperparams).to(device)

        fold_history, fold_states, fold_preds = _train_one_fold(
            model, train_loader, test_ds,
            max_epochs=max_epochs, checkpoint_every=checkpoint_every,
            lr=lr, weight_decay=weight_decay, device=device,
            batch_size=batch_size,
            fold_id=fold_id, meta=meta,
            test_df=test_df, y_te_raw=y_te_raw, y_sc=y_sc,
            seq_len=seq_len, int_stride_sec=int_stride_sec,
            int_stride_rows=int_stride_rows,
        )

        for row in fold_history:
            history_rows.append({
                "model_id": meta.model_id,
                "architecture_id": meta.architecture_id,
                "fold_id": fold_id,
                **row,
            })
        for ep, state in fold_states.items():
            epoch_states.setdefault(ep, {})[fold_id] = state
        pred_rows.extend(fold_preds)

        # Метрика последней эпохи для лога.
        final_df = fold_preds[-1] if fold_preds else None
        if final_df is not None:
            mae_min = float(np.mean(np.abs(final_df["y_true"] - final_df["y_pred"]))) / 60.0
        else:
            mae_min = float("nan")
        elapsed = time.perf_counter() - t0
        print(f"  fold {fold_id:<22s}  n={len(final_df) if final_df is not None else 0:<4d}  "
              f"MAE_final={mae_min:.3f} мин  "
              f"ckpts={len(fold_states)}  ({elapsed:.1f}s)")

    # ── Сохранение артефактов ────────────────────────────────────────────────
    save_history_csv(history_rows, md, meta)
    if pred_rows:
        preds = pd.concat(pred_rows, ignore_index=True)
        save_predictions_parquet(preds, md, meta)
    else:
        preds = pd.DataFrame()

    for epoch in sorted(epoch_states.keys()):
        save_grouped_checkpoint(epoch_states[epoch], md, meta, epoch=epoch)

    # Краткая сводка по эпохам.
    if not preds.empty:
        per_epoch = preds.groupby("epoch").apply(
            lambda g: float(np.mean(np.abs(g["y_true"] - g["y_pred"]))) / 60.0
        )
        best_epoch = int(per_epoch.idxmin())
        print(f"\n  Всего {len(subjects)} folds, "
              f"{time.perf_counter() - t_total:.1f}s")
        print(f"  MAE по эпохам (мин):")
        for ep, mae in per_epoch.items():
            mark = "  ← best" if ep == best_epoch else ""
            print(f"    epoch={ep:>3d}  MAE={mae:.3f}{mark}")
        print(f"  → {md}/history.csv  ({len(history_rows)} строк)")
        print(f"  → {md}/predictions_{meta.model_id}.parquet  ({len(preds)} строк, "
              f"{preds['epoch'].nunique()} эпох × ~{len(preds) // max(preds['epoch'].nunique(), 1)} samples)")
        print(f"  → {md}/model_{meta.model_id}_epoch-*.pt  ({len(epoch_states)} файлов)")


if __name__ == "__main__":
    run(parse_args())
