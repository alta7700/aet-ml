"""lstm_stateful_runner — обучение stateful-LSTM с LOSO и TBPTT.

Один элемент train-набора = одна запись subject'а целиком (после
субсэмплирования с шагом internal_stride_sec). Внутри записи делаются
TBPTT-чанки длины chunk_size; (h, c) пробрасывается между чанками
с detach (для batch=1).

На test-subject: одна запись целиком, тот же chunking с пробросом state.

Без val-split, фиксированное число эпох, per-epoch grouped checkpoint,
per-epoch predictions.

Артефакты:
  results/{architecture_id}/models.csv
  results/{architecture_id}/{model_id}/history.csv
  results/{architecture_id}/{model_id}/model_{model_id}_epoch-{NN}.pt
  results/{architecture_id}/{model_id}/predictions_{model_id}.parquet
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

from architectures import get_architecture
from common_lib import (
    ExperimentMetadata, arch_dir, build_fold_id, model_dir,
    save_grouped_checkpoint, save_history_csv, save_models_csv,
    save_predictions_parquet,
)
from dataset_pipeline.common import DEFAULT_DATASET_DIR
from features import get_feature_cols, prepare_data
from models.lstm import LSTMStatefulRegressor
from training_utils import CwtCache, get_device, prepare_X_for_fold

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
BASE_STEP_SEC = 5

TARGET_COLS = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stateful LSTM runner (TBPTT, LOSO)")
    p.add_argument("--architecture", required=True, help="например LSTM7")
    p.add_argument("--target", choices=["lt1", "lt2"], default="lt1")
    p.add_argument("--feature-set",
                   choices=["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default="EMG+NIRS+HRV")
    p.add_argument("--with-abs", dest="with_abs", action="store_true", default=True)
    p.add_argument("--no-abs", dest="with_abs", action="store_false")
    p.add_argument("--wavelet-mode",
                   choices=["none", "dwt", "cwt", "wavelet_features", "wavelet_cnn"],
                   default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
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


def _subsample(X: np.ndarray, y: np.ndarray, stride_rows: int):
    """Субсэмпл по строкам с шагом stride_rows."""
    sel = np.arange(0, len(X), stride_rows, dtype=np.int64)
    return X[sel], y[sel], sel


def _build_train_records(df_train: pd.DataFrame, X_tr: np.ndarray, y_tr: np.ndarray,
                          stride_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-subject записи после субсэмплирования."""
    groups = df_train.groupby("subject_id", sort=False).indices
    records: list[tuple[np.ndarray, np.ndarray]] = []
    for _, idx in groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        X_sub, y_sub, _ = _subsample(X_tr[idx], y_tr[idx], stride_rows)
        if len(X_sub) >= 2:
            records.append((X_sub, y_sub))
    return records


def _run_record(model: nn.Module, X: np.ndarray, y: np.ndarray, chunk_size: int,
                opt, criterion, device: str, *, train: bool):
    """Один subject-record: TBPTT chunks с пробросом state.

    Возвращает (loss_per_step, predictions_concat_for_record).
    """
    h = c = None
    loss_sum = 0.0; n_steps = 0
    preds: list[np.ndarray] = []
    for st in range(0, len(X), chunk_size):
        end = min(st + chunk_size, len(X))
        x_chunk = torch.from_numpy(X[st:end]).float().unsqueeze(0).to(device)
        y_chunk = torch.from_numpy(y[st:end]).float().unsqueeze(0).to(device)
        state = (h, c) if h is not None else None
        pred, (h, c) = model(x_chunk, state, return_all=True)
        loss = criterion(pred, y_chunk)
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        preds.append(pred.detach().cpu().numpy().ravel())
        loss_sum += float(loss.item()) * (end - st)
        n_steps += (end - st)
        h = h.detach(); c = c.detach()
    return (loss_sum / max(n_steps, 1), np.concatenate(preds))


def run(args: argparse.Namespace) -> None:
    arch = get_architecture(args.architecture)
    if arch.family != "LSTM" or arch.model_class_name != "LSTMStatefulRegressor":
        raise SystemExit(
            f"{arch.architecture_id} не stateful LSTM "
            f"(model_class={arch.model_class_name})"
        )

    hp = arch.hyperparams
    int_stride_sec = int(hp["internal_stride_sec"])
    assert int_stride_sec % BASE_STEP_SEC == 0
    int_stride_rows = int_stride_sec // BASE_STEP_SEC
    hidden = int(hp["hidden_size"])
    nlayers = int(hp["num_layers"])
    dropout = float(hp["dropout"])
    lr = float(hp["lr"])
    weight_decay = float(hp["weight_decay"])
    chunk_size = int(hp.get("chunk_size", 10))
    max_epochs = int(args.max_epochs or hp["max_epochs"])
    checkpoint_every = int(args.checkpoint_every or hp["checkpoint_every_epochs"])
    checkpoint_every = max(1, checkpoint_every)

    wavelet_mode = args.wavelet_mode or arch.forced_wavelet_mode or "none"
    meta = ExperimentMetadata.from_arch(
        arch,
        target=args.target, feature_set=args.feature_set,
        with_abs=args.with_abs, wavelet_mode=wavelet_mode,
    )
    # Stateful LSTM с пробросом (h, c) ломает MPS sliceDimension; CUDA либо CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_all(args.seed)

    print(f"[{arch.architecture_id}] {arch.architecture_name}")
    print(f"  model_id={meta.model_id}  device={device}")
    print(f"  target={meta.target}  feature_set={meta.feature_set}  "
          f"with_abs={meta.with_abs}  wavelet_mode={meta.wavelet_mode}")
    print(f"  stride={int_stride_sec}s  chunk_size={chunk_size}  "
          f"hidden={hidden}x{nlayers}")
    print(f"  max_epochs={max_epochs}  checkpoint_every={checkpoint_every}")

    md = model_dir(args.results_root, meta)
    md.mkdir(parents=True, exist_ok=True)
    save_models_csv(meta, args.results_root)

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
          f"n_features={len(feat_cols)}{f' (+CWT {cwt.n_features})' if cwt else ''}")

    subjects = sorted(df_prep["subject_id"].unique())
    criterion = nn.MSELoss()

    pred_rows: list[pd.DataFrame] = []
    history_rows: list[dict] = []
    epoch_states: dict[int, dict[str, dict]] = {}

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

        train_records = _build_train_records(train_df, X_tr, y_tr, int_stride_rows)
        if not train_records:
            print(f"  [skip {fold_id}] нет train-records")
            continue

        # Test subsample.
        X_te_sub, _, te_sel = _subsample(X_te, y_te_raw, int_stride_rows)
        if len(X_te_sub) < 2:
            print(f"  [skip {fold_id}] нет test-records")
            continue
        y_te_sub = y_te_raw[te_sel]
        win_start_sub = test_df["window_start_sec"].astype(float).values[te_sel]
        subj_te = test_df["subject_id"].values[te_sel]

        model = LSTMStatefulRegressor(
            input_size=X_tr.shape[1],
            hidden_size=hidden, num_layers=nlayers, dropout=dropout,
        ).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        rng = np.random.default_rng(args.seed + hash(fold_id) % 10000)
        n_records = len(train_records)
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = rng.permutation(n_records)
            loss_sum_e = 0.0; mae_sum_e = 0.0; n_e = 0
            for i in order:
                X, y = train_records[i]
                rec_loss, rec_pred = _run_record(
                    model, X, y, chunk_size, opt, criterion, device, train=True,
                )
                loss_sum_e += rec_loss * len(X)
                mae_sum_e += float(np.mean(np.abs(rec_pred - y))) * len(X)
                n_e += len(X)

            history_rows.append({
                "model_id": meta.model_id,
                "architecture_id": meta.architecture_id,
                "fold_id": fold_id,
                "epoch": epoch,
                "train_loss": loss_sum_e / max(n_e, 1),
                "val_loss": float("nan"),
                "train_mae": mae_sum_e / max(n_e, 1),
                "val_mae": float("nan"),
                "lr": lr,
            })

            is_ckpt = epoch % checkpoint_every == 0 or epoch == max_epochs
            if is_ckpt:
                epoch_states.setdefault(epoch, {})[fold_id] = {
                    k: v.detach().to("cpu").clone()
                    for k, v in model.state_dict().items()
                }
                model.eval()
                with torch.no_grad():
                    _, pred_norm = _run_record(
                        model, X_te_sub, y_sc.transform(y_te_sub.reshape(-1, 1)).ravel().astype(np.float32),
                        chunk_size, opt, criterion, device, train=False,
                    )
                y_pred_real = y_sc.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
                n_pred = len(y_pred_real)
                pred_rows.append(pd.DataFrame({
                    "model_id": meta.model_id,
                    "fold_id": fold_id,
                    "subject_id": subj_te,
                    "epoch": int(epoch),
                    "window_size_sec": meta.window_size_sec,
                    "sequence_length": meta.sequence_length,
                    "stride_sec": meta.stride_sec,
                    "sample_stride_sec": meta.sample_stride_sec,
                    "sample_index": np.arange(n_pred, dtype=np.int64),
                    "sample_start_sec": win_start_sub.astype(float),
                    "sample_end_sec": (win_start_sub + float(meta.window_size_sec)).astype(float),
                    "y_true": y_te_sub.astype(float),
                    "y_pred": y_pred_real.astype(float),
                }))

        # Финальный MAE на последней эпохе.
        final_df = pred_rows[-1] if pred_rows else None
        mae_min = (float(np.mean(np.abs(final_df["y_true"] - final_df["y_pred"]))) / 60.0
                   if final_df is not None and final_df["fold_id"].iloc[0] == fold_id
                   else float("nan"))
        elapsed = time.perf_counter() - t0
        n_ckpts = sum(1 for e in range(1, max_epochs + 1)
                      if e % checkpoint_every == 0 or e == max_epochs)
        print(f"  fold {fold_id:<22s}  n={len(X_te_sub):<4d}  "
              f"MAE_final={mae_min:.3f} мин  ckpts={n_ckpts}  ({elapsed:.1f}s)")

    save_history_csv(history_rows, md, meta)
    if pred_rows:
        preds = pd.concat(pred_rows, ignore_index=True)
        save_predictions_parquet(preds, md, meta)
    else:
        preds = pd.DataFrame()

    for epoch in sorted(epoch_states.keys()):
        save_grouped_checkpoint(epoch_states[epoch], md, meta, epoch=epoch)

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
        print(f"  → {md}/predictions_{meta.model_id}.parquet  ({len(preds)} строк)")
        print(f"  → {md}/model_{meta.model_id}_epoch-*.pt  ({len(epoch_states)} файлов)")


if __name__ == "__main__":
    run(parse_args())
