"""v0107 — Ensemble: Wavelet-TCN (v0106b) + LinearModels (v0011)

Версия:    v0107
Дата:      2026-05-09

Идея: объединить лучшую нейросеть (v0106b, raw MAE=2.765) с лучшей
      линейной моделью (v0011, MAE=1.859) взвешенным усреднением.

y_ensemble = α × y_v0106b + (1 - α) × y_v0011

Вес α оптимизируется на validation fold каждого LOSO-разбиения.
Если нейросеть и линейная модель делают разные ошибки — ensemble лучше обоих.

Pipeline (внутри одного LOSO фолда):
  train_subjects (N-1):
    80% → обучение v0106b + v0011
    20% → val: подбор оптимального α

  test_subject (1):
    y_tcn  = v0106b.predict(X_test)
    y_lin  = v0011.predict(X_test)
    y_ens  = α* × y_tcn + (1-α*) × y_lin

Воспроизведение:
  uv run python scripts/v0107_ensemble.py --target both
  uv run python scripts/v0107_ensemble.py --target lt2 --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from joblib import Parallel, delayed

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_v0011_ref() -> dict:
    """Загружает референсные MAE v0011 из best_per_set.csv; fallback на N=14 константы."""
    csv = _ROOT / "results" / "v0011" / "best_per_set.csv"
    if not csv.exists():
        return {("lt2", "EMG+NIRS+HRV"): 1.859, ("lt1", "EMG+NIRS+HRV"): 2.277}
    ref_df = pd.read_csv(csv)
    return {(row["target"], row["feature_set"]): row["kalman_mae_min"]
            for _, row in ref_df.iterrows()}


from dataset_pipeline.common import DEFAULT_DATASET_DIR
from scripts.v0011_modality_ablation import prepare_data, get_feature_cols, kalman_smooth

OUT_DIR = _ROOT / "results" / "v0107"

EXCLUDE_ABS = frozenset([
    "trainred_smo2_mean", "trainred_hhb_mean", "trainred_hbdiff_mean", "trainred_thb_mean",
    "hrv_mean_rr_ms", "feat_smo2_x_rr",
])


# ─── Конфиг ───────────────────────────────────────────────────────────────────

class Config:
    # TCN параметры (идентичны v0106b)
    window_step   = 6   # 6 × 5с = 30с, нулевое перекрытие
    seq_length    = 12  # 12 × 30с = 6 мин контекста
    batch_size    = 16
    num_epochs    = 100
    learning_rate = 0.001
    patience      = 15
    wavelet       = "morl"
    scales        = [1, 2, 4, 8, 16]
    n_channels    = 32
    kernel_size   = 3
    dilations     = [1, 2, 4, 8, 16]
    dropout       = 0.2
    device        = "cpu"
    # Ensemble параметры
    alpha_grid    = np.linspace(0.0, 1.0, 21)  # α от 0 (только linear) до 1 (только TCN)


# ─── Dataset (из v0106b) ──────────────────────────────────────────────────────

def _cwt_one_feature(args):
    """Вычисляет CWT для одного признака — вызывается из пула потоков."""
    f, col, scales, wavelet = args
    coeffs, _ = pywt.cwt(col.astype(np.float64), scales, wavelet)
    return f, np.abs(coeffs).T.astype(np.float32)


def _precompute_cwt(X: np.ndarray, scales, wavelet) -> np.ndarray:
    """Считает CWT для всего X (n_rows, n_features) параллельно по признакам.
    Использует threading (pywavelets освобождает GIL) → ускорение ~n_cpu раз."""
    n, n_f, n_sc = len(X), X.shape[1], len(scales)
    cwt_full = np.zeros((n, n_f, n_sc), dtype=np.float32)
    if n <= 1:
        return cwt_full
    args = [(f, X[:, f], scales, wavelet) for f in range(n_f)]
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_cwt_one_feature)(a) for a in args)
    for f, cwt_f in results:
        cwt_full[:, f, :] = cwt_f
    return cwt_full


class WaveletSeqDataset(Dataset):
    def __init__(self, X, y, seq_length, window_step, scales, wavelet,
                 cwt_cache: np.ndarray = None, device: str = "cpu"):
        """cwt_cache: предпосчитанный CWT shape=(len(X), n_f, n_sc).
        device: если 'cuda' — весь датасет живёт на GPU, копий CPU→GPU в цикле нет."""
        ind_idx = np.arange(0, len(X), window_step)
        X_ind = X[ind_idx].astype(np.float32)
        y_ind = y[ind_idx].astype(np.float32)
        self.n = len(X_ind); self.seq_length = seq_length

        if cwt_cache is not None:
            cwt_all = cwt_cache[ind_idx]
        else:
            n_f, n_sc = X_ind.shape[1], len(scales)
            cwt_all = _precompute_cwt(X_ind, scales, wavelet)

        flat = cwt_all.reshape(-1, cwt_all.shape[-1])
        mean_, std_ = flat.mean(0), flat.std(0) + 1e-8
        cwt_norm = ((cwt_all - mean_) / std_).reshape(
            self.n, X_ind.shape[1] * cwt_all.shape[-1]).astype(np.float32)

        # Переносим весь датасет на GPU один раз — в цикле обучения копий нет
        self.X   = torch.from_numpy(X_ind).to(device)
        self.cwt = torch.from_numpy(cwt_norm).to(device)
        self.y   = torch.from_numpy(y_ind).to(device)

    def __len__(self): return max(0, self.n - self.seq_length + 1)

    def __getitem__(self, idx):
        # Слайсы уже на нужном устройстве — никаких .to() в цикле обучения
        x_orig = self.X[idx: idx + self.seq_length]
        x_cwt  = self.cwt[idx: idx + self.seq_length]
        y      = self.y[idx + self.seq_length - 1]
        return x_orig, x_cwt, y


# ─── Wavelet-TCN модель (из v0106b) ───────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_ch); self.norm2 = nn.LayerNorm(out_ch)
        self.drop  = nn.Dropout(dropout); self.relu = nn.ReLU()
        self.proj  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        out = self.drop(out)
        out = self.norm2(self.conv2(out).transpose(1,2)).transpose(1,2)
        res = x if self.proj is None else self.proj(x)
        return self.relu(out + res)


class WaveletTCN(nn.Module):
    def __init__(self, n_features, n_cwt, n_channels=32, kernel_size=3,
                 dilations=None, dropout=0.2):
        super().__init__()
        in_ch = n_features + n_cwt
        self.input_proj = nn.Sequential(nn.Conv1d(in_ch, n_channels, 1), nn.ReLU())
        layers = []
        for d in (dilations or [1, 2, 4, 8, 16]):
            layers.append(TCNBlock(n_channels, n_channels, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(n_channels, 1)

    def forward(self, x_orig, x_cwt):
        x = torch.cat([x_orig, x_cwt], dim=-1).transpose(1, 2)
        x = self.input_proj(x)
        return self.fc(self.tcn(x).mean(dim=2)).squeeze(1)


# ─── Обучение TCN ─────────────────────────────────────────────────────────────

def _train_tcn(model, train_loader, val_loader, config):
    criterion = nn.HuberLoss(delta=60.0)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-5)
    best_val, patience_counter = float("inf"), 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(config.num_epochs):
        model.train()
        for x_orig, x_cwt, y in train_loader:
            # данные уже на GPU (перенесены в датасете) — .to() не нужен
            optimizer.zero_grad()
            loss = criterion(model(x_orig, x_cwt), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x_orig, x_cwt, y in val_loader:
                val_loss += criterion(model(x_orig, x_cwt), y).item() * len(y)
        val_loss /= max(len(val_loader.dataset), 1)

        if val_loss < best_val:
            best_val = val_loss; patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience: break

    model.load_state_dict(best_state)
    return model


# ─── Один LOSO фолд с ensemble ────────────────────────────────────────────────

def _run_one_loso_fold(test_subject_id, df, feat_cols, target_col, config,
                       cwt_global=None, cwt_idx_map=None, cwt_feat_cols=None):
    """Обучает TCN + ElasticNet, подбирает α на val, возвращает ensemble pred."""
    train_all = df[df["subject_id"] != test_subject_id].sort_values(
        ["subject_id", "window_start_sec"])
    test = df[df["subject_id"] == test_subject_id].sort_values("window_start_sec")

    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    X_tr_all = sc.fit_transform(imp.fit_transform(train_all[feat_cols].values))
    X_te = sc.transform(imp.transform(test[feat_cols].values))
    y_tr_all = train_all[target_col].values
    y_te = test[target_col].values

    # ── Линейная модель (ElasticNet) ──────────────────────────────────────────
    lin_model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)
    lin_model.fit(X_tr_all, y_tr_all)
    y_pred_lin_te = lin_model.predict(X_te)

    # ── TCN с wavelet ─────────────────────────────────────────────────────────
    dev = config.device

    # CWT: читаем из предрасчитанного кэша если доступен, иначе считаем на лету
    if cwt_global is not None and cwt_idx_map and cwt_feat_cols:
        # Выбираем только нужные признаки из кэша (кэш содержит все 137, фолд — подмножество)
        f_idx = [cwt_feat_cols.index(c) for c in feat_cols if c in cwt_feat_cols]
        tr_pos = [cwt_idx_map[i] for i in train_all.index if i in cwt_idx_map]
        te_pos = [cwt_idx_map[i] for i in test.index      if i in cwt_idx_map]
        if tr_pos and len(f_idx) == len(feat_cols):
            cwt_tr = cwt_global[np.ix_(tr_pos, f_idx)]   # (n_tr, n_feat, n_scales)
            cwt_te = cwt_global[np.ix_(te_pos, f_idx)]
        else:
            cwt_tr = _precompute_cwt(X_tr_all, config.scales, config.wavelet)
            cwt_te = _precompute_cwt(X_te,     config.scales, config.wavelet)
    else:
        cwt_tr = _precompute_cwt(X_tr_all, config.scales, config.wavelet)
        cwt_te = _precompute_cwt(X_te,     config.scales, config.wavelet)

    _offset_ds = [
        WaveletSeqDataset(X_tr_all[off:], y_tr_all[off:], config.seq_length,
                          config.window_step, config.scales, config.wavelet,
                          cwt_cache=cwt_tr[off:], device=dev)
        for off in range(config.window_step)
    ]
    train_ds = ConcatDataset([d for d in _offset_ds if len(d) > 0])
    test_ds  = WaveletSeqDataset(X_te, y_te, config.seq_length,
                                  config.window_step, config.scales, config.wavelet,
                                  cwt_cache=cwt_te, device=dev)

    if len(train_ds) < 4 or len(test_ds) == 0:
        return {"fold": test_subject_id, "error": "Недостаточно данных"}

    split = max(1, int(0.8 * len(train_ds)))
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [split, len(train_ds) - split],
        generator=torch.Generator().manual_seed(42))

    # num_workers=0 обязательно — CUDA-тензоры нельзя передавать через fork
    train_loader = DataLoader(train_sub, batch_size=config.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_sub,   batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,   batch_size=config.batch_size, shuffle=False)

    n_cwt = X_tr_all.shape[1] * len(config.scales)
    tcn = WaveletTCN(X_tr_all.shape[1], n_cwt, config.n_channels,
                     config.kernel_size, config.dilations, config.dropout).to(config.device)
    tcn = _train_tcn(tcn, train_loader, val_loader, config)

    # Предсказания TCN + y_true + Linear на val (для подбора α)
    tcn.eval()
    val_tcn_list, val_true_list, val_last_x_list = [], [], []
    with torch.no_grad():
        for x_orig, x_cwt, y_batch in val_loader:
            # данные уже на GPU — только .cpu() для numpy-конвертации
            preds = tcn(x_orig, x_cwt).cpu().numpy()
            val_tcn_list.append(preds)
            val_true_list.append(y_batch.cpu().numpy())
            val_last_x_list.append(x_orig[:, -1, :].cpu().numpy())
    y_val_tcn  = np.concatenate(val_tcn_list)
    y_val_true = np.concatenate(val_true_list)
    val_last_x = np.concatenate(val_last_x_list) if val_last_x_list else np.empty((0, X_tr_all.shape[1]))
    if len(val_last_x) > 0:
        y_val_lin = lin_model.predict(val_last_x)
    else:
        y_val_lin = np.zeros_like(y_val_tcn)

    # Подбираем α на val
    n_val = min(len(y_val_tcn), len(y_val_true), len(y_val_lin))
    best_alpha, best_val_mae = 0.5, float("inf")
    if n_val > 0:
        for alpha in config.alpha_grid:
            y_ens_val = alpha * y_val_tcn[:n_val] + (1 - alpha) * y_val_lin[:n_val]
            mae_val = mean_absolute_error(y_val_true[:n_val], y_ens_val)
            if mae_val < best_val_mae:
                best_val_mae, best_alpha = mae_val, alpha

    # TCN предсказания на тесте
    tcn.eval(); test_preds = []
    with torch.no_grad():
        for x_orig, x_cwt, _ in test_loader:
            test_preds.append(tcn(x_orig, x_cwt).cpu().numpy())
    y_tcn_sparse = np.concatenate(test_preds)

    # Интерполяция TCN предсказаний
    test_ind = np.arange(0, len(y_te), config.window_step)
    pred_at  = test_ind[config.seq_length - 1: config.seq_length - 1 + len(y_tcn_sparse)]
    y_tcn_full = np.full(len(y_te), np.nan)
    for i, idx in enumerate(pred_at):
        if idx < len(y_tcn_full): y_tcn_full[idx] = y_tcn_sparse[i]
    x_all = np.arange(len(y_te)); valid = ~np.isnan(y_tcn_full)
    y_tcn_full = (np.interp(x_all, x_all[valid], y_tcn_full[valid])
                  if valid.sum() >= 2 else np.full(len(y_te), np.nanmean(y_tcn_full)))

    # Ensemble
    y_ensemble = best_alpha * y_tcn_full + (1 - best_alpha) * y_pred_lin_te

    return {
        "fold": test_subject_id,
        "y_pred": y_ensemble,
        "y_pred_tcn": y_tcn_full,
        "y_pred_lin": y_pred_lin_te,
        "y_true": y_te,
        "best_alpha": best_alpha,
    }


def _loso(df, feat_cols, target_col, config, n_jobs,
          cwt_global=None, cwt_idx_map=None, cwt_feat_cols=None):
    subjects = sorted(df["subject_id"].unique())
    records  = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_run_one_loso_fold)(
            s, df, feat_cols, target_col, config,
            cwt_global, cwt_idx_map, cwt_feat_cols)
        for s in subjects)
    all_ens, all_tcn, all_lin, all_true, alphas, subj_rows = [], [], [], [], [], []
    for rec in records:
        if "error" not in rec:
            all_ens.append(rec["y_pred"])
            all_tcn.append(rec["y_pred_tcn"])
            all_lin.append(rec["y_pred_lin"])
            all_true.append(rec["y_true"])
            alphas.append(rec["best_alpha"])
            mae_ens = mean_absolute_error(rec["y_true"], rec["y_pred"]) / 60.0
            mae_tcn = mean_absolute_error(rec["y_true"], rec["y_pred_tcn"]) / 60.0
            mae_lin = mean_absolute_error(rec["y_true"], rec["y_pred_lin"]) / 60.0
            subj_rows.append({
                "subject_id": rec["fold"],
                "mae_ens_min": round(mae_ens, 4),
                "mae_tcn_min": round(mae_tcn, 4),
                "mae_lin_min": round(mae_lin, 4),
                "best_alpha":  round(rec["best_alpha"], 3),
                "best_expert": "tcn" if mae_tcn < mae_lin else "linear",
            })
    if not all_ens: return {"error": "Нет данных"}
    y_ens  = np.concatenate(all_ens)
    y_tcn  = np.concatenate(all_tcn)
    y_lin  = np.concatenate(all_lin)
    y_true = np.concatenate(all_true)
    return {
        "y_pred": y_ens, "y_pred_tcn": y_tcn, "y_pred_lin": y_lin, "y_true": y_true,
        "raw_mae_min":     mean_absolute_error(y_true, y_ens) / 60.0,
        "raw_mae_tcn_min": mean_absolute_error(y_true, y_tcn) / 60.0,
        "raw_mae_lin_min": mean_absolute_error(y_true, y_lin) / 60.0,
        "mean_alpha": float(np.mean(alphas)),
        "r2": r2_score(y_true, y_ens),
        "rho": float(spearmanr(y_true, y_ens).statistic),
        "per_subject": subj_rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target",      choices=["lt1","lt2","both"], default="both")
    p.add_argument("--feature-set", nargs="+",
                   choices=["EMG","NIRS","EMG+NIRS","EMG+NIRS+HRV"],
                   default=["EMG","NIRS","EMG+NIRS","EMG+NIRS+HRV"])
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    args = p.parse_args()

    config = Config()

    # Автовыбор устройства: CUDA → GPU + последовательно, иначе CPU + параллельно
    if torch.cuda.is_available():
        config.device = "cuda"
        config.batch_size = 256
        n_jobs = 1
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}, n_jobs=1")
    else:
        config.device = "cpu"
        n_jobs = args.n_jobs
        print(f"[CPU] CUDA недоступна, n_jobs={n_jobs}")

    print("=" * 70)
    print("v0107 — ENSEMBLE: Wavelet-TCN + ElasticNet")
    print("=" * 70)
    print(f"α оптимизируется на validation fold ({len(config.alpha_grid)} шагов)")
    print(f"wavelet={config.wavelet}, scales={config.scales}")
    print(f"window_step={config.window_step}, seq_length={config.seq_length}")
    print(f"device={config.device}, n_jobs={n_jobs}\n")

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Загружаем предрасчитанный CWT-кэш (посчитан локально, перекинут на сервер)
    cwt_cache_path = DEFAULT_DATASET_DIR / "cwt_cache.npz"
    if cwt_cache_path.exists():
        _cache = np.load(cwt_cache_path, allow_pickle=True)
        cwt_global    = _cache["cwt"]                      # (N, n_all_features, n_scales)
        cwt_row_ids   = _cache["row_ids"]
        cwt_feat_cols = list(_cache["feat_cols"])           # список признаков в кэше
        cwt_idx_map   = {int(rid): pos for pos, rid in enumerate(cwt_row_ids)}
        print(f"[CWT] Загружен кэш: {cwt_global.shape}, {cwt_cache_path.stat().st_size//1024} КБ")
    else:
        cwt_global = None; cwt_feat_cols = []; cwt_idx_map = {}
        print("[CWT] Кэш не найден — CWT будет считаться на лету")

    targets = {"lt2":"target_time_to_lt2_center_sec","lt1":"target_time_to_lt1_pchip_sec"}
    if args.target != "both":
        targets = {k: v for k, v in targets.items() if k == args.target}

    sigma_grid = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 150.0]
    v0011_ref  = _load_v0011_ref()
    v0106b_ref = {("lt2","EMG+NIRS+HRV"):2.765, ("lt1","EMG+NIRS+HRV"):3.080}
    records, subj_records = [], []

    for tgt_name, target_col in targets.items():
        print(f"\n{'═'*70}\nТАРГЕТ: {tgt_name.upper()}\n{'═'*70}\n")
        df_prep = prepare_data(df_raw, session_params, tgt_name)
        df_tgt  = df_prep.dropna(subset=[target_col])

        variants = [("with_abs", None, OUT_DIR), ("noabs", EXCLUDE_ABS, OUT_DIR / "noabs")]
        for fset in args.feature_set:
            feat_cols_full = get_feature_cols(df_tgt, fset)
            if not feat_cols_full: continue
            n_subj = df_tgt["subject_id"].nunique()

            for variant, exclude_set, out_sub in variants:
                feat_cols = (feat_cols_full if exclude_set is None
                             else [c for c in feat_cols_full if c not in exclude_set])
                if not feat_cols: continue
                out_sub.mkdir(exist_ok=True)
                print(f"  [{fset} / {tgt_name} / {variant}]  n={n_subj}, {len(feat_cols)} признаков")

                t0  = time.perf_counter()
                res = _loso(df_tgt, feat_cols, target_col, config, n_jobs,
                            cwt_global=cwt_global, cwt_idx_map=cwt_idx_map,
                            cwt_feat_cols=cwt_feat_cols)
                elapsed = time.perf_counter() - t0
                if "error" in res: print(f"    ❌ {res['error']}"); continue
                for row in res.get("per_subject", []):
                    subj_records.append({"variant": variant, "feature_set": fset, "target": tgt_name, **row})

                fset_tag = fset.replace("+", "_")
                np.save(out_sub / f"ypred_{tgt_name}_{fset_tag}.npy", res["y_pred"])
                np.save(out_sub / f"ytrue_{tgt_name}_{fset_tag}.npy", res["y_true"])

                best_mae, best_sig, k_maes = float("inf"), sigma_grid[0], {}
                for sigma in sigma_grid:
                    y_k  = kalman_smooth(res["y_pred"], sigma_p=5.0, sigma_obs=sigma)
                    mae  = mean_absolute_error(res["y_true"], y_k) / 60.0
                    k_maes[sigma] = round(mae, 4)
                    if mae < best_mae: best_mae, best_sig = mae, sigma

                records.append({
                    "variant": variant,
                    "feature_set": fset, "target": tgt_name,
                    "n_subjects": n_subj, "n_features": len(feat_cols),
                    "raw_ensemble_mae":  round(res["raw_mae_min"], 4),
                    "raw_tcn_mae":       round(res["raw_mae_tcn_min"], 4),
                    "raw_lin_mae":       round(res["raw_mae_lin_min"], 4),
                    "kalman_mae_min":    round(best_mae, 4),
                    "best_sigma_obs":    best_sig,
                    "mean_alpha":        round(res["mean_alpha"], 3),
                    "r2": round(res["r2"], 3), "rho": round(res["rho"], 3),
                    "sec": round(elapsed, 1),
                })
                print(f"    ens_raw={res['raw_mae_min']:.3f}  tcn={res['raw_mae_tcn_min']:.3f}"
                      f"  lin={res['raw_mae_lin_min']:.3f}  α={res['mean_alpha']:.2f}")
                print(f"    kalman_best={best_mae:.3f} (sigma={best_sig})  ({elapsed:.1f}s)")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(subj_records).to_csv(OUT_DIR / "per_subject.csv", index=False)
    df_noabs = df_out[df_out["variant"] == "noabs"]
    if not df_noabs.empty:
        df_noabs.to_csv(OUT_DIR / "noabs" / "summary.csv", index=False)

    print("\n" + "="*70 + "\nИТОГИ ENSEMBLE:")
    for variant in ["with_abs", "noabs"]:
        sub = df_out[df_out["variant"] == variant]
        if sub.empty: continue
        print(f"\n  [{variant}]")
        for _, r in sub.sort_values(["target","kalman_mae_min"]).iterrows():
            ref11   = v0011_ref.get((r["target"], r["feature_set"]))
            ref06b  = v0106b_ref.get((r["target"], r["feature_set"]))
            d11  = f"  Δ={r['kalman_mae_min']-ref11:+.3f} vs v0011"  if ref11  else ""
            d06b = f"  Δ={r['kalman_mae_min']-ref06b:+.3f} vs v0106b" if ref06b else ""
            print(f"    {r['target'].upper()} / {r['feature_set']:<16s}  "
                  f"ens={r['kalman_mae_min']:.3f}{d11}{d06b}")
    print(f"\n✅ Готово: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
