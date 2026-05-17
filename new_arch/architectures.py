"""Явные списки ArchitectureSpec для трёх семейств.

Каждый список фиксирует базовые архитектуры; конкретный model_id формируется
из ArchitectureSpec + (target, feature_set, with_abs, wavelet_mode).

build_estimator(arch) — фабрика sklearn-эстиматора для Lin*-архитектур.
"""

from __future__ import annotations

import itertools
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.svm import SVR

from common_lib import ArchitectureSpec


# ─── Linear / classical зоопарк (зеркало scripts/v0011_modality_ablation._build_zoo) ──

def _lin_specs() -> list[ArchitectureSpec]:
    """29 классических моделей: Ridge, Huber, ElasticNet, GBM, SVR."""
    base = dict(
        family="Lin",
        window_size_sec=30,
        sequence_length=1,
        stride_sec=0,
        sample_stride_sec=5,
        forced_wavelet_mode=None,
    )
    out: list[ArchitectureSpec] = []
    n = 0

    def add(model_class: str, hp: dict[str, Any], full: str, short: str) -> None:
        nonlocal n
        n += 1
        out.append(ArchitectureSpec(
            architecture_id=f"Lin{n}",
            architecture_name=full,
            short_architecture_name=short,
            model_class_name=model_class,
            hyperparams=hp,
            **base,
        ))

    for alpha in [1, 10, 100, 1000]:
        add("Ridge", {"alpha": alpha},
            f"Ridge regression (alpha={alpha})", f"Ridge α={alpha}")

    for eps in [1.1, 1.35, 1.5, 2.0]:
        add("HuberRegressor", {"epsilon": eps, "max_iter": 2000},
            f"Huber regression (epsilon={eps})", f"Huber ε={eps}")

    for alpha, l1 in itertools.product([0.01, 0.1, 1.0], [0.2, 0.5, 0.9]):
        add("ElasticNet",
            {"alpha": alpha, "l1_ratio": l1, "max_iter": 5000, "random_state": 42},
            f"ElasticNet (alpha={alpha}, l1_ratio={l1})",
            f"EN α={alpha}/l1={l1}")

    for n_est, depth in itertools.product([50, 100, 200], [2, 3]):
        add("GradientBoostingRegressor",
            {"n_estimators": n_est, "max_depth": depth, "random_state": 42},
            f"Gradient Boosting (n={n_est}, depth={depth})",
            f"GBM n={n_est}/d={depth}")

    for C, eps in itertools.product([1, 10, 100], [0.1, 1.0]):
        add("SVR",
            {"kernel": "rbf", "C": C, "epsilon": eps},
            f"SVR rbf (C={C}, epsilon={eps})",
            f"SVR C={C}/ε={eps}")

    return out


LINEAR_ARCHS: list[ArchitectureSpec] = _lin_specs()

# ─── LSTM семейство ─────────────────────────────────────────────────────────

_BASE_STEP_SEC = 5

_LSTM_COMMON_HP = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 20,
    "checkpoint_every_epochs": 4,
}

# Stateful-only гиперпараметры.
_STATEFUL_HP = {**_LSTM_COMMON_HP, "chunk_size": 10}


def _lstm_stateless(arch_id: str, internal_stride_sec: int,
                    wavelet_mode: str) -> ArchitectureSpec:
    """Stateless LSTM, seq_len=6, выбираемый internal_stride, outer=5."""
    wav_tag = "+CWT" if wavelet_mode == "cwt" else ""
    return ArchitectureSpec(
        architecture_id=arch_id,
        family="LSTM",
        architecture_name=f"Stateless LSTM, 6 × {internal_stride_sec} сек{wav_tag}",
        short_architecture_name=f"SLSTM 6x{internal_stride_sec}{wav_tag}",
        model_class_name="LSTMRegressor",
        window_size_sec=30,
        sequence_length=6,
        stride_sec=internal_stride_sec,
        sample_stride_sec=5,
        forced_wavelet_mode=wavelet_mode,
        hyperparams={
            "seq_len": 6,
            "internal_stride_sec": internal_stride_sec,
            "outer_stride_sec": 5,
            **_LSTM_COMMON_HP,
        },
    )


def _lstm_attention(arch_id: str, internal_stride_sec: int,
                    wavelet_mode: str) -> ArchitectureSpec:
    """Attention LSTM, seq_len=12, выбираемый internal_stride, outer=5."""
    wav_tag = "+CWT" if wavelet_mode == "cwt" else ""
    return ArchitectureSpec(
        architecture_id=arch_id,
        family="LSTM",
        architecture_name=f"Attention LSTM, 12 × {internal_stride_sec} сек{wav_tag}",
        short_architecture_name=f"AttnLSTM 12x{internal_stride_sec}{wav_tag}",
        model_class_name="AttentionLSTMRegressor",
        window_size_sec=30,
        sequence_length=12,
        stride_sec=internal_stride_sec,
        sample_stride_sec=5,
        forced_wavelet_mode=wavelet_mode,
        hyperparams={
            "seq_len": 12,
            "internal_stride_sec": internal_stride_sec,
            "outer_stride_sec": 5,
            **_LSTM_COMMON_HP,
        },
    )


def _lstm_stateful(arch_id: str, internal_stride_sec: int,
                   wavelet_mode: str) -> ArchitectureSpec:
    """Stateful LSTM, целая запись, internal_stride=шаг субсэмплирования.

    sequence_length=null концептуально (full recording),
    в schema используем 0 для отличия от stateless/attention.
    """
    wav_tag = "+CWT" if wavelet_mode == "cwt" else ""
    return ArchitectureSpec(
        architecture_id=arch_id,
        family="LSTM",
        architecture_name=f"Stateful LSTM, full record, шаг {internal_stride_sec} сек{wav_tag}",
        short_architecture_name=f"StLSTM full s{internal_stride_sec}{wav_tag}",
        model_class_name="LSTMStatefulRegressor",
        window_size_sec=30,
        sequence_length=0,   # 0 ≡ "full recording", не None (чтобы не было NaN в parquet)
        stride_sec=internal_stride_sec,
        sample_stride_sec=internal_stride_sec,
        forced_wavelet_mode=wavelet_mode,
        hyperparams={
            "internal_stride_sec": internal_stride_sec,
            **_STATEFUL_HP,
        },
    )


def _lstm_specs() -> list[ArchitectureSpec]:
    """Все 16 LSTM-архитектур."""
    specs: list[ArchitectureSpec] = []
    # LSTM1..LSTM6: stateless 6×{30,15,5} × {none, cwt}
    n = 0
    for stride in [30, 15, 5]:
        for wav in ["none", "cwt"]:
            n += 1
            specs.append(_lstm_stateless(f"LSTM{n}", stride, wav))
    # LSTM7..LSTM12: stateful full × {30,15,5} × {none, cwt}
    for stride in [30, 15, 5]:
        for wav in ["none", "cwt"]:
            n += 1
            specs.append(_lstm_stateful(f"LSTM{n}", stride, wav))
    # LSTM13..LSTM16: attention 12×{30,15} × {none, cwt}
    for stride in [30, 15]:
        for wav in ["none", "cwt"]:
            n += 1
            specs.append(_lstm_attention(f"LSTM{n}", stride, wav))
    return specs


LSTM_ARCHS: list[ArchitectureSpec] = _lstm_specs()


# ─── TCN семейство ──────────────────────────────────────────────────────────

def _tcn_specs() -> list[ArchitectureSpec]:
    """TCN-архитектуры (causal, без val-split).

    Все свёртки причинные (pad слева на (k-1)*d) — нет утечки "будущего" внутри окна.
    """
    common_hp = {
        "n_channels": 32,
        "kernel_size": 3,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "max_epochs": 80,
        "checkpoint_every_epochs": 4,
    }
    specs: list[ArchitectureSpec] = []

    # TCN1: pure dilated TCN, seq=30 строк (150 сек), RF=31.
    specs.append(ArchitectureSpec(
        architecture_id="TCN1",
        family="TCN",
        architecture_name="Causal PureTCN, seq=30 строк (150 сек), RF=31",
        short_architecture_name="PureTCN 30",
        model_class_name="PureTCN",
        window_size_sec=30,
        sequence_length=30,
        stride_sec=5,
        sample_stride_sec=5,
        forced_wavelet_mode="none",
        hyperparams={
            "seq_len": 30,
            "internal_stride_sec": 5,
            "outer_stride_sec": 5,
            "dilations": [1, 2, 4, 8],
            **common_hp,
        },
    ))

    # TCN2: medium — seq=60 строк (300 сек), dilations=[1,2,4,8,16], RF=63.
    specs.append(ArchitectureSpec(
        architecture_id="TCN2",
        family="TCN",
        architecture_name="Causal MediumTCN, seq=60 строк (300 сек), RF=63",
        short_architecture_name="MediumTCN 60",
        model_class_name="PureTCN",
        window_size_sec=30,
        sequence_length=60,
        stride_sec=5,
        sample_stride_sec=5,
        forced_wavelet_mode="none",
        hyperparams={
            "seq_len": 60,
            "internal_stride_sec": 5,
            "outer_stride_sec": 5,
            "dilations": [1, 2, 4, 8, 16],
            **common_hp,
        },
    ))

    # TCN3: DWT-TCN — Haar DWT → 2 causal TCN-ветви, dilations=[1,2,4].
    specs.append(ArchitectureSpec(
        architecture_id="TCN3",
        family="TCN",
        architecture_name="Causal DWT-TCN, seq=30 (Haar DWT → cA/cD ветви)",
        short_architecture_name="DwtTCN 30",
        model_class_name="DwtTCN",
        window_size_sec=30,
        sequence_length=30,
        stride_sec=5,
        sample_stride_sec=5,
        forced_wavelet_mode="dwt",
        hyperparams={
            "seq_len": 30,
            "internal_stride_sec": 5,
            "outer_stride_sec": 5,
            "dilations": [1, 2, 4],
            **common_hp,
        },
    ))

    # TCN4: WaveNet-style TCN — gated activation, kernel=2, dilations=[1,2,4,8,16].
    specs.append(ArchitectureSpec(
        architecture_id="TCN4",
        family="TCN",
        architecture_name="Causal WaveNet-TCN, seq=30, gated activation",
        short_architecture_name="WaveNetTCN 30",
        model_class_name="WaveNetTCN",
        window_size_sec=30,
        sequence_length=30,
        stride_sec=5,
        sample_stride_sec=5,
        forced_wavelet_mode="none",
        hyperparams={
            "seq_len": 30,
            "internal_stride_sec": 5,
            "outer_stride_sec": 5,
            "kernel_size": 2,
            "dilations": [1, 2, 4, 8, 16],
            "skip_channels_mult": 2,
            **{k: v for k, v in common_hp.items() if k != "kernel_size"},
        },
    ))

    return specs


TCN_ARCHS: list[ArchitectureSpec] = _tcn_specs()


# ─── Реестр и фабрика эстиматоров ───────────────────────────────────────────

_SKLEARN_CLASSES: dict[str, type] = {
    "Ridge": Ridge,
    "HuberRegressor": HuberRegressor,
    "ElasticNet": ElasticNet,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
}


def get_architecture(architecture_id: str) -> ArchitectureSpec:
    """Возвращает ArchitectureSpec по architecture_id (поиск во всех семействах)."""
    for spec in LINEAR_ARCHS + LSTM_ARCHS + TCN_ARCHS:
        if spec.architecture_id == architecture_id:
            return spec
    raise KeyError(
        f"Неизвестная architecture_id={architecture_id!r}. "
        f"Доступны Lin*: {[s.architecture_id for s in LINEAR_ARCHS]}"
    )


def build_estimator(arch: ArchitectureSpec):
    """Создаёт sklearn-эстиматор по ArchitectureSpec.

    Поддерживает только семейство Lin (классические модели sklearn).
    """
    if arch.family != "Lin":
        raise ValueError(
            f"build_estimator пока поддерживает только family='Lin', "
            f"получено {arch.family!r}"
        )
    cls = _SKLEARN_CLASSES.get(arch.model_class_name)
    if cls is None:
        raise KeyError(
            f"Неизвестный model_class_name={arch.model_class_name!r}. "
            f"Доступны: {list(_SKLEARN_CLASSES)}"
        )
    return cls(**arch.hyperparams)
