#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensorless FPV temperature-field forecasting experiments.

This script is a clean experimental fork inspired by the old
train_fpv_sensorless_reduced_sensor_1min workflow. It changes the formal target
from scalar Tmean/DeltaT forecasting to multi-output temperature-field
forecasting:

    X(t-lookback+1 ... t): meteorological / operating / time features only
    Y(t+horizon):          15 module-temperature sensor values

The formal strategy is factorized:
    factorized  two independent models predict Tmean and spatial residuals

The direct strategy is retained for ablation:
    direct      one model predicts the full 15-point temperature field

DeltaT/range is reported only as a secondary derived diagnostic. The primary
derived spatial metrics are Tmean, Tstd, P95-P5 spread, and max deviation from
the field mean.

Example smoke run:
    E:\\anaconda3\\envs\\patchtst\\python.exe code\\scripts\\train_fpv_sensorless_field_models.py --smoke

Formal Oakville comparison:
    E:\\anaconda3\\envs\\patchtst\\python.exe code\\scripts\\train_fpv_sensorless_field_models.py ^
      --site oakville --horizons 1,5,10,30 --seeds 42,3407,2026 ^
      --models mscnn_lstm_attn,transformer,patch_transformer,itransformer ^
      --strategies direct,factorized --epochs 80
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SENTINELS = {32767.0, 32766.0, 6999.0, 7999.0, -6999.0, -7999.0}


SITE_CONFIGS: Dict[str, Dict[str, object]] = {
    "altamonte": {
        "data_file": "data/FPV_Altamonte_FL_data.csv",
        "temp_cols": [
            "SEPTMA", "SEPTMB", "SEPTMC",
            "SWPTMA", "SWPTMB", "SWPTMC",
            "MDPTMA", "MDPTMB", "MDPTMC",
            "NWPTMA", "NWPTMB", "NWPTMC",
            # NEPTMB is present in the file, but has no usable validation/test
            # coverage after chronological splitting, so it is excluded rather
            # than imputed as a target.
            "NEPTMA", "NEPTMC",
        ],
        "met_cols": [
            "FPOAI1", "FPOAI2", "FSPIRR",
            "WTPLOW", "WTPMID", "WTPDEP",
            "FPVDBT", "FPV_RH", "FRAIRP",
            "FWINDA", "FWINDM", "FWNDAC",
        ],
        "wind_dir_cols": ["FWDIRA"],
        "op_cols": [
            "ACPWRT", "DCVOLT",
            "INV1PW", "INV2PW", "INV3PW", "INV4PW", "INV5PW",
            "IN1DCV", "IN2DCV", "IN3DCV", "IN4DCV", "IN5DCV",
        ],
        "irradiance_candidates": ["FPOAI1", "FPOAI2", "FSPIRR"],
    },
    "oakville": {
        "data_file": "data/FPV_Oakville_CA_data.csv",
        "temp_cols": [
            "SEPTMA", "SEPTMB", "SEPTMC",
            "SWPTMA", "SWPTMB", "SWPTMC",
            "MDPTMA", "MDPTMB", "MDPTMC",
            "NWPTMA", "NWPTMB", "NWPTMC",
            "NEPTMA", "NEPTMB", "NEPTMC",
        ],
        "met_cols": [
            "FPAIRR", "FHZIRR", "FPVDBT", "FPV_RH", "FRAIRP",
            "FWINDA", "FWINDM",
        ],
        "wind_dir_cols": ["FWDIRA"],
        "op_cols": [
            "INVPWR",
            "MPPT1P", "MPPT1V", "MPPT1C",
            "MPPT2P", "MPPT2V", "MPPT2C",
        ],
        "irradiance_candidates": ["FPAIRR", "FHZIRR"],
    },
    "orlando": {
        "data_file": "data/FPV_Orlando_FL_data.csv",
        "temp_cols": [
            "SEPTMA", "SEPTMB", "SEPTMC",
            "NEPTMA", "NEPTMB", "NEPTMC",
            "MDPTMA", "MDPTMB", "MDPTMC",
            "NWPTMA", "NWPTMB", "NWPTMC",
            "SWPTMA", "SWPTMB", "SWPTMC",
        ],
        "met_cols": [
            "POAIR1", "POAIR2", "POAIR3", "FPHIRR", "FAMBTM",
            "FPV_RH", "FBPRES", "FWINDS", "FWINDM", "RAINMM",
        ],
        "wind_dir_cols": ["FWINDD"],
        "op_cols": [],
        "irradiance_candidates": ["POAIR1", "POAIR2", "POAIR3", "FPHIRR"],
    },
    "windsor": {
        "data_file": "data/FPV_Windsor_CA_data.csv",
        "temp_cols": [
            "NWPTMA", "NWPTMB", "NWPTMC",
            "NOPTMA", "NOPTMB", "NOPTMC",
            "NEPTMA", "NEPTMB", "NEPTMC",
            "MDPTMA", "MDPTMB", "MDPTMC",
            "SOPTMA", "SOPTMB", "SOPTMC",
        ],
        "met_cols": [
            "FPAIRR", "FHZIRR", "FPVDBT", "FPV_RH", "FRAIRP",
            "FWINDA", "FWINDM", "FWINDV",
        ],
        "wind_dir_cols": [],
        "op_cols": [],
        "irradiance_candidates": ["FPAIRR", "FHZIRR"],
    },
}


@dataclass
class Config:
    site: str = "oakville"
    data_file: str = "data/FPV_Oakville_CA_data.csv"
    output_root: str = "code/runs"
    run_name: str = ""
    resample_minutes: int = 1
    lookback_minutes: int = 60
    horizons: Tuple[int, ...] = (1,)
    seeds: Tuple[int, ...] = (42,)
    models: Tuple[str, ...] = (
        "mscnn_lstm_attn",
        "transformer",
        "patch_transformer",
        "itransformer",
    )
    strategies: Tuple[str, ...] = ("factorized",)
    feature_set: str = "metop"
    feature_modes: Tuple[str, ...] = ("raw",)
    xgb_top_k: int = 12
    nmf_components: int = 8
    lowrank_rank: int = 6
    selector_sample_limit: int = 80000
    epochs: int = 60
    patience: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_size: int = 96
    d_model: int = 96
    num_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.10
    loss: str = "mse"
    spatial_loss_weight: float = 0.0
    epoch_pause_sec: float = 0.0
    torch_num_threads: int = 0
    cuda_memory_fraction: float = 0.0
    num_workers: int = 0
    min_irradiance: float = 200.0
    input_min_mean_irradiance: float = 100.0
    short_gap_limit_steps: int = 2
    temp_min: float = -30.0
    temp_max: float = 100.0
    missing_feature_threshold: float = 0.30
    max_rows: int = 0
    smoke: bool = False

    @property
    def lookback_steps(self) -> int:
        if self.lookback_minutes % self.resample_minutes != 0:
            raise ValueError("lookback_minutes must be divisible by resample_minutes")
        return self.lookback_minutes // self.resample_minutes

    @property
    def resample_rule(self) -> str:
        return f"{self.resample_minutes}min"


class ArrayScaler:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "ArrayScaler":
        self.mean_ = np.nanmean(x, axis=0).astype(np.float32)
        self.std_ = np.nanstd(x, axis=0).astype(np.float32)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        return (x * self.std_ + self.mean_).astype(np.float32)


class MinMaxFiller:
    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.fill_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "MinMaxFiller":
        self.fill_ = np.nanmedian(x, axis=0).astype(np.float32)
        self.fill_[~np.isfinite(self.fill_)] = 0.0
        filled = np.where(np.isfinite(x), x, self.fill_)
        self.min_ = np.nanmin(filled, axis=0).astype(np.float32)
        max_ = np.nanmax(filled, axis=0).astype(np.float32)
        self.scale_ = (max_ - self.min_).astype(np.float32)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.scale_ is None or self.fill_ is None:
            raise RuntimeError("MinMaxFiller is not fitted")
        filled = np.where(np.isfinite(x), x, self.fill_)
        z = (filled - self.min_) / self.scale_
        return np.clip(z, 0.0, None).astype(np.float32)


def parse_csv_tuple(value: str, cast=str) -> Tuple:
    if value is None or str(value).strip() == "":
        return tuple()
    return tuple(cast(x.strip()) for x in str(value).split(",") if x.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def infer_datetime(df: pd.DataFrame) -> pd.Series:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    if "day" not in cols_lower or "hour" not in cols_lower:
        raise ValueError("DAY and HOUR columns are required")

    day_col = cols_lower["day"]
    hour_col = cols_lower["hour"]
    day_str = df[day_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    day_dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    mask_yyyyddd = day_str.str.match(r"^\d{7}$", na=False)
    mask_yyyymmdd = day_str.str.match(r"^\d{8}$", na=False)
    if mask_yyyyddd.any():
        day_dt.loc[mask_yyyyddd] = pd.to_datetime(
            day_str.loc[mask_yyyyddd], format="%Y%j", errors="coerce"
        )
    if mask_yyyymmdd.any():
        day_dt.loc[mask_yyyymmdd] = pd.to_datetime(
            day_str.loc[mask_yyyymmdd], format="%Y%m%d", errors="coerce"
        )
    remain = ~(mask_yyyyddd | mask_yyyymmdd)
    if remain.any():
        day_dt.loc[remain] = pd.to_datetime(day_str.loc[remain], errors="coerce")

    hour_str = df[hour_col].astype(str).str.strip()
    time_delta = pd.to_timedelta(hour_str, errors="coerce")
    bad = time_delta.isna()
    if bad.any():
        hour_num = pd.to_numeric(hour_str[bad], errors="coerce")
        time_delta.loc[bad] = pd.to_timedelta(hour_num, unit="h")

    dt = day_dt + time_delta
    nat_ratio = float(dt.isna().mean())
    if nat_ratio > 0.1:
        raise ValueError(f"Too many invalid datetimes: {nat_ratio:.2%}")
    return dt


def clean_numeric_frame(df: pd.DataFrame, temp_cols: Sequence[str], cfg: Config) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in ("DAY", "HOUR", "datetime"):
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col].isin(SENTINELS), col] = np.nan
        out.loc[out[col].abs() > 1e5, col] = np.nan

    for col in temp_cols:
        out.loc[(out[col] < cfg.temp_min) | (out[col] > cfg.temp_max), col] = np.nan
    return out


def select_existing(columns: Iterable[str], available: Sequence[str]) -> List[str]:
    available_set = set(available)
    return [c for c in columns if c in available_set]


def load_site_dataframe(cfg: Config) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    site_info = SITE_CONFIGS[cfg.site]
    temp_cols = list(site_info["temp_cols"])  # type: ignore[index]
    met_cols = list(site_info["met_cols"])  # type: ignore[index]
    op_cols = list(site_info["op_cols"])  # type: ignore[index]
    wind_dir_cols = list(site_info["wind_dir_cols"])  # type: ignore[index]

    data_path = Path(cfg.data_file)
    header = pd.read_csv(data_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing_temp = [c for c in temp_cols if c not in header]
    if missing_temp:
        raise ValueError(f"Missing temperature columns for {cfg.site}: {missing_temp}")

    met_cols = select_existing(met_cols, header)
    op_cols = select_existing(op_cols, header)
    wind_dir_cols = select_existing(wind_dir_cols, header)
    op_used = op_cols if cfg.feature_set.lower() == "metop" else []
    base_cols = ["DAY", "HOUR"] + temp_cols + met_cols + op_used + wind_dir_cols
    base_cols = list(dict.fromkeys([c for c in base_cols if c in header]))

    read_kwargs = {"usecols": base_cols, "encoding": "utf-8-sig"}
    if cfg.max_rows and cfg.max_rows > 0:
        read_kwargs["nrows"] = cfg.max_rows
    df = pd.read_csv(data_path, **read_kwargs)
    df["datetime"] = infer_datetime(df)
    df = clean_numeric_frame(df, temp_cols, cfg)
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"]).set_index("datetime")

    numeric_cols = [c for c in df.columns if c not in ("DAY", "HOUR")]
    df = df[numeric_cols].resample(cfg.resample_rule).mean()
    df = df.interpolate(
        method="time",
        limit=cfg.short_gap_limit_steps,
        limit_direction="both",
    )

    for col in wind_dir_cols:
        radians = np.deg2rad(df[col])
        df[f"{col}_sin"] = np.sin(radians)
        df[f"{col}_cos"] = np.cos(radians)
        df = df.drop(columns=[col])

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    doy = df.index.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)

    feature_base = met_cols + op_used
    feature_cols = select_existing(feature_base, df.columns)
    for col in feature_base:
        if col in df.columns:
            dcol = f"d{col}"
            df[dcol] = df[col].diff()
            feature_cols.append(dcol)

    feature_cols += [f"{c}_sin" for c in wind_dir_cols if f"{c}_sin" in df.columns]
    feature_cols += [f"{c}_cos" for c in wind_dir_cols if f"{c}_cos" in df.columns]
    feature_cols += ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    feature_cols = list(dict.fromkeys(feature_cols))

    keep_features: List[str] = []
    dropped_features: List[Tuple[str, float]] = []
    for col in feature_cols:
        miss = float(df[col].isna().mean())
        if miss <= cfg.missing_feature_threshold:
            keep_features.append(col)
        else:
            dropped_features.append((col, miss))
    if dropped_features:
        print("Dropped high-missing sensorless features:")
        for col, miss in dropped_features:
            print(f"  {col}: {miss:.2%}")
    feature_cols = keep_features

    irr_candidates = list(site_info["irradiance_candidates"])  # type: ignore[index]
    irr_col = next((c for c in irr_candidates if c in df.columns), "")
    return df, temp_cols, feature_cols, irr_col


def rolling_count_valid(valid_rows: np.ndarray, window: int) -> np.ndarray:
    valid_int = valid_rows.astype(np.int32)
    cumsum = np.concatenate([[0], np.cumsum(valid_int)])
    idx = np.arange(len(valid_rows))
    start = np.maximum(0, idx - window + 1)
    return cumsum[idx + 1] - cumsum[start]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(values).astype(np.float32)
    vals = np.nan_to_num(values, nan=0.0).astype(np.float32)
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    cnum = np.concatenate([[0.0], np.cumsum(finite)])
    idx = np.arange(len(values))
    start = np.maximum(0, idx - window + 1)
    total = csum[idx + 1] - csum[start]
    count = cnum[idx + 1] - cnum[start]
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def build_anchors(
    df: pd.DataFrame,
    x_values: np.ndarray,
    y_values: np.ndarray,
    irr_col: str,
    horizon_steps: int,
    cfg: Config,
) -> np.ndarray:
    n = len(df)
    lookback = cfg.lookback_steps
    anchor_idx = np.arange(n)
    target_idx = anchor_idx + horizon_steps
    in_range = (anchor_idx >= lookback - 1) & (target_idx < n)

    x_row_valid = np.isfinite(x_values).all(axis=1)
    window_valid = rolling_count_valid(x_row_valid, lookback) == lookback
    y_valid = np.isfinite(y_values).all(axis=1)
    target_valid = np.zeros(n, dtype=bool)
    target_valid[in_range] = y_valid[target_idx[in_range]]

    irr_valid = np.ones(n, dtype=bool)
    if irr_col and cfg.min_irradiance > 0:
        irr = df[irr_col].to_numpy(dtype=np.float32)
        irr_mean = rolling_mean(irr, lookback)
        irr_target_valid = np.zeros(n, dtype=bool)
        irr_target_valid[in_range] = np.isfinite(irr[target_idx[in_range]]) & (
            irr[target_idx[in_range]] >= cfg.min_irradiance
        )
        irr_valid = (
            np.isfinite(irr)
            & (irr >= cfg.min_irradiance)
            & np.isfinite(irr_mean)
            & (irr_mean >= cfg.input_min_mean_irradiance)
            & irr_target_valid
        )

    anchors = anchor_idx[in_range & window_valid & target_valid & irr_valid]
    return anchors.astype(np.int64)


def chronological_split(
    anchors: np.ndarray, n_total: int, horizon_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)
    target_indices = anchors + horizon_steps
    train = anchors[target_indices < train_end]
    val = anchors[(target_indices >= train_end) & (target_indices < val_end)]
    test = anchors[target_indices >= val_end]
    return train, val, test


class WindowDataset(Dataset):
    def __init__(
        self,
        x_scaled: np.ndarray,
        y_scaled: np.ndarray,
        anchors: np.ndarray,
        lookback_steps: int,
        horizon_steps: int,
    ) -> None:
        self.x_scaled = x_scaled.astype(np.float32)
        self.y_scaled = y_scaled.astype(np.float32)
        self.anchors = anchors.astype(np.int64)
        self.lookback_steps = int(lookback_steps)
        self.horizon_steps = int(horizon_steps)

    def __len__(self) -> int:
        return int(len(self.anchors))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor = int(self.anchors[idx])
        start = anchor - self.lookback_steps + 1
        target = anchor + self.horizon_steps
        x = self.x_scaled[start : anchor + 1, :]
        y = self.y_scaled[target, :]
        return torch.from_numpy(x), torch.from_numpy(y)


class FlattenMLP(nn.Module):
    def __init__(self, lookback: int, input_dim: int, output_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * input_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DLinear(nn.Module):
    def __init__(self, lookback: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(lookback * input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(start_dim=1))


class AttentionPool(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(torch.tanh(h)), dim=1)
        return torch.sum(weights * h, dim=1)


class LSTMAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.pool = AttentionPool(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(self.pool(h))


class MultiScaleCNNLSTMAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = max(16, hidden // 3)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(input_dim, channels, kernel_size=k, padding=k // 2)
                for k in (3, 5, 9)
            ]
        )
        conv_dim = channels * len(self.convs)
        self.norm = nn.LayerNorm(conv_dim)
        self.lstm = nn.LSTM(
            conv_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.pool = AttentionPool(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        z = torch.cat([torch.relu(conv(xt)) for conv in self.convs], dim=1)
        z = z.transpose(1, 2)
        z = self.norm(z)
        h, _ = self.lstm(z)
        return self.head(self.pool(h))


class SinusoidalPosition(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        lookback: int,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = SinusoidalPosition(lookback, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.proj(x))
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


class PatchTransformerModel(nn.Module):
    def __init__(
        self,
        lookback: int,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
        patch_len: int = 12,
        stride: int = 6,
    ) -> None:
        super().__init__()
        self.patch_len = min(patch_len, lookback)
        self.stride = min(stride, self.patch_len)
        n_patches = 1 + max(0, (lookback - self.patch_len) // self.stride)
        self.proj = nn.Linear(self.patch_len * input_dim, d_model)
        self.pos = SinusoidalPosition(max(n_patches, 1), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().flatten(start_dim=2)
        z = self.pos(self.proj(patches))
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


class ITransformerLite(nn.Module):
    def __init__(
        self,
        lookback: int,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.var_proj = nn.Linear(lookback, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, input_dim, d_model))
        nn.init.normal_(self.var_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Flatten(),
            nn.Linear(input_dim * d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.var_proj(z) + self.var_embed
        z = self.encoder(z)
        return self.head(z)


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        if trend.size(1) != x.size(1):
            trend = trend[:, : x.size(1), :]
        seasonal = x - trend
        return seasonal, trend


class AutoformerLite(nn.Module):
    """Series decomposition plus temporal attention, adapted for exogenous inputs."""

    def __init__(
        self,
        lookback: int,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        kernel = min(25, lookback if lookback % 2 == 1 else max(1, lookback - 1))
        self.decomp = MovingAverageDecomposition(kernel)
        self.season_proj = nn.Linear(input_dim, d_model)
        self.trend_proj = nn.Linear(input_dim, d_model)
        self.pos = SinusoidalPosition(lookback, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)
        z = self.pos(self.season_proj(seasonal))
        z = self.encoder(z).mean(dim=1)
        t = self.trend_proj(trend).mean(dim=1)
        return self.head(torch.cat([z, t], dim=-1))


class CrossformerLite(nn.Module):
    """Dimension-segment-wise tokens for cross-time and cross-variable attention."""

    def __init__(
        self,
        lookback: int,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
        segment_len: int = 12,
        stride: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.segment_len = min(segment_len, lookback)
        self.stride = min(stride, self.segment_len)
        self.n_segments = 1 + max(0, (lookback - self.segment_len) // self.stride)
        self.value_proj = nn.Linear(self.segment_len, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, input_dim, 1, d_model))
        self.seg_embed = nn.Parameter(torch.zeros(1, 1, self.n_segments, d_model))
        nn.init.normal_(self.var_embed, std=0.02)
        nn.init.normal_(self.seg_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, T, C -> B, C, n_segments, segment_len
        z = x.transpose(1, 2).unfold(dimension=2, size=self.segment_len, step=self.stride)
        z = self.value_proj(z)
        z = z + self.var_embed[:, :, : z.size(2), :] + self.seg_embed[:, :, : z.size(2), :]
        z = z.flatten(start_dim=1, end_dim=2)
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


class PathformerLite(nn.Module):
    """Multi-scale patch branches with adaptive pathway gating."""

    def __init__(
        self,
        lookback: int,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        scales = [4, 8, 16, 32]
        self.scales = [s for s in scales if s <= lookback]
        if not self.scales:
            self.scales = [lookback]
        self.branches = nn.ModuleList(
            [
                PatchTransformerModel(
                    lookback,
                    input_dim,
                    d_model,
                    d_model,
                    n_heads,
                    max(1, layers),
                    dropout,
                    patch_len=s,
                    stride=max(1, s // 2),
                )
                for s in self.scales
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(self.scales)),
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)
        weights = torch.softmax(self.gate(stats), dim=-1)
        branch_outputs = torch.stack([branch(x) for branch in self.branches], dim=1)
        z = torch.sum(branch_outputs * weights.unsqueeze(-1), dim=1)
        return self.head(z)


class TimeXerLite(nn.Module):
    """Patch-wise exogenous encoding plus target-query cross attention."""

    def __init__(
        self,
        lookback: int,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
        patch_len: int = 12,
        stride: int = 6,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.patch_len = min(patch_len, lookback)
        self.stride = min(stride, self.patch_len)
        n_patches = 1 + max(0, (lookback - self.patch_len) // self.stride)
        self.exog_proj = nn.Linear(self.patch_len * input_dim, d_model)
        self.exog_pos = SinusoidalPosition(max(n_patches, 1), d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, layers))
        self.target_queries = nn.Parameter(torch.zeros(1, output_dim, d_model))
        nn.init.normal_(self.target_queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.scalar_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().flatten(start_dim=2)
        memory = self.encoder(self.exog_pos(self.exog_proj(patches)))
        queries = self.target_queries.expand(x.size(0), -1, -1)
        z, _ = self.cross_attn(queries, memory, memory)
        z = self.norm(z + self.ffn(z))
        return self.scalar_head(z).squeeze(-1)


def downsample_time(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 1:
        return x
    if x.size(1) < scale:
        return x
    return F.avg_pool1d(
        x.transpose(1, 2),
        kernel_size=scale,
        stride=scale,
        ceil_mode=False,
    ).transpose(1, 2)


class CoreFusion(nn.Module):
    """Global-core feature fusion inspired by SOFTS/STar-style mixing."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.core = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        core = self.core(tokens.mean(dim=1, keepdim=True))
        core = core.expand(-1, tokens.size(1), -1)
        return tokens + self.mix(torch.cat([tokens, core], dim=-1))


class TimeCardSensorLite(nn.Module):
    """
    Hybrid for sensorless temperature-field residuals.

    It combines multi-scale decomposition tokens (TimeMixer/Autoformer idea),
    feature-channel tokens (CARD/iTransformer idea), and sensor/output queries
    cross-attending to exogenous memory (TimeXer idea). This is a task-adapted
    lite model, not a faithful reproduction of any single paper.
    """

    def __init__(
        self,
        lookback: int,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.scales = [s for s in (1, 2, 4, 8) if s <= lookback]
        if not self.scales:
            self.scales = [1]

        kernel = min(25, lookback if lookback % 2 == 1 else max(1, lookback - 1))
        self.decomp = MovingAverageDecomposition(kernel)
        self.season_proj = nn.Linear(input_dim, d_model)
        self.trend_proj = nn.Linear(input_dim, d_model)
        self.var_proj = nn.Linear(lookback, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, input_dim, d_model))
        self.scale_embed = nn.Parameter(torch.zeros(len(self.scales), d_model))
        nn.init.normal_(self.var_embed, std=0.02)
        nn.init.normal_(self.scale_embed, std=0.02)

        self.pos = SinusoidalPosition(lookback, d_model)
        self.core_fusion = CoreFusion(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, layers))
        self.target_queries = nn.Parameter(torch.zeros(1, output_dim, d_model))
        nn.init.normal_(self.target_queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.query_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.scalar_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory_parts: List[torch.Tensor] = []
        for idx, scale in enumerate(self.scales):
            xs = downsample_time(x, scale)
            seasonal, trend = self.decomp(xs)
            tokens = self.season_proj(seasonal) + self.trend_proj(trend)
            tokens = self.pos(tokens) + self.scale_embed[idx].view(1, 1, -1)
            memory_parts.append(tokens)

        var_tokens = self.var_proj(x.transpose(1, 2)) + self.var_embed
        memory_parts.append(var_tokens)
        memory = torch.cat(memory_parts, dim=1)
        memory = self.encoder(self.core_fusion(memory))

        queries = self.target_queries.expand(x.size(0), -1, -1)
        z, _ = self.cross_attn(queries, memory, memory)
        z = self.query_norm(z + self.query_ffn(z))
        return self.scalar_head(z).squeeze(-1)


class SoftsSensorLite(nn.Module):
    """
    SOFTS-style channel-core fusion plus TimeXer-style output queries.

    The model keeps feature-channel tokens explicit, blends them through a
    global core, and adds patch tokens so short transients are still visible.
    """

    def __init__(
        self,
        lookback: int,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
        patch_len: int = 12,
        stride: int = 6,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.patch_len = min(patch_len, lookback)
        self.stride = min(stride, self.patch_len)
        n_patches = 1 + max(0, (lookback - self.patch_len) // self.stride)

        self.var_proj = nn.Linear(lookback, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, input_dim, d_model))
        nn.init.normal_(self.var_embed, std=0.02)
        self.patch_proj = nn.Linear(self.patch_len * input_dim, d_model)
        self.patch_pos = SinusoidalPosition(max(n_patches, 1), d_model)
        self.core_fusion = CoreFusion(d_model, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, layers))
        self.target_queries = nn.Parameter(torch.zeros(1, output_dim, d_model))
        nn.init.normal_(self.target_queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.scalar_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var_tokens = self.var_proj(x.transpose(1, 2)) + self.var_embed
        var_tokens = self.core_fusion(var_tokens)

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().flatten(start_dim=2)
        patch_tokens = self.patch_pos(self.patch_proj(patches))

        memory = self.encoder(torch.cat([var_tokens, patch_tokens], dim=1))
        queries = self.target_queries.expand(x.size(0), -1, -1)
        z, _ = self.cross_attn(queries, memory, memory)
        z = self.norm(z + self.ffn(z))
        return self.scalar_head(z).squeeze(-1)


def split_factorized_model_name(name: str) -> Tuple[str, str]:
    if "__" in name:
        mean_name, resid_name = name.split("__", 1)
        return mean_name.strip(), resid_name.strip()
    return name, name


def split_residual_model_spec(name: str) -> Tuple[str, str]:
    if name.lower().startswith("pca_"):
        return "pca", name[4:].strip()
    if name.lower().startswith("ens_"):
        return "ensemble", name[4:].strip()
    return "direct", name


def build_model(
    name: str,
    lookback: int,
    input_dim: int,
    output_dim: int,
    cfg: Config,
) -> nn.Module:
    key = name.lower()
    if key == "mlp":
        return FlattenMLP(lookback, input_dim, output_dim, cfg.hidden_size, cfg.dropout)
    if key == "dlinear":
        return DLinear(lookback, input_dim, output_dim)
    if key == "lstm_attn":
        return LSTMAttention(input_dim, output_dim, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    if key == "mscnn_lstm_attn":
        return MultiScaleCNNLSTMAttention(
            input_dim, output_dim, cfg.hidden_size, cfg.num_layers, cfg.dropout
        )
    if key == "transformer":
        return TransformerEncoderModel(
            lookback, input_dim, output_dim, cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.dropout
        )
    if key == "patch_transformer":
        return PatchTransformerModel(
            lookback, input_dim, output_dim, cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.dropout
        )
    if key == "itransformer":
        return ITransformerLite(
            lookback, input_dim, output_dim, cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.dropout
        )
    if key == "autoformer_lite":
        return AutoformerLite(
            lookback, input_dim, output_dim, cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.dropout
        )
    if key == "crossformer_lite":
        return CrossformerLite(
            lookback, input_dim, output_dim, cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.dropout
        )
    if key == "pathformer_lite":
        return PathformerLite(
            lookback, input_dim, output_dim, cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.dropout
        )
    if key == "timexer_lite":
        return TimeXerLite(
            lookback, input_dim, output_dim, cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.dropout
        )
    if key == "timecard_sensor_lite":
        return TimeCardSensorLite(
            lookback, input_dim, output_dim, cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.dropout
        )
    if key == "softs_sensor_lite":
        return SoftsSensorLite(
            lookback, input_dim, output_dim, cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.dropout
        )
    raise ValueError(f"Unknown model: {name}")


class CompositeRegressionLoss(nn.Module):
    def __init__(self, base: str = "mse", spatial_weight: float = 0.0) -> None:
        super().__init__()
        self.base = base.lower()
        self.spatial_weight = float(spatial_weight)
        if self.base not in {"mse", "smoothl1", "huber", "mae"}:
            raise ValueError(f"Unknown loss: {base}")

    def _base_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.base == "mae":
            return F.l1_loss(pred, target)
        if self.base in {"smoothl1", "huber"}:
            return F.smooth_l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self._base_loss(pred, target)
        if self.spatial_weight <= 0.0 or pred.ndim != 2 or pred.size(1) <= 1:
            return loss

        pred_c = pred - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)
        pred_std = pred_c.std(dim=1, unbiased=False)
        target_std = target_c.std(dim=1, unbiased=False)
        pred_range = pred_c.max(dim=1).values - pred_c.min(dim=1).values
        target_range = target_c.max(dim=1).values - target_c.min(dim=1).values
        pred_maxdev = pred_c.abs().max(dim=1).values
        target_maxdev = target_c.abs().max(dim=1).values
        spatial = (
            F.mse_loss(pred_std, target_std)
            + 0.5 * F.mse_loss(pred_range, target_range)
            + 0.5 * F.mse_loss(pred_maxdev, target_maxdev)
        )
        return loss + self.spatial_weight * spatial


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    criterion = CompositeRegressionLoss(cfg.loss, cfg.spatial_loss_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model = model.to(device)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= cfg.patience:
            break
        if cfg.epoch_pause_sec > 0:
            time.sleep(cfg.epoch_pause_sec)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        losses.append(float(criterion(model(xb), yb).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    parts: List[np.ndarray] = []
    for xb, _ in loader:
        pred = model(xb.to(device)).detach().cpu().numpy()
        parts.append(pred)
    return np.vstack(parts) if parts else np.empty((0, 0), dtype=np.float32)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def field_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    true_mean = np.mean(y_true, axis=1)
    pred_mean = np.mean(y_pred, axis=1)
    true_std = np.std(y_true, axis=1)
    pred_std = np.std(y_pred, axis=1)
    true_spread95 = np.percentile(y_true, 95, axis=1) - np.percentile(y_true, 5, axis=1)
    pred_spread95 = np.percentile(y_pred, 95, axis=1) - np.percentile(y_pred, 5, axis=1)
    true_maxdev = np.max(np.abs(y_true - true_mean[:, None]), axis=1)
    pred_maxdev = np.max(np.abs(y_pred - pred_mean[:, None]), axis=1)
    true_range = np.max(y_true, axis=1) - np.min(y_true, axis=1)
    pred_range = np.max(y_pred, axis=1) - np.min(y_pred, axis=1)

    out = {
        "field_MAE": float(np.mean(np.abs(err))),
        "field_RMSE": float(np.sqrt(np.mean(err**2))),
        "field_Bias": float(np.mean(err)),
        "per_sensor_MAE_mean": float(np.mean(np.mean(np.abs(err), axis=0))),
        "per_sensor_MAE_max": float(np.max(np.mean(np.abs(err), axis=0))),
        "Tmean_MAE": float(np.mean(np.abs(pred_mean - true_mean))),
        "Tmean_RMSE": float(np.sqrt(np.mean((pred_mean - true_mean) ** 2))),
        "Tmean_R2": r2_score_np(true_mean, pred_mean),
        "Tstd_MAE": float(np.mean(np.abs(pred_std - true_std))),
        "Tstd_RMSE": float(np.sqrt(np.mean((pred_std - true_std) ** 2))),
        "Tstd_R2": r2_score_np(true_std, pred_std),
        "Spread95_MAE": float(np.mean(np.abs(pred_spread95 - true_spread95))),
        "Spread95_RMSE": float(np.sqrt(np.mean((pred_spread95 - true_spread95) ** 2))),
        "Spread95_R2": r2_score_np(true_spread95, pred_spread95),
        "MaxDev_MAE": float(np.mean(np.abs(pred_maxdev - true_maxdev))),
        "MaxDev_RMSE": float(np.sqrt(np.mean((pred_maxdev - true_maxdev) ** 2))),
        "MaxDev_R2": r2_score_np(true_maxdev, pred_maxdev),
        "Range_MAE_secondary": float(np.mean(np.abs(pred_range - true_range))),
        "Range_R2_secondary": r2_score_np(true_range, pred_range),
    }
    return out


def make_loaders(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    train_anchors: np.ndarray,
    val_anchors: np.ndarray,
    test_anchors: np.ndarray,
    horizon_steps: int,
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = WindowDataset(x_scaled, y_scaled, train_anchors, cfg.lookback_steps, horizon_steps)
    val_ds = WindowDataset(x_scaled, y_scaled, val_anchors, cfg.lookback_steps, horizon_steps)
    test_ds = WindowDataset(x_scaled, y_scaled, test_anchors, cfg.lookback_steps, horizon_steps)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def prepare_feature_mode(
    mode: str,
    x_raw: np.ndarray,
    y_temp: np.ndarray,
    feature_cols: Sequence[str],
    train_anchors: np.ndarray,
    horizon_steps: int,
    cfg: Config,
    seed: int,
    run_dir: Path,
) -> Tuple[np.ndarray, List[str]]:
    """Fit feature screening/decomposition on training data only."""
    mode_key = mode.lower()
    train_targets = train_anchors + horizon_steps
    train_rows = train_anchors
    rng = np.random.default_rng(seed)

    if len(train_rows) == 0:
        raise ValueError("Cannot prepare features with empty train split")
    sample_rows = train_rows
    if cfg.selector_sample_limit > 0 and len(sample_rows) > cfg.selector_sample_limit:
        sample_rows = rng.choice(sample_rows, size=cfg.selector_sample_limit, replace=False)

    if mode_key == "raw":
        scaler = ArrayScaler().fit(x_raw[: int(len(x_raw) * 0.70)])
        return scaler.transform(x_raw), list(feature_cols)

    if mode_key.startswith("xgb"):
        k = min(cfg.xgb_top_k, x_raw.shape[1])
        filler = MinMaxFiller().fit(x_raw[train_rows])
        x_train = filler.transform(x_raw[sample_rows])
        # Use field mean as the feature-selection target. It is stable and avoids
        # privileging one sensor position when scoring sensorless input channels.
        row_to_target = {int(r): int(r + horizon_steps) for r in sample_rows}
        y = np.array(
            [np.mean(y_temp[row_to_target[int(r)]]) for r in sample_rows],
            dtype=np.float32,
        )
        model_name = "xgboost"
        try:
            from xgboost import XGBRegressor

            selector = XGBRegressor(
                n_estimators=180,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=4,
                tree_method="hist",
            )
            selector.fit(x_train, y)
            importance = np.asarray(selector.feature_importances_, dtype=np.float64)
        except Exception:
            from sklearn.ensemble import RandomForestRegressor

            model_name = "random_forest_fallback"
            selector = RandomForestRegressor(
                n_estimators=180,
                max_depth=12,
                random_state=seed,
                n_jobs=4,
            )
            selector.fit(x_train, y)
            importance = np.asarray(selector.feature_importances_, dtype=np.float64)

        order = np.argsort(importance)[::-1]
        selected_idx = order[:k]
        selected_names = [feature_cols[int(i)] for i in selected_idx]
        score_df = pd.DataFrame(
            {
                "feature": list(feature_cols),
                "importance": importance,
                "rank": pd.Series(np.argsort(np.argsort(-importance)) + 1),
                "selector": model_name,
            }
        ).sort_values("rank")
        score_df.to_csv(run_dir / f"feature_importance_{mode_key}_seed{seed}.csv", index=False)

        x_selected = x_raw[:, selected_idx]
        scaler = ArrayScaler().fit(x_selected[: int(len(x_selected) * 0.70)])
        return scaler.transform(x_selected), selected_names

    if mode_key in ("nmf", "nonnegative", "nonnegative_factor"):
        n_comp = min(cfg.nmf_components, x_raw.shape[1])
        filler = MinMaxFiller().fit(x_raw[train_rows])
        x_nonneg = filler.transform(x_raw)
        x_fit = x_nonneg[sample_rows]
        from sklearn.decomposition import NMF

        nmf = NMF(
            n_components=n_comp,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            max_iter=500,
            random_state=seed,
        )
        nmf.fit(x_fit)
        x_latent = nmf.transform(x_nonneg).astype(np.float32)
        names = [f"nmf_{i + 1:02d}" for i in range(n_comp)]
        pd.DataFrame(
            nmf.components_,
            columns=list(feature_cols),
            index=names,
        ).to_csv(run_dir / f"nmf_components_seed{seed}.csv")
        scaler = ArrayScaler().fit(x_latent[: int(len(x_latent) * 0.70)])
        return scaler.transform(x_latent), names

    raise ValueError(f"Unknown feature mode: {mode}")


def train_direct(
    model_name: str,
    x_scaled: np.ndarray,
    y_temp: np.ndarray,
    train_anchors: np.ndarray,
    val_anchors: np.ndarray,
    test_anchors: np.ndarray,
    horizon_steps: int,
    cfg: Config,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    train_targets = train_anchors + horizon_steps
    y_scaler = ArrayScaler().fit(y_temp[train_targets])
    y_scaled = y_scaler.transform(y_temp)
    loaders = make_loaders(
        x_scaled, y_scaled, train_anchors, val_anchors, test_anchors, horizon_steps, cfg
    )
    model = build_model(
        model_name,
        cfg.lookback_steps,
        x_scaled.shape[1],
        y_temp.shape[1],
        cfg,
    )
    model, best_val = train_one_model(model, loaders[0], loaders[1], cfg, device)
    pred_scaled = predict_model(model, loaders[2], device)
    return y_scaler.inverse_transform(pred_scaled), best_val


def train_factorized(
    model_name: str,
    x_scaled: np.ndarray,
    y_temp: np.ndarray,
    train_anchors: np.ndarray,
    val_anchors: np.ndarray,
    test_anchors: np.ndarray,
    horizon_steps: int,
    cfg: Config,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    mean_raw = np.mean(y_temp, axis=1, keepdims=True)
    resid_raw = y_temp - mean_raw
    train_targets = train_anchors + horizon_steps

    mean_scaler = ArrayScaler().fit(mean_raw[train_targets])
    mean_scaled = mean_scaler.transform(mean_raw)

    mean_loaders = make_loaders(
        x_scaled, mean_scaled, train_anchors, val_anchors, test_anchors, horizon_steps, cfg
    )

    mean_model_name, resid_spec = split_factorized_model_name(model_name)
    resid_decomp, resid_model_name = split_residual_model_spec(resid_spec)
    mean_model = build_model(mean_model_name, cfg.lookback_steps, x_scaled.shape[1], 1, cfg)

    pca_model = None
    coeff_scaler = None
    resid_scaler = None
    if resid_decomp == "pca":
        from sklearn.decomposition import PCA

        n_comp = max(1, min(cfg.lowrank_rank, y_temp.shape[1], len(train_targets)))
        pca_model = PCA(n_components=n_comp, random_state=0)
        pca_model.fit(resid_raw[train_targets])
        finite_rows = np.isfinite(resid_raw).all(axis=1)
        coeff_raw = np.full((len(y_temp), n_comp), np.nan, dtype=np.float32)
        coeff_raw[finite_rows] = pca_model.transform(resid_raw[finite_rows]).astype(np.float32)
        coeff_scaler = ArrayScaler().fit(coeff_raw[train_targets])
        resid_target_scaled = coeff_scaler.transform(coeff_raw)
        resid_output_dim = n_comp
    else:
        resid_scaler = ArrayScaler().fit(resid_raw[train_targets])
        resid_target_scaled = resid_scaler.transform(resid_raw)
        resid_output_dim = y_temp.shape[1]

    resid_loaders = make_loaders(
        x_scaled,
        resid_target_scaled,
        train_anchors,
        val_anchors,
        test_anchors,
        horizon_steps,
        cfg,
    )
    resid_model = build_model(
        resid_model_name,
        cfg.lookback_steps,
        x_scaled.shape[1],
        resid_output_dim,
        cfg,
    )
    mean_model, mean_val = train_one_model(mean_model, mean_loaders[0], mean_loaders[1], cfg, device)
    resid_model, resid_val = train_one_model(
        resid_model, resid_loaders[0], resid_loaders[1], cfg, device
    )

    pred_mean = mean_scaler.inverse_transform(predict_model(mean_model, mean_loaders[2], device))
    pred_resid_scaled = predict_model(resid_model, resid_loaders[2], device)
    if resid_decomp == "pca":
        if pca_model is None or coeff_scaler is None:
            raise RuntimeError("PCA residual model was not fitted")
        pred_coeff = coeff_scaler.inverse_transform(pred_resid_scaled)
        pred_resid = (pred_coeff @ pca_model.components_ + pca_model.mean_).astype(np.float32)
    else:
        if resid_scaler is None:
            raise RuntimeError("Residual scaler was not fitted")
        pred_resid = resid_scaler.inverse_transform(pred_resid_scaled)
    pred_resid = pred_resid - np.mean(pred_resid, axis=1, keepdims=True)
    return pred_mean + pred_resid, float((mean_val + resid_val) / 2.0)


def save_prediction_sample(
    out_path: Path,
    target_times: Sequence[pd.Timestamp],
    temp_cols: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_rows: int = 2000,
) -> None:
    n = min(max_rows, len(y_true))
    data = {"target_time": [str(t) for t in target_times[:n]]}
    for idx, col in enumerate(temp_cols):
        data[f"true_{col}"] = y_true[:n, idx]
        data[f"pred_{col}"] = y_pred[:n, idx]
    pd.DataFrame(data).to_csv(out_path, index=False)


def run_experiments(cfg: Config) -> Path:
    if cfg.site not in SITE_CONFIGS:
        raise ValueError(f"Unknown site {cfg.site}. Choices: {sorted(SITE_CONFIGS)}")

    if cfg.data_file == "":
        cfg.data_file = str(SITE_CONFIGS[cfg.site]["data_file"])
    if cfg.smoke:
        cfg.max_rows = cfg.max_rows or 40000
        cfg.epochs = min(cfg.epochs, 2)
        cfg.patience = min(cfg.patience, 2)
        cfg.horizons = (cfg.horizons[0],)
        cfg.seeds = (cfg.seeds[0],)
        cfg.models = cfg.models[:2]
        cfg.feature_modes = cfg.feature_modes[:2]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"field_{cfg.site}_{cfg.feature_set}_{stamp}"
    run_dir = Path(cfg.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.torch_num_threads > 0:
        torch.set_num_threads(cfg.torch_num_threads)
        torch.set_num_interop_threads(max(1, min(2, cfg.torch_num_threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and cfg.cuda_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(min(cfg.cuda_memory_fraction, 1.0))
    df, temp_cols, feature_cols, irr_col = load_site_dataframe(cfg)
    x_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_temp = df[temp_cols].to_numpy(dtype=np.float32)

    records: List[Dict[str, object]] = []
    config_payload = asdict(cfg)
    config_payload.update(
        {
            "run_dir": str(run_dir),
            "device": str(device),
            "n_rows": int(len(df)),
            "temp_cols": temp_cols,
            "feature_cols": feature_cols,
            "irradiance_col": irr_col,
        }
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Rows: {len(df)} | features: {len(feature_cols)} | temps: {len(temp_cols)}")
    print(f"Irradiance filter column: {irr_col or 'none'}")

    for seed in cfg.seeds:
        set_seed(seed)
        for horizon_min in cfg.horizons:
            if horizon_min % cfg.resample_minutes != 0:
                raise ValueError("horizon must be divisible by resample_minutes")
            horizon_steps = horizon_min // cfg.resample_minutes
            anchors = build_anchors(df, x_raw, y_temp, irr_col, horizon_steps, cfg)
            train_anchors, val_anchors, test_anchors = chronological_split(
                anchors, len(df), horizon_steps
            )
            if min(len(train_anchors), len(val_anchors), len(test_anchors)) == 0:
                raise ValueError(
                    f"Empty split for horizon {horizon_min}: "
                    f"train={len(train_anchors)}, val={len(val_anchors)}, test={len(test_anchors)}"
                )

            y_true = y_temp[test_anchors + horizon_steps]
            target_times = df.index[test_anchors + horizon_steps]

            for feature_mode in cfg.feature_modes:
                x_scaled, active_features = prepare_feature_mode(
                    feature_mode,
                    x_raw,
                    y_temp,
                    feature_cols,
                    train_anchors,
                    horizon_steps,
                    cfg,
                    seed,
                    run_dir,
                )
                print(
                    f"\nFeature mode: {feature_mode} | active features: "
                    f"{len(active_features)}"
                )
                (run_dir / f"active_features_{feature_mode}_seed{seed}_h{horizon_min}.json").write_text(
                    json.dumps(active_features, indent=2), encoding="utf-8"
                )

                for strategy in cfg.strategies:
                    for model_name in cfg.models:
                        print(
                            f"\nseed={seed} horizon={horizon_min} feature_mode={feature_mode} "
                            f"strategy={strategy} model={model_name}"
                        )
                        start = time.time()
                        if strategy == "direct":
                            y_pred, best_val = train_direct(
                                model_name,
                                x_scaled,
                                y_temp,
                                train_anchors,
                                val_anchors,
                                test_anchors,
                                horizon_steps,
                                cfg,
                                device,
                            )
                        elif strategy == "factorized":
                            y_pred, best_val = train_factorized(
                                model_name,
                                x_scaled,
                                y_temp,
                                train_anchors,
                                val_anchors,
                                test_anchors,
                                horizon_steps,
                                cfg,
                                device,
                            )
                        else:
                            raise ValueError(f"Unknown strategy: {strategy}")

                        metrics = field_metrics(y_true, y_pred)
                        rec: Dict[str, object] = {
                            "site": cfg.site,
                            "seed": seed,
                            "horizon_min": horizon_min,
                            "feature_mode": feature_mode,
                            "strategy": strategy,
                            "model": model_name,
                            "feature_set": cfg.feature_set,
                            "n_features": len(active_features),
                            "n_temp_inputs": 0,
                            "loss": cfg.loss,
                            "spatial_loss_weight": cfg.spatial_loss_weight,
                            "train_samples": len(train_anchors),
                            "val_samples": len(val_anchors),
                            "test_samples": len(test_anchors),
                            "best_val_loss": best_val,
                            "elapsed_sec": round(time.time() - start, 2),
                        }
                        rec.update(metrics)
                        records.append(rec)
                        pd.DataFrame(records).to_csv(run_dir / "metrics_field.csv", index=False)

                        pred_name = (
                            f"pred_{cfg.site}_h{horizon_min}_{feature_mode}_"
                            f"{strategy}_{model_name}_seed{seed}.csv"
                        )
                        save_prediction_sample(
                            run_dir / pred_name,
                            target_times,
                            temp_cols,
                            y_true,
                            y_pred,
                        )
                        print(
                            f"field_MAE={metrics['field_MAE']:.3f} "
                            f"Tmean_MAE={metrics['Tmean_MAE']:.3f} "
                            f"Tstd_MAE={metrics['Tstd_MAE']:.3f} "
                            f"Spread95_MAE={metrics['Spread95_MAE']:.3f}"
                        )

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="oakville", choices=sorted(SITE_CONFIGS.keys()))
    parser.add_argument("--data-file", default="", help="Override CSV path.")
    parser.add_argument("--output-root", default="code/runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--feature-set", default="metop", choices=["met", "metop"])
    parser.add_argument("--resample-minutes", type=int, default=1)
    parser.add_argument("--lookback-minutes", type=int, default=60)
    parser.add_argument("--horizons", default="1")
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--models",
        default="mscnn_lstm_attn,transformer,patch_transformer,itransformer",
        help=(
            "Comma-separated: mlp,dlinear,lstm_attn,mscnn_lstm_attn,transformer,"
            "patch_transformer,itransformer,autoformer_lite,crossformer_lite,"
            "pathformer_lite,timexer_lite,timecard_sensor_lite,softs_sensor_lite. "
            "For factorized runs, mean__residual combines two model names; "
            "prefix residual with pca_ to predict low-rank spatial residual coefficients."
        ),
    )
    parser.add_argument("--strategies", default="factorized")
    parser.add_argument(
        "--feature-modes",
        default="raw",
        help="Comma-separated: raw,xgb_topk,nmf",
    )
    parser.add_argument("--xgb-top-k", type=int, default=12)
    parser.add_argument("--nmf-components", type=int, default=8)
    parser.add_argument("--lowrank-rank", type=int, default=6)
    parser.add_argument("--selector-sample-limit", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--loss", default="mse", choices=["mse", "smoothl1", "huber", "mae"])
    parser.add_argument("--spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--epoch-pause-sec", type=float, default=0.0)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-irradiance", type=float, default=200.0)
    parser.add_argument("--input-min-mean-irradiance", type=float, default=100.0)
    parser.add_argument("--short-gap-limit-steps", type=int, default=2)
    parser.add_argument("--missing-feature-threshold", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = Config(
        site=args.site,
        data_file=args.data_file,
        output_root=args.output_root,
        run_name=args.run_name,
        feature_set=args.feature_set,
        resample_minutes=args.resample_minutes,
        lookback_minutes=args.lookback_minutes,
        horizons=parse_csv_tuple(args.horizons, int),
        seeds=parse_csv_tuple(args.seeds, int),
        models=parse_csv_tuple(args.models, str),
        strategies=parse_csv_tuple(args.strategies, str),
        feature_modes=parse_csv_tuple(args.feature_modes, str),
        xgb_top_k=args.xgb_top_k,
        nmf_components=args.nmf_components,
        lowrank_rank=args.lowrank_rank,
        selector_sample_limit=args.selector_sample_limit,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        loss=args.loss,
        spatial_loss_weight=args.spatial_loss_weight,
        epoch_pause_sec=args.epoch_pause_sec,
        torch_num_threads=args.torch_num_threads,
        cuda_memory_fraction=args.cuda_memory_fraction,
        num_workers=args.num_workers,
        min_irradiance=args.min_irradiance,
        input_min_mean_irradiance=args.input_min_mean_irradiance,
        short_gap_limit_steps=args.short_gap_limit_steps,
        missing_feature_threshold=args.missing_feature_threshold,
        max_rows=args.max_rows,
        smoke=args.smoke,
    )
    run_dir = run_experiments(cfg)
    print(f"\nCompleted. Metrics: {run_dir / 'metrics_field.csv'}")


if __name__ == "__main__":
    main()
