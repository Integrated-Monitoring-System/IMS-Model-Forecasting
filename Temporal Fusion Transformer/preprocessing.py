"""
preprocessing.py
==========
Pipeline preprocessing data sensor upstream (pressure, flowrate, temperature,
vibration) dari file CSV, untuk persiapan input model Temporal Fusion
Transformer (TFT).

Scope script ini HANYA sampai tahap data siap pakai (cleaned, feature-engineered,
tersimpan sebagai parquet + metadata.json). Pembangunan `TimeSeriesDataSet`
(pytorch_forecasting) dan training model TFT dilakukan di script terpisah.

Cara pakai (CLI):
    python preprocessing.py --input data/sensor_pep_prod.csv \
        --output-dir output/ --group-cols well_id \
        --target-cols pressure,flowrate,temperature,vibration \
        --datetime-col timestamp --freq H

Bisa juga dipakai sebagai module (di notebook / script lain):
    from preprocessing import Config, run_pipeline
    cfg = Config(input_path="data.csv", output_dir="output/", ...)
    result = run_pipeline(cfg)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocessing")


# =============================================================================
# 1. KONFIGURASI
# =============================================================================
@dataclass
class Config:
    # --- I/O ---
    input_path: str
    output_dir: str

    # --- kolom-kolom penting ---
    datetime_col: str = "timestamp"
    group_cols: list = field(default_factory=lambda: ["well_id"])
    target_cols: list = field(
        default_factory=lambda: ["pressure", "flowrate", "temperature", "vibration"]
    )
    static_categoricals: list = field(default_factory=list)  # mis. ["field_name", "equipment_type"]

    # --- resampling & cleaning ---
    freq: str = "H"                 # frekuensi resample: "H"=jam, "D"=hari, "15min", dst.
    max_missing_ratio: float = 0.4  # drop grup kalau rasio NaN > ini (dihitung sebelum imputasi)
    outlier_method: str = "iqr"     # "iqr" atau "zscore"
    outlier_threshold: float = 3.0  # multiplier IQR atau jumlah std dev

    # --- split (berbasis rasio, dipetakan ke time_idx cutoff per grup) ---
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    # test_ratio = 1 - train_ratio - val_ratio

    # --- info untuk TFT (disimpan di metadata, dipakai script model) ---
    max_encoder_length: int = 168     # panjang histori/lookback window
    max_prediction_length: int = 24   # panjang horizon forecast

    def __post_init__(self):
        if not (0 < self.train_ratio < 1) or not (0 < self.val_ratio < 1):
            raise ValueError("train_ratio dan val_ratio harus di antara 0 dan 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio harus < 1 (sisanya untuk test)")


# =============================================================================
# 2. LOAD DATA
# =============================================================================
def load_data(cfg: Config) -> pd.DataFrame:
    logger.info(f"Load data dari CSV: {cfg.input_path}")
    df = pd.read_csv(cfg.input_path)
    logger.info(f"  -> {len(df):,} baris, {len(df.columns)} kolom")
    return df


# =============================================================================
# 3. UTILITAS GROUP KEY
# =============================================================================
def _make_group_key(df: pd.DataFrame, group_cols: list) -> pd.Series:
    """Gabungkan group_cols jadi satu kolom string, memudahkan groupby internal."""
    if len(group_cols) == 1:
        return df[group_cols[0]].astype(str)
    return df[group_cols].astype(str).agg("_".join, axis=1)


# =============================================================================
# 4. CLEANING
# =============================================================================
def parse_and_sort(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=[cfg.datetime_col])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"Drop {n_dropped:,} baris dengan timestamp tidak valid/kosong")

    df = df.sort_values(cfg.group_cols + [cfg.datetime_col]).reset_index(drop=True)
    return df


def drop_duplicates(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=cfg.group_cols + [cfg.datetime_col], keep="last")
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info(f"Drop {n_dropped:,} baris duplikat (kombinasi grup + timestamp sama)")
    return df.reset_index(drop=True)


def resample_per_group(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Resample tiap grup ke frekuensi seragam (cfg.freq) supaya time_idx nanti kontinu."""
    logger.info(f"Resample per grup ke frekuensi '{cfg.freq}'")
    df = df.copy()
    df["_group_key"] = _make_group_key(df, cfg.group_cols)

    out_frames = []
    for key, g in df.groupby("_group_key"):
        g = g.set_index(cfg.datetime_col)

        numeric_agg = g[cfg.target_cols].resample(cfg.freq).mean()

        # kolom identitas grup & static categorical -> ambil nilai pertama tiap bin waktu
        id_cols = cfg.group_cols + cfg.static_categoricals
        id_agg = g[id_cols].resample(cfg.freq).first()

        merged = pd.concat([numeric_agg, id_agg], axis=1)
        merged["_group_key"] = key
        out_frames.append(merged.reset_index())

    result = pd.concat(out_frames, ignore_index=True)

    # isi ulang kolom identitas yang mungkin kosong di bin waktu tanpa data mentah
    for col in cfg.group_cols + cfg.static_categoricals:
        result[col] = result.groupby("_group_key")[col].transform(lambda s: s.ffill().bfill())

    logger.info(f"  -> {len(result):,} baris setelah resample ({df['_group_key'].nunique()} grup)")
    return result


def filter_sparse_groups(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Buang grup yang rasio missing-nya (di salah satu target) melebihi ambang batas."""
    ratios = df.groupby("_group_key")[cfg.target_cols].apply(lambda g: g.isna().mean().max())
    keep_keys = ratios[ratios <= cfg.max_missing_ratio].index
    dropped_keys = ratios[ratios > cfg.max_missing_ratio].index

    if len(dropped_keys):
        logger.warning(
            f"Drop {len(dropped_keys)} grup karena missing ratio > {cfg.max_missing_ratio:.0%}: "
            f"{list(dropped_keys)[:10]}{'...' if len(dropped_keys) > 10 else ''}"
        )

    df = df[df["_group_key"].isin(keep_keys)].reset_index(drop=True)
    return df


def handle_outliers(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clip outlier per grup per kolom target (bukan drop baris, supaya time_idx tetap kontinu)."""
    logger.info(f"Handle outlier dengan metode '{cfg.outlier_method}' (threshold={cfg.outlier_threshold})")
    df = df.copy()

    def _bounds(s: pd.Series):
        s_valid = s.dropna()
        if len(s_valid) < 4:
            return -np.inf, np.inf
        if cfg.outlier_method == "iqr":
            q1, q3 = s_valid.quantile(0.25), s_valid.quantile(0.75)
            iqr = q3 - q1
            return q1 - cfg.outlier_threshold * iqr, q3 + cfg.outlier_threshold * iqr
        elif cfg.outlier_method == "zscore":
            mean, std = s_valid.mean(), s_valid.std()
            if std == 0 or np.isnan(std):
                return -np.inf, np.inf
            return mean - cfg.outlier_threshold * std, mean + cfg.outlier_threshold * std
        else:
            raise ValueError(f"outlier_method tidak dikenal: {cfg.outlier_method}")

    total_clipped = 0
    for col in cfg.target_cols:
        def _clip(s):
            nonlocal total_clipped
            lower, upper = _bounds(s)
            n_out = ((s < lower) | (s > upper)).sum()
            total_clipped += int(n_out)
            return s.clip(lower, upper)

        df[col] = df.groupby("_group_key")[col].transform(_clip)

    logger.info(f"  -> total {total_clipped:,} nilai di-clip di seluruh kolom target")
    return df


def impute_missing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Interpolasi linear per grup untuk mengisi gap kecil, lalu ffill/bfill sisa di ujung."""
    logger.info("Imputasi missing value (interpolasi linear per grup)")
    df = df.sort_values(["_group_key", cfg.datetime_col]).copy()

    n_missing_before = df[cfg.target_cols].isna().sum().sum()

    # NB: pakai transform per kolom (bukan groupby().apply()) karena beberapa versi
    # pandas membuang kolom pengelompokan ("_group_key") saat apply() dipakai untuk
    # mengembalikan DataFrame utuh dengan group_keys=False.
    for col in cfg.target_cols:
        df[col] = df.groupby("_group_key")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )

    n_missing_after = df[cfg.target_cols].isna().sum().sum()
    logger.info(f"  -> missing value: {n_missing_before:,} -> {n_missing_after:,}")

    # kalau masih ada NaN (grup dengan seluruh datanya kosong), drop baris tsb
    if n_missing_after > 0:
        n_before = len(df)
        df = df.dropna(subset=cfg.target_cols)
        logger.warning(f"  -> drop {n_before - len(df):,} baris sisa NaN yang tidak bisa diinterpolasi")

    return df.reset_index(drop=True)


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
def add_time_idx(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """time_idx: integer berurutan per grup, syarat wajib TimeSeriesDataSet (pytorch_forecasting)."""
    df = df.sort_values(["_group_key", cfg.datetime_col]).copy()
    df["time_idx"] = df.groupby("_group_key").cumcount()
    return df


def add_calendar_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fitur kalender -> kandidat time_varying_known_reals (diketahui di masa depan)."""
    df = df.copy()
    dt = df[cfg.datetime_col]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return df


KNOWN_CALENDAR_FEATURES = ["hour", "dayofweek", "day", "month", "is_weekend"]


# =============================================================================
# 6. SPLIT (cutoff time_idx per grup, sesuai kebutuhan TFT)
# =============================================================================
def compute_split_cutoffs(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Hitung cutoff time_idx per grup untuk train/val, berdasarkan rasio.
    Cutoff ini yang dipakai script model untuk TimeSeriesDataSet.from_dataset(...),
    BUKAN dengan memotong dataframe jadi 3 file terpisah (karena TFT butuh histori
    sebelum cutoff sebagai encoder context untuk val/test).
    """
    rows = []
    for key, g in df.groupby("_group_key"):
        n = g["time_idx"].max() + 1
        train_cutoff = int(n * cfg.train_ratio) - 1
        val_cutoff = int(n * (cfg.train_ratio + cfg.val_ratio)) - 1
        rows.append({
            "_group_key": key,
            "n_timesteps": n,
            "train_cutoff_time_idx": max(train_cutoff, 0),
            "val_cutoff_time_idx": max(val_cutoff, train_cutoff + 1),
        })
    return pd.DataFrame(rows)


def make_reference_split(df: pd.DataFrame, cutoffs: pd.DataFrame, cfg: Config):
    """
    Versi split sederhana (potong dataframe jadi 3) HANYA untuk keperluan EDA/sanity-check
    cepat (mis. cek distribusi train vs test). Untuk training TFT sesungguhnya, pakai
    train_cutoff_time_idx / val_cutoff_time_idx dari metadata + TimeSeriesDataSet.from_dataset.
    """
    merged = df.merge(cutoffs, on="_group_key", how="left")
    train = merged[merged["time_idx"] <= merged["train_cutoff_time_idx"]]
    val = merged[
        (merged["time_idx"] > merged["train_cutoff_time_idx"])
        & (merged["time_idx"] <= merged["val_cutoff_time_idx"])
    ]
    test = merged[merged["time_idx"] > merged["val_cutoff_time_idx"]]

    drop_cols = ["train_cutoff_time_idx", "val_cutoff_time_idx", "n_timesteps"]
    train = train.drop(columns=drop_cols).reset_index(drop=True)
    val = val.drop(columns=drop_cols).reset_index(drop=True)
    test = test.drop(columns=drop_cols).reset_index(drop=True)
    return train, val, test


# =============================================================================
# 7. SIMPAN OUTPUT
# =============================================================================
def save_outputs(
    df_clean: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cutoffs: pd.DataFrame,
    cfg: Config,
) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_clean_save = df_clean.drop(columns=["_group_key"], errors="ignore")
    train_save = train.drop(columns=["_group_key"], errors="ignore")
    val_save = val.drop(columns=["_group_key"], errors="ignore")
    test_save = test.drop(columns=["_group_key"], errors="ignore")

    df_clean_save.to_parquet(out_dir / "full_clean.parquet", index=False)
    train_save.to_parquet(out_dir / "reference_train.parquet", index=False)
    val_save.to_parquet(out_dir / "reference_val.parquet", index=False)
    test_save.to_parquet(out_dir / "reference_test.parquet", index=False)
    cutoffs.to_csv(out_dir / "split_cutoffs.csv", index=False)

    metadata = {
        "datetime_col": cfg.datetime_col,
        "group_cols": cfg.group_cols,
        "static_categoricals": cfg.static_categoricals,
        "target_cols": cfg.target_cols,
        "time_varying_known_reals": ["time_idx"] + KNOWN_CALENDAR_FEATURES,
        "time_varying_unknown_reals": cfg.target_cols,
        "freq": cfg.freq,
        "max_encoder_length": cfg.max_encoder_length,
        "max_prediction_length": cfg.max_prediction_length,
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "n_groups": df_clean["_group_key"].nunique(),
        "n_rows_clean": len(df_clean),
        "date_range": [
            str(df_clean[cfg.datetime_col].min()),
            str(df_clean[cfg.datetime_col].max()),
        ],
        "note_split": (
            "reference_train/val/test.parquet HANYA untuk EDA/sanity-check cepat. "
            "Untuk training TFT, gunakan full_clean.parquet + train_cutoff_time_idx / "
            "val_cutoff_time_idx per grup di split_cutoffs.csv, bersama "
            "TimeSeriesDataSet.from_dataset() dari pytorch_forecasting supaya val/test "
            "tetap punya encoder context (histori) yang benar."
        ),
        "note_scaling": (
            "Scaling/normalisasi SENGAJA tidak dilakukan di tahap ini. Gunakan "
            "GroupNormalizer (pytorch_forecasting) di script model, di-fit HANYA dari "
            "training set, supaya tidak ada data leakage."
        ),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Output tersimpan di: {out_dir.resolve()}")
    logger.info(
        f"  - full_clean.parquet          ({len(df_clean_save):,} baris)  <- pakai ini untuk modeling"
    )
    logger.info(f"  - reference_train.parquet     ({len(train_save):,} baris)  [EDA only]")
    logger.info(f"  - reference_val.parquet       ({len(val_save):,} baris)  [EDA only]")
    logger.info(f"  - reference_test.parquet      ({len(test_save):,} baris)  [EDA only]")
    logger.info("  - split_cutoffs.csv, metadata.json")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline(cfg: Config) -> dict:
    logger.info("=" * 70)
    logger.info("MULAI PIPELINE PREPROCESSING")
    logger.info("=" * 70)

    df = load_data(cfg)
    df = parse_and_sort(df, cfg)
    df = drop_duplicates(df, cfg)
    df = resample_per_group(df, cfg)
    df = filter_sparse_groups(df, cfg)
    df = handle_outliers(df, cfg)
    df = impute_missing(df, cfg)
    df = add_time_idx(df, cfg)
    df = add_calendar_features(df, cfg)

    cutoffs = compute_split_cutoffs(df, cfg)
    train, val, test = make_reference_split(df, cutoffs, cfg)

    save_outputs(df, train, val, test, cutoffs, cfg)

    logger.info("=" * 70)
    logger.info("PIPELINE SELESAI")
    logger.info("=" * 70)

    return {"full_clean": df, "train": train, "val": val, "test": test, "cutoffs": cutoffs}


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Preprocessing data sensor CSV untuk TFT")

    p.add_argument("--input", dest="input_path", required=True, help="path file CSV")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--datetime-col", default="timestamp")
    p.add_argument("--group-cols", default="well_id", help="pisahkan dengan koma kalau lebih dari 1")
    p.add_argument(
        "--target-cols",
        default="pressure,flowrate,temperature,vibration",
        help="pisahkan dengan koma",
    )
    p.add_argument("--static-categoricals", default="", help="pisahkan dengan koma, boleh kosong")

    p.add_argument("--freq", default="H")
    p.add_argument("--max-missing-ratio", type=float, default=0.4)
    p.add_argument("--outlier-method", choices=["iqr", "zscore"], default="iqr")
    p.add_argument("--outlier-threshold", type=float, default=3.0)

    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)

    p.add_argument("--max-encoder-length", type=int, default=168)
    p.add_argument("--max-prediction-length", type=int, default=24)

    args = p.parse_args()

    def _split(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    cfg = Config(
        input_path=args.input_path,
        output_dir=args.output_dir,
        datetime_col=args.datetime_col,
        group_cols=_split(args.group_cols),
        target_cols=_split(args.target_cols),
        static_categoricals=_split(args.static_categoricals),
        freq=args.freq,
        max_missing_ratio=args.max_missing_ratio,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
    )
    return cfg


if __name__ == "__main__":
    config = _parse_args()
    logger.info(f"Konfigurasi:\n{json.dumps(asdict(config), indent=2, default=str)}")
    run_pipeline(config)