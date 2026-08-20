"""
train.py
========
NBEATSx Forecasting dengan fixed temporal split.

Tujuan:
    TRAIN:
        Semua data sampai dengan 30 Juni 2026.

    TEST:
        Data mulai 1 Juli 2026 sampai data terakhir yang tersedia
        di dataset.

    Model:
        NBEATSx

    Target:
        pressure / flowrate / temperature

    Exogenous:
        FUTURE KNOWN ONLY:
            - hour
            - dayofweek
            - day
            - month
            - is_weekend

    Catatan penting:
        flowrate dan temperature TIDAK digunakan sebagai hist_exog
        dalam versi ini.

        Alasannya:
        Jika kita ingin memprediksi pressure pada Juli 2026,
        nilai flowrate dan temperature Juli 2026 belum diketahui
        pada saat forecasting. Menggunakannya akan menyebabkan
        leakage.

    Forecast:
        Forecast dilakukan secara recursive dengan horizon kecil.
        Setelah menghasilkan prediksi, prediksi tersebut dimasukkan
        kembali sebagai histori untuk forecast berikutnya.

    Evaluasi:
        Actual Juli 2026 dibandingkan dengan prediction Juli 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("train")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    data_dir: str
    output_dir: str
    target: str

    # Fixed temporal split
    train_end: str = "2026-06-30 23:59:59"
    test_start: str = "2026-07-01 00:00:00"
    test_end: str = "2026-07-31 23:59:59"

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    horizon: int | None = None
    input_size: int | None = None

    scaler_type: str = "standard"

    max_steps: int = 300
    learning_rate: float = 1e-3
    batch_size: int = 32

    val_check_steps: int = 25
    early_stop_patience_steps: int = 5

    random_seed: int = 42


# =============================================================================
# RANDOM SEED
# =============================================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except Exception:
        pass


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(cfg: Config):

    logger.info("=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)

    data_dir = Path(cfg.data_dir)

    parquet_file = data_dir / "full_clean.parquet"
    metadata_file = data_dir / "metadata.json"

    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{parquet_file}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"metadata.json tidak ditemukan:\n{metadata_file}"
        )

    df = pd.read_parquet(parquet_file)

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # -------------------------------------------------------------------------
    # DATETIME
    # -------------------------------------------------------------------------

    datetime_col = metadata.get("datetime_col", "DateTime")

    if datetime_col not in df.columns:
        raise ValueError(
            f"Kolom datetime '{datetime_col}' tidak ditemukan."
        )

    df[datetime_col] = pd.to_datetime(
        df[datetime_col],
        errors="coerce"
    )

    df = df.dropna(subset=[datetime_col])

    # -------------------------------------------------------------------------
    # TARGET
    # -------------------------------------------------------------------------

    if cfg.target not in metadata["target_cols"]:
        raise ValueError(
            f"Target '{cfg.target}' tidak tersedia.\n"
            f"Target tersedia: {metadata['target_cols']}"
        )

    if cfg.target not in df.columns:
        raise ValueError(
            f"Kolom target '{cfg.target}' tidak ditemukan di dataset."
        )

    # -------------------------------------------------------------------------
    # SORT
    # -------------------------------------------------------------------------

    group_col = metadata.get(
        "group_col",
        "equipment_id"
    )

    df = df.sort_values(
        [group_col, datetime_col]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # BASIC CLEANING
    # -------------------------------------------------------------------------

    df[cfg.target] = pd.to_numeric(
        df[cfg.target],
        errors="coerce"
    )

    df = df.dropna(
        subset=[cfg.target]
    ).reset_index(drop=True)

    logger.info(f"Rows       : {len(df):,}")
    logger.info(
        f"Equipment  : {df[group_col].nunique()}"
    )
    logger.info(
        f"Date range : "
        f"{df[datetime_col].min()} -> {df[datetime_col].max()}"
    )

    return df, metadata


# =============================================================================
# CREATE CALENDAR FEATURES
# =============================================================================

def create_calendar_features(
    df: pd.DataFrame,
    datetime_col: str
) -> pd.DataFrame:

    df = df.copy()

    dt = pd.to_datetime(
        df[datetime_col]
    )

    df["hour"] = dt.dt.hour.astype(float)

    df["dayofweek"] = dt.dt.dayofweek.astype(float)

    df["day"] = dt.dt.day.astype(float)

    df["month"] = dt.dt.month.astype(float)

    df["is_weekend"] = (
        dt.dt.dayofweek >= 5
    ).astype(float)

    return df


# =============================================================================
# FIXED TRAIN / TEST SPLIT
# =============================================================================

def split_train_test(
    df: pd.DataFrame,
    cfg: Config,
    metadata: dict
):

    logger.info("=" * 70)
    logger.info("FIXED TRAIN / TEST SPLIT")
    logger.info("=" * 70)

    datetime_col = metadata["datetime_col"]
    group_col = metadata["group_col"]

    train_end = pd.Timestamp(
        cfg.train_end
    )

    test_start = pd.Timestamp(
        cfg.test_start
    )

    test_end_requested = pd.Timestamp(
        cfg.test_end
    )

    # -------------------------------------------------------------------------
    # Important:
    # Dataset mungkin hanya tersedia sampai 22 Juli.
    #
    # Jadi jangan membuat data palsu sampai 31 Juli.
    # -------------------------------------------------------------------------

    actual_data_end = df[datetime_col].max()

    test_end = min(
        test_end_requested,
        actual_data_end
    )

    train_df = df[
        df[datetime_col] <= train_end
    ].copy()

    test_df = df[
        (df[datetime_col] >= test_start)
        &
        (df[datetime_col] <= test_end)
    ].copy()

    if len(train_df) == 0:
        raise ValueError(
            "TRAIN kosong. Periksa train_end."
        )

    if len(test_df) == 0:
        raise ValueError(
            "TEST kosong. Periksa test_start/test_end."
        )

    logger.info(
        f"TRAIN : "
        f"{train_df[datetime_col].min()} -> "
        f"{train_df[datetime_col].max()}"
    )

    logger.info(
        f"TEST  : "
        f"{test_df[datetime_col].min()} -> "
        f"{test_df[datetime_col].max()}"
    )

    logger.info(
        f"Train rows : {len(train_df):,}"
    )

    logger.info(
        f"Test rows  : {len(test_df):,}"
    )

    logger.info(
        f"Test points/equipment:"
    )

    for uid, g in test_df.groupby(group_col):

        logger.info(
            f"  {uid}: {len(g)} points"
        )

    return train_df, test_df


# =============================================================================
# PREPARE NIXTLA FORMAT
# =============================================================================

def prepare_nixtla_dataframe(
    df: pd.DataFrame,
    metadata: dict,
    target: str
):

    datetime_col = metadata["datetime_col"]
    group_col = metadata["group_col"]

    df = df.copy()

    # -------------------------------------------------------------------------
    # Calendar
    # -------------------------------------------------------------------------

    df = create_calendar_features(
        df,
        datetime_col
    )

    # -------------------------------------------------------------------------
    # Nixtla format
    # -------------------------------------------------------------------------

    nf_df = pd.DataFrame()

    nf_df["unique_id"] = (
        df[group_col].astype(str)
    )

    nf_df["ds"] = pd.to_datetime(
        df[datetime_col]
    )

    nf_df["y"] = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    # Future-known exogenous
    future_exog = [
        "hour",
        "dayofweek",
        "day",
        "month",
        "is_weekend",
    ]

    for col in future_exog:

        nf_df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    nf_df = nf_df.dropna(
        subset=["unique_id", "ds", "y"]
    )

    nf_df = nf_df.sort_values(
        ["unique_id", "ds"]
    ).reset_index(drop=True)

    return nf_df


# =============================================================================
# VALIDATE TRAIN SERIES
# =============================================================================

def validate_training_series(
    train_nf: pd.DataFrame,
    input_size: int,
    horizon: int
):

    minimum_required = (
        input_size + horizon
    )

    lengths = (
        train_nf
        .groupby("unique_id")
        .size()
    )

    logger.info(
        f"Minimum training samples per equipment: "
        f"{minimum_required}"
    )

    too_short = lengths[
        lengths < minimum_required
    ]

    if len(too_short) > 0:

        raise ValueError(
            "Ada equipment yang datanya terlalu pendek:\n"
            f"{too_short.to_dict()}"
        )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    error = (
        y_true - y_pred
    )

    mae = np.mean(
        np.abs(error)
    )

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(
        mse
    )

    # -------------------------------------------------------------------------
    # MAPE
    # -------------------------------------------------------------------------

    mask = (
        np.abs(y_true) > 1e-6
    )

    if mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    y_true[mask]
                    -
                    y_pred[mask]
                )
                /
                y_true[mask]
            )
        ) * 100

    else:

        mape = np.nan

    # -------------------------------------------------------------------------
    # R2
    # -------------------------------------------------------------------------

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot > 1e-12:

        r2 = (
            1
            -
            ss_res / ss_tot
        )

    else:

        r2 = np.nan

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape_pct": float(mape),
        "r2": float(r2),
        "n_points": len(y_true),
    }


# =============================================================================
# TRAIN NBEATSX
# =============================================================================

def train_model(
    train_nf: pd.DataFrame,
    horizon: int,
    input_size: int,
    cfg: Config,
    freq: str
):

    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx

    logger.info("=" * 70)
    logger.info("TRAINING NBEATSx")
    logger.info("=" * 70)

    logger.info(
        f"Target       : {cfg.target}"
    )

    logger.info(
        f"Input size   : {input_size}"
    )

    logger.info(
        f"Horizon      : {horizon}"
    )

    logger.info(
        f"Max steps    : {cfg.max_steps}"
    )

    logger.info(
        f"Learning rate: {cfg.learning_rate}"
    )

    logger.info(
        "Historical exogenous: NONE"
    )

    logger.info(
        "Future exogenous: "
        "['hour', 'dayofweek', 'day', 'month', 'is_weekend']"
    )

    model = NBEATSx(

        h=horizon,

        input_size=input_size,

        # ---------------------------------------------------------------------
        # IMPORTANT:
        # Hanya future-known exogenous.
        # ---------------------------------------------------------------------

        futr_exog_list=[
            "hour",
            "dayofweek",
            "day",
            "month",
            "is_weekend",
        ],

        # Tidak menggunakan historical exogenous.
        hist_exog_list=None,

        scaler_type=cfg.scaler_type,

        max_steps=cfg.max_steps,

        learning_rate=cfg.learning_rate,

        batch_size=cfg.batch_size,

        val_check_steps=cfg.val_check_steps,

        early_stop_patience_steps=cfg.early_stop_patience_steps,

        random_seed=cfg.random_seed,

        alias="NBEATSx",
    )

    nf = NeuralForecast(
        models=[model],
        freq=freq
    )

    # -------------------------------------------------------------------------
    # Validation hanya berasal dari data TRAIN.
    #
    # Ini penting:
    # Juli tidak boleh masuk validation.
    # -------------------------------------------------------------------------

    nf.fit(
        df=train_nf,
        val_size=horizon
    )

    logger.info(
        "Training selesai."
    )

    return nf


# =============================================================================
# BUILD FUTURE DATAFRAME
# =============================================================================

def build_future_dataframe(
    history_nf: pd.DataFrame,
    steps: int,
    freq: str
):

    """
    Membuat future dataframe untuk NeuralForecast.

    Sangat penting bahwa future dataframe berisi:
        unique_id
        ds
        seluruh future exogenous

    untuk SETIAP equipment.
    """

    rows = []

    for uid, g in history_nf.groupby(
        "unique_id"
    ):

        g = g.sort_values("ds")

        last_date = g["ds"].max()

        # ---------------------------------------------------------------------
        # Buat timestamp masa depan.
        #
        # Dataset kita daily, jadi freq = D.
        # ---------------------------------------------------------------------

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=freq
        )

        future = pd.DataFrame({
            "unique_id": uid,
            "ds": future_dates,
        })

        # Calendar
        future["hour"] = (
            future["ds"]
            .dt.hour
            .astype(float)
        )

        future["dayofweek"] = (
            future["ds"]
            .dt.dayofweek
            .astype(float)
        )

        future["day"] = (
            future["ds"]
            .dt.day
            .astype(float)
        )

        future["month"] = (
            future["ds"]
            .dt.month
            .astype(float)
        )

        future["is_weekend"] = (
            future["ds"]
            .dt.dayofweek >= 5
        ).astype(float)

        rows.append(future)

    if not rows:
        raise ValueError(
            "Tidak dapat membuat future dataframe."
        )

    future_df = pd.concat(
        rows,
        ignore_index=True
    )

    return future_df


# =============================================================================
# RECURSIVE FORECAST
# =============================================================================

def recursive_forecast(
    nf,
    train_nf: pd.DataFrame,
    test_nf: pd.DataFrame,
    horizon: int,
    freq: str
):
    """
    Forecast Juli secara recursive.
    ...
    """

    logger.info("=" * 70)
    logger.info("FORECAST JULY 2026")
    logger.info("=" * 70)

    history = train_nf.copy()
    all_predictions = []

    equipment_ids = test_nf["unique_id"].unique()
    test_dates = test_nf["ds"].sort_values().unique()
    total_test_points = len(test_dates)
    current_position = 0

    while current_position < total_test_points:

        remaining = total_test_points - current_position
        current_horizon = min(horizon, remaining)  # jumlah tanggal yang KITA BUTUHKAN dari step ini

        # PENTING: future_df yang dikirim ke nf.predict() HARUS selalu berisi
        # persis `horizon` (h) langkah ke depan, karena itu arsitektur tetap
        # model NBEATSx -- BUKAN current_horizon, meskipun di window terakhir
        # kita cuma butuh sebagian dari hasilnya.
        future_df = build_future_dataframe(
            history_nf=history,
            steps=horizon,
            freq=freq
        )

        # Tanggal yang benar-benar ingin kita ambil dari hasil prediksi kali ini
        expected_dates = pd.to_datetime(
            test_dates[current_position: current_position + current_horizon]
        )

        # Predict
        forecast_df = nf.predict(
            df=history,
            futr_df=future_df
        )

        forecast_df = forecast_df[
            ["unique_id", "ds", "NBEATSx"]
        ].copy()

        forecast_df = forecast_df.rename(
            columns={"NBEATSx": "prediction"}
        )

        # Ambil HANYA tanggal yang memang kita butuhkan (di window terakhir,
        # ini akan lebih sedikit dari horizon penuh -- sisanya dibuang karena
        # melewati batas data test yang tersedia)
        used_predictions = forecast_df[
            forecast_df["ds"].isin(expected_dates)
        ].copy()

        all_predictions.append(used_predictions)

        # Masukkan HANYA prediksi yang dipakai ke history (bukan seluruh
        # horizon penuh), supaya posisi "last_date" di iterasi berikutnya
        # tetap konsisten dengan tanggal test yang sesungguhnya.
        history_append = used_predictions.rename(columns={"prediction": "y"})

        history_append["hour"] = history_append["ds"].dt.hour.astype(float)
        history_append["dayofweek"] = history_append["ds"].dt.dayofweek.astype(float)
        history_append["day"] = history_append["ds"].dt.day.astype(float)
        history_append["month"] = history_append["ds"].dt.month.astype(float)
        history_append["is_weekend"] = (history_append["ds"].dt.dayofweek >= 5).astype(float)

        history = pd.concat(
            [history, history_append],
            ignore_index=True
        )

        history = history.sort_values(
            ["unique_id", "ds"]
        ).reset_index(drop=True)

        logger.info(
            f"Forecasted {expected_dates.min()} -> {expected_dates.max()}"
        )

        current_position += current_horizon

    predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    return predictions


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_forecast(
    predictions: pd.DataFrame,
    test_nf: pd.DataFrame
):

    merged = pd.merge(
        test_nf[
            [
                "unique_id",
                "ds",
                "y",
            ]
        ],
        predictions,
        on=[
            "unique_id",
            "ds",
        ],
        how="inner"
    )

    if len(merged) == 0:

        raise ValueError(
            "Tidak ada timestamp yang berhasil "
            "di-merge antara prediction dan actual."
        )

    if len(merged) != len(test_nf):

        logger.warning(
            f"Jumlah actual test : {len(test_nf)}"
        )

        logger.warning(
            f"Jumlah prediction  : {len(merged)}"
        )

    metrics = calculate_metrics(
        merged["y"].values,
        merged["prediction"].values
    )

    return merged, metrics


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    output_dir: Path,
    predictions_df: pd.DataFrame,
    metrics: dict,
    cfg: Config,
    horizon: int,
    input_size: int,
    metadata: dict
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    predictions_file = (
        output_dir
        /
        "predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics_file = (
        output_dir
        /
        "metrics.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Run info
    # -------------------------------------------------------------------------

    run_info = {

        "target": cfg.target,

        "train_end": cfg.train_end,

        "test_start": cfg.test_start,

        "test_end": cfg.test_end,

        "actual_test_end": (
            str(
                predictions_df["ds"].max()
            )
            if len(predictions_df)
            else None
        ),

        "horizon": horizon,

        "input_size": input_size,

        "freq": metadata["freq"],

        "scaler_type": cfg.scaler_type,

        "max_steps": cfg.max_steps,

        "learning_rate": cfg.learning_rate,

        "batch_size": cfg.batch_size,

        "random_seed": cfg.random_seed,

        # ---------------------------------------------------------------------
        # Explicitly document exogenous configuration.
        # ---------------------------------------------------------------------

        "hist_exog_list": [],

        "futr_exog_list": [
            "hour",
            "dayofweek",
            "day",
            "month",
            "is_weekend",
        ],

        "data_leakage_policy": (
            "Actual target values after train_end are never "
            "fed back into the model during forecasting. "
            "Only previous predictions are recursively used."
        ),

        "metrics": metrics,
    }

    run_info_file = (
        output_dir
        /
        "run_info.json"
    )

    with open(
        run_info_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            run_info,
            f,
            indent=2
        )

    logger.info(
        f"Saved predictions : {predictions_file}"
    )

    logger.info(
        f"Saved metrics     : {metrics_file}"
    )

    logger.info(
        f"Saved run info    : {run_info_file}"
    )


# =============================================================================
# PLOT
# =============================================================================

def save_plot(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    target: str
):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_file = (
        output_dir
        /
        "prediction_plot.png"
    )

    plt.figure(
        figsize=(14, 6)
    )

    for uid, g in predictions_df.groupby(
        "unique_id"
    ):

        g = g.sort_values("ds")

        plt.plot(
            g["ds"],
            g["y"],
            marker="o",
            label=f"{uid} - Actual"
        )

        plt.plot(
            g["ds"],
            g["prediction"],
            marker="x",
            linestyle="--",
            label=f"{uid} - NBEATSx"
        )

    plt.title(
        f"NBEATSx Forecast vs Actual - {target}"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        target
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        f"Saved plot        : {plot_file}"
    )


# =============================================================================
# SAVE MODEL
# =============================================================================

def save_model(
    nf,
    output_dir: Path
):

    model_dir = (
        output_dir
        /
        "nf_checkpoint"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        nf.save(
            path=str(model_dir),
            overwrite=True,
            save_dataset=False
        )

        logger.info(
            f"Saved model       : {model_dir}"
        )

    except Exception as e:

        logger.warning(
            f"Gagal menyimpan NeuralForecast checkpoint: {e}"
        )


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def run_training(
    cfg: Config
):

    set_seed(
        cfg.random_seed
    )

    logger.info(
        "CONFIGURATION:"
    )

    logger.info(
        json.dumps(
            asdict(cfg),
            indent=2
        )
    )

    logger.info("")

    logger.info("=" * 70)
    logger.info(
        f"NBEATSx FORECASTING - {cfg.target}"
    )
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    df, metadata = load_data(
        cfg
    )

    datetime_col = metadata[
        "datetime_col"
    ]

    freq = metadata[
        "freq"
    ]

    # -------------------------------------------------------------------------
    # Resolve model parameters
    # -------------------------------------------------------------------------

    horizon = (
        cfg.horizon
        if cfg.horizon is not None
        else metadata["horizon"]
    )

    input_size = (
        cfg.input_size
        if cfg.input_size is not None
        else metadata["input_size"]
    )

    logger.info(
        f"Input size : {input_size}"
    )

    logger.info(
        f"Horizon    : {horizon}"
    )

    # -------------------------------------------------------------------------
    # CREATE CALENDAR
    # -------------------------------------------------------------------------

    df = create_calendar_features(
        df,
        datetime_col
    )

    # -------------------------------------------------------------------------
    # SPLIT
    # -------------------------------------------------------------------------

    train_df, test_df = split_train_test(
        df,
        cfg,
        metadata
    )

    # -------------------------------------------------------------------------
    # Convert to Nixtla
    # -------------------------------------------------------------------------

    train_nf = prepare_nixtla_dataframe(
        train_df,
        metadata,
        cfg.target
    )

    test_nf = prepare_nixtla_dataframe(
        test_df,
        metadata,
        cfg.target
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_training_series(
        train_nf,
        input_size,
        horizon
    )

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    nf = train_model(
        train_nf=train_nf,
        horizon=horizon,
        input_size=input_size,
        cfg=cfg,
        freq=freq
    )

    # -------------------------------------------------------------------------
    # FORECAST
    # -------------------------------------------------------------------------

    predictions = recursive_forecast(
        nf=nf,
        train_nf=train_nf,
        test_nf=test_nf,
        horizon=horizon,
        freq=freq
    )

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------

    predictions_df, metrics = evaluate_forecast(
        predictions,
        test_nf
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 70)

    logger.info(
        f"MAE      : {metrics['mae']:.6f}"
    )

    logger.info(
        f"MSE      : {metrics['mse']:.6f}"
    )

    logger.info(
        f"RMSE     : {metrics['rmse']:.6f}"
    )

    logger.info(
        f"MAPE (%) : {metrics['mape_pct']:.6f}"
    )

    logger.info(
        f"R2       : {metrics['r2']:.6f}"
    )

    logger.info(
        f"N Points : {metrics['n_points']}"
    )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    output_dir = Path(
        cfg.output_dir
    )

    save_results(
        output_dir=output_dir,
        predictions_df=predictions_df,
        metrics=metrics,
        cfg=cfg,
        horizon=horizon,
        input_size=input_size,
        metadata=metadata
    )

    save_plot(
        predictions_df,
        output_dir,
        cfg.target
    )

    save_model(
        nf,
        output_dir
    )

    # -------------------------------------------------------------------------
    # Print predictions
    # -------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("PREDICTIONS")
    logger.info("=" * 70)

    logger.info(
        "\n"
        +
        predictions_df.to_string(
            index=False
        )
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING + TESTING SELESAI")
    logger.info("=" * 70)

    return {
        "model": nf,
        "predictions": predictions_df,
        "metrics": metrics,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "NBEATSx forecasting dengan "
            "fixed train/test split."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--target",
        required=True,
        choices=[
            "pressure",
            "flowrate",
            "temperature",
        ]
    )

    parser.add_argument(
        "--train-end",
        default="2026-06-30 23:59:59"
    )

    parser.add_argument(
        "--test-start",
        default="2026-07-01 00:00:00"
    )

    parser.add_argument(
        "--test-end",
        default="2026-07-31 23:59:59"
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=None
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=None
    )

    parser.add_argument(
        "--scaler-type",
        default="standard"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--val-check-steps",
        type=int,
        default=25
    )

    parser.add_argument(
        "--early-stop-patience-steps",
        type=int,
        default=5
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target=args.target,

        train_end=args.train_end,

        test_start=args.test_start,

        test_end=args.test_end,

        horizon=args.horizon,

        input_size=args.input_size,

        scaler_type=args.scaler_type,

        max_steps=args.max_steps,

        learning_rate=args.learning_rate,

        batch_size=args.batch_size,

        val_check_steps=args.val_check_steps,

        early_stop_patience_steps=(
            args.early_stop_patience_steps
        ),

        random_seed=args.random_seed,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    config = parse_args()

    run_training(
        config
    )