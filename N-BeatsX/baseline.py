from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Folder tempat baseline.py berada
# IMS-Model-Analytic/N-BeatsX/
MODEL_DIR = Path(__file__).resolve().parent

# Root project
# IMS-Model-Analytic/
PROJECT_ROOT = MODEL_DIR.parent

# Dataset
DATASET_DIR = PROJECT_ROOT / "dataset" / "N-BeatsX processed"

# Gunakan FULL DATA, bukan reference_train/reference_test
DATA_FILE = DATASET_DIR / "full_clean.parquet"

# Output baseline
OUTPUT_DIR = MODEL_DIR / "artifacts" / "baseline"

# Target
TARGET_COL = "pressure"

# Datetime
DATETIME_COL = "DateTime"

# ============================================================
# TIME-BASED HOLD-OUT
# ============================================================

# Training hanya sampai 30 Juni 2026
TRAIN_END = pd.Timestamp("2026-06-30 23:59:59")

# Testing mulai 1 Juli 2026
TEST_START = pd.Timestamp("2026-07-01 00:00:00")

EPSILON = 1e-8


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, target_name):
    """
    Mencari nama kolom secara case-insensitive.
    """

    target_lower = target_name.lower()

    for col in df.columns:
        if str(col).lower() == target_lower:
            return col

    return None


def find_datetime_column(df):
    """
    Mencari kolom datetime secara fleksibel.
    """

    possible_names = [
        "datetime",
        "date_time",
        "timestamp",
        "time",
    ]

    for col in df.columns:
        if str(col).lower() in possible_names:
            return col

    return None


def calculate_mape(y_true, y_pred):
    """
    Menghitung MAPE dalam persen.

    Nilai aktual yang sangat dekat dengan 0
    dikeluarkan dari perhitungan agar MAPE stabil.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.abs(y_true) > EPSILON

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING FULL DATASET")
    print("=" * 70)

    print(f"Dataset : {DATA_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{DATA_FILE}"
        )

    df = pd.read_parquet(DATA_FILE)

    print(f"\nDataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # Cari datetime column
    # --------------------------------------------------------

    datetime_col = find_column(
        df,
        DATETIME_COL
    )

    if datetime_col is None:
        datetime_col = find_datetime_column(df)

    if datetime_col is None:
        raise ValueError(
            "Kolom DateTime tidak ditemukan di dataset."
        )

    # --------------------------------------------------------
    # Convert datetime
    # --------------------------------------------------------

    df[datetime_col] = pd.to_datetime(
        df[datetime_col],
        errors="coerce"
    )

    # Hapus datetime invalid
    df = df.dropna(
        subset=[datetime_col]
    ).copy()

    # Sort berdasarkan waktu
    df = df.sort_values(
        datetime_col
    ).reset_index(drop=True)

    print(f"\nDatetime column : {datetime_col}")
    print(f"Dataset start   : {df[datetime_col].min()}")
    print(f"Dataset end     : {df[datetime_col].max()}")

    return df, datetime_col


# ============================================================
# CREATE TIME-BASED SPLIT
# ============================================================

def create_time_split(df, datetime_col):

    print("\n" + "=" * 70)
    print("CREATING TIME-BASED TRAIN / TEST SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_df = df[
        df[datetime_col] <= TRAIN_END
    ].copy()

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    test_df = df[
        df[datetime_col] >= TEST_START
    ].copy()

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    train_target_col = find_column(
        train_df,
        TARGET_COL
    )

    test_target_col = find_column(
        test_df,
        TARGET_COL
    )

    if train_target_col is None:
        raise ValueError(
            f"Kolom target '{TARGET_COL}' "
            "tidak ditemukan pada training data."
        )

    if test_target_col is None:
        raise ValueError(
            f"Kolom target '{TARGET_COL}' "
            "tidak ditemukan pada testing data."
        )

    train_df = train_df.dropna(
        subset=[train_target_col]
    ).copy()

    test_df = test_df.dropna(
        subset=[test_target_col]
    ).copy()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if train_df.empty:
        raise ValueError(
            "Training data kosong."
        )

    if test_df.empty:
        raise ValueError(
            "Testing data kosong."
        )

    # Pastikan tidak ada overlap
    latest_train = train_df[datetime_col].max()
    earliest_test = test_df[datetime_col].min()

    if latest_train >= earliest_test:
        raise ValueError(
            "TRAIN dan TEST overlap."
        )

    # --------------------------------------------------------
    # Print split information
    # --------------------------------------------------------

    print("\nTRAIN")
    print("-" * 70)
    print(f"Start : {train_df[datetime_col].min()}")
    print(f"End   : {train_df[datetime_col].max()}")
    print(f"Rows  : {len(train_df)}")

    print("\nTEST")
    print("-" * 70)
    print(f"Start : {test_df[datetime_col].min()}")
    print(f"End   : {test_df[datetime_col].max()}")
    print(f"Rows  : {len(test_df)}")

    print("\nExpected cutoff:")
    print(f"Train <= {TRAIN_END}")
    print(f"Test  >= {TEST_START}")

    return train_df, test_df


# ============================================================
# NAIVE BASELINE
# ============================================================

def create_naive_forecast(
    train_df,
    test_df,
):

    print("\n" + "=" * 70)
    print("CREATING NAIVE / PERSISTENCE BASELINE")
    print("=" * 70)

    train_target_col = find_column(
        train_df,
        TARGET_COL
    )

    test_target_col = find_column(
        test_df,
        TARGET_COL
    )

    if train_target_col is None:
        raise ValueError(
            f"Kolom target '{TARGET_COL}' "
            "tidak ditemukan di training data."
        )

    if test_target_col is None:
        raise ValueError(
            f"Kolom target '{TARGET_COL}' "
            "tidak ditemukan di testing data."
        )

    # --------------------------------------------------------
    # Ambil nilai terakhir dari TRAIN
    # --------------------------------------------------------

    train_target = (
        train_df[train_target_col]
        .astype(float)
        .dropna()
    )

    if train_target.empty:
        raise ValueError(
            "Tidak ada nilai target valid di training data."
        )

    last_train_value = float(
        train_target.iloc[-1]
    )

    print(f"\nTarget column     : {train_target_col}")
    print(f"Last train value : {last_train_value}")

    # --------------------------------------------------------
    # Naive / Persistence forecast
    # --------------------------------------------------------

    # Semua nilai TEST diprediksi menggunakan
    # nilai terakhir TRAIN.
    predictions = np.full(
        len(test_df),
        last_train_value,
        dtype=float,
    )

    actual = (
        test_df[test_target_col]
        .astype(float)
        .to_numpy()
    )

    return (
        actual,
        predictions,
        last_train_value,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(mse)

    mape = calculate_mape(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": r2,
        "n_points": len(y_true),
    }

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    test_df,
    datetime_col,
    y_true,
    y_pred,
    metrics,
    last_train_value,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({
        datetime_col: test_df[
            datetime_col
        ].values,

        "actual": y_true,

        "prediction": y_pred,

        "error": y_true - y_pred,

        "absolute_error": np.abs(
            y_true - y_pred
        ),
    })

    predictions_file = (
        OUTPUT_DIR
        / "baseline_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame([
        metrics
    ])

    metrics_file = (
        OUTPUT_DIR
        / "baseline_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config_df = pd.DataFrame({
        "parameter": [
            "baseline",
            "target",
            "train_end",
            "test_start",
            "last_train_value",
            "test_points",
        ],

        "value": [
            "Naive / Persistence",
            TARGET_COL,
            TRAIN_END,
            TEST_START,
            last_train_value,
            len(test_df),
        ],
    })

    config_file = (
        OUTPUT_DIR
        / "baseline_config.csv"
    )

    config_df.to_csv(
        config_file,
        index=False,
    )

    print("\nResults saved:")
    print(f"- {predictions_file}")
    print(f"- {metrics_file}")
    print(f"- {config_file}")

    return predictions_df


# ============================================================
# PLOT
# ============================================================

def plot_results(
    test_df,
    datetime_col,
    y_true,
    y_pred,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = pd.to_datetime(
        test_df[datetime_col]
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        x,
        y_true,
        marker="o",
        label="Actual Pressure",
    )

    plt.plot(
        x,
        y_pred,
        marker="x",
        linestyle="--",
        label="Naive Baseline",
    )

    plt.title(
        "Naive Baseline vs Actual Pressure\n"
        "Hold-out Test: July 2026"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Pressure"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plot_file = (
        OUTPUT_DIR
        / "baseline_vs_actual.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"- {plot_file}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print("NAIVE BASELINE - N-BEATSX DATASET")
    print("=" * 70)

    print("\nEvaluation strategy:")
    print(
        "TRAIN : <= 2026-06-30"
    )

    print(
        "TEST  : >= 2026-07-01"
    )

    # --------------------------------------------------------
    # 1. Load full dataset
    # --------------------------------------------------------

    df, datetime_col = load_data()

    # --------------------------------------------------------
    # 2. Time-based split
    # --------------------------------------------------------

    train_df, test_df = create_time_split(
        df,
        datetime_col,
    )

    # --------------------------------------------------------
    # 3. Create baseline
    # --------------------------------------------------------

    (
        y_true,
        y_pred,
        last_train_value,
    ) = create_naive_forecast(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # 4. Evaluation
    # --------------------------------------------------------

    metrics = evaluate_model(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # 5. Print metrics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(
        f"MAE      : {metrics['mae']:.6f}"
    )

    print(
        f"MSE      : {metrics['mse']:.6f}"
    )

    print(
        f"RMSE     : {metrics['rmse']:.6f}"
    )

    print(
        f"MAPE (%) : {metrics['mape_pct']:.6f}"
    )

    print(
        f"R²       : {metrics['r2']:.6f}"
    )

    print(
        f"N Points : {metrics['n_points']}"
    )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    predictions_df = save_results(
        test_df,
        datetime_col,
        y_true,
        y_pred,
        metrics,
        last_train_value,
    )

    # --------------------------------------------------------
    # 7. Plot
    # --------------------------------------------------------

    plot_results(
        test_df,
        datetime_col,
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # 8. Print predictions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)

    print(
        predictions_df.to_string(
            index=False
        )
    )

    print("\nDone.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()