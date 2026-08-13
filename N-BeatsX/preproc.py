"""
preproc.py
==========
Pipeline preprocessing data sensor upstream (pressure, flowrate, temperature)
dari 3 file CSV historian terpisah (format long/tag-based, mis. export dari
IMS/OSIsoft PI-style historian), untuk persiapan input model NBEATSx
(neuralforecast).

Beda utama dibanding preprocessing untuk TFT/PatchTST:
    - Sumber data berupa 3 FILE TERPISAH (pressure, flowrate, temperature),
      masing-masing dalam format LONG per-tag (kolom TagName, DateTime, Value,
      QualityStatus, dst.), bukan satu file wide seperti sebelumnya.
    - Ada tahap filter kualitas data (buang baris QualityStatus selain "Good",
      mis. "Cannot convert" / "Bad") -- nilai 0/"(null)" pada baris semacam itu
      BUKAN pembacaan sensor asli.
    - Kolom "Value" bisa berisi string "(null)" -- perlu dikonversi ke numerik
      dengan coercion (baris yang gagal dikonversi otomatis dibuang).
    - group/equipment id diekstrak otomatis dari TagName (mis.
      "PTG_A3_CTK_FT_002" -> equipment "PTG_A3_CTK_002", instrumen "FT"),
      supaya ketiga file bisa digabung jadi satu tabel per equipment yang sama.
    - Timestamp di 3 file historian ini TIDAK akan pernah sama persis satu
      sama lain (event-based, per-tag). Jadi tiap sumber di-resample ke
      frekuensi seragam DULU (per equipment), baru digabung (outer join) pada
      timestamp yang sudah beraturan.
    - Scaling diserahkan ke tahap model (scaler_type bawaan neuralforecast,
      mirip prinsip GroupNormalizer di TFT) supaya konsisten & tidak leakage.
    - Disediakan utilitas to_nixtla_format() untuk mengubah output jadi format
      (unique_id, ds, y, ...exog) yang dipakai library neuralforecast.

Scope script ini HANYA sampai tahap data siap pakai (cleaned, merged,
feature-engineered, tersimpan sebagai parquet + metadata.json). Pembangunan
dataset NBEATSx dan training modelnya dilakukan di script terpisah.

Cara pakai (CLI), dijalankan dari dalam folder N-BeatsX/ (struktur project:
dataset/ berisi CSV mentah, sejajar dengan folder tiap model):
    python preproc.py \
        --pressure-csv "../dataset/IMS Pres Data 20260722.csv" \
        --flowrate-csv "../dataset/IMS Flow Data 20260722.csv" \
        --temperature-csv "../dataset/IMS Temp Data 20260722.csv" \
        --output-dir "../dataset/N-BeatsX processed" \
        --freq H --input-size 168 --horizon 24

Bisa juga dipakai sebagai module (di notebook / script lain):
    from preproc import Config, run_pipeline
    cfg = Config(
        input_paths={
            "pressure": "../dataset/IMS Pres Data 20260722.csv",
            "flowrate": "../dataset/IMS Flow Data 20260722.csv",
            "temperature": "../dataset/IMS Temp Data 20260722.csv",
        },
        output_dir="../dataset/N-BeatsX processed",
    )
    result = run_pipeline(cfg)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preproc")


# =============================================================================
# 1. KONFIGURASI
# =============================================================================
@dataclass
class Config:
    # --- I/O: satu file CSV historian per parameter ---
    input_paths: dict  # mis. {"pressure": "...", "flowrate": "...", "temperature": "..."}
    output_dir: str

    # --- kolom-kolom pada file historian (format long/tag-based) ---
    tag_col: str = "TagName"
    datetime_col: str = "DateTime"
    value_col: str = "Value"
    quality_status_col: str = "QualityStatus"
    valid_quality_values: list = field(default_factory=lambda: ["Good"])

    # --- resampling & cleaning ---
    freq: str = "H"                 # frekuensi resample: "H"=jam, "D"=hari, "15min", dst.
    max_missing_ratio: float = 0.4  # drop equipment kalau rasio NaN > ini (sebelum imputasi)
    outlier_method: str = "iqr"     # "iqr" atau "zscore"
    outlier_threshold: float = 3.0

    # --- fitur kalender opsional (bisa dipakai sebagai futr_exog di NBEATSx) ---
    include_calendar_features: bool = True

    # --- split (berbasis rasio, dipetakan ke time_idx cutoff per equipment) ---
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # --- parameter NBEATSx (istilah menyesuaikan neuralforecast) ---
    input_size: int = 168   # panjang lookback window
    horizon: int = 24       # panjang horizon forecast

    def __post_init__(self):
        if not (0 < self.train_ratio < 1) or not (0 < self.val_ratio < 1):
            raise ValueError("train_ratio dan val_ratio harus di antara 0 dan 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio harus < 1 (sisanya untuk test)")
        if not self.input_paths:
            raise ValueError("input_paths tidak boleh kosong, mis. {'pressure': 'a.csv', ...}")

    @property
    def target_cols(self) -> list:
        """Daftar parameter target, diturunkan dari key input_paths."""
        return list(self.input_paths.keys())


KNOWN_CALENDAR_FEATURES = ["hour", "dayofweek", "day", "month", "is_weekend"]


# =============================================================================
# 2. EKSTRAKSI EQUIPMENT ID DARI TAG NAME
# =============================================================================
_INSTRUMENT_CODE_RE = re.compile(r"^[A-Z]{2}$")  # kode instrumen 2 huruf: FT, TT, PT, LT, dst.


def extract_equipment_id(tag_name: str) -> tuple:
    """
    Ekstrak equipment_id (identitas equipment/tangki/sumur yang sama) dan
    instrument_code (jenis instrumen: FT/TT/PT/dst.) dari TagName historian.

    Contoh: "PTG_A3_CTK_FT_002.ValueQuery" -> ("PTG_A3_CTK_002", "FT")

    Asumsi: token kode instrumen adalah SATU token yang persis 2 huruf besar
    (FT, TT, PT, LT, dst.) di antara token-token lain yang menyusun tag name.
    Kalau tidak ditemukan token semacam itu, seluruh tag (minus suffix
    ".ValueQuery") dipakai apa adanya sebagai equipment_id.
    """
    base = str(tag_name).split(".")[0]  # buang suffix seperti ".ValueQuery"
    tokens = base.split("_")

    code_idx = None
    for i, tok in enumerate(tokens):
        if _INSTRUMENT_CODE_RE.fullmatch(tok):
            code_idx = i
            break  # ambil kemunculan pertama yang cocok

    if code_idx is not None:
        instrument_code = tokens[code_idx]
        equip_tokens = tokens[:code_idx] + tokens[code_idx + 1:]
    else:
        instrument_code = "UNKNOWN"
        equip_tokens = tokens

    equipment_id = "_".join(equip_tokens)
    return equipment_id, instrument_code


# =============================================================================
# 3. LOAD + CLEAN PER SUMBER (satu parameter historian)
# =============================================================================
def load_and_clean_source(path: str, param_name: str, cfg: Config) -> pd.DataFrame:
    """
    Load satu file historian, filter kualitas data, konversi Value ke numerik,
    parse datetime, ekstrak equipment_id dari TagName, dan buang duplikat.
    Return dataframe dengan kolom: ["equipment_id", cfg.datetime_col, param_name].
    """
    logger.info(f"[{param_name}] Load dari: {path}")
    df = pd.read_csv(path)
    n_raw = len(df)

    # 1. filter kualitas data -- baris "Cannot convert"/"Bad" BUKAN pembacaan sensor asli
    df = df[df[cfg.quality_status_col].isin(cfg.valid_quality_values)].copy()
    logger.info(
        f"  -> {len(df):,}/{n_raw:,} baris lolos filter kualitas "
        f"({cfg.valid_quality_values})"
    )

    # 2. konversi Value ke numerik (ada string seperti "(null)")
    df["_value_numeric"] = pd.to_numeric(df[cfg.value_col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["_value_numeric"])
    if n_before - len(df):
        logger.info(f"  -> drop {n_before - len(df):,} baris dengan Value tidak bisa dikonversi ke angka")

    # 3. parse datetime
    df["_timestamp"] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["_timestamp"])
    if n_before - len(df):
        logger.warning(f"  -> drop {n_before - len(df):,} baris dengan timestamp tidak valid")

    # 4. ekstrak equipment_id dari TagName
    extracted = df[cfg.tag_col].apply(extract_equipment_id)
    df["equipment_id"] = [e[0] for e in extracted]
    df["_instrument_code"] = [e[1] for e in extracted]

    n_unknown = (df["_instrument_code"] == "UNKNOWN").sum()
    if n_unknown:
        logger.warning(
            f"  -> {n_unknown:,} baris gagal mendeteksi kode instrumen dari TagName "
            "(equipment_id dipakai apa adanya dari tag lengkap)"
        )

    # 5. dedupe -- kombinasi equipment_id + timestamp sama persis, ambil yang terakhir
    df = df.sort_values(["equipment_id", "_timestamp"])
    n_before = len(df)
    df = df.drop_duplicates(subset=["equipment_id", "_timestamp"], keep="last")
    if n_before - len(df):
        logger.info(f"  -> drop {n_before - len(df):,} baris duplikat (equipment_id + timestamp sama)")

    result = df[["equipment_id", "_timestamp", "_value_numeric"]].rename(
        columns={"_timestamp": cfg.datetime_col, "_value_numeric": param_name}
    ).reset_index(drop=True)

    logger.info(
        f"  -> hasil akhir '{param_name}': {len(result):,} baris, "
        f"{result['equipment_id'].nunique()} equipment: {sorted(result['equipment_id'].unique())}"
    )
    return result


def resample_source(df: pd.DataFrame, param_name: str, cfg: Config) -> pd.DataFrame:
    """Resample satu sumber (sudah bersih) ke frekuensi seragam, per equipment_id."""
    out_frames = []
    for key, g in df.groupby("equipment_id"):
        g = g.set_index(cfg.datetime_col)
        resampled = g[[param_name]].resample(cfg.freq).mean()
        resampled["equipment_id"] = key
        out_frames.append(resampled.reset_index())
    return pd.concat(out_frames, ignore_index=True)


def merge_sources(resampled: dict, cfg: Config) -> pd.DataFrame:
    """Gabungkan (outer join) semua sumber yang sudah di-resample, pada (equipment_id, timestamp)."""
    logger.info("Gabungkan semua sumber parameter jadi satu tabel (outer join per equipment + waktu)")
    dfs = list(resampled.values())
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["equipment_id", cfg.datetime_col], how="outer")
    merged = merged.sort_values(["equipment_id", cfg.datetime_col]).reset_index(drop=True)
    logger.info(f"  -> hasil merge: {len(merged):,} baris, {merged['equipment_id'].nunique()} equipment")
    return merged


# =============================================================================
# 4. UTILITAS GROUP KEY (equipment_id dipakai langsung sebagai group key)
# =============================================================================
def _add_group_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_group_key"] = df["equipment_id"].astype(str)
    return df


# =============================================================================
# 5. CLEANING LANJUTAN (setelah merge): filter sparse, outlier, imputasi
# =============================================================================
def filter_sparse_groups(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ratios = df.groupby("_group_key")[cfg.target_cols].apply(lambda g: g.isna().mean().max())
    keep_keys = ratios[ratios <= cfg.max_missing_ratio].index
    dropped_keys = ratios[ratios > cfg.max_missing_ratio].index

    if len(dropped_keys):
        logger.warning(
            f"Drop {len(dropped_keys)} equipment karena missing ratio > {cfg.max_missing_ratio:.0%}: "
            f"{list(dropped_keys)[:10]}{'...' if len(dropped_keys) > 10 else ''}"
        )

    df = df[df["_group_key"].isin(keep_keys)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "Semua equipment ter-drop karena missing ratio > max_missing_ratio "
            f"({cfg.max_missing_ratio:.0%}). Data historian sumbernya kemungkinan jarang "
            "(on-change logging), sehingga resample ke freq="
            f"'{cfg.freq}' menghasilkan terlalu banyak gap. Coba perbesar --max-missing-ratio "
            "atau pakai --freq yang lebih kasar (mis. 'D' bukan 'H')."
        )
    return df


def handle_outliers(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clip outlier per equipment per kolom target (bukan drop baris, supaya time_idx tetap kontinu)."""
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
    """Interpolasi linear per equipment untuk mengisi gap (termasuk gap hasil outer join antar sumber)."""
    logger.info("Imputasi missing value (interpolasi linear per equipment)")
    df = df.sort_values(["_group_key", cfg.datetime_col]).copy()

    n_missing_before = df[cfg.target_cols].isna().sum().sum()

    for col in cfg.target_cols:
        df[col] = df.groupby("_group_key")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )

    n_missing_after = df[cfg.target_cols].isna().sum().sum()
    logger.info(f"  -> missing value: {n_missing_before:,} -> {n_missing_after:,}")

    if n_missing_after > 0:
        n_before = len(df)
        df = df.dropna(subset=cfg.target_cols)
        logger.warning(f"  -> drop {n_before - len(df):,} baris sisa NaN yang tidak bisa diinterpolasi")

    return df.reset_index(drop=True)


# =============================================================================
# 6. TIME INDEX & FITUR KALENDER
# =============================================================================
def add_time_idx(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.sort_values(["_group_key", cfg.datetime_col]).copy()
    df["time_idx"] = df.groupby("_group_key").cumcount()
    return df


def add_calendar_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fitur kalender -> kandidat futr_exog_list di NBEATSx (diketahui di masa depan)."""
    if not cfg.include_calendar_features:
        return df
    df = df.copy()
    dt = df[cfg.datetime_col]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return df


# =============================================================================
# 7. VALIDASI KECUKUPAN PANJANG DATA (input_size + horizon)
# =============================================================================
def check_window_feasibility(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """NBEATSx butuh tiap equipment punya panjang data >= input_size + horizon."""
    min_len_required = cfg.input_size + cfg.horizon
    lengths = df.groupby("_group_key")["time_idx"].agg(lambda s: s.max() + 1)

    too_short = lengths[lengths < min_len_required]
    if len(too_short):
        logger.warning(
            f"Drop {len(too_short)} equipment karena panjang data < input_size + horizon "
            f"({min_len_required} timestep): {list(too_short.index)}"
        )

    keep_keys = lengths[lengths >= min_len_required].index
    df = df[df["_group_key"].isin(keep_keys)].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"Semua equipment ter-drop karena panjang data < input_size + horizon "
            f"({min_len_required} timestep). Coba perkecil --input-size/--horizon, atau pakai "
            "--freq yang lebih kasar supaya jumlah timestep per equipment lebih banyak."
        )

    logger.info(f"  -> {len(keep_keys)} equipment lolos validasi panjang data")
    return df


# =============================================================================
# 8. SPLIT (cutoff time_idx per equipment)
# =============================================================================
def compute_split_cutoffs(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if df.empty:
        raise ValueError(
            "Tidak ada equipment tersisa setelah tahap cleaning/filtering -- tidak ada yang bisa "
            "di-split. Kemungkinan penyebab: 'max_missing_ratio' terlalu ketat untuk frekuensi "
            "resample ('freq') yang dipilih, atau 'input_size + horizon' lebih panjang dari data "
            "yang tersedia. Coba perbesar --max-missing-ratio, pakai --freq yang lebih kasar "
            "(mis. 'D' bukan 'H') jika data historian sumbernya jarang (on-change logging), atau "
            "perkecil --input-size/--horizon."
        )
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
    Versi split sederhana HANYA untuk EDA/sanity-check cepat. Untuk training NBEATSx
    sesungguhnya, gunakan full_clean.parquet + train_cutoff_time_idx/val_cutoff_time_idx
    supaya window val/test tetap bisa mengambil input_size histori sebelum cutoff-nya.
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
# 9. UTILITAS FORMAT NIXTLA (untuk neuralforecast / NBEATSx)
# =============================================================================
def to_nixtla_format(
    df: pd.DataFrame,
    target_col: str,
    datetime_col: str = "DateTime",
    group_col: str = "equipment_id",
    exog_cols=None,
) -> pd.DataFrame:
    """
    Ubah dataframe wide (satu baris per timestamp per equipment) jadi format long
    yang dipakai library `neuralforecast` untuk NBEATSx: kolom unique_id, ds, y,
    plus kolom exogenous tambahan (kalau ada).

    Contoh pemakaian di script model:
        df_full = pd.read_parquet("output/full_clean.parquet")
        nf_df = to_nixtla_format(
            df_full, target_col="pressure",
            exog_cols=["flowrate", "temperature", "hour", "dayofweek"],
        )
        # nf_df siap dipakai sebagai input NeuralForecast(...).fit(df=nf_df)
    """
    cols = [group_col, datetime_col, target_col] + (exog_cols or [])
    out = df[cols].rename(columns={group_col: "unique_id", datetime_col: "ds", target_col: "y"})
    return out


# =============================================================================
# 10. SIMPAN OUTPUT
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

    known_reals = ["time_idx"] + (KNOWN_CALENDAR_FEATURES if cfg.include_calendar_features else [])

    metadata = {
        "datetime_col": cfg.datetime_col,
        "group_col": "equipment_id",
        "target_cols": cfg.target_cols,
        "time_varying_known_reals": known_reals,
        "time_varying_unknown_reals": cfg.target_cols,
        "freq": cfg.freq,
        "input_size": cfg.input_size,
        "horizon": cfg.horizon,
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "n_equipment": df_clean["_group_key"].nunique(),
        "equipment_list": sorted(df_clean["_group_key"].unique().tolist()),
        "n_rows_clean": len(df_clean),
        "date_range": [
            str(df_clean[cfg.datetime_col].min()),
            str(df_clean[cfg.datetime_col].max()),
        ],
        "note_scaling": (
            "Scaling SENGAJA tidak dilakukan di tahap preprocessing ini. neuralforecast "
            "(library untuk NBEATSx) punya parameter scaler_type bawaan model (mis. "
            "'standard', 'robust') yang otomatis menormalisasi y dan exogenous per window "
            "saat training, supaya statistik tidak leakage dan konsisten dengan cara kerja "
            "library-nya."
        ),
        "note_split": (
            "reference_train/val/test.parquet HANYA untuk EDA/sanity-check cepat. Untuk "
            "training NBEATSx sesungguhnya, gunakan full_clean.parquet + "
            "train_cutoff_time_idx/val_cutoff_time_idx per equipment di split_cutoffs.csv, "
            "lalu ubah ke format Nixtla dengan to_nixtla_format() sebelum dipakai "
            "NeuralForecast(...).fit()."
        ),
        "note_windowing": (
            f"Tiap sample window butuh input_size ({cfg.input_size}) + horizon "
            f"({cfg.horizon}) = {cfg.input_size + cfg.horizon} timestep berurutan tanpa "
            "gap. Equipment dengan data lebih pendek dari itu sudah di-drop di tahap "
            "preprocessing ini."
        ),
        "note_format": (
            "full_clean.parquet masih dalam format WIDE (satu baris per timestamp per "
            "equipment, kolom target sejajar). Pakai to_nixtla_format() dari preproc.py "
            "untuk mengubahnya ke format long (unique_id, ds, y, exog) sebelum dipakai "
            "NBEATSx, target_col dipilih sesuai parameter yang mau diforecast di run "
            "tersebut (pressure / flowrate / temperature)."
        ),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Output tersimpan di: {out_dir.resolve()}")
    logger.info(f"  - full_clean.parquet          ({len(df_clean_save):,} baris)  <- pakai ini untuk modeling")
    logger.info(f"  - reference_train.parquet     ({len(train_save):,} baris)  [EDA only]")
    logger.info(f"  - reference_val.parquet       ({len(val_save):,} baris)  [EDA only]")
    logger.info(f"  - reference_test.parquet      ({len(test_save):,} baris)  [EDA only]")
    logger.info("  - split_cutoffs.csv, metadata.json")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline(cfg: Config) -> dict:
    logger.info("=" * 70)
    logger.info("MULAI PIPELINE PREPROCESSING (NBEATSx)")
    logger.info("=" * 70)

    # 1. load + clean tiap sumber, lalu resample per sumber (sebelum digabung)
    resampled_sources = {}
    for param_name, path in cfg.input_paths.items():
        cleaned = load_and_clean_source(path, param_name, cfg)
        resampled_sources[param_name] = resample_source(cleaned, param_name, cfg)

    # 2. gabungkan semua sumber jadi satu tabel wide
    df = merge_sources(resampled_sources, cfg)
    df = _add_group_key(df)

    # 3. cleaning lanjutan
    df = filter_sparse_groups(df, cfg)
    df = handle_outliers(df, cfg)
    df = impute_missing(df, cfg)

    # 4. feature engineering
    df = add_time_idx(df, cfg)
    df = add_calendar_features(df, cfg)
    df = check_window_feasibility(df, cfg)

    # 5. split & simpan
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
    p = argparse.ArgumentParser(
        description="Preprocessing 3 file CSV historian (pressure/flowrate/temperature) untuk NBEATSx"
    )

    p.add_argument("--pressure-csv", required=True, help="path file CSV historian pressure")
    p.add_argument("--flowrate-csv", required=True, help="path file CSV historian flowrate")
    p.add_argument("--temperature-csv", required=True, help="path file CSV historian temperature")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--tag-col", default="TagName")
    p.add_argument("--datetime-col", default="DateTime")
    p.add_argument("--value-col", default="Value")
    p.add_argument("--quality-status-col", default="QualityStatus")
    p.add_argument(
        "--valid-quality-values", default="Good",
        help="pisahkan dengan koma kalau lebih dari satu nilai valid, mis. 'Good,OK'",
    )

    p.add_argument("--freq", default="H")
    p.add_argument("--max-missing-ratio", type=float, default=0.4)
    p.add_argument("--outlier-method", choices=["iqr", "zscore"], default="iqr")
    p.add_argument("--outlier-threshold", type=float, default=3.0)
    p.add_argument("--no-calendar-features", action="store_true")

    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)

    p.add_argument("--input-size", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)

    args = p.parse_args()

    cfg = Config(
        input_paths={
            "pressure": args.pressure_csv,
            "flowrate": args.flowrate_csv,
            "temperature": args.temperature_csv,
        },
        output_dir=args.output_dir,
        tag_col=args.tag_col,
        datetime_col=args.datetime_col,
        value_col=args.value_col,
        quality_status_col=args.quality_status_col,
        valid_quality_values=[x.strip() for x in args.valid_quality_values.split(",") if x.strip()],
        freq=args.freq,
        max_missing_ratio=args.max_missing_ratio,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold,
        include_calendar_features=not args.no_calendar_features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        input_size=args.input_size,
        horizon=args.horizon,
    )
    return cfg


if __name__ == "__main__":
    config = _parse_args()
    logger.info(f"Konfigurasi:\n{json.dumps(asdict(config), indent=2, default=str)}")
    run_pipeline(config)