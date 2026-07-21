import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import (
    TimeSeriesDataSet,
    NBeats,
    GroupNormalizer
)
from pytorch_forecasting.metrics import MAE, RMSE

# =========================================================
# CONFIG
# =========================================================

SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

ENCODER_LENGTH = 144      # 144 menit histori
PREDICTION_LENGTH = 12    # prediksi 12 menit

pl.seed_everything(SEED)

# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR.parent /
    'dataset' /
    'N-BeatsX processed' /
    'nbeatsx_dataset.csv'
)

SCALER_PATH = BASE_DIR / 'artifacts' / 'scaler.pkl'

MODEL_DIR = BASE_DIR / 'artifacts'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f'Data path: {DATA_PATH}')

# =========================================================
# LOAD DATA
# =========================================================

full_df = pd.read_csv(DATA_PATH)
scaler = joblib.load(SCALER_PATH)

print('Dataset shape:', full_df.shape)
print(full_df.head())

# =========================================================
# SPLIT
# =========================================================

# train sampai 30 Juni 2026
training_cutoff = full_df.loc[
    full_df['DateTime'] < '2026-07-01',
    'time_idx'
].max()

train_df = full_df[full_df.time_idx <= training_cutoff]
eval_df = full_df[full_df.time_idx > training_cutoff]

print(f'Train: {train_df.shape}')
print(f'Eval : {eval_df.shape}')

# =========================================================
# DATASET
# =========================================================

training = TimeSeriesDataSet(
    train_df,
    time_idx='time_idx',
    target='pressure',
    group_ids=['series_id'],

    min_encoder_length=ENCODER_LENGTH,
    max_encoder_length=ENCODER_LENGTH,

    min_prediction_length=PREDICTION_LENGTH,
    max_prediction_length=PREDICTION_LENGTH,

    # N-BEATS klasik: TIDAK BOLEH ada fitur tambahan
    static_categoricals=[],
    static_reals=[],
    time_varying_known_reals=[],
    time_varying_unknown_reals=['pressure'],

    target_normalizer=GroupNormalizer(
        groups=['series_id']
    ),

    add_relative_time_idx=False,
    add_target_scales=False,
    add_encoder_length=False,
)

# validation untuk seluruh Juli
validation = TimeSeriesDataSet.from_dataset(
    training,
    full_df[full_df.time_idx > training_cutoff - ENCODER_LENGTH],
    predict=False,
    stop_randomization=True
)

# =========================================================
# DATALOADER
# =========================================================

train_loader = training.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    num_workers=0
)

val_loader = validation.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=0
)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches  : {len(val_loader)}')

# =========================================================
# MODEL
# =========================================================

nbeats = NBeats.from_dataset(
    training,

    learning_rate=LEARNING_RATE,

    widths=[256, 256],

    stack_types=['trend', 'seasonality'],

    num_blocks=[3, 3],

    num_block_layers=[4, 4],

    dropout=0.1,

    loss=RMSE(),

    logging_metrics=[MAE(), RMSE()]
)

print(nbeats)

# =========================================================
# CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

checkpoint = ModelCheckpoint(
    dirpath=MODEL_DIR,
    filename='nbeatsx-best',
    monitor='val_loss',
    mode='min',
    save_top_k=1
)

# =========================================================
# TRAINER
# =========================================================

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator='auto',
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[early_stop, checkpoint],
    logger=False,
    enable_progress_bar=True
)

# =========================================================
# TRAINING
# =========================================================

print('\\nStarting N-BEATSX training...')

trainer.fit(
    nbeats,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)

# =========================================================
# SAVE MODEL
# =========================================================

MODEL_PATH = MODEL_DIR / 'nbeatsx_model.ckpt'
trainer.save_checkpoint(MODEL_PATH)

print(f'Model saved: {MODEL_PATH}')

# =========================================================
# PREDICTION ON FULL EVAL SET
# =========================================================

predictions = nbeats.predict(
    val_loader,
    mode='prediction'
)

# [n_windows, prediction_length]
y_pred = predictions.cpu().numpy()

# ambil target sebenarnya
actuals = []
for batch in val_loader:
    x, y = batch
    actuals.append(y[0].numpy())

y_true = np.concatenate(actuals, axis=0)

print('Prediction shape:', y_pred.shape)
print('Target shape    :', y_true.shape)

# flatten
y_pred = y_pred.reshape(-1)
y_true = y_true.reshape(-1)

# inverse scaling
y_pred_inv = scaler.inverse_transform(
    y_pred.reshape(-1, 1)
).flatten()

y_true_inv = scaler.inverse_transform(
    y_true.reshape(-1, 1)
).flatten()

# remove NaN
mask = ~np.isnan(y_true_inv) & ~np.isnan(y_pred_inv)
y_true_eval = y_true_inv[mask]
y_pred_eval = y_pred_inv[mask]

print(f'Aligned samples: {len(y_true_eval)}')

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(y_true_eval, y_pred_eval)
mse = mean_squared_error(y_true_eval, y_pred_eval)
rmse = np.sqrt(mse)

smape = np.mean(
    2 * np.abs(y_pred_eval - y_true_eval) /
    (np.abs(y_true_eval) + np.abs(y_pred_eval) + 1e-8)
) * 100

r2 = r2_score(y_true_eval, y_pred_eval)

# =========================================================
# DEBUG
# =========================================================

print('\\nVariance check:')
print('Actual variance     :', np.var(y_true_eval))
print('Prediction variance :', np.var(y_pred_eval))
print('Correlation         :', np.corrcoef(y_true_eval, y_pred_eval)[0,1])

# =========================================================
# RESULTS
# =========================================================

results = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'SMAPE (%)', 'R-Squared'],
    'Value': [mae, mse, rmse, smape, r2]
})

print('\\n' + '='*60)
print('N-BEATSX RESULTS')
print('='*60)
print(results.to_string(index=False))

# =========================================================
# SAVE RESULTS
# =========================================================

pred_df = pd.DataFrame({
    'step': np.arange(len(y_true_eval)),
    'actual': y_true_eval,
    'predicted': y_pred_eval
})

pred_df.to_csv(MODEL_DIR / 'predictions.csv', index=False)
results.to_csv(MODEL_DIR / 'metrics.csv', index=False)

# =========================================================
# PLOT PREDICTION
# =========================================================

plt.figure(figsize=(14,5))

n_plot = min(500, len(y_true_eval))

plt.plot(
    y_true_eval[:n_plot],
    label='Actual',
    linewidth=1.5
)

plt.plot(
    y_pred_eval[:n_plot],
    label='N-BEATSX',
    linewidth=1.5
)

plt.title('N-BEATSX: Actual vs Predicted Pressure')
plt.xlabel('Time Step')
plt.ylabel('Pressure')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'prediction_plot.png', dpi=300)
plt.show()

print('\\nTraining completed successfully!')