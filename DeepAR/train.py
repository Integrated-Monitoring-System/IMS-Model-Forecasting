import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer

# =========================================================
# CONFIG
# =========================================================

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 20

ENCODER_LENGTH = 144
PREDICTION_LENGTH = 12

L.seed_everything(SEED)

# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'dataset' / 'DeepAR processed'

TRAIN_PATH = DATA_DIR / 'train.csv'
EVAL_PATH = DATA_DIR / 'eval.csv'
SCALER_PATH = DATA_DIR / 'scaler.pkl'

ARTIFACT_DIR = BASE_DIR / 'artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

train_df = pd.read_csv(TRAIN_PATH, parse_dates=['time'])
eval_df = pd.read_csv(EVAL_PATH, parse_dates=['time'])

scaler = joblib.load(SCALER_PATH)

full_df = (
    pd.concat([train_df, eval_df], ignore_index=True)
    .sort_values('time_idx')
    .reset_index(drop=True)
)

print('Train:', train_df.shape)
print('Eval :', eval_df.shape)

# =========================================================
# DATASET
# =========================================================

training_cutoff = train_df['time_idx'].max()

training = TimeSeriesDataSet(
    full_df[full_df.time_idx <= training_cutoff],
    time_idx='time_idx',
    target='pressure',
    group_ids=['series_id'],

    max_encoder_length=ENCODER_LENGTH,
    max_prediction_length=PREDICTION_LENGTH,

    static_categoricals=['series_id'],

    time_varying_known_reals=['time_idx'],
    time_varying_unknown_reals=['pressure'],

    target_normalizer=GroupNormalizer(
        groups=['series_id']
    ),

    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training,
    full_df[full_df.time_idx > training_cutoff - ENCODER_LENGTH],
    predict=True,
    stop_randomization=True
)

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

# =========================================================
# MODEL
# =========================================================

deepar = DeepAR.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=32,
    rnn_layers=2,
    dropout=0.2
)

print(deepar)

# =========================================================
# TRAINER
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='auto',
    devices=1,
    callbacks=[early_stop],
    logger=False,
    enable_checkpointing=True
)

print('\\nStarting DeepAR training...')

trainer.fit(
    deepar,
    train_loader,
    val_loader
)

# =========================================================
# SAVE MODEL
# =========================================================

MODEL_PATH = ARTIFACT_DIR / 'deepar_model.ckpt'
trainer.save_checkpoint(MODEL_PATH)

print(f'Model saved: {MODEL_PATH}')

# =========================================================
# PREDICTION ON FULL EVAL SET
# =========================================================

# ambil prediksi untuk semua window
predictions = deepar.predict(
    val_loader,
    mode='prediction'
)

# shape: [n_windows, prediction_length]
y_pred = predictions.cpu().numpy()

# ambil target sebenarnya dari validation dataset
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

# debug
print('\\nVariance check:')
print('Actual variance     :', np.var(y_true_eval))
print('Prediction variance :', np.var(y_pred_eval))
print('Correlation         :', np.corrcoef(y_true_eval, y_pred_eval)[0,1])

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
# DEBUG CHECK
# =========================================================

print('\\nVariance check:')
print('Actual variance     :', np.var(y_true_eval))
print('Prediction variance :', np.var(y_pred_eval))
print('Correlation         :', np.corrcoef(y_true_eval, y_pred_eval)[0,1])

print('\\nSample pairs:')
check_df = pd.DataFrame({
    'actual': y_true_eval[:10],
    'predicted': y_pred_eval[:10],
    'error': np.abs(y_true_eval[:10] - y_pred_eval[:10])
})
print(check_df)

results = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'SMAPE (%)', 'R-Squared'],
    'Value': [mae, mse, rmse, smape, r2]
})

print('\\n' + '=' * 60)
print('DEEPAR RESULTS')
print('=' * 60)
print(results.to_string(index=False))

# =========================================================
# SAVE RESULTS
# =========================================================

results.to_csv(ARTIFACT_DIR / 'metrics.csv', index=False)

pred_df = pd.DataFrame({
    'step': np.arange(len(y_true_inv)),
    'actual': y_true_inv,
    'predicted': y_pred_inv
})

pred_df.to_csv(ARTIFACT_DIR / 'predictions.csv', index=False)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(14,5))

n_plot = min(500, len(y_true_inv))

plt.plot(y_true_inv[:n_plot], label='Actual')
plt.plot(y_pred_inv[:n_plot], label='DeepAR')

plt.title('DeepAR: Actual vs Predicted Pressure')
plt.xlabel('Time Step')
plt.ylabel('Pressure')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'prediction_plot.png', dpi=300)
plt.show()

print('\\nTraining completed successfully!')