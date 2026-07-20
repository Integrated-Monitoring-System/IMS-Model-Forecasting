# IMS-Model-Forecasting

Forecasting and anomaly-aware modeling pipeline for **Integrated Monitoring System (IMS)** industrial sensor data using deep learning and time-series forecasting approaches.

## Overview

**IMS-Model-Forecasting** is a research-oriented forecasting project developed during an internship at **Pertamina EP**. The project focuses on predicting future sensor behavior (initially **pressure**) from historical time-series data and evaluating multiple forecasting architectures for industrial monitoring applications.

The current implementation includes:

- Data preprocessing and window generation
- Univariate time-series forecasting
- Model training and evaluation
- Metric benchmarking
- Experiment tracking for multiple forecasting models

---

## Objectives

- Forecast industrial sensor values for the next **24 minutes** using historical observations.
- Compare modern deep learning forecasting architectures on IMS sensor data.
- Develop a production-oriented forecasting pipeline for monitoring and early warning applications.

### Target Performance

| Metric | Target |
|--------|--------|
| MAE | ≤ 10 |
| MSE | ≤ 200 |
| RMSE | ≤ 25 |
| MAPE | ≤ 5% |
| R² Score | ≥ 0.70 |

---

## Current Models

| Model | Status |
|------|--------|
| PatchTST | Implemented |
| N-BEATSx | Planned |
| Temporal Convolutional Network (TCN) | Planned |
| DeepAR | Planned |
| Temporal Fusion Transformer (TFT) | Planned |

---

## Current Experiment: PatchTST

### Configuration

- **Context Length:** 96 time steps
- **Prediction Horizon:** 24 time steps
- **Input:** Univariate pressure sensor
- **Framework:** PyTorch + Hugging Face Transformers

### Initial Evaluation Result

| Metric | Value |
|--------|------:|
| MAE | 29.47 |
| MSE | 5257.73 |
| RMSE | 72.51 |
| MAPE | 851.01% |
| R² Score | 0.269 |

### Preliminary Analysis

The initial PatchTST model successfully learns part of the temporal pattern from the pressure data, but the current performance is **not yet production-ready**.

**Key observations:**

- **R² = 0.269** indicates the model currently explains only about **26.9%** of the variance in the target series.
- **RMSE = 72.51** shows that prediction deviations are still relatively large.
- **MAPE is extremely high**, suggesting that percentage-based error is heavily affected by values close to zero or strong fluctuations in the dataset.
- The current result should be treated as a **baseline experiment** rather than a final forecasting solution.

---

## Dataset

The current experiment uses **processed IMS pressure sensor data** exported into NumPy arrays for efficient training.

### Data Split

- **Train:** 4,774 samples
- **Test:** 437 samples

The preprocessing workflow includes:

- Timestamp alignment
- Data cleaning
- Scaling
- Sliding window generation
- Train/test separation without temporal leakage

---

## Why Multiple Models?

Industrial sensor forecasting has different characteristics from typical business time-series data:

- High-frequency measurements
- Sudden spikes and regime changes
- Operational transitions (startup/shutdown)
- Strong cross-sensor dependencies

Therefore, several architectures are evaluated for different strengths:

| Model | Main Strength |
|------|---------------|
| N-BEATSx | Strong univariate forecasting |
| TCN | Spike and local pattern detection |
| PatchTST | Long-range multivariate dependencies |
| TFT | Interpretable forecasting with operational metadata |
| DeepAR | Probabilistic forecasting and uncertainty estimation |

---

## Future Work

### Short Term

- Implement N-BEATSx baseline
- Implement TCN baseline
- Perform hyperparameter optimization with Optuna
- Investigate high MAPE root cause

### Medium Term

Integrate additional IMS sensor parameters:

- Temperature
- Flow rate
- Vibration
- Valve position
- Motor current
- Production rate

### Long Term

- Build a **multivariate PatchTST forecasting pipeline**
- Add **TFT with operational metadata**
- Develop **anomaly-aware forecasting** for early warning systems
- Deploy real-time inference for IMS monitoring dashboards

---

## Tech Stack

- Python 3.10
- PyTorch
- Hugging Face Transformers
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Research Status

> This repository is an **active research and experimentation project**. Model architectures, preprocessing strategies, and evaluation procedures may evolve as additional sensor data becomes available and benchmarking experiments progress.

---

## Contributors

- **Fakhri Muhammad Al Hisyam** — [@zepunnn](https://github.com/zepunnn)
- **Ivan Febrianto Lalo** — [@schroerizaki](https://github.com/schroerizaki)

---

## License

This project is intended for research, educational, and industrial experimentation purposes related to **Integrated Monitoring System (IMS) forecasting and analytics**.
