# IMS-Model-Forecasting

Benchmarking and anomaly-aware forecasting pipeline for **Integrated Monitoring System (IMS)** industrial sensor data using modern deep learning time-series architectures.

---

## Overview

**IMS-Model-Forecasting** is a research-oriented forecasting project developed during an internship at **Pertamina EP**. The project focuses on predicting future industrial sensor behavior from historical time-series data and benchmarking multiple forecasting architectures for monitoring, anomaly detection, and early warning applications.

The current benchmark uses **pressure sensor data** sampled at **1-minute frequency** and compares five deep learning forecasting models under the same preprocessing and evaluation pipeline.

### Current Project Status

- Dataset cleaning and reconstruction: **Completed**
- Preprocessing pipeline: **Completed**
- Model benchmarking (5 architectures): **Completed**
- Evaluation and artifact export: **Completed**
- Multivariate forecasting phase: **In Progress**
- Ensemble and hybrid modeling phase: **Planned**

---

## Dataset Pipeline

### Data Cleaning Strategy

The following cleaning procedure is applied consistently across all models:

- Remove invalid records from **2025-08-01 to 2026-04-30**
- Merge valid historical and recent segments
- Resample to **1-minute frequency**
- Interpolate small temporal gaps
- Preserve chronological order to avoid temporal leakage

### Final Processed Dataset

| Split | Samples |
|------|--------:|
| Total | 526,115 |
| Train | 498,240 |
| Evaluation (July 2026) | 27,875 |

The evaluation set uses **July 2026** as an unseen forecasting period for fair model comparison.

---

## Forecasting Objective

Predict future pressure sensor values using historical observations.

### Forecast Configuration

- **Input window:** 144 minutes
- **Forecast horizon:** 12–24 minutes (depending on architecture)
- **Sampling frequency:** 1 minute
- **Target:** Pressure sensor

---

# Implemented Models

| Model | Status | Main Strength |
|------|--------|---------------|
| PatchTST | Implemented | Long-range temporal dependencies |
| N-BEATS | Implemented | Strong univariate trend and seasonality modeling |
| Temporal Convolutional Network (TCN) | Implemented | Local pattern and spike modeling |
| DeepAR | Implemented | Probabilistic short-horizon forecasting |
| Temporal Fusion Transformer (TFT) | Implemented | Interpretable sequence forecasting |

---

# Benchmark Results

All models are evaluated on the same IMS pressure forecasting pipeline.

| Model | MAE | RMSE | SMAPE | R² |
|------|----:|-----:|------:|---:|
| **PatchTST** | **21.97** | **53.93** | 19.50% | 0.557 |
| TFT | 16.68 | 16.69 | 5.27%* | -4630.66 |
| TCN | 27.18 | 60.98 | 18.68% | 0.405 |
| DeepAR | 3.29 | 3.69 | **0.92%** | 0.190 |
| N-BEATS | 47.58 | 245.87 | 1.45% | **0.982** |

> **Note:** TFT currently uses MAPE and still requires additional evaluation because the R² value is inconsistent with the error metrics.

---

# Model Comparison Analysis

## PatchTST — Best Operational Trade-off

**Current best production-oriented candidate**

**Strengths**
- Lowest practical absolute forecasting error
- Stable generalization on large-scale data
- Strong potential for future multivariate expansion

**Best for**
- Real-time operational forecasting
- Dashboard prediction
- KPI monitoring

---

## N-BEATS — Best Temporal Dynamics Representation

N-BEATS achieved:

- **R² = 0.9815**
- **Signal correlation ≈ 0.992**

This indicates exceptional capability in modeling the underlying temporal structure of the pressure signal, although absolute error metrics are still affected by evaluation alignment and scaling considerations.

**Best for**
- Trend reconstruction
- Long-term temporal dynamics
- Residual and ensemble modeling

---

## TCN — Robust Baseline

**Strengths**
- Simple architecture
- Good local pattern modeling
- Competitive baseline performance

**Best for**
- Spike detection
- Short local dependencies
- Lightweight deployment

---

## DeepAR — Short-Horizon Specialist

**Strengths**
- Excellent short-window forecasting metrics
- Probabilistic output
- Uncertainty-aware forecasting

**Limitations**
- Lower explanatory power on long evaluation periods

**Best for**
- Confidence interval prediction
- Early warning probability estimation

---

## TFT — Requires Further Investigation

TFT shows low MAE and RMSE but highly unstable R², suggesting that additional hyperparameter tuning and evaluation alignment are required before drawing conclusions.

---

# Current Research Insight

A key observation from the benchmark is the difference between **absolute forecasting accuracy** and **temporal structure modeling**.

### Absolute Error Perspective

**PatchTST** currently provides the best balance between MAE, RMSE, and generalization.

### Temporal Dynamics Perspective

**N-BEATS** captures the shape and evolution of the pressure signal with extremely high correlation and R².

This suggests that the two models may be **complementary rather than mutually exclusive**.

---

# Planned Hybrid Architecture

The next research phase explores an ensemble of the strongest complementary models.

### Proposed Ensemble

- **N-BEATS** → Trend and seasonality backbone
- **PatchTST** → Long-range dependency correction
- **TCN** → Local spike and residual correction

This hybrid design is expected to improve both:

- Absolute forecasting accuracy
- Temporal pattern fidelity

---

# Multivariate Expansion (Next Phase)

Additional IMS sensor parameters will be integrated:

- Pressure
- Temperature
- Flowrate
- Vibration
- Valve position
- Motor current
- Production rate

This phase will enable true **N-BEATSX** and **multivariate PatchTST/TFT** experiments.

---

# Tech Stack

- Python 3.10
- PyTorch
- PyTorch Forecasting
- Lightning / PyTorch Lightning
- Hugging Face Transformers
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

# Research Status

**Current phase:** Large-scale univariate benchmarking on IMS pressure sensor data (**526k+ observations**).

**Next phase:** Multivariate forecasting and ensemble modeling using pressure, temperature, flowrate, and vibration sensors for industrial anomaly-aware forecasting and early warning systems.

---

# Contributors

- **Fakhri Muhammad Al Hisyam** — [@zepunnn](https://github.com/zepunnn)
- **Ivan Febrianto Lalo** — [@schroerizaki](https://github.com/schroerizaki)

---

# License

This repository is intended for **research, educational, and industrial experimentation purposes** related to **Integrated Monitoring System (IMS) forecasting, analytics, and anomaly-aware monitoring**.
