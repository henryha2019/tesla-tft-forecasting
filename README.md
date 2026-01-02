# 📈 TSLA Stock Price Forecasting with Temporal Fusion Transformer (TFT)

This project implements a **Temporal Fusion Transformer (TFT)** model to perform **multi-horizon time-series forecasting** on **Tesla (TSLA) daily stock prices**, with a strong focus on:

* Proper **time-series validation**
* **Expanding-window cross-validation**
* Comparison against a **naive baseline**
* **Interpretability** (attention + variable importance)
* Clear, reproducible evaluation and visualization

---

## 🔍 Problem Statement

Given historical TSLA price data, the goal is to:

* Forecast **future closing prices** over a **30-day horizon**
* Evaluate model performance across **multiple forecast horizons**
* Understand **which features and time steps drive predictions**

This is **not** a point-forecast toy example — it is a **realistic time-series modeling pipeline** with rigorous validation.

---

## 📊 Data Source

* **Source:** Yahoo Finance via `yfinance`
* **Ticker:** `TSLA`
* **Frequency:** Daily
* **Span:** Maximum available historical data (`period="max"`)

If Yahoo Finance fails (e.g., VPN / API issues), the pipeline **automatically falls back to synthetic data** so development can continue without blocking.

---

## 🧹 Data Preparation

### Core Steps

1. Robust download via `yf.Ticker().history()`
2. Column normalization and cleanup
3. Time ordering and missing-value handling
4. Creation of TFT-required fields:

   * `time_idx`
   * `group_id`
5. Categorical encoding:

   * `day_of_week`
   * `month`

### Feature Engineering

* **Rolling mean (7-day)**
* **Log returns**
* Relative time indices and target scaling (handled internally by TFT)

All transformations are **Pandas 2.x safe** and reproducible.

---

## 🧠 Model: Temporal Fusion Transformer (TFT)

The TFT architecture is chosen because it:

* Handles **static**, **known**, and **unknown** variables
* Produces **multi-horizon forecasts**
* Offers **built-in interpretability**

### Key Configuration

* Encoder length: **60 days**
* Prediction horizon: **30 days**
* Loss: **Quantile Loss**
* Normalization: **GroupNormalizer**
* Lightweight architecture (~21K parameters)

---

## ✅ Validation Strategy (Critical)

### ❌ What We Avoid

* Random train/validation splits
* Shuffled windows
* Information leakage

### ✅ What We Use: Expanding-Window Cross-Validation

Each fold:

* Trains on **all data up to time *t***
* Validates on a **future block**
* Advances forward in time

This mimics **real-world forecasting deployment**.

**Why this matters:**
Time-series models can look “great” with improper validation and fail in production. This setup prevents that.

---

## 📏 Baseline Model

A **naive persistence baseline** is used:

> Forecast = last observed value, repeated across the horizon

This is a **strong baseline** in financial time series and is required for honest evaluation.

---

## 📐 Evaluation Metrics

Metrics are computed:

### 1️⃣ Per Horizon (1 → 30 days)

* RMSE
* MAE
* R²

### 2️⃣ Overall (Flattened)

* RMSE across all horizons
* MAE across all horizons
* R² across all horizons

This avoids misleading single-step reporting.

---

## 📈 Results Summary

### Cross-Validation (Overall)

* TFT **consistently outperforms** the naive baseline on RMSE and MAE
* R² is modest or slightly negative (expected for noisy financial data)

### Per-Horizon Behavior

* Error **increases with forecast horizon**
* TFT degrades **more gracefully** than naive forecasts

> This is the expected and correct behavior for realistic stock forecasting.

---

## 🧪 Final Model Training

After cross-validation:

* Model is retrained on **all available data**
* Learning rate is fixed (no LR finder noise)
* Used for final forecasting and interpretation

---

## 🔮 Holdout Forecast (Last 30 Days)

A final holdout window compares:

* Actual prices
* TFT median forecast
* Naive baseline

This provides an **intuitive sanity check** on model behavior.

---

## 🔍 Interpretability & Explainability

One of TFT’s strengths is interpretability.

### Global Explanations

* Static variable importance
* Encoder variable importance
* Decoder variable importance

These reveal:

* Time indices and rolling statistics dominate
* Calendar effects (day/month) contribute modestly

### Attention Analysis

* Attention weights show **recent history is most influential**
* Smooth decay across the encoder window (expected and healthy)

### Local Explanation

* Single-window prediction plot
* Shows how the model combines history + attention to produce forecasts

---

## 📁 Project Structure

```
.
├── data/
├── notebooks/
│   └── tsla_tft_forecasting.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Key Takeaways

* TFT **does not magically beat markets**, but:

  * Outperforms naive baselines
  * Produces stable multi-horizon forecasts
  * Provides **transparent reasoning**
* Expanding-window CV is **non-negotiable** for time series
* Interpretation tools are critical for trust, not just accuracy

---

## 🚀 Future Improvements

* Add volatility-aware targets (e.g., returns)
* Incorporate macro or sector indicators
* Hyperparameter tuning across folds
* Probabilistic backtesting strategies

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **not financial advice**.

