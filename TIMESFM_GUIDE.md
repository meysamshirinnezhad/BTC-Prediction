# TimesFM Integration Guide

This guide explains how to use Google's **TimesFM** (Time Series Foundation Model) for Bitcoin price prediction.

## What is TimesFM?

TimesFM is a pre-trained foundation model developed by Google Research specifically for time series forecasting. Unlike traditional models that need to be trained from scratch, TimesFM comes pre-trained on a massive dataset and can be used directly for predictions.

### Key Features

- **Pre-trained**: No training required - ready to use out of the box
- **Lightweight**: 200M parameters (TimesFM 2.5)
- **Extended context**: Supports up to 16,384 context length
- **Quantile forecasting**: Provides prediction intervals with confidence bands
- **Zero-shot forecasting**: Works on new data without fine-tuning

## Installation

### Option 1: PyTorch Backend (Recommended)

```bash
pip install timesfm[torch]
```

For GPU support:
```bash
pip install timesfm[torch] torch --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: JAX/Flax Backend

```bash
pip install timesfm[flax]
```

## Usage

### 1. Using the Training Script (Evaluation Mode)

Since TimesFM is pre-trained, "training" actually means loading and evaluating the model:

```bash
# Evaluate TimesFM with default settings
python src/train.py --model timesfm --days 365

# Custom context length
python src/train.py --model timesfm --context-length 1024 --days 365

# Use JAX backend instead of PyTorch
python src/train.py --model timesfm --backend flax --days 365
```

**Parameters:**
- `--context-length`: Number of historical points to use for prediction (default: 512, max: 16384)
- `--backend`: Choose 'pytorch' or 'flax' (default: pytorch)
- `--days`: Days of historical data to fetch

### 2. Making Predictions

```bash
# Make predictions using TimesFM
python src/predict.py \
    --model models/trained/timesfm_config_20231115_143022.pkl \
    --model-type timesfm \
    --horizon 24 \
    --plot
```

**Note**: For TimesFM, the `--model` argument should point to the saved configuration file, not a model weights file (since the model is pre-trained).

### 3. Python API

#### Basic Usage

```python
from models.timesfm_model import create_timesfm_model
from src.data_collector import BTCDataCollector
from src.preprocessing import prepare_data_for_timesfm

# Collect data
collector = BTCDataCollector()
df = collector.fetch_yahoo_finance(days=365, interval='1h')

# Prepare data for TimesFM
data = prepare_data_for_timesfm(
    df,
    context_length=512,
    forecast_horizon=1
)

# Create and load model
model = create_timesfm_model(
    backend='pytorch',
    max_context=1024,
    max_horizon=256
)

model.load_model()

# Evaluate
metrics = model.evaluate(
    data['X_test'],
    data['y_test'],
    horizon=1,
    context_length=512
)

print(f"MAE: {metrics['mae']}")
print(f"RMSE: {metrics['rmse']}")
```

#### Predictions with Confidence Intervals

TimesFM provides quantile forecasts, giving you prediction intervals:

```python
# Get predictions with confidence intervals
predictions = model.predict_with_quantiles(
    data['X_test'],
    horizon=24,
    context_length=512
)

# Access different outputs
point_forecast = predictions['point']        # Point predictions
quantiles = predictions['quantiles']         # All quantiles
lower_bound = predictions['lower_bound']     # 10th percentile
upper_bound = predictions['upper_bound']     # 90th percentile
```

## How TimesFM Works

### Architecture

TimesFM uses a decoder-only transformer architecture with:
- 200M parameters (version 2.5)
- Patching mechanism for efficient processing
- Continuous quantile head for prediction intervals

### Input Format

TimesFM expects **univariate** time series data (single column). For Bitcoin prediction, we use only the **close price**:

```python
# TimesFM uses only the target column (close price)
# Not the engineered features used by LSTM/GRU/XGBoost
inputs = [
    np.array([price1, price2, price3, ...]),  # Series 1
    np.array([price4, price5, price6, ...]),  # Series 2
    # ... more series
]
```

### Context Length

- **Minimum**: 1
- **Maximum**: 16,384 (TimesFM 2.5)
- **Recommended**: 512-1024 for hourly Bitcoin data
  - 512 hours ≈ 21 days
  - 1024 hours ≈ 42 days

Longer context captures more historical patterns but increases computation time.

## Performance Comparison

Comparing TimesFM with other models on Bitcoin price prediction:

| Model    | Training Required | MAE    | RMSE   | Advantages                           |
|----------|------------------|---------|--------|--------------------------------------|
| TimesFM  | No (pre-trained) | ~TBD    | ~TBD   | No training, confidence intervals    |
| LSTM     | Yes              | ~TBD    | ~TBD   | Learns patterns, uses features       |
| GRU      | Yes              | ~TBD    | ~TBD   | Faster than LSTM, uses features      |
| XGBoost  | Yes              | ~TBD    | ~TBD   | Feature importance, interpretable    |

*Note: Run benchmarks on your data to get actual performance metrics.*

## Advanced Configuration

### ForecastConfig Parameters

TimesFM supports advanced configuration through `ForecastConfig`:

```python
import timesfm

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,              # Maximum context length
        max_horizon=256,               # Maximum forecast horizon
        normalize_inputs=True,         # Normalize input data
        use_continuous_quantile_head=True,  # Enable quantile forecasts
        force_flip_invariance=True,    # Enforce flip invariance
        infer_is_positive=True,        # Infer if series is positive
        fix_quantile_crossing=True,    # Ensure monotonic quantiles
    )
)
```

### Custom Backend Selection

```python
# PyTorch backend (default)
model = create_timesfm_model(backend='pytorch')

# JAX/Flax backend
model = create_timesfm_model(backend='flax')
```

## Best Practices

### 1. Data Preparation

- **Clean data**: Remove outliers and handle missing values
- **Sufficient history**: Use at least 30 days of hourly data
- **Consistent intervals**: Ensure regular time intervals

### 2. Context Length Selection

- **Short-term predictions** (1-24 hours): Use 512-1024 context
- **Medium-term predictions** (1-7 days): Use 1024-2048 context
- **Long-term predictions** (>7 days): Use 2048-4096 context

### 3. Interpreting Quantile Forecasts

```python
predictions = model.predict_with_quantiles(X, horizon=24)

# 90% confidence interval
lower = predictions['lower_bound']  # 10th percentile
upper = predictions['upper_bound']  # 90th percentile

# There's a 90% chance the actual price will be in [lower, upper]
```

### 4. When to Use TimesFM

**Use TimesFM when:**
- You need quick predictions without training time
- You want confidence intervals
- You have limited computational resources
- You need predictions on new/unseen data immediately

**Use LSTM/GRU/XGBoost when:**
- You can afford training time
- You have rich feature engineering
- You need custom architecture for your specific use case
- You want to leverage technical indicators

## Troubleshooting

### Common Issues

**1. Import Error: `ModuleNotFoundError: No module named 'timesfm'`**

Solution:
```bash
pip install timesfm[torch]
```

**2. CUDA Out of Memory**

Solution:
- Reduce `max_context` length
- Use CPU instead of GPU
- Process data in smaller batches

**3. Slow Predictions**

Solution:
- Reduce context length
- Use PyTorch backend (usually faster than JAX for small batches)
- Enable GPU if available

**4. Poor Predictions**

Solution:
- Increase context length
- Clean and normalize input data
- Check for data quality issues
- Try different horizons

### GPU vs CPU

TimesFM can run on both GPU and CPU:

```python
# Check if GPU is available (PyTorch)
import torch
print(f"GPU available: {torch.cuda.is_available()}")

# Force CPU usage
torch.set_default_device('cpu')
```

## Example: Complete Workflow

```python
from src.data_collector import BTCDataCollector
from src.preprocessing import prepare_data_for_timesfm
from models.timesfm_model import create_timesfm_model
import matplotlib.pyplot as plt

# 1. Collect data
collector = BTCDataCollector()
df = collector.fetch_yahoo_finance(days=180, interval='1h')

# 2. Prepare data
data = prepare_data_for_timesfm(
    df,
    context_length=1024,
    forecast_horizon=24  # Predict 24 hours ahead
)

# 3. Create and load model
model = create_timesfm_model(
    backend='pytorch',
    max_context=1024,
    max_horizon=256
)
model.load_model()

# 4. Make predictions with confidence intervals
predictions = model.predict_with_quantiles(
    data['X_test'][:100],  # First 100 test samples
    horizon=24,
    context_length=1024
)

# 5. Visualize results
fig, ax = plt.subplots(figsize=(15, 6))

# Plot point forecasts
ax.plot(predictions['point'][:, 0], label='Predicted', color='red')

# Plot confidence intervals
ax.fill_between(
    range(len(predictions['point'])),
    predictions['lower_bound'][:, 0],
    predictions['upper_bound'][:, 0],
    alpha=0.3,
    color='red',
    label='90% CI'
)

# Plot actual values
ax.plot(data['y_test'][:100], label='Actual', color='blue')

ax.legend()
ax.set_title('TimesFM Bitcoin Price Predictions')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Price (USD)')
plt.show()

# 6. Evaluate
metrics = model.evaluate(data['X_test'], data['y_test'], horizon=1)
print(f"\nPerformance Metrics:")
print(f"  MAE: {metrics['mae']:.2f}")
print(f"  RMSE: {metrics['rmse']:.2f}")
print(f"  MAPE: {metrics['mape']:.2f}%")
print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
```

## Resources

- [TimesFM GitHub Repository](https://github.com/google-research/timesfm)
- [TimesFM Paper](https://arxiv.org/abs/2310.10688)
- [Hugging Face Model](https://huggingface.co/google/timesfm-2.5-200m-pytorch)

## Comparison with Other Approaches

### TimesFM vs Traditional ML

| Aspect | TimesFM | LSTM/GRU | XGBoost |
|--------|---------|----------|---------|
| Training Time | None (pre-trained) | Hours | Minutes |
| Feature Engineering | Not needed | Optional | Critical |
| Confidence Intervals | ✓ Built-in | ✗ Need ensemble | ✗ Need quantile regression |
| Interpretability | Low | Low | High (feature importance) |
| Data Requirements | Univariate series | Multivariate | Multivariate |

### When to Combine Models

Consider using **ensemble methods** combining TimesFM with other models:

```python
# Simple ensemble
timesfm_pred = timesfm_model.predict(X)
lstm_pred = lstm_model.predict(X)
xgb_pred = xgb_model.predict(X)

# Weighted average
ensemble_pred = 0.4 * timesfm_pred + 0.3 * lstm_pred + 0.3 * xgb_pred
```

## Conclusion

TimesFM is a powerful foundation model that provides:
- ✓ Zero-shot forecasting capabilities
- ✓ Built-in uncertainty quantification
- ✓ State-of-the-art performance without training
- ✓ Easy integration into existing pipelines

Try it for your Bitcoin price predictions and compare with other models!
