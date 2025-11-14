# Usage Guide

This guide provides detailed instructions on how to use the BTC Prediction system.

## Table of Contents

1. [Installation](#installation)
2. [Data Collection](#data-collection)
3. [Feature Engineering](#feature-engineering)
4. [Training Models](#training-models)
5. [Making Predictions](#making-predictions)
6. [Evaluation](#evaluation)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd BTC-Prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Collection

### Command Line Usage

Fetch 365 days of hourly BTC data:
```bash
python src/data_collector.py --days 365 --interval 1h
```

Available intervals:
- `1m`, `2m`, `5m`, `15m`, `30m` - Minute intervals
- `1h`, `2h`, `4h` - Hourly intervals
- `1d` - Daily
- `1wk` - Weekly
- `1mo` - Monthly

### Python API

```python
from src.data_collector import BTCDataCollector

collector = BTCDataCollector()
df = collector.fetch_yahoo_finance(days=365, interval='1h')

# Save data
collector.save_data(df, 'btc_data.csv')

# Load data
df = collector.load_data('btc_data.csv')
```

## Feature Engineering

### Using the FeatureEngineer Class

```python
from src.feature_engineering import FeatureEngineer

# Create feature engineer
engineer = FeatureEngineer(df)

# Add all features
df_features = engineer.add_all_features()

# Or add specific feature groups
engineer.add_returns()
engineer.add_moving_averages()
engineer.add_rsi()
engineer.add_macd()
# ... etc
```

### Using the Convenience Function

```python
from src.feature_engineering import create_features

# Create all features
df_features = create_features(df, feature_set='all')

# Create basic features only
df_features = create_features(df, feature_set='basic')

# Create technical indicators only
df_features = create_features(df, feature_set='technical')
```

## Training Models

### Using the Training Script

Train LSTM model:
```bash
python src/train.py --model lstm --epochs 100 --batch-size 32
```

Train GRU model:
```bash
python src/train.py --model gru --epochs 100
```

Train XGBoost model:
```bash
python src/train.py --model xgboost
```

Train all models:
```bash
python src/train.py --model all --days 365
```

### Python API

#### LSTM Model

```python
from src.preprocessing import prepare_data_for_lstm
from models.lstm_model import create_lstm_model

# Prepare data
data = prepare_data_for_lstm(
    df,
    sequence_length=60,
    forecast_horizon=1
)

# Create and train model
model = create_lstm_model(
    sequence_length=60,
    n_features=data['n_features'],
    lstm_units=[128, 64, 32],
    dropout_rate=0.2
)

model.build_model()
model.train(
    data['X_train'],
    data['y_train'],
    data['X_val'],
    data['y_val'],
    epochs=100
)

# Save model
model.save_model('models/trained/my_lstm_model.h5')
```

#### GRU Model

```python
from models.gru_model import create_gru_model

model = create_gru_model(
    sequence_length=60,
    n_features=data['n_features'],
    gru_units=[128, 64, 32]
)

model.build_model()
model.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
```

#### XGBoost Model

```python
from src.preprocessing import prepare_data_for_xgboost
from models.xgboost_model import create_xgboost_model

# Prepare data
data = prepare_data_for_xgboost(df, forecast_horizon=1)

# Create and train model
model = create_xgboost_model(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.01
)

model.train(
    data['X_train'],
    data['y_train'],
    data['X_val'],
    data['y_val']
)

# Save model
model.save_model('models/trained/my_xgboost_model.pkl')
```

## Making Predictions

### Using the Prediction Script

```bash
python src/predict.py \
    --model models/trained/lstm_model.h5 \
    --model-type lstm \
    --preprocessor models/trained/lstm_preprocessor.pkl \
    --horizon 24 \
    --plot
```

### Python API

```python
from src.predict import BTCPredictor

# Initialize predictor
predictor = BTCPredictor(
    model_path='models/trained/lstm_model.h5',
    model_type='lstm',
    preprocessor_path='models/trained/lstm_preprocessor.pkl'
)

# Make predictions
predictions = predictor.predict_future(df, horizon=24)

print(predictions)
```

## Evaluation

### Evaluate Model Performance

```python
# For LSTM/GRU models
metrics = model.evaluate(X_test, y_test)

print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
```

### Get Feature Importance (XGBoost)

```python
feature_importance = model.get_feature_importance(feature_names)

# Print top 10 features
for feature, importance in list(feature_importance.items())[:10]:
    print(f"{feature}: {importance:.4f}")
```

## Configuration

You can modify default settings in `src/config.py`:

```python
# Example: Change sequence length for LSTM
PREPROCESSING_CONFIG['sequence_length'] = 120

# Example: Change LSTM architecture
LSTM_CONFIG['lstm_units'] = [256, 128, 64]

# Example: Change learning rate
LSTM_CONFIG['learning_rate'] = 0.0005
```

## Tips and Best Practices

1. **Data Quality**: Ensure you have sufficient historical data (at least 6 months recommended)

2. **Feature Selection**: Not all features may be useful. Use feature importance analysis to select the most relevant ones

3. **Hyperparameter Tuning**: Experiment with different hyperparameters for better performance

4. **Ensemble Methods**: Combine predictions from multiple models for better accuracy

5. **Validation**: Always validate on unseen data to avoid overfitting

6. **Regular Retraining**: Retrain models periodically with fresh data

7. **Risk Management**: Never use predictions as the sole basis for trading decisions

## Troubleshooting

### Common Issues

**Issue**: Out of memory error during training
- **Solution**: Reduce batch size or sequence length

**Issue**: Model not converging
- **Solution**: Try different learning rates or model architectures

**Issue**: Poor directional accuracy
- **Solution**: Add more relevant features or increase training data

**Issue**: Data fetching fails
- **Solution**: Check internet connection and try different data sources

## Examples

See the `notebooks/` directory for comprehensive examples:
- `quickstart.ipynb` - Complete walkthrough of the system
- More notebooks coming soon!

## Support

For issues, questions, or contributions, please open an issue on GitHub.
