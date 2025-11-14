# Bitcoin Price Prediction System

A comprehensive machine learning system for predicting Bitcoin (BTC) prices using multiple deep learning and traditional ML models.

## Overview

This project implements a complete end-to-end pipeline for Bitcoin price prediction, including:
- Historical data collection from multiple sources
- Advanced feature engineering with technical indicators
- Multiple model architectures (LSTM, GRU, XGBoost)
- Comprehensive evaluation metrics
- Visualization and prediction tools

## Features

- **Data Collection**: Automated fetching of historical BTC price data from Yahoo Finance and cryptocurrency exchanges
- **Feature Engineering**: 50+ technical indicators including:
  - Moving Averages (SMA, EMA)
  - Momentum indicators (RSI, MACD, Stochastic)
  - Volatility indicators (Bollinger Bands, ATR)
  - Volume indicators (OBV, Volume SMA)
  - Custom features (price changes, rolling statistics)

- **Multiple Models**:
  - LSTM (Long Short-Term Memory) networks
  - GRU (Gated Recurrent Unit) networks
  - XGBoost for comparison with traditional ML

- **Evaluation**: Comprehensive metrics including MAE, RMSE, MAPE, and directional accuracy

## Project Structure

```
BTC-Prediction/
├── data/
│   ├── raw/              # Raw data from APIs
│   └── processed/        # Processed and feature-engineered data
├── models/
│   ├── trained/          # Saved model weights
│   ├── lstm_model.py     # LSTM model definition
│   ├── gru_model.py      # GRU model definition
│   └── xgboost_model.py  # XGBoost model definition
├── src/
│   ├── data_collector.py      # Data fetching utilities
│   ├── feature_engineering.py # Technical indicator calculation
│   ├── preprocessing.py       # Data preprocessing
│   ├── train.py              # Model training script
│   └── predict.py            # Prediction script
├── notebooks/
│   └── exploration.ipynb      # Data exploration
├── tests/
│   └── test_models.py         # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BTC-Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Data

```bash
python src/data_collector.py --days 365 --interval 1h
```

### 2. Train Models

```bash
# Train LSTM model
python src/train.py --model lstm --epochs 100

# Train GRU model
python src/train.py --model gru --epochs 100

# Train XGBoost model
python src/train.py --model xgboost
```

### 3. Make Predictions

```bash
python src/predict.py --model lstm --horizon 24
```

### 4. Evaluate Models

```bash
python src/train.py --model lstm --evaluate
```

## Model Performance

| Model | MAE | RMSE | MAPE | Directional Accuracy |
|-------|-----|------|------|---------------------|
| LSTM  | TBD | TBD  | TBD  | TBD                |
| GRU   | TBD | TBD  | TBD  | TBD                |
| XGBoost| TBD | TBD  | TBD  | TBD                |

## Configuration

Key parameters can be configured in `src/config.py`:
- Data collection parameters (interval, lookback period)
- Feature engineering settings
- Model hyperparameters
- Training parameters

## Technical Indicators

The system uses the following technical indicators:
- **Trend**: SMA, EMA, MACD
- **Momentum**: RSI, Stochastic Oscillator, ROC
- **Volatility**: Bollinger Bands, ATR, Standard Deviation
- **Volume**: OBV, Volume SMA, Volume ROC
- **Custom**: Price changes, rolling statistics, lagged features

## Prediction Methodology

1. **Data Preprocessing**: Normalize data, handle missing values, create sequences
2. **Feature Selection**: Select most informative features based on correlation and importance
3. **Model Training**: Train multiple models with cross-validation
4. **Ensemble**: Combine predictions from multiple models
5. **Evaluation**: Assess performance on held-out test set

## Future Improvements

- [ ] Add sentiment analysis from social media and news
- [ ] Implement transformer-based models
- [ ] Add real-time prediction API
- [ ] Incorporate on-chain metrics
- [ ] Ensemble methods for improved accuracy
- [ ] AutoML for hyperparameter optimization

## Disclaimer

This project is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Do not use these predictions for actual trading without proper risk management and understanding of the limitations.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.
