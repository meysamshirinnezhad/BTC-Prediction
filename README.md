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
  - **LSTM** (Long Short-Term Memory) networks
  - **GRU** (Gated Recurrent Unit) networks
  - **XGBoost** for comparison with traditional ML
  - **TimesFM** - Google's pre-trained Time Series Foundation Model (NEW!)

- **Evaluation**: Comprehensive metrics including MAE, RMSE, MAPE, and directional accuracy

- **TimesFM Integration**:
  - Pre-trained foundation model - no training required!
  - Built-in quantile forecasting with confidence intervals
  - Supports up to 16k context length
  - Zero-shot predictions on new data

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
│   ├── xgboost_model.py  # XGBoost model definition
│   └── timesfm_model.py  # TimesFM model wrapper
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

## VPS Deployment & Quick Start

### Quick Start on VPS

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd BTC-Prediction

# 2. Run interactive setup
chmod +x quick_start.sh
./quick_start.sh

# 3. Or automated training
chmod +x auto_train.sh
./auto_train.sh timesfm 180  # Fast demo
./auto_train.sh xgboost 365  # Quick training
./auto_train.sh lstm 365     # Full training
```

### Background Training (For Long Sessions)

```bash
# Option 1: Using nohup
nohup ./auto_train.sh lstm 365 > training.log 2>&1 &

# Option 2: Using screen
screen -S btc-training
./auto_train.sh lstm 365
# Press Ctrl+A, then D to detach
# Reattach: screen -r btc-training

# Option 3: Using tmux
tmux new -s btc-training
./auto_train.sh lstm 365
# Press Ctrl+B, then D to detach
# Reattach: tmux attach -t btc-training
```

📖 **For complete VPS deployment guide, see [VPS_DEPLOYMENT.md](VPS_DEPLOYMENT.md)**

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

# Evaluate TimesFM (pre-trained, no training needed!)
python src/train.py --model timesfm --context-length 512

# Train all models
python src/train.py --model all --days 365
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

| Model | Training | MAE | RMSE | MAPE | Directional Accuracy |
|-------|----------|-----|------|------|---------------------|
| LSTM  | Required | TBD | TBD  | TBD  | TBD                |
| GRU   | Required | TBD | TBD  | TBD  | TBD                |
| XGBoost| Required| TBD | TBD  | TBD  | TBD                |
| TimesFM| **Pre-trained** | TBD | TBD  | TBD  | TBD                |

**Note**: TimesFM is a pre-trained model and doesn't require training time, making it ideal for quick predictions!

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

## Documentation

- **[USAGE.md](USAGE.md)** - Detailed usage guide with code examples
- **[TIMESFM_GUIDE.md](TIMESFM_GUIDE.md)** - Complete guide for using Google's TimesFM model
- **[notebooks/quickstart.ipynb](notebooks/quickstart.ipynb)** - Interactive tutorial

## Future Improvements

- [ ] Add sentiment analysis from social media and news
- [x] ~~Implement transformer-based models~~ (TimesFM integrated!)
- [ ] Add real-time prediction API
- [ ] Incorporate on-chain metrics
- [ ] Ensemble methods combining TimesFM with other models
- [ ] AutoML for hyperparameter optimization
- [ ] Fine-tuning TimesFM on Bitcoin-specific data

## Disclaimer

This project is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Do not use these predictions for actual trading without proper risk management and understanding of the limitations.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.
