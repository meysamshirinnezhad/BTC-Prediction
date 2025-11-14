"""
Configuration file for BTC Prediction system.
"""

import os

# Data Collection Settings
DATA_CONFIG = {
    'default_days': 365,
    'default_interval': '1h',
    'default_symbol': 'BTC-USD',
    'data_dir': 'data/raw',
}

# Feature Engineering Settings
FEATURE_CONFIG = {
    'ma_windows': [7, 14, 21, 30, 50, 100, 200],
    'rsi_window': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_window': 20,
    'bollinger_std': 2.0,
    'atr_window': 14,
    'stochastic_window': 14,
    'stochastic_smooth': 3,
}

# Preprocessing Settings
PREPROCESSING_CONFIG = {
    'sequence_length': 60,
    'forecast_horizon': 1,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'normalization_method': 'minmax',
}

# LSTM Model Settings
LSTM_CONFIG = {
    'lstm_units': [128, 64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 5,
}

# GRU Model Settings
GRU_CONFIG = {
    'gru_units': [128, 64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 5,
}

# XGBoost Model Settings
XGBOOST_CONFIG = {
    'n_estimators': 1000,
    'max_depth': 7,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'early_stopping_rounds': 50,
}

# Model Paths
MODEL_PATHS = {
    'trained_models_dir': 'models/trained',
    'results_dir': 'results',
    'plots_dir': 'plots',
}

# Ensure directories exist
for dir_path in [DATA_CONFIG['data_dir'], MODEL_PATHS['trained_models_dir'],
                 MODEL_PATHS['results_dir'], MODEL_PATHS['plots_dir']]:
    os.makedirs(dir_path, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging Settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}
