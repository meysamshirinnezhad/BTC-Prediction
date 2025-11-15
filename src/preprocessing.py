"""
Data preprocessing and preparation module for Bitcoin price prediction.

This module handles data cleaning, normalization, sequence creation, and
train/test splitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import os


class DataPreprocessor:
    """Preprocessor for time series data."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.scalers = {}
        self.feature_columns = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates(subset=['date'], keep='last')

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Handle missing values
        # For technical indicators, forward fill then backward fill
        df = df.ffill().bfill()

        # Replace infinity values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Remove any remaining NaN rows (usually at the beginning due to rolling windows)
        initial_len = len(df)
        df = df.dropna()
        removed = initial_len - len(df)

        if removed > 0:
            print(f"Removed {removed} rows with missing values and infinity values")

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
        include_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features for modeling.

        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude
            include_cols: Specific columns to include (if None, use all except excluded)

        Returns:
            Tuple of (DataFrame with selected features, list of feature names)
        """
        if exclude_cols is None:
            exclude_cols = ['date']

        if include_cols is None:
            # Use all columns except excluded ones
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        else:
            feature_cols = include_cols

        self.feature_columns = feature_cols

        return df[feature_cols], feature_cols

    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = 'minmax',
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize the data.

        Args:
            df: Input DataFrame
            method: Normalization method ('minmax' or 'standard')
            feature_cols: Columns to normalize (if None, normalize all numeric columns)

        Returns:
            Normalized DataFrame
        """
        df = df.copy()

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Safety check: Replace any remaining infinity values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN after infinity replacement
        if df[feature_cols].isnull().any().any():
            print(f"Warning: Found NaN values in features, dropping affected rows")
            df = df.dropna(subset=feature_cols)

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Fit and transform
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        # Store scaler for inverse transform
        self.scalers[method] = scaler

        return df

    def inverse_transform(
        self,
        data: np.ndarray,
        method: str = 'minmax',
        feature_index: int = 0
    ) -> np.ndarray:
        """
        Inverse transform normalized data.

        Args:
            data: Normalized data
            method: Normalization method used
            feature_index: Index of the feature in the scaler

        Returns:
            Original scale data
        """
        if method not in self.scalers:
            raise ValueError(f"No scaler found for method: {method}")

        scaler = self.scalers[method]

        # Handle different input shapes
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Create a dummy array with the same number of features as the scaler
        n_features = scaler.n_features_in_
        dummy = np.zeros((len(data), n_features))
        dummy[:, feature_index] = data.flatten()

        # Inverse transform
        inversed = scaler.inverse_transform(dummy)

        return inversed[:, feature_index]

    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int,
        target_column_index: int = 0,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            data: Input data array
            sequence_length: Length of input sequences
            target_column_index: Index of target column
            forecast_horizon: Number of steps to predict ahead

        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []

        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            # Input sequence
            X.append(data[i:i + sequence_length])

            # Target (price after forecast_horizon steps)
            target_idx = i + sequence_length + forecast_horizon - 1
            y.append(data[target_idx, target_column_index])

        return np.array(X), np.array(y)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Input sequences
            y: Target values
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
            "Ratios must sum to 1.0"

        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]

        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]

        print(f"Data split:")
        print(f"  Train: {len(X_train)} samples ({train_ratio*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({val_ratio*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({test_ratio*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_scalers(self, filepath: str):
        """
        Save scalers to file.

        Args:
            filepath: Path to save scalers
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'feature_columns': self.feature_columns
            }, f)

        print(f"Scalers saved to {filepath}")

    def load_scalers(self, filepath: str):
        """
        Load scalers from file.

        Args:
            filepath: Path to load scalers from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scalers = data['scalers']
            self.feature_columns = data['feature_columns']

        print(f"Scalers loaded from {filepath}")


def prepare_data_for_lstm(
    df: pd.DataFrame,
    target_col: str = 'close',
    sequence_length: int = 60,
    forecast_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    exclude_cols: Optional[List[str]] = None
) -> dict:
    """
    Prepare data for LSTM model training.

    Args:
        df: Input DataFrame with features
        target_col: Name of target column
        sequence_length: Length of input sequences
        forecast_horizon: Number of steps to predict ahead
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        exclude_cols: Columns to exclude from features

    Returns:
        Dictionary containing prepared data and preprocessor
    """
    print("Preparing data for LSTM model...")

    preprocessor = DataPreprocessor()

    # Clean data
    df = preprocessor.clean_data(df)
    print(f"Data after cleaning: {len(df)} samples")

    # Ensure target column is present
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Select features
    if exclude_cols is None:
        exclude_cols = ['date']

    df_features, feature_cols = preprocessor.select_features(df, exclude_cols=exclude_cols)

    # Reorder columns to put target first (for easier indexing)
    if target_col in feature_cols:
        feature_cols = [target_col] + [col for col in feature_cols if col != target_col]
        df_features = df_features[feature_cols]

    print(f"Using {len(feature_cols)} features")

    # Normalize data
    df_normalized = preprocessor.normalize_data(df_features, method='minmax', feature_cols=feature_cols)

    # Convert to numpy array
    data = df_normalized.values

    # Create sequences
    print(f"Creating sequences with length {sequence_length} and forecast horizon {forecast_horizon}...")
    target_idx = feature_cols.index(target_col)
    X, y = preprocessor.create_sequences(
        data,
        sequence_length=sequence_length,
        target_column_index=target_idx,
        forecast_horizon=forecast_horizon
    )

    print(f"Created {len(X)} sequences with shape {X.shape}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y, train_ratio, val_ratio, test_ratio
    )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_columns': feature_cols,
        'target_column': target_col,
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon,
        'n_features': len(feature_cols)
    }


def prepare_data_for_xgboost(
    df: pd.DataFrame,
    target_col: str = 'close',
    forecast_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    exclude_cols: Optional[List[str]] = None
) -> dict:
    """
    Prepare data for XGBoost model training.

    Args:
        df: Input DataFrame with features
        target_col: Name of target column
        forecast_horizon: Number of steps to predict ahead
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        exclude_cols: Columns to exclude from features

    Returns:
        Dictionary containing prepared data and preprocessor
    """
    print("Preparing data for XGBoost model...")

    preprocessor = DataPreprocessor()

    # Clean data
    df = preprocessor.clean_data(df)
    print(f"Data after cleaning: {len(df)} samples")

    # Create target (future price)
    df['target'] = df[target_col].shift(-forecast_horizon)

    # Remove rows with NaN target
    df = df.dropna(subset=['target'])

    # Select features
    if exclude_cols is None:
        exclude_cols = ['date', 'target']
    else:
        exclude_cols = list(exclude_cols) + ['target']

    # Exclude target column from features
    if target_col not in exclude_cols:
        exclude_cols.append(target_col)

    df_features, feature_cols = preprocessor.select_features(df, exclude_cols=exclude_cols)

    print(f"Using {len(feature_cols)} features")

    # Get target
    y = df['target'].values

    # Convert to numpy array
    X = df_features.values

    # Split data (XGBoost doesn't require normalization, but we can apply it)
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    print(f"Data split:")
    print(f"  Train: {len(X_train)} samples ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({test_ratio*100:.1f}%)")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_columns': feature_cols,
        'target_column': target_col,
        'forecast_horizon': forecast_horizon,
        'n_features': len(feature_cols)
    }


def prepare_data_for_timesfm(
    df: pd.DataFrame,
    target_col: str = 'close',
    context_length: int = 512,
    forecast_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> dict:
    """
    Prepare data for TimesFM model.

    TimesFM is a pre-trained foundation model that works differently from LSTM/GRU.
    It doesn't require sequence creation but uses variable-length time series.

    Args:
        df: Input DataFrame with features
        target_col: Name of target column (usually 'close')
        context_length: Maximum context length to use
        forecast_horizon: Number of steps to predict ahead
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data

    Returns:
        Dictionary containing prepared data
    """
    print("Preparing data for TimesFM model...")

    preprocessor = DataPreprocessor()

    # Clean data
    df = preprocessor.clean_data(df)
    print(f"Data after cleaning: {len(df)} samples")

    # Ensure target column is present
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Extract target column (TimesFM works with univariate time series)
    target_series = df[target_col].values

    print(f"Target series length: {len(target_series)}")

    # For evaluation, we need to create sequences
    # Each sequence will be used to predict the next value(s)
    X_list = []
    y_list = []

    # Create sliding windows
    for i in range(context_length, len(target_series) - forecast_horizon + 1):
        X_list.append(target_series[i-context_length:i])
        y_list.append(target_series[i + forecast_horizon - 1])

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Created {len(X)} sequences with context length {context_length}")

    # Split data
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    print(f"Data split:")
    print(f"  Train: {len(X_train)} samples ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({test_ratio*100:.1f}%)")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'target_column': target_col,
        'context_length': context_length,
        'forecast_horizon': forecast_horizon,
        'full_series': target_series,
        'dates': df['date'].values if 'date' in df.columns else None
    }


if __name__ == "__main__":
    print("Data Preprocessing Module")
    print("This module provides data preprocessing and preparation utilities.")
