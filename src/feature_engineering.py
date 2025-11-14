"""
Feature engineering module for Bitcoin price prediction.

This module provides functions to calculate technical indicators and create
features for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Optional, List


class FeatureEngineer:
    """Feature engineering for time series data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer.

        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()

    def add_returns(self) -> pd.DataFrame:
        """
        Add return-based features.

        Returns:
            DataFrame with return features
        """
        # Simple returns
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))

        # Multiple period returns
        for period in [3, 7, 14, 30]:
            self.df[f'returns_{period}d'] = self.df['close'].pct_change(period)

        return self.df

    def add_moving_averages(self, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Add moving average features.

        Args:
            windows: List of window sizes (default: [7, 14, 21, 30, 50, 100, 200])

        Returns:
            DataFrame with moving average features
        """
        if windows is None:
            windows = [7, 14, 21, 30, 50, 100, 200]

        for window in windows:
            # Simple Moving Average
            self.df[f'sma_{window}'] = self.df['close'].rolling(window=window).mean()

            # Exponential Moving Average
            self.df[f'ema_{window}'] = self.df['close'].ewm(span=window, adjust=False).mean()

            # Price relative to MA
            self.df[f'close_to_sma_{window}'] = self.df['close'] / self.df[f'sma_{window}'] - 1
            self.df[f'close_to_ema_{window}'] = self.df['close'] / self.df[f'ema_{window}'] - 1

        return self.df

    def add_bollinger_bands(self, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands features.

        Args:
            window: Rolling window size
            num_std: Number of standard deviations

        Returns:
            DataFrame with Bollinger Bands features
        """
        # Calculate Bollinger Bands
        sma = self.df['close'].rolling(window=window).mean()
        std = self.df['close'].rolling(window=window).std()

        self.df['bb_upper'] = sma + (std * num_std)
        self.df['bb_middle'] = sma
        self.df['bb_lower'] = sma - (std * num_std)

        # Bollinger Band width and position
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
        self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])

        return self.df

    def add_rsi(self, window: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).

        Args:
            window: Window size for RSI calculation

        Returns:
            DataFrame with RSI feature
        """
        delta = self.df['close'].diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        self.df[f'rsi_{window}'] = 100 - (100 / (1 + rs))

        return self.df

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD features
        """
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()

        self.df['macd'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal, adjust=False).mean()
        self.df['macd_diff'] = self.df['macd'] - self.df['macd_signal']

        return self.df

    def add_stochastic(self, window: int = 14, smooth: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.

        Args:
            window: Lookback period
            smooth: Smoothing period

        Returns:
            DataFrame with Stochastic features
        """
        low_min = self.df['low'].rolling(window=window).min()
        high_max = self.df['high'].rolling(window=window).max()

        self.df['stoch_k'] = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=smooth).mean()

        return self.df

    def add_atr(self, window: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR).

        Args:
            window: Window size

        Returns:
            DataFrame with ATR feature
        """
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df[f'atr_{window}'] = true_range.rolling(window=window).mean()

        # Normalized ATR
        self.df[f'atr_{window}_pct'] = self.df[f'atr_{window}'] / self.df['close']

        return self.df

    def add_obv(self) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV).

        Returns:
            DataFrame with OBV feature
        """
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
                obv.append(obv[-1] + self.df['volume'].iloc[i])
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                obv.append(obv[-1] - self.df['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        self.df['obv'] = obv

        # OBV moving average
        self.df['obv_sma_20'] = self.df['obv'].rolling(window=20).mean()

        return self.df

    def add_volume_features(self) -> pd.DataFrame:
        """
        Add volume-based features.

        Returns:
            DataFrame with volume features
        """
        # Volume moving averages
        self.df['volume_sma_20'] = self.df['volume'].rolling(window=20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma_20']

        # Volume rate of change
        self.df['volume_roc'] = self.df['volume'].pct_change(periods=5)

        return self.df

    def add_price_patterns(self) -> pd.DataFrame:
        """
        Add price pattern features.

        Returns:
            DataFrame with price pattern features
        """
        # Daily range
        self.df['daily_range'] = self.df['high'] - self.df['low']
        self.df['daily_range_pct'] = self.df['daily_range'] / self.df['close']

        # Body size (difference between open and close)
        self.df['body'] = self.df['close'] - self.df['open']
        self.df['body_pct'] = self.df['body'] / self.df['open']

        # Upper and lower shadows
        self.df['upper_shadow'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['lower_shadow'] = self.df[['open', 'close']].min(axis=1) - self.df['low']

        return self.df

    def add_volatility_features(self) -> pd.DataFrame:
        """
        Add volatility-based features.

        Returns:
            DataFrame with volatility features
        """
        # Rolling standard deviation
        for window in [7, 14, 30]:
            self.df[f'volatility_{window}d'] = self.df['returns'].rolling(window=window).std()

        # Parkinson's volatility (uses high-low range)
        for window in [7, 14, 30]:
            hl_ratio = np.log(self.df['high'] / self.df['low'])
            self.df[f'parkinson_vol_{window}d'] = np.sqrt(
                hl_ratio.pow(2).rolling(window=window).mean() / (4 * np.log(2))
            )

        return self.df

    def add_momentum_features(self) -> pd.DataFrame:
        """
        Add momentum-based features.

        Returns:
            DataFrame with momentum features
        """
        # Rate of Change (ROC)
        for period in [7, 14, 21, 30]:
            self.df[f'roc_{period}'] = (
                (self.df['close'] - self.df['close'].shift(period)) /
                self.df['close'].shift(period) * 100
            )

        return self.df

    def add_lagged_features(self, lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Add lagged price features.

        Args:
            lags: List of lag periods (default: [1, 2, 3, 7, 14])

        Returns:
            DataFrame with lagged features
        """
        if lags is None:
            lags = [1, 2, 3, 7, 14]

        for lag in lags:
            self.df[f'close_lag_{lag}'] = self.df['close'].shift(lag)
            self.df[f'volume_lag_{lag}'] = self.df['volume'].shift(lag)

        return self.df

    def add_time_features(self) -> pd.DataFrame:
        """
        Add time-based features.

        Returns:
            DataFrame with time features
        """
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])

            self.df['hour'] = self.df['date'].dt.hour
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            self.df['day_of_month'] = self.df['date'].dt.day
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter

            # Cyclical encoding for time features
            self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
            self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
            self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

        return self.df

    def add_all_features(self) -> pd.DataFrame:
        """
        Add all available features.

        Returns:
            DataFrame with all features
        """
        print("Adding return features...")
        self.add_returns()

        print("Adding moving averages...")
        self.add_moving_averages()

        print("Adding Bollinger Bands...")
        self.add_bollinger_bands()

        print("Adding RSI...")
        self.add_rsi()

        print("Adding MACD...")
        self.add_macd()

        print("Adding Stochastic Oscillator...")
        self.add_stochastic()

        print("Adding ATR...")
        self.add_atr()

        print("Adding OBV...")
        self.add_obv()

        print("Adding volume features...")
        self.add_volume_features()

        print("Adding price patterns...")
        self.add_price_patterns()

        print("Adding volatility features...")
        self.add_volatility_features()

        print("Adding momentum features...")
        self.add_momentum_features()

        print("Adding lagged features...")
        self.add_lagged_features()

        print("Adding time features...")
        self.add_time_features()

        print(f"Feature engineering complete! Total features: {len(self.df.columns)}")

        return self.df

    def get_feature_names(self, exclude_ohlcv: bool = True) -> List[str]:
        """
        Get list of feature names.

        Args:
            exclude_ohlcv: Whether to exclude OHLCV columns

        Returns:
            List of feature names
        """
        if exclude_ohlcv:
            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            return [col for col in self.df.columns if col not in exclude_cols]
        else:
            return list(self.df.columns)


def create_features(df: pd.DataFrame, feature_set: str = 'all') -> pd.DataFrame:
    """
    Convenience function to create features.

    Args:
        df: Input DataFrame with OHLCV data
        feature_set: Feature set to create ('all', 'basic', 'technical')

    Returns:
        DataFrame with features
    """
    engineer = FeatureEngineer(df)

    if feature_set == 'all':
        return engineer.add_all_features()
    elif feature_set == 'basic':
        engineer.add_returns()
        engineer.add_moving_averages([7, 14, 30])
        engineer.add_volume_features()
        return engineer.df
    elif feature_set == 'technical':
        engineer.add_returns()
        engineer.add_moving_averages()
        engineer.add_bollinger_bands()
        engineer.add_rsi()
        engineer.add_macd()
        engineer.add_stochastic()
        engineer.add_atr()
        return engineer.df
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("This module provides technical indicators for time series data.")
    print("\nAvailable features:")
    print("  - Returns (simple, log, multi-period)")
    print("  - Moving Averages (SMA, EMA)")
    print("  - Bollinger Bands")
    print("  - RSI (Relative Strength Index)")
    print("  - MACD (Moving Average Convergence Divergence)")
    print("  - Stochastic Oscillator")
    print("  - ATR (Average True Range)")
    print("  - OBV (On-Balance Volume)")
    print("  - Volume features")
    print("  - Price patterns")
    print("  - Volatility features")
    print("  - Momentum features")
    print("  - Lagged features")
    print("  - Time-based features")
