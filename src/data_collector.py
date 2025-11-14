"""
Data collection module for fetching Bitcoin historical price data.

This module provides functions to download BTC price data from various sources
including Yahoo Finance and cryptocurrency exchanges.
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm


class BTCDataCollector:
    """Collector for Bitcoin historical price data."""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data collector.

        Args:
            data_dir: Directory to save raw data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def fetch_yahoo_finance(
        self,
        days: int = 365,
        interval: str = "1h",
        symbol: str = "BTC-USD"
    ) -> pd.DataFrame:
        """
        Fetch BTC data from Yahoo Finance.

        Args:
            days: Number of days of historical data to fetch
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            symbol: Trading pair symbol

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {days} days of {symbol} data with {interval} interval from Yahoo Finance...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )

            if df.empty:
                raise ValueError(f"No data received for {symbol}")

            # Reset index to make datetime a column
            df.reset_index(inplace=True)

            # Rename columns to standard format
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            # For hourly or minute data, the column might be 'datetime'
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)

            print(f"Successfully fetched {len(df)} records from {df['date'].min()} to {df['date'].max()}")

            return df

        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            raise

    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save dataframe to CSV file.

        Args:
            df: DataFrame to save
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"btc_data_{timestamp}.csv"

        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

        return filepath

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filename: Name of file to load

        Returns:
            DataFrame with loaded data
        """
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])

        print(f"Loaded {len(df)} records from {filepath}")

        return df

    def get_latest_data_file(self) -> Optional[str]:
        """
        Get the most recently saved data file.

        Returns:
            Filename of latest data file or None
        """
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        if not files:
            return None

        # Sort by modification time
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)), reverse=True)

        return files[0]

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate the integrity of the data.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check for required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for missing values
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            issues.append(f"Missing values found: {null_cols.to_dict()}")

        # Check for duplicates
        if df.duplicated(subset=['date']).any():
            dup_count = df.duplicated(subset=['date']).sum()
            issues.append(f"Found {dup_count} duplicate timestamps")

        # Check for negative values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")

        # Check OHLC logic (High >= Low, etc.)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            if (df['high'] < df['low']).any():
                issues.append("Invalid OHLC: High < Low found")
            if (df['high'] < df['close']).any():
                issues.append("Invalid OHLC: High < Close found")
            if (df['low'] > df['close']).any():
                issues.append("Invalid OHLC: Low > Close found")

        is_valid = len(issues) == 0

        return is_valid, issues

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the data.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'price_stats': {
                'min': df['low'].min(),
                'max': df['high'].max(),
                'mean': df['close'].mean(),
                'std': df['close'].std()
            },
            'volume_stats': {
                'total': df['volume'].sum(),
                'mean': df['volume'].mean(),
                'max': df['volume'].max()
            }
        }

        return summary


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Fetch Bitcoin historical price data")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch (default: 365)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"],
        help="Data interval (default: 1h)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC-USD",
        help="Trading pair symbol (default: BTC-USD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (auto-generated if not specified)"
    )

    args = parser.parse_args()

    # Initialize collector
    collector = BTCDataCollector()

    # Fetch data
    df = collector.fetch_yahoo_finance(
        days=args.days,
        interval=args.interval,
        symbol=args.symbol
    )

    # Validate data
    is_valid, issues = collector.validate_data(df)
    if not is_valid:
        print("\nData validation warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nData validation: PASSED")

    # Print summary
    summary = collector.get_data_summary(df)
    print("\nData Summary:")
    print(f"  Records: {summary['records']}")
    print(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"  Price Range: ${summary['price_stats']['min']:.2f} - ${summary['price_stats']['max']:.2f}")
    print(f"  Average Price: ${summary['price_stats']['mean']:.2f} (±${summary['price_stats']['std']:.2f})")
    print(f"  Total Volume: {summary['volume_stats']['total']:.2f}")

    # Save data
    filepath = collector.save_data(df, args.output)
    print(f"\nData collection complete! File saved to: {filepath}")


if __name__ == "__main__":
    main()
