"""
Prediction and visualization module for Bitcoin price prediction.

This script provides utilities for making predictions and visualizing results.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector import BTCDataCollector
from src.feature_engineering import create_features
from src.preprocessing import DataPreprocessor
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from models.xgboost_model import XGBoostModel

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)


class BTCPredictor:
    """Bitcoin price predictor."""

    def __init__(self, model_path: str, model_type: str, preprocessor_path: str = None):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model
            model_type: Type of model ('lstm', 'gru', 'xgboost')
            preprocessor_path: Path to preprocessor (for LSTM/GRU)
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = None

        # Load model
        self.load_model(model_path, preprocessor_path)

    def load_model(self, model_path: str, preprocessor_path: str = None):
        """
        Load trained model.

        Args:
            model_path: Path to model
            preprocessor_path: Path to preprocessor
        """
        print(f"Loading {self.model_type} model from {model_path}...")

        if self.model_type == 'lstm':
            self.model = LSTMModel(input_shape=(60, 1))  # Placeholder
            self.model.load_model(model_path)
        elif self.model_type == 'gru':
            self.model = GRUModel(input_shape=(60, 1))  # Placeholder
            self.model.load_model(model_path)
        elif self.model_type == 'xgboost':
            self.model = XGBoostModel()
            self.model.load_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load preprocessor for LSTM/GRU
        if preprocessor_path and self.model_type in ['lstm', 'gru']:
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_scalers(preprocessor_path)

        print("Model loaded successfully!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input data

        Returns:
            Predictions
        """
        predictions = self.model.predict(X)

        # Inverse transform if preprocessor is available
        if self.preprocessor and self.model_type in ['lstm', 'gru']:
            predictions = self.preprocessor.inverse_transform(predictions, method='minmax', feature_index=0)

        return predictions

    def predict_future(
        self,
        df: pd.DataFrame,
        horizon: int = 24,
        sequence_length: int = 60
    ) -> pd.DataFrame:
        """
        Predict future prices.

        Args:
            df: Historical data with features
            horizon: Number of steps to predict
            sequence_length: Sequence length (for LSTM/GRU)

        Returns:
            DataFrame with predictions
        """
        print(f"\nPredicting next {horizon} time steps...")

        predictions = []
        dates = []

        if self.model_type in ['lstm', 'gru']:
            # Get last sequence
            last_sequence = df.iloc[-sequence_length:].copy()

            # Make predictions iteratively
            for i in range(horizon):
                # Prepare input
                X = last_sequence.values.reshape(1, sequence_length, -1)

                # Predict
                pred = self.model.predict(X)[0, 0]
                predictions.append(pred)

                # Update sequence (simplified - in practice would update all features)
                new_row = last_sequence.iloc[-1:].copy()
                new_row.iloc[0, 0] = pred  # Assuming first column is close price
                last_sequence = pd.concat([last_sequence.iloc[1:], new_row])

                # Generate future date
                last_date = df['date'].iloc[-1] if 'date' in df.columns else datetime.now()
                future_date = last_date + timedelta(hours=i+1)
                dates.append(future_date)

        else:  # XGBoost
            # For XGBoost, we need to prepare features differently
            # This is a simplified version
            for i in range(horizon):
                X = df.iloc[-1:].drop(columns=['date'], errors='ignore').values
                pred = self.model.predict(X)[0]
                predictions.append(pred)

                last_date = df['date'].iloc[-1] if 'date' in df.columns else datetime.now()
                future_date = last_date + timedelta(hours=i+1)
                dates.append(future_date)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates,
            'predicted_price': predictions
        })

        return results_df


def plot_predictions(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
    title: str = "BTC Price Predictions",
    save_path: str = None
):
    """
    Plot actual vs predicted prices.

    Args:
        actual: DataFrame with actual prices
        predictions: DataFrame with predictions
        title: Plot title
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot actual prices
    if 'date' in actual.columns:
        ax.plot(actual['date'], actual['close'], label='Actual Price', linewidth=2, color='blue')
    else:
        ax.plot(actual['close'].values, label='Actual Price', linewidth=2, color='blue')

    # Plot predictions
    if 'date' in predictions.columns:
        ax.plot(predictions['date'], predictions['predicted_price'],
                label='Predicted Price', linewidth=2, color='red', linestyle='--')
    else:
        ax.plot(predictions['predicted_price'].values,
                label='Predicted Price', linewidth=2, color='red', linestyle='--')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    plt.show()


def plot_prediction_intervals(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
    confidence: float = 0.95,
    title: str = "BTC Price Predictions with Confidence Intervals",
    save_path: str = None
):
    """
    Plot predictions with confidence intervals.

    Args:
        actual: DataFrame with actual prices
        predictions: DataFrame with predictions
        confidence: Confidence level
        title: Plot title
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    # Calculate prediction intervals (simplified)
    pred_values = predictions['predicted_price'].values
    std = np.std(pred_values) * 1.5  # Simplified
    lower_bound = pred_values - std
    upper_bound = pred_values + std

    # Plot actual prices
    if 'date' in actual.columns:
        ax.plot(actual['date'], actual['close'], label='Actual Price', linewidth=2, color='blue')
        pred_dates = predictions['date']
    else:
        ax.plot(actual['close'].values, label='Actual Price', linewidth=2, color='blue')
        pred_dates = range(len(predictions))

    # Plot predictions
    ax.plot(pred_dates, pred_values, label='Predicted Price', linewidth=2, color='red', linestyle='--')

    # Plot confidence intervals
    ax.fill_between(pred_dates, lower_bound, upper_bound, alpha=0.2, color='red',
                     label=f'{int(confidence*100)}% Confidence Interval')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    plt.show()


def plot_feature_importance(feature_importance: dict, top_n: int = 20, save_path: str = None):
    """
    Plot feature importance.

    Args:
        feature_importance: Dictionary of feature importance
        top_n: Number of top features to show
        save_path: Path to save plot
    """
    # Get top N features
    top_features = dict(list(feature_importance.items())[:top_n])

    fig, ax = plt.subplots(figsize=(12, 8))

    features = list(top_features.keys())
    importance = list(top_features.values())

    ax.barh(features, importance, color='steelblue')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    plt.show()


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Make Bitcoin price predictions")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['lstm', 'gru', 'xgboost'],
        required=True,
        help="Type of model"
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default=None,
        help="Path to preprocessor (for LSTM/GRU)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (if None, will fetch new data)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Prediction horizon (number of steps ahead)"
    )
    parser.add_argument(
        "--plot",
        action='store_true',
        help="Show plots"
    )

    args = parser.parse_args()

    # Load data
    if args.data:
        print(f"Loading data from {args.data}...")
        df_raw = pd.read_csv(args.data)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
    else:
        print("Fetching latest BTC data...")
        collector = BTCDataCollector()
        df_raw = collector.fetch_yahoo_finance(days=30, interval='1h')

    # Create features
    print("Creating features...")
    df = create_features(df_raw, feature_set='all')

    # Initialize predictor
    predictor = BTCPredictor(
        model_path=args.model,
        model_type=args.model_type,
        preprocessor_path=args.preprocessor
    )

    # Make predictions
    predictions = predictor.predict_future(df, horizon=args.horizon)

    # Display predictions
    print("\n" + "="*80)
    print("Predictions")
    print("="*80)
    print(predictions.to_string(index=False))

    # Save predictions
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file = os.path.join(output_dir, f"predictions_{args.model_type}_{timestamp}.csv")
    predictions.to_csv(pred_file, index=False)
    print(f"\nPredictions saved to {pred_file}")

    # Plot if requested
    if args.plot:
        plot_predictions(
            df_raw.tail(100),
            predictions,
            title=f"{args.model_type.upper()} BTC Price Predictions",
            save_path=os.path.join(output_dir, f"predictions_plot_{args.model_type}_{timestamp}.png")
        )

    print("\n" + "="*80)
    print("Prediction Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
