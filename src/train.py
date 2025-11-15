"""
Training script for Bitcoin price prediction models.

This script provides a unified interface for training LSTM, GRU, XGBoost, and TimesFM models.
"""

import argparse
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector import BTCDataCollector
from src.feature_engineering import create_features
from src.preprocessing import prepare_data_for_lstm, prepare_data_for_xgboost, prepare_data_for_timesfm
from models.lstm_model import create_lstm_model
from models.gru_model import create_gru_model
from models.xgboost_model import create_xgboost_model
from models.timesfm_model import create_timesfm_model


def train_lstm_model(
    df: pd.DataFrame,
    sequence_length: int = 60,
    epochs: int = 100,
    batch_size: int = 32,
    model_type: str = 'lstm'
) -> dict:
    """
    Train LSTM model.

    Args:
        df: DataFrame with features
        sequence_length: Sequence length for LSTM
        epochs: Number of training epochs
        batch_size: Batch size
        model_type: Type of LSTM ('lstm' or 'bidirectional')

    Returns:
        Dictionary with model and results
    """
    print("\n" + "="*80)
    print(f"Training {model_type.upper()} Model")
    print("="*80)

    # Prepare data
    data = prepare_data_for_lstm(
        df,
        sequence_length=sequence_length,
        forecast_horizon=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Create model
    model = create_lstm_model(
        sequence_length=data['sequence_length'],
        n_features=data['n_features'],
        model_type=model_type,
        lstm_units=[128, 64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )

    # Build and display model
    model.build_model()
    print("\nModel Architecture:")
    model.get_model_summary()

    # Train model
    history = model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "-"*80)
    print("Evaluating on test set...")
    metrics = model.evaluate(data['X_test'], data['y_test'])

    print("\nTest Set Performance:")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")

    # Save model
    model_dir = "models/trained"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{model_type}_model_{timestamp}.h5")
    model.save_model(model_path)

    # Save preprocessor
    preprocessor_path = os.path.join(model_dir, f"{model_type}_preprocessor_{timestamp}.pkl")
    data['preprocessor'].save_scalers(preprocessor_path)

    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'data': data,
        'model_path': model_path,
        'preprocessor_path': preprocessor_path
    }


def train_gru_model(
    df: pd.DataFrame,
    sequence_length: int = 60,
    epochs: int = 100,
    batch_size: int = 32,
    model_type: str = 'gru'
) -> dict:
    """
    Train GRU model.

    Args:
        df: DataFrame with features
        sequence_length: Sequence length for GRU
        epochs: Number of training epochs
        batch_size: Batch size
        model_type: Type of GRU ('gru' or 'bidirectional')

    Returns:
        Dictionary with model and results
    """
    print("\n" + "="*80)
    print(f"Training {model_type.upper()} Model")
    print("="*80)

    # Prepare data
    data = prepare_data_for_lstm(
        df,
        sequence_length=sequence_length,
        forecast_horizon=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Create model
    model = create_gru_model(
        sequence_length=data['sequence_length'],
        n_features=data['n_features'],
        model_type=model_type,
        gru_units=[128, 64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )

    # Build and display model
    model.build_model()
    print("\nModel Architecture:")
    model.get_model_summary()

    # Train model
    history = model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "-"*80)
    print("Evaluating on test set...")
    metrics = model.evaluate(data['X_test'], data['y_test'])

    print("\nTest Set Performance:")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")

    # Save model
    model_dir = "models/trained"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{model_type}_model_{timestamp}.h5")
    model.save_model(model_path)

    # Save preprocessor
    preprocessor_path = os.path.join(model_dir, f"{model_type}_preprocessor_{timestamp}.pkl")
    data['preprocessor'].save_scalers(preprocessor_path)

    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'data': data,
        'model_path': model_path,
        'preprocessor_path': preprocessor_path
    }


def train_xgboost_model(df: pd.DataFrame) -> dict:
    """
    Train XGBoost model.

    Args:
        df: DataFrame with features

    Returns:
        Dictionary with model and results
    """
    print("\n" + "="*80)
    print("Training XGBoost Model")
    print("="*80)

    # Prepare data
    data = prepare_data_for_xgboost(
        df,
        forecast_horizon=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Create model
    model = create_xgboost_model(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8
    )

    # Train model
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        early_stopping_rounds=50,
        verbose=True
    )

    # Evaluate on test set
    print("\n" + "-"*80)
    print("Evaluating on test set...")
    metrics = model.evaluate(data['X_test'], data['y_test'])

    print("\nTest Set Performance:")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")

    # Get feature importance
    feature_importance = model.get_feature_importance(data['feature_columns'])
    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"  {i}. {feature}: {importance:.4f}")

    # Save model
    model_dir = "models/trained"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"xgboost_model_{timestamp}.pkl")
    model.save_model(model_path)

    return {
        'model': model,
        'metrics': metrics,
        'data': data,
        'feature_importance': feature_importance,
        'model_path': model_path
    }


def train_timesfm_model(
    df: pd.DataFrame,
    context_length: int = 512,
    horizon: int = 1,
    backend: str = "pytorch"
) -> dict:
    """
    Evaluate TimesFM model (pre-trained, no training needed).

    TimesFM is a pre-trained foundation model, so we load it and evaluate
    its performance directly on the data.

    Args:
        df: DataFrame with price data
        context_length: Context length for predictions
        horizon: Forecast horizon
        backend: Backend to use ('pytorch' or 'flax')

    Returns:
        Dictionary with model and results
    """
    print("\n" + "="*80)
    print("Loading and Evaluating TimesFM Model")
    print("="*80)
    print("Note: TimesFM is a pre-trained foundation model.")
    print("It will be loaded and evaluated directly without additional training.")

    # Prepare data (we only need the target column for TimesFM)
    data = prepare_data_for_timesfm(
        df,
        context_length=context_length,
        forecast_horizon=horizon,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Create and load model
    model = create_timesfm_model(
        backend=backend,
        max_context=min(context_length, 1024),  # TimesFM 2.5 supports up to 16k, but start with 1024
        max_horizon=max(horizon, 256),
        normalize_inputs=True,
        use_continuous_quantile_head=True
    )

    # Load pre-trained model
    model.load_model()

    # Evaluate on test set
    print("\n" + "-"*80)
    print("Evaluating on test set...")
    metrics = model.evaluate(
        data['X_test'],
        data['y_test'],
        horizon=horizon,
        context_length=context_length
    )

    print("\nTest Set Performance:")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")

    # Make sample predictions with quantiles
    print("\n" + "-"*80)
    print("Generating sample predictions with confidence intervals...")
    sample_predictions = model.predict_with_quantiles(
        data['X_test'][:5],  # First 5 test samples
        horizon=horizon,
        context_length=context_length
    )

    print(f"\nSample predictions (first 5):")
    for i in range(min(5, len(sample_predictions['point']))):
        point = sample_predictions['point'][i, 0] if horizon == 1 else sample_predictions['point'][i]
        lower = sample_predictions['lower_bound'][i, 0] if horizon == 1 else sample_predictions['lower_bound'][i]
        upper = sample_predictions['upper_bound'][i, 0] if horizon == 1 else sample_predictions['upper_bound'][i]
        actual = data['y_test'][i]

        print(f"  Sample {i+1}:")
        print(f"    Actual: {actual:.2f}")
        print(f"    Predicted: {point:.2f}")
        print(f"    90% CI: [{lower:.2f}, {upper:.2f}]")

    # Save model configuration
    model_dir = "models/trained"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(model_dir, f"timesfm_config_{timestamp}.pkl")
    model.save_config(config_path)

    return {
        'model': model,
        'metrics': metrics,
        'data': data,
        'config_path': config_path,
        'sample_predictions': sample_predictions
    }


def plot_training_history(history, model_name: str, save_path: str = None):
    """
    Plot training history.

    Args:
        history: Training history
        model_name: Name of model
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True)

    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title(f'{model_name} - MAE')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.close()


def save_results(results: dict, model_name: str, output_dir: str = "results"):
    """
    Save training results.

    Args:
        results: Results dictionary
        model_name: Name of model
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics
    metrics_file = os.path.join(output_dir, f"{model_name}_metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            'model': model_name,
            'timestamp': timestamp,
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in results['metrics'].items()},
            'model_path': results.get('model_path', '')
        }, f, indent=2)

    print(f"\nMetrics saved to {metrics_file}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Bitcoin price prediction models")
    parser.add_argument(
        "--model",
        type=str,
        choices=['lstm', 'gru', 'xgboost', 'timesfm', 'all'],
        default='lstm',
        help="Model to train (default: lstm)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (if None, will fetch new data)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of data to fetch (default: 365)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length for LSTM/GRU (default: 60)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for deep learning models (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length for TimesFM (default: 512)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=['pytorch', 'flax'],
        default='pytorch',
        help="Backend for TimesFM (default: pytorch)"
    )

    args = parser.parse_args()

    # Load or fetch data
    if args.data:
        print(f"Loading data from {args.data}...")
        df_raw = pd.read_csv(args.data)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
    else:
        print(f"Fetching {args.days} days of BTC data...")
        collector = BTCDataCollector()
        df_raw = collector.fetch_yahoo_finance(days=args.days, interval='1h')

    print(f"\nRaw data shape: {df_raw.shape}")

    # Create features
    print("\nCreating features...")
    df = create_features(df_raw, feature_set='all')
    print(f"Data with features shape: {df.shape}")

    # Train model(s)
    results = {}

    if args.model in ['lstm', 'all']:
        results['lstm'] = train_lstm_model(
            df,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        if 'history' in results['lstm']:
            plot_training_history(
                results['lstm']['history'],
                'LSTM',
                save_path='results/lstm_training_history.png'
            )
        save_results(results['lstm'], 'lstm')

    if args.model in ['gru', 'all']:
        results['gru'] = train_gru_model(
            df,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        if 'history' in results['gru']:
            plot_training_history(
                results['gru']['history'],
                'GRU',
                save_path='results/gru_training_history.png'
            )
        save_results(results['gru'], 'gru')

    if args.model in ['xgboost', 'all']:
        results['xgboost'] = train_xgboost_model(df)
        save_results(results['xgboost'], 'xgboost')

    if args.model in ['timesfm', 'all']:
        # TimesFM doesn't use features, just raw price data
        results['timesfm'] = train_timesfm_model(
            df_raw,  # Use raw data for TimesFM
            context_length=args.context_length,
            horizon=1,
            backend=args.backend
        )
        save_results(results['timesfm'], 'timesfm')

    # Print summary
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)

    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in result['metrics'].items():
            if isinstance(value, float):
                if metric == 'directional_accuracy':
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value:.6f}")

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
