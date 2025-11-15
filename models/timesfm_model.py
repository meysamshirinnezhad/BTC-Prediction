"""
TimesFM model for Bitcoin price prediction.

This module implements a wrapper for Google's TimesFM (Time Series Foundation Model)
for time series forecasting of Bitcoin prices.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import os
import pickle


class TimesFMModel:
    """TimesFM model wrapper for time series prediction."""

    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        backend: str = "pytorch",
        max_context: int = 1024,
        max_horizon: int = 256,
        normalize_inputs: bool = True,
        use_continuous_quantile_head: bool = True,
        name: str = "timesfm_model"
    ):
        """
        Initialize TimesFM model.

        Args:
            model_name: Pre-trained model name/path
            backend: Backend to use ('pytorch' or 'flax')
            max_context: Maximum input sequence length (up to 16k)
            max_horizon: Maximum forecast horizon
            normalize_inputs: Whether to normalize inputs
            use_continuous_quantile_head: Enable quantile forecasting
            name: Model name
        """
        self.model_name = model_name
        self.backend = backend
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.normalize_inputs = normalize_inputs
        self.use_continuous_quantile_head = use_continuous_quantile_head
        self.name = name
        self.model = None
        self.is_loaded = False

        # Try to import TimesFM
        try:
            import timesfm
            self.timesfm = timesfm
            if backend == "pytorch":
                import torch
                self.torch = torch
                torch.set_float32_matmul_precision("high")
        except ImportError:
            raise ImportError(
                "TimesFM not installed. Install with: pip install timesfm[torch] or timesfm[flax]"
            )

    def load_model(self):
        """
        Load the pre-trained TimesFM model.
        """
        if self.is_loaded:
            print("Model already loaded.")
            return

        print(f"Loading TimesFM model: {self.model_name}...")

        try:
            # Load pre-trained TimesFM model using from_pretrained()
            if self.backend == "pytorch":
                self.model = self.timesfm.TimesFm.from_pretrained(
                    pretrained_model_name_or_path=self.model_name
                )
            elif self.backend == "flax":
                # For JAX/Flax backend (if needed)
                self.model = self.timesfm.TimesFm.from_pretrained(
                    pretrained_model_name_or_path=self.model_name
                )
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            # Compile model with configuration
            print("Compiling model with configuration...")
            self.model.compile(
                self.timesfm.ForecastConfig(
                    max_context=self.max_context,
                    max_horizon=self.max_horizon,
                    normalize_inputs=self.normalize_inputs,
                    use_continuous_quantile_head=self.use_continuous_quantile_head,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )

            self.is_loaded = True
            backend_str = "GPU" if self.backend == "pytorch" and hasattr(self, 'torch') and self.torch.cuda.is_available() else "CPU"
            print(f"Model loaded and compiled successfully on {backend_str}!")

        except Exception as e:
            print(f"Error loading TimesFM model: {e}")
            raise

    def prepare_input(
        self,
        data: np.ndarray,
        context_length: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Prepare input data for TimesFM.

        Args:
            data: Input data (can be 2D or 3D array)
            context_length: Length of context to use (if None, use all data)

        Returns:
            List of numpy arrays (one per time series)
        """
        # If data is 3D (batch, sequence, features), we'll use only the target feature
        if data.ndim == 3:
            # Assume first feature is the target (close price)
            data = data[:, :, 0]

        # If data is 2D (batch, sequence) or 1D (sequence)
        if data.ndim == 2:
            # Each row is a separate time series
            inputs = [row for row in data]
        elif data.ndim == 1:
            # Single time series
            inputs = [data]
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        # Trim to context length if specified
        if context_length is not None:
            inputs = [series[-context_length:] for series in inputs]

        return inputs

    def forecast(
        self,
        inputs: List[np.ndarray],
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make forecasts using TimesFM.

        Args:
            inputs: List of input time series (numpy arrays)
            horizon: Forecast horizon

        Returns:
            Tuple of (point_forecast, quantile_forecast)
        """
        if not self.is_loaded:
            self.load_model()

        if horizon > self.max_horizon:
            print(f"Warning: horizon {horizon} > max_horizon {self.max_horizon}")
            print(f"Using max_horizon {self.max_horizon}")
            horizon = self.max_horizon

        print(f"Generating forecast with horizon {horizon}...")

        # TimesFM forecast API: forecast(horizon, inputs)
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=horizon,
            inputs=inputs
        )

        return point_forecast, quantile_forecast

    def predict(
        self,
        X: np.ndarray,
        horizon: int = 1,
        context_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions (compatibility method for existing pipeline).

        Args:
            X: Input data
            horizon: Forecast horizon
            context_length: Length of context to use

        Returns:
            Point predictions
        """
        inputs = self.prepare_input(X, context_length)
        point_forecast, _ = self.forecast(inputs, horizon)

        # Return only the first step if horizon is 1 (for compatibility)
        if horizon == 1:
            return point_forecast[:, 0:1]
        else:
            return point_forecast

    def predict_with_quantiles(
        self,
        X: np.ndarray,
        horizon: int = 1,
        context_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with quantile forecasts.

        Args:
            X: Input data
            horizon: Forecast horizon
            context_length: Length of context to use

        Returns:
            Dictionary with 'point', 'quantiles', 'lower_bound', 'upper_bound'
        """
        inputs = self.prepare_input(X, context_length)
        point_forecast, quantile_forecast = self.forecast(inputs, horizon)

        # Extract confidence intervals (assume 10th and 90th percentiles)
        # quantile_forecast shape: (batch, horizon, quantiles)
        lower_bound = quantile_forecast[:, :, 0]  # 10th percentile
        upper_bound = quantile_forecast[:, :, -1]  # 90th percentile

        return {
            'point': point_forecast,
            'quantiles': quantile_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        horizon: int = 1,
        context_length: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test input data
            y_test: Test target values
            horizon: Forecast horizon
            context_length: Length of context to use

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_loaded:
            self.load_model()

        # Make predictions
        y_pred = self.predict(X_test, horizon=horizon, context_length=context_length)

        # Flatten predictions if necessary
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Calculate directional accuracy
        if len(y_test) > 1:
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == pred_direction)
        else:
            directional_accuracy = 0.0

        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }

        return metrics

    def save_config(self, filepath: str):
        """
        Save model configuration.

        Args:
            filepath: Path to save configuration
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        config = {
            'model_name': self.model_name,
            'backend': self.backend,
            'max_context': self.max_context,
            'max_horizon': self.max_horizon,
            'normalize_inputs': self.normalize_inputs,
            'use_continuous_quantile_head': self.use_continuous_quantile_head,
            'name': self.name
        }

        with open(filepath, 'wb') as f:
            pickle.dump(config, f)

        print(f"Configuration saved to {filepath}")

    def load_config(self, filepath: str):
        """
        Load model configuration.

        Args:
            filepath: Path to load configuration from
        """
        with open(filepath, 'rb') as f:
            config = pickle.load(f)

        self.model_name = config['model_name']
        self.backend = config['backend']
        self.max_context = config['max_context']
        self.max_horizon = config['max_horizon']
        self.normalize_inputs = config['normalize_inputs']
        self.use_continuous_quantile_head = config['use_continuous_quantile_head']
        self.name = config['name']

        print(f"Configuration loaded from {filepath}")

        # Reload model with new config
        self.is_loaded = False
        self.load_model()


def create_timesfm_model(
    backend: str = "pytorch",
    max_context: int = 1024,
    max_horizon: int = 256,
    **kwargs
) -> TimesFMModel:
    """
    Factory function to create TimesFM model.

    Args:
        backend: Backend to use ('pytorch' or 'flax')
        max_context: Maximum context length
        max_horizon: Maximum forecast horizon
        **kwargs: Additional arguments for TimesFM model

    Returns:
        TimesFM model instance
    """
    return TimesFMModel(
        backend=backend,
        max_context=max_context,
        max_horizon=max_horizon,
        **kwargs
    )


if __name__ == "__main__":
    print("TimesFM Model for BTC Prediction")
    print("This module provides Google's TimesFM foundation model for time series forecasting.")
    print("\nTimesFM is a pre-trained model that can be used directly for forecasting without training.")
    print("It supports up to 16k context length and provides quantile forecasts.")
