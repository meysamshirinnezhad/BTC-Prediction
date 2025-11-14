"""
GRU model for Bitcoin price prediction.

This module implements a Gated Recurrent Unit (GRU) neural network
for time series forecasting of Bitcoin prices.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from typing import Tuple, Optional, Dict
import os


class GRUModel:
    """GRU model for time series prediction."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        gru_units: list = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        name: str = "gru_model"
    ):
        """
        Initialize GRU model.

        Args:
            input_shape: Shape of input (sequence_length, n_features)
            gru_units: List of units for each GRU layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            name: Model name
        """
        self.input_shape = input_shape
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.name = name
        self.model = None
        self.history = None

    def build_model(self) -> Model:
        """
        Build the GRU model architecture.

        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape, name='input')

        x = inputs

        # Stack GRU layers
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1

            x = layers.GRU(
                units,
                return_sequences=return_sequences,
                name=f'gru_{i+1}'
            )(x)

            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

        # Dense layers for output
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_final')(x)

        # Output layer
        outputs = layers.Dense(1, name='output')(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.name)

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )

        self.model = model

        return model

    def get_model_summary(self) -> str:
        """
        Get model summary.

        Returns:
            Model summary as string
        """
        if self.model is None:
            self.build_model()

        return self.model.summary()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        callbacks_list: Optional[list] = None
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training input sequences
            y_train: Training targets
            X_val: Validation input sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            callbacks_list: List of Keras callbacks

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        if callbacks_list is None:
            callbacks_list = self._get_default_callbacks()

        print(f"\nTraining {self.name}...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Input shape: {X_train.shape}")

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks_list
        )

        self.history = history

        return history

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input sequences
            batch_size: Batch size for prediction

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test input sequences
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        # Get raw metrics from Keras
        results = self.model.evaluate(X_test, y_test, verbose=0)

        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mape': results[2]
        }

        # Make predictions for additional metrics
        y_pred = self.predict(X_test)

        # Calculate RMSE
        metrics['rmse'] = np.sqrt(np.mean((y_test - y_pred.flatten()) ** 2))

        # Calculate directional accuracy
        if len(y_test) > 1:
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(y_pred.flatten()))
            metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction)
        else:
            metrics['directional_accuracy'] = 0.0

        return metrics

    def save_model(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from file.

        Args:
            filepath: Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def _get_default_callbacks(self) -> list:
        """
        Get default callbacks for training.

        Returns:
            List of Keras callbacks
        """
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks_list


class BidirectionalGRUModel(GRUModel):
    """Bidirectional GRU model for time series prediction."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        gru_units: list = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        name: str = "bidirectional_gru_model"
    ):
        """
        Initialize Bidirectional GRU model.

        Args:
            input_shape: Shape of input (sequence_length, n_features)
            gru_units: List of units for each GRU layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            name: Model name
        """
        super().__init__(input_shape, gru_units, dropout_rate, learning_rate, name)

    def build_model(self) -> Model:
        """
        Build the Bidirectional GRU model architecture.

        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape, name='input')

        x = inputs

        # Stack Bidirectional GRU layers
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1

            x = layers.Bidirectional(
                layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    name=f'gru_{i+1}'
                ),
                name=f'bidirectional_{i+1}'
            )(x)

            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

        # Dense layers for output
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_final')(x)

        # Output layer
        outputs = layers.Dense(1, name='output')(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.name)

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )

        self.model = model

        return model


def create_gru_model(
    sequence_length: int,
    n_features: int,
    model_type: str = 'gru',
    **kwargs
) -> GRUModel:
    """
    Factory function to create GRU models.

    Args:
        sequence_length: Length of input sequences
        n_features: Number of features
        model_type: Type of model ('gru' or 'bidirectional')
        **kwargs: Additional arguments for model

    Returns:
        GRU model instance
    """
    input_shape = (sequence_length, n_features)

    if model_type == 'gru':
        return GRUModel(input_shape, **kwargs)
    elif model_type == 'bidirectional':
        return BidirectionalGRUModel(input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("GRU Model for BTC Prediction")
    print("This module provides GRU-based models for time series forecasting.")
