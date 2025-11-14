"""
Unit tests for BTC prediction models.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import create_lstm_model
from models.gru_model import create_gru_model
from models.xgboost_model import create_xgboost_model


class TestLSTMModel(unittest.TestCase):
    """Test LSTM model."""

    def setUp(self):
        """Set up test data."""
        self.sequence_length = 10
        self.n_features = 5
        self.n_samples = 100

        # Create dummy data
        self.X_train = np.random.randn(self.n_samples, self.sequence_length, self.n_features)
        self.y_train = np.random.randn(self.n_samples)
        self.X_val = np.random.randn(20, self.sequence_length, self.n_features)
        self.y_val = np.random.randn(20)

    def test_model_creation(self):
        """Test model creation."""
        model = create_lstm_model(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            lstm_units=[32, 16],
            dropout_rate=0.2
        )

        self.assertIsNotNone(model)
        self.assertIsNone(model.model)  # Not built yet

    def test_model_build(self):
        """Test model building."""
        model = create_lstm_model(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            lstm_units=[32, 16]
        )

        keras_model = model.build_model()
        self.assertIsNotNone(keras_model)
        self.assertIsNotNone(model.model)

    def test_model_prediction_shape(self):
        """Test prediction output shape."""
        model = create_lstm_model(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            lstm_units=[32, 16]
        )

        model.build_model()

        # Make prediction
        predictions = model.predict(self.X_val)

        self.assertEqual(predictions.shape, (len(self.X_val), 1))


class TestGRUModel(unittest.TestCase):
    """Test GRU model."""

    def setUp(self):
        """Set up test data."""
        self.sequence_length = 10
        self.n_features = 5
        self.n_samples = 100

        self.X_train = np.random.randn(self.n_samples, self.sequence_length, self.n_features)
        self.y_train = np.random.randn(self.n_samples)

    def test_model_creation(self):
        """Test model creation."""
        model = create_gru_model(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            gru_units=[32, 16]
        )

        self.assertIsNotNone(model)

    def test_model_build(self):
        """Test model building."""
        model = create_gru_model(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            gru_units=[32, 16]
        )

        keras_model = model.build_model()
        self.assertIsNotNone(keras_model)


class TestXGBoostModel(unittest.TestCase):
    """Test XGBoost model."""

    def setUp(self):
        """Set up test data."""
        self.n_samples = 100
        self.n_features = 10

        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.randn(self.n_samples)
        self.X_val = np.random.randn(20, self.n_features)
        self.y_val = np.random.randn(20)

    def test_model_creation(self):
        """Test model creation."""
        model = create_xgboost_model(
            n_estimators=10,
            max_depth=3
        )

        self.assertIsNotNone(model)

    def test_model_build(self):
        """Test model building."""
        model = create_xgboost_model(n_estimators=10)
        xgb_model = model.build_model()

        self.assertIsNotNone(xgb_model)

    def test_model_training(self):
        """Test model training."""
        model = create_xgboost_model(n_estimators=10)

        # Train model
        model.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            verbose=False
        )

        self.assertIsNotNone(model.model)

    def test_model_prediction(self):
        """Test model prediction."""
        model = create_xgboost_model(n_estimators=10)

        model.train(
            self.X_train,
            self.y_train,
            verbose=False
        )

        predictions = model.predict(self.X_val)

        self.assertEqual(len(predictions), len(self.X_val))


if __name__ == '__main__':
    unittest.main()
