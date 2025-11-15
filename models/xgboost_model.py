"""
XGBoost model for Bitcoin price prediction.

This module implements an XGBoost regressor for time series forecasting
of Bitcoin prices.
"""

import numpy as np
import xgboost as xgb
from xgboost.callback import EarlyStopping
from typing import Dict, Optional
import joblib
import os


class XGBoostModel:
    """XGBoost model for time series prediction."""

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 7,
        learning_rate: float = 0.01,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        name: str = "xgboost_model"
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            gamma: Minimum loss reduction required to make a further partition
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed
            name: Model name
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.name = name
        self.model = None
        self.feature_importance = None

    def build_model(self) -> xgb.XGBRegressor:
        """
        Build the XGBoost model.

        Returns:
            XGBoost regressor
        """
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            tree_method='hist',
            objective='reg:squarederror'
        )

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Early stopping patience
            verbose: Whether to print training progress
        """
        if self.model is None:
            self.build_model()

        print(f"\nTraining {self.name}...")
        print(f"Training samples: {len(X_train)}")

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            print(f"Validation samples: {len(X_val)}")

        # Create callbacks for early stopping (XGBoost 2.0+ API)
        callbacks_list = []
        if early_stopping_rounds is not None:
            callbacks_list.append(EarlyStopping(rounds=early_stopping_rounds, save_best=True))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks_list if callbacks_list else None,
            verbose=verbose
        )

        # Store feature importance
        self.feature_importance = self.model.feature_importances_

        print(f"Training complete! Best iteration: {self.model.best_iteration}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        # Make predictions
        y_pred = self.predict(X_test)

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

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary of feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]

        importance_dict = dict(zip(feature_names, self.feature_importance))

        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return importance_dict

    def save_model(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save using joblib
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'gamma': self.gamma,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'random_state': self.random_state
            }
        }, filepath)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from file.

        Args:
            filepath: Path to load model from
        """
        data = joblib.load(filepath)

        self.model = data['model']
        self.feature_importance = data['feature_importance']

        params = data['params']
        self.n_estimators = params['n_estimators']
        self.max_depth = params['max_depth']
        self.learning_rate = params['learning_rate']
        self.subsample = params['subsample']
        self.colsample_bytree = params['colsample_bytree']
        self.gamma = params['gamma']
        self.reg_alpha = params['reg_alpha']
        self.reg_lambda = params['reg_lambda']
        self.random_state = params['random_state']

        print(f"Model loaded from {filepath}")


class LightGBMModel:
    """LightGBM model for time series prediction (alternative to XGBoost)."""

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 7,
        learning_rate: float = 0.01,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        name: str = "lightgbm_model"
    ):
        """
        Initialize LightGBM model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            num_leaves: Maximum number of leaves in one tree
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed
            name: Model name
        """
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.name = name
        self.model = None
        self.feature_importance = None

    def build_model(self):
        """Build the LightGBM model."""
        self.model = self.lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            objective='regression',
            verbose=-1
        )

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """Train the model."""
        if self.model is None:
            self.build_model()

        print(f"\nTraining {self.name}...")
        print(f"Training samples: {len(X_train)}")

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            print(f"Validation samples: {len(X_val)}")

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[self.lgb.early_stopping(early_stopping_rounds, verbose=verbose)]
        )

        self.feature_importance = self.model.feature_importances_

        print(f"Training complete!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built or loaded")

        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError("Model not built or loaded")

        y_pred = self.predict(X_test)

        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        if len(y_test) > 1:
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == pred_direction)
        else:
            directional_accuracy = 0.0

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }


def create_xgboost_model(**kwargs) -> XGBoostModel:
    """
    Factory function to create XGBoost model.

    Args:
        **kwargs: Arguments for XGBoost model

    Returns:
        XGBoost model instance
    """
    return XGBoostModel(**kwargs)


if __name__ == "__main__":
    print("XGBoost Model for BTC Prediction")
    print("This module provides XGBoost-based models for time series forecasting.")
