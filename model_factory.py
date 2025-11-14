"""
ML Model Factory - Support Multiple Model Types
===============================================

Allows users to choose between:
- GradientBoosting (default)
- RandomForest (more robust)
- XGBoost (best performance)
- Ensemble (combines all three)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


class ModelFactory:
    """Factory for creating different ML models"""

    @staticmethod
    def create_model(model_type: str = 'gradientboost', random_state: int = 42):
        """
        Create ML model based on type

        Args:
            model_type: One of 'gradientboost', 'randomforest', 'xgboost', 'ensemble'
            random_state: Random seed for reproducibility

        Returns:
            Trained model instance
        """
        model_type = model_type.lower()

        if model_type == 'gradientboost':
            return GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                max_features='sqrt',
                random_state=random_state
            )

        elif model_type == 'randomforest':
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight='balanced',  # Handle imbalanced data
                n_jobs=-1,  # Use all CPU cores
                random_state=random_state
            )

        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                print("XGBoost not available, falling back to GradientBoosting")
                return ModelFactory.create_model('gradientboost', random_state)

            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.2,  # Handle class imbalance
                gamma=0.1,  # Regularization
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=random_state,
                n_jobs=-1
            )

        elif model_type == 'ensemble':
            # Combine all three for maximum accuracy
            models = []

            # Add GradientBoosting
            models.append(('gb', ModelFactory.create_model('gradientboost', random_state)))

            # Add RandomForest
            models.append(('rf', ModelFactory.create_model('randomforest', random_state)))

            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                models.append(('xgb', ModelFactory.create_model('xgboost', random_state)))
                weights = [1, 1, 2]  # XGBoost gets 2x weight
            else:
                weights = [1, 1]

            return VotingClassifier(
                estimators=models,
                voting='soft',  # Use probabilities
                weights=weights
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: gradientboost, randomforest, xgboost, ensemble")

    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, str]:
        """Get information about a model type"""
        info = {
            'gradientboost': {
                'name': 'Gradient Boosting',
                'speed': 'Fast',
                'accuracy': 'Good',
                'overfitting_risk': 'Medium',
                'description': 'Default choice. Fast training, good performance.'
            },
            'randomforest': {
                'name': 'Random Forest',
                'speed': 'Medium',
                'accuracy': 'Good',
                'overfitting_risk': 'Low',
                'description': 'Most robust. Less prone to overfitting.'
            },
            'xgboost': {
                'name': 'XGBoost',
                'speed': 'Medium',
                'accuracy': 'Excellent',
                'overfitting_risk': 'Low-Medium',
                'description': 'Best performance. Built-in regularization.'
            },
            'ensemble': {
                'name': 'Ensemble (GB + RF + XGB)',
                'speed': 'Slow',
                'accuracy': 'Excellent',
                'overfitting_risk': 'Very Low',
                'description': 'Combines all models. Most accurate but slowest.'
            }
        }

        return info.get(model_type.lower(), {'name': 'Unknown', 'description': 'Unknown model type'})


class EnhancedMLModel:
    """Enhanced ML model with confidence thresholding and better predictions"""

    def __init__(self, model_type: str = 'gradientboost', min_confidence: float = 0.65):
        """
        Initialize enhanced ML model

        Args:
            model_type: Type of model to use
            min_confidence: Minimum confidence threshold for trading (0.0-1.0)
        """
        self.model_type = model_type
        self.min_confidence = min_confidence
        self.model = ModelFactory.create_model(model_type)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the model

        Args:
            X: Feature DataFrame
            y: Target variable (forward returns)

        Returns:
            Training metrics
        """
        # Store feature names
        self.feature_names = list(X.columns)

        # Create binary target (success = positive return)
        y_binary = (y > 0).astype(int)

        # Handle NaNs
        X_clean = X.fillna(0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)

        # Train model
        self.model.fit(X_scaled, y_binary)
        self.is_trained = True

        # Calculate training accuracy
        train_pred = self.model.predict(X_scaled)
        accuracy = (train_pred == y_binary).mean()

        return {
            'accuracy': accuracy,
            'model_type': self.model_type,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }

    def predict_with_confidence(self, features: pd.Series) -> Dict:
        """
        Predict trade quality with confidence scoring

        Args:
            features: Feature Series for current market state

        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            return {
                'should_trade': False,
                'reason': 'Model not trained yet',
                'confidence_score': 0.0,
                'success_probability': 0.5
            }

        try:
            # Prepare features
            X = features[self.feature_names].values.reshape(1, -1)
            X = np.nan_to_num(X, nan=0.0)

            # Scale
            X_scaled = self.scaler.transform(X)

            # Get prediction probabilities
            proba = self.model.predict_proba(X_scaled)[0]
            success_prob = proba[1]  # Probability of success (class 1)

            # Confidence score (how sure the model is)
            confidence = max(proba)

            # Decision logic
            should_trade = (
                success_prob >= 0.60 and  # At least 60% predicted success
                confidence >= self.min_confidence  # At least minimum confidence
            )

            reason = "Trade approved"
            if not should_trade:
                if success_prob < 0.60:
                    reason = f"Success probability too low: {success_prob:.1%}"
                elif confidence < self.min_confidence:
                    reason = f"Confidence too low: {confidence:.1%} < {self.min_confidence:.1%}"

            return {
                'should_trade': should_trade,
                'reason': reason,
                'confidence_score': confidence,
                'success_probability': success_prob
            }

        except Exception as e:
            return {
                'should_trade': False,
                'reason': f'Prediction error: {str(e)}',
                'confidence_score': 0.0,
                'success_probability': 0.5
            }

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            return pd.DataFrame()

        # Get importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            # Ensemble - average importance across estimators
            importances = []
            for name, estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            importance = np.mean(importances, axis=0)
        else:
            return pd.DataFrame()

        # Create DataFrame
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return feature_imp.head(top_n)
