"""
shap_service.py
SHAP-based explanation generator for Ethical AI Auditor
Alternative to SAP Joule's explainability features
Implements the 'Explainable AI' component from your SAP Hackfest proposal
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import shap
from sklearn.ensemble import RandomForestClassifier
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class FeatureImportance:
    """Data class to hold feature importance information for role-based insights"""
    def __init__(self, feature: str, importance: float, direction: str):
        self.feature = feature
        self.importance = importance
        self.direction = direction

class SHAPService:
    """
    Service for generating SHAP explanations for AI models.
    
    This implements the 'Explainability Layer' from the Ethical AI Auditor proposal,
    providing role-based model insights as an alternative to SAP Joule.
    
    Key Features:
    - SHAP values for local and global explanations
    - Feature importance ranking
    - Integration-ready with LLM service for natural language explanations
    - SAP BTP-compatible microservice design
    - Full compatibility with SAP AI Core model analysis
    """
    
    # In-memory cache for demo (would use Redis in production)
    _model_cache = {}
    _data_cache = {}
    _encoder_cache = {}

    @staticmethod
    def train_model(
        dataset: List[Dict[str, Any]],
        target_variable: str
    ) -> Tuple[RandomForestClassifier, pd.DataFrame, np.ndarray]:
        """
        Train a lightweight model on the dataset for explanation purposes.
        
        Args:
            dataset: Input dataset as list of dicts
            target_variable: Target variable name (e.g., 'hired')
            
        Returns:
            model, X, y
        """
        try:
            cache_key = f"{len(dataset)}_{target_variable}"
            
            if cache_key in SHAPService._model_cache:
                logger.info("Using cached model for faster response")
                return (
                    SHAPService._model_cache[cache_key],
                    SHAPService._data_cache[cache_key]['X'],
                    SHAPService._data_cache[cache_key]['y']
                )
            
            logger.info(f"Training model for explanation on {len(dataset)} records")
            
            # Convert to DataFrame
            df = pd.DataFrame(dataset)
            
            # Validate target variable
            if target_variable not in df.columns:
                raise ValueError(f"Target variable '{target_variable}' not found in dataset")
            
            # Separate features and target
            X = df.drop(columns=[target_variable])
            y = df[target_variable].astype(int).values  # Ensure y is int array
            
            # Handle categorical features with proper encoding
            X_encoded = X.copy()
            encoders = {}
            
            for col in X_encoded.select_dtypes(include=['object']).columns:
                if col == target_variable:
                    continue
                    
                le = LabelEncoder()
                # Handle missing values before fitting
                non_null_mask = X_encoded[col].notna()
                if non_null_mask.any():
                    le.fit(X_encoded.loc[non_null_mask, col])
                    X_encoded.loc[non_null_mask, col] = le.transform(X_encoded.loc[non_null_mask, col])
                    # Fill NA with 0 after encoding
                    X_encoded[col] = X_encoded[col].fillna(0)
                else:
                    X_encoded[col] = 0
                
                encoders[col] = le
            
            # Ensure all features are float
            X_encoded = X_encoded.astype(float)
            
            # Cache encoders and data
            SHAPService._encoder_cache[cache_key] = encoders
            SHAPService._data_cache[cache_key] = {'X': X_encoded, 'y': y}
            
            # Train lightweight model for fast explanation
            model = RandomForestClassifier(
                n_estimators=10,
                max_depth=6,
                random_state=42
            )
            model.fit(X_encoded, y)
            
            # Cache model
            SHAPService._model_cache[cache_key] = model
            
            # Log model accuracy for debugging
            train_acc = model.score(X_encoded, y)
            logger.info(f"Model trained | Train accuracy: {train_acc:.3f}")
            
            return model, X_encoded, y
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    @staticmethod
    def get_shap_values(
        model: RandomForestClassifier,
        X: pd.DataFrame,
        instance_index: int
    ) -> Dict[str, float]:
        """
        Get SHAP values for a specific instance.
        
        Args:
            model: Trained model
            X: Feature DataFrame (encoded)
            instance_index: Index of the instance to explain
            
        Returns:
            Dictionary of feature: shap_value
        """
        try:
            logger.info(f"Computing SHAP values for instance {instance_index}")
            
            # Validate inputs
            if instance_index >= len(X):
                raise ValueError(f"Instance index {instance_index} out of range for dataset size {len(X)}")
            
            # Create explainer
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X)
            
            # Handle binary classification
            if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                shap_values = np.array(shap_values_raw[1])  # Positive class
            else:
                shap_values = np.array(shap_values_raw)
            
            # Ensure 2D
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Get values for the requested instance
            instance_shap = shap_values[instance_index]
            
            # Convert to dictionary with safe scalar conversion
            shap_dict = {}
            for feature, value in zip(X.columns, instance_shap):
                try:
                    # Handle numpy types safely
                    if isinstance(value, (np.generic, np.ndarray)):
                        # Convert to scalar
                        scalar_value = float(np.array(value).flatten()[0])
                    else:
                        scalar_value = float(value)
                    
                    shap_dict[feature] = scalar_value
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Could not convert SHAP value for {feature}: {e}")
                    shap_dict[feature] = 0.0  # Default fallback
            
            # Debug: Log total SHAP magnitude
            total_impact = sum(abs(v) for v in shap_dict.values())
            if total_impact < 1e-6:
                logger.warning("All SHAP values are near zero — model may not be learning")
            
            return shap_dict
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {str(e)}")
            raise

    @staticmethod
    def get_feature_importance(
        shap_values: Dict[str, float]
    ) -> List[FeatureImportance]:
        """
        Get feature importance from SHAP values.
        
        Args:
            shap_values: Dictionary of feature: shap_value
            
        Returns:
            List of FeatureImportance objects sorted by absolute importance
        """
        try:
            logger.info("Computing feature importance from SHAP values")
            
            features = []
            for feature, value in shap_values.items():
                direction = "positive" if value > 0 else "negative"
                importance = abs(float(value))
                
                features.append(
                    FeatureImportance(feature=feature, importance=importance, direction=direction)
                )
            
            # Sort by importance (descending)
            features.sort(key=lambda x: x.importance, reverse=True)
            
            logger.info(f"Generated {len(features)} feature importance entries")
            return features
            
        except Exception as e:
            logger.error(f"Feature importance computation failed: {str(e)}")
            raise