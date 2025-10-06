import pickle
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ExoplanetPredictor:
    """ML model predictor for exoplanet classification with LightGBM support"""
    
    def __init__(self, model_path: str = "model/model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.is_loaded = False
        self.model_type = None
        
    def load_model(self) -> bool:
        """Load the ML model from pickle file"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Handle different model formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names', [])
            else:
                self.model = model_data
                # Default feature names based on the schema
                self.feature_names = [
                    'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1',
                    'koi_time0bk_err2', 'koi_impact', 'koi_duration', 'koi_duration_err1',
                    'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 'koi_model_snr',
                    'koi_steff_err1', 'koi_srad_err1', 'delivname_q1_q16_tce',
                    'delivname_q1_q17_dr24_tce', 'delivname_q1_q17_dr25_tce'
                ]
            
            # Detect model type
            model_class_name = self.model.__class__.__name__
            if 'LGBM' in model_class_name or 'LightGBM' in model_class_name:
                self.model_type = 'lightgbm'
                logger.info("Detected LightGBM model")
            elif 'Booster' in model_class_name:
                # Native LightGBM Booster
                self.model_type = 'lightgbm'
                logger.info("Detected native LightGBM Booster model")
            elif 'RandomForest' in model_class_name:
                self.model_type = 'random_forest'
                logger.info("Detected Random Forest model")
            else:
                self.model_type = 'generic'
                logger.info(f"Detected {model_class_name} model")
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Make prediction on exoplanet data
        
        Args:
            features: Array of feature values
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Ensure features are in the right shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Make prediction
            if self.model_type == 'lightgbm':
                # For native LightGBM models, predict returns probabilities
                probability = self.model.predict(features)[0]
                prediction = 1 if probability > 0.5 else 0
                confidence = float(probability) if prediction == 1 else float(1 - probability)
            else:
                # For scikit-learn models
                prediction = self.model.predict(features)[0]
                confidence = self._get_confidence_score(features, prediction)
            
            # Convert prediction to string
            if isinstance(prediction, (int, np.integer)):
                prediction_str = "CONFIRMED" if prediction == 1 else "CANDIDATE"
            else:
                prediction_str = str(prediction)
            
            logger.info(f"Prediction: {prediction_str}, Confidence: {confidence:.3f}")
            return prediction_str, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _get_confidence_score(self, features: np.ndarray, prediction: int) -> float:
        """Get confidence score based on model type"""
        try:
            if hasattr(self.model, 'predict_proba'):
                # Most models support predict_proba
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                
                # For LightGBM, sometimes we need to handle binary classification differently
                if self.model_type == 'lightgbm' and len(probabilities) == 2:
                    # For binary classification, use the probability of the predicted class
                    predicted_class_idx = int(prediction)
                    confidence = float(probabilities[predicted_class_idx])
                
                return confidence
                
            elif hasattr(self.model, 'decision_function'):
                # SVM-like models
                decision_scores = self.model.decision_function(features)
                confidence = float(np.abs(decision_scores[0]) / (np.abs(decision_scores[0]) + 1))
                return confidence
                
            else:
                # Fallback confidence
                logger.warning("No probability method available, using default confidence")
                return 0.8
                
        except Exception as e:
            logger.warning(f"Error getting confidence score: {str(e)}, using default")
            return 0.8
    
    def get_feature_names(self) -> list:
        """Get the feature names used by the model"""
        return self.feature_names or []


# Global predictor instance
predictor = ExoplanetPredictor()
