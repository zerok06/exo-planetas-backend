"""
Script to create a sample ML model for demonstration purposes.
This creates a simple Random Forest classifier trained on synthetic data.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=1000):
    """Generate synthetic exoplanet data for training"""
    np.random.seed(42)
    
    # Generate synthetic features based on typical Kepler data ranges
    data = {
        'koi_period_err1': np.random.exponential(0.1, n_samples),
        'koi_period_err2': np.random.exponential(0.1, n_samples),
        'koi_time0bk': np.random.uniform(100, 2000, n_samples),
        'koi_time0bk_err1': np.random.exponential(0.5, n_samples),
        'koi_time0bk_err2': np.random.exponential(0.5, n_samples),
        'koi_impact': np.random.uniform(0, 1, n_samples),
        'koi_duration': np.random.uniform(1, 20, n_samples),
        'koi_duration_err1': np.random.exponential(0.1, n_samples),
        'koi_duration_err2': np.random.exponential(0.1, n_samples),
        'koi_depth': np.random.uniform(0.001, 0.1, n_samples),
        'koi_depth_err1': np.random.exponential(0.001, n_samples),
        'koi_model_snr': np.random.uniform(5, 50, n_samples),
        'koi_steff_err1': np.random.exponential(50, n_samples),
        'koi_srad_err1': np.random.exponential(0.1, n_samples),
        'delivname_q1_q16_tce': np.random.randint(0, 1000, n_samples),
        'delivname_q1_q17_dr24_tce': np.random.randint(0, 1000, n_samples),
        'delivname_q1_q17_dr25_tce': np.random.randint(0, 1000, n_samples)
    }
    
    # Create feature matrix
    X = np.column_stack([data[col] for col in data.keys()])
    
    # Generate labels based on some logical rules
    # Higher SNR and lower errors tend to be CONFIRMED
    confirmed_prob = (
        (data['koi_model_snr'] > 20) * 0.7 +
        (data['koi_depth_err1'] < 0.01) * 0.3 +
        (data['koi_duration_err1'] < 0.5) * 0.2 +
        np.random.random(n_samples) * 0.1
    )
    
    y = (confirmed_prob > 0.5).astype(int)
    
    # Convert to string labels
    y_labels = ['CANDIDATE' if label == 0 else 'CONFIRMED' for label in y]
    
    return X, y_labels, list(data.keys())


def create_and_save_model():
    """Create and save the sample ML model"""
    logger.info("Generating synthetic training data...")
    X, y, feature_names = generate_synthetic_data(2000)
    
    logger.info("Training Random Forest classifier...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    logger.info("Model performance:")
    logger.info(classification_report(y_test, y_pred))
    
    # Save the model
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info("✅ Sample model saved to model/model.pkl")
    return model


if __name__ == "__main__":
    create_and_save_model()

