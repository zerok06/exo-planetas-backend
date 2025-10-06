"""
Create a working LightGBM model for testing
"""

import lightgbm as lgb
import numpy as np
import pickle
import os

def create_lightgbm_model():
    """Create a simple LightGBM model for testing"""
    print("Creating LightGBM model...")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 17
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (binary classification)
    # Higher values in certain features lead to class 1 (CONFIRMED)
    y = (X[:, 5] > 0.2) & (X[:, 11] > 0.1)  # Based on koi_impact and koi_model_snr
    y = y.astype(int)
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X, label=y)
    
    # Set parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Train the model
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # Save the model
    os.makedirs('model', exist_ok=True)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("SUCCESS: LightGBM model created and saved!")
    
    # Test the model
    test_data = np.random.randn(1, n_features)
    prediction = model.predict(test_data)
    print(f"Test prediction: {prediction[0]:.3f}")
    
    return True

if __name__ == "__main__":
    create_lightgbm_model()

