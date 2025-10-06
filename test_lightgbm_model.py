"""
Script to test the LightGBM model loading and prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.predict import predictor
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the LightGBM model loads correctly"""
    print("🔍 Testing LightGBM model loading...")
    
    success = predictor.load_model()
    
    if success:
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: {predictor.model_type}")
        print(f"🔧 Feature names: {len(predictor.get_feature_names())} features")
        return True
    else:
        print("❌ Model loading failed!")
        return False

def test_prediction():
    """Test model prediction with sample data"""
    print("\n🧪 Testing model prediction...")
    
    # Create sample feature array (17 features as expected by the model)
    sample_features = np.array([
        0.1,  # koi_period_err1
        0.1,  # koi_period_err2
        1000.0,  # koi_time0bk
        0.5,  # koi_time0bk_err1
        0.5,  # koi_time0bk_err2
        0.5,  # koi_impact
        10.0,  # koi_duration
        0.1,  # koi_duration_err1
        0.1,  # koi_duration_err2
        0.01,  # koi_depth
        0.001,  # koi_depth_err1
        20.0,  # koi_model_snr
        50.0,  # koi_steff_err1
        0.1,  # koi_srad_err1
        100.0,  # delivname_q1_q16_tce (encoded)
        200.0,  # delivname_q1_q17_dr24_tce (encoded)
        300.0   # delivname_q1_q17_dr25_tce (encoded)
    ])
    
    try:
        prediction, confidence = predictor.predict(sample_features)
        print(f"✅ Prediction successful!")
        print(f"🎯 Prediction: {prediction}")
        print(f"📈 Confidence: {confidence:.3f}")
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return False

def test_multiple_predictions():
    """Test multiple predictions with different data"""
    print("\n🔄 Testing multiple predictions...")
    
    # Generate multiple test cases
    test_cases = []
    for i in range(5):
        # Generate random but realistic values
        features = np.array([
            np.random.exponential(0.1),  # koi_period_err1
            np.random.exponential(0.1),  # koi_period_err2
            np.random.uniform(100, 2000),  # koi_time0bk
            np.random.exponential(0.5),  # koi_time0bk_err1
            np.random.exponential(0.5),  # koi_time0bk_err2
            np.random.uniform(0, 1),  # koi_impact
            np.random.uniform(1, 20),  # koi_duration
            np.random.exponential(0.1),  # koi_duration_err1
            np.random.exponential(0.1),  # koi_duration_err2
            np.random.uniform(0.001, 0.1),  # koi_depth
            np.random.exponential(0.001),  # koi_depth_err1
            np.random.uniform(5, 50),  # koi_model_snr
            np.random.exponential(50),  # koi_steff_err1
            np.random.exponential(0.1),  # koi_srad_err1
            np.random.randint(0, 1000),  # delivname_q1_q16_tce
            np.random.randint(0, 1000),  # delivname_q1_q17_dr24_tce
            np.random.randint(0, 1000)   # delivname_q1_q17_dr25_tce
        ])
        test_cases.append(features)
    
    success_count = 0
    for i, features in enumerate(test_cases):
        try:
            prediction, confidence = predictor.predict(features)
            print(f"  Test {i+1}: {prediction} (confidence: {confidence:.3f})")
            success_count += 1
        except Exception as e:
            print(f"  Test {i+1}: Failed - {str(e)}")
    
    print(f"✅ {success_count}/{len(test_cases)} predictions successful")
    return success_count == len(test_cases)

def main():
    """Run all tests"""
    print("🚀 LightGBM Model Testing Suite")
    print("=" * 50)
    
    # Test 1: Model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed. Exiting.")
        return False
    
    # Test 2: Single prediction
    if not test_prediction():
        print("\n❌ Single prediction test failed.")
        return False
    
    # Test 3: Multiple predictions
    if not test_multiple_predictions():
        print("\n❌ Multiple predictions test failed.")
        return False
    
    print("\n🎉 All tests passed! LightGBM model is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

