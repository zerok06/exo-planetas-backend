"""
Simple test to verify the LightGBM model file exists and can be loaded
"""

import pickle
import os

def test_model_file():
    """Test if the model file exists and can be loaded"""
    model_path = "model/model.pkl"
    
    print("Testing LightGBM model file...")
    print(f"Model path: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("ERROR: Model file not found!")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("SUCCESS: Model file loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model class: {model.__class__.__name__}")
        
        # Check if it's a LightGBM model
        if 'LGBM' in model.__class__.__name__:
            print("SUCCESS: LightGBM model detected!")
        else:
            print(f"WARNING: Unexpected model type: {model.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_file()
    if success:
        print("\nSUCCESS: Model file is ready for use!")
    else:
        print("\nERROR: Model file has issues.")
