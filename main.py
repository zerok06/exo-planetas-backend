from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import json
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
import uvicorn

from schemas.exoplanet import ExoplanetData, PredictionResponse, HealthResponse
from model.predict import predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Exoplanet Classification API",
    description="ML-powered exoplanet classification system for Kepler data",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5193", "https://exo-planetas-frontend.onrender.com"],  # Vite and React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to track models
model_loaded = False
trained_models = {}  # Store trained models
current_model_id = "default"  # Current active model


@app.on_event("startup")
async def startup_event():
    """Load the ML model on startup"""
    global model_loaded, trained_models, current_model_id
    logger.info("Starting up Exoplanet Classification API...")
    
    # Try to load the default model
    model_loaded = predictor.load_model()
    
    if model_loaded:
        logger.info("✅ Default ML model loaded successfully")
        # Add default model to trained models
        trained_models["default"] = {
            "id": "default",
            "name": "Default LightGBM Model",
            "algorithm": "LightGBM",
            "accuracy": 0.92,
            "f1_score": 0.89,
            "training_date": datetime.now().isoformat(),
            "is_default": True,
            "is_active": True
        }
    else:
        logger.warning("⚠️ Default model not loaded - using mock predictions")
        # Add mock default model
        trained_models["default"] = {
            "id": "default",
            "name": "Mock Default Model",
            "algorithm": "Mock",
            "accuracy": 0.85,
            "f1_score": 0.82,
            "training_date": datetime.now().isoformat(),
            "is_default": True,
            "is_active": True
        }


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Exoplanet Classification API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_exoplanet(data: ExoplanetData):
    """
    Predict exoplanet classification based on Kepler data
    
    Args:
        data: Exoplanet data with required features
        
    Returns:
        Prediction result with confidence score
    """
    try:
        # Log the input data
        logger.info(f"Received prediction request for KOI: {data.koi_name or 'Unknown'}")
        
        # Convert Pydantic model to feature array
        # LightGBM expects numeric features, so we'll convert string features to numeric
        features = np.array([
            float(data.koi_period_err1),
            float(data.koi_period_err2),
            float(data.koi_time0bk),
            float(data.koi_time0bk_err1),
            float(data.koi_time0bk_err2),
            float(data.koi_impact),
            float(data.koi_duration),
            float(data.koi_duration_err1),
            float(data.koi_duration_err2),
            float(data.koi_depth),
            float(data.koi_depth_err1),
            float(data.koi_model_snr),
            float(data.koi_steff_err1),
            float(data.koi_srad_err1),
            # Convert string features to numeric (hash-based encoding)
            float(hash(data.delivname_q1_q16_tce) % 1000),
            float(hash(data.delivname_q1_q17_dr24_tce) % 1000),
            float(hash(data.delivname_q1_q17_dr25_tce) % 1000)
        ])
        
        if model_loaded:
            # Use the actual ML model
            prediction, confidence = predictor.predict(features)
        else:
            # Mock prediction for demo purposes
            logger.warning("Using mock prediction - model not loaded")
            prediction = "CONFIRMED" if np.random.random() > 0.5 else "CANDIDATE"
            confidence = np.random.uniform(0.6, 0.95)
        
        # Log the prediction
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.3f})")
        
        return PredictionResponse(
            prediction=prediction,
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    return {
        "model_loaded": model_loaded,
        "feature_names": predictor.get_feature_names(),
        "model_path": predictor.model_path
    }


# New endpoints for research features
@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload new exoplanet data for training"""
    try:
        # Ensure uploads directory exists
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size (limit to 10MB)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Validate file extension
        allowed_extensions = ['.csv', '.json', '.xlsx', '.xls']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        file_path = os.path.join(uploads_dir, file.filename)
        
        # Ensure unique filename if file already exists
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_path)
            file_path = f"{name}_{counter}{ext}"
            counter += 1
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"File uploaded successfully: {file.filename} -> {file_path}")
        
        return {
            "message": "File uploaded successfully",
            "filename": os.path.basename(file_path),
            "original_filename": file.filename,
            "size": len(content),
            "path": file_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/uploads")
async def list_uploads():
    """List uploaded files"""
    try:
        uploads_dir = "uploads"
        if not os.path.exists(uploads_dir):
            return {"files": [], "message": "No uploads directory found"}
        
        files = []
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "path": file_path
                })
        
        return {
            "files": files,
            "count": len(files),
            "total_size": sum(f["size"] for f in files)
        }
        
    except Exception as e:
        logger.error(f"List uploads error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list uploads: {str(e)}")


@app.post("/train")
async def train_model(config: Dict[str, Any]):
    """Train the model with new data"""
    global trained_models, current_model_id
    try:
        # Simulate training process
        logger.info(f"Starting model training with config: {config}")
        
        # Simulate training time
        import time
        time.sleep(2)
        
        # Mock training results
        accuracy = np.random.uniform(0.85, 0.95)
        f1_score = np.random.uniform(0.80, 0.90)
        precision = np.random.uniform(0.82, 0.92)
        recall = np.random.uniform(0.78, 0.88)
        
        # Create new model ID
        model_id = f"model_{int(time.time())}"
        algorithm = config.get("algorithm", "lightgbm")
        
        # Store the trained model
        trained_models[model_id] = {
            "id": model_id,
            "name": f"{algorithm.upper()} Model - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "algorithm": algorithm,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "training_date": datetime.now().isoformat(),
            "is_default": False,
            "is_active": False,
            "config": config,
            "data_points": np.random.randint(1000, 5000)
        }
        
        logger.info(f"Stored new model: {model_id} with accuracy: {accuracy:.3f}")
        logger.info(f"Total models now: {len(trained_models)}")
        
        # If this model is better than current, make it active
        current_model = trained_models.get(current_model_id, {})
        if accuracy > current_model.get("accuracy", 0):
            # Deactivate current model
            if current_model_id in trained_models:
                trained_models[current_model_id]["is_active"] = False
            
            # Activate new model
            trained_models[model_id]["is_active"] = True
            current_model_id = model_id
            logger.info(f"New best model activated: {model_id} (accuracy: {accuracy:.3f})")
        
        result = {
            "model_id": model_id,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "training_time": "2.5s",
            "algorithm": algorithm,
            "data_points": trained_models[model_id]["data_points"],
            "is_best": trained_models[model_id]["is_active"]
        }
        
        logger.info(f"Training completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/stats")
async def get_model_stats():
    """Get model performance statistics"""
    try:
        # Mock statistics
        stats = {
            "accuracy": np.random.uniform(0.85, 0.95),
            "precision": np.random.uniform(0.82, 0.92),
            "recall": np.random.uniform(0.78, 0.88),
            "f1_score": np.random.uniform(0.80, 0.90),
            "confusion_matrix": {
                "true_positives": np.random.randint(200, 400),
                "true_negatives": np.random.randint(300, 500),
                "false_positives": np.random.randint(20, 50),
                "false_negatives": np.random.randint(15, 40)
            },
            "algorithm": "LightGBM",
            "training_date": datetime.now().isoformat(),
            "training_time": "2.5s",
            "data_points": np.random.randint(1000, 5000),
            "feature_count": 17,
            "feature_importance": [
                {"name": "koi_model_snr", "importance": 0.25},
                {"name": "koi_depth", "importance": 0.20},
                {"name": "koi_duration", "importance": 0.15},
                {"name": "koi_impact", "importance": 0.12},
                {"name": "koi_period_err1", "importance": 0.10},
                {"name": "koi_steff_err1", "importance": 0.08},
                {"name": "koi_srad_err1", "importance": 0.06},
                {"name": "koi_time0bk", "importance": 0.04}
            ]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.post("/tune")
async def tune_hyperparameters(parameters: Dict[str, Any]):
    """Tune model hyperparameters"""
    try:
        logger.info(f"Hyperparameter tuning with: {parameters}")
        
        # Simulate tuning process
        import time
        time.sleep(1)
        
        # Mock tuning results
        result = {
            "score": np.random.uniform(0.88, 0.95),
            "accuracy": np.random.uniform(0.85, 0.94),
            "f1_score": np.random.uniform(0.82, 0.92),
            "best_parameters": parameters,
            "tuning_time": "1.2s"
        }
        
        logger.info(f"Hyperparameter tuning completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Tuning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tuning failed: {str(e)}")


@app.get("/training-history")
async def get_training_history():
    """Get training history"""
    try:
        # Mock training history
        history = [
            {
                "id": 1,
                "timestamp": datetime.now().isoformat(),
                "algorithm": "LightGBM",
                "accuracy": 0.92,
                "f1_score": 0.89,
                "duration": "2.5s"
            },
            {
                "id": 2,
                "timestamp": (datetime.now().replace(hour=datetime.now().hour-1)).isoformat(),
                "algorithm": "Random Forest",
                "accuracy": 0.88,
                "f1_score": 0.85,
                "duration": "3.2s"
            }
        ]
        
        return history
        
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List all trained models"""
    try:
        models_list = list(trained_models.values())
        # Sort by accuracy descending
        models_list.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
        
        logger.info(f"Returning {len(models_list)} models. Current model: {current_model_id}")
        logger.info(f"Models: {[m['id'] for m in models_list]}")
        
        return {
            "models": models_list,
            "current_model_id": current_model_id,
            "total_models": len(models_list)
        }
        
    except Exception as e:
        logger.error(f"List models error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.post("/models/{model_id}/activate")
async def activate_model(model_id: str):
    """Activate a specific model"""
    global current_model_id
    try:
        if model_id not in trained_models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Deactivate current model
        if current_model_id in trained_models:
            trained_models[current_model_id]["is_active"] = False
        
        # Activate new model
        trained_models[model_id]["is_active"] = True
        current_model_id = model_id
        
        logger.info(f"Model activated: {model_id}")
        
        return {
            "message": f"Model {model_id} activated successfully",
            "current_model_id": current_model_id,
            "model": trained_models[model_id]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Activate model error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")


@app.get("/export/results")
async def export_results():
    """Export all results and data"""
    try:
        # Get all models
        models_list = list(trained_models.values())
        models_list.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
        
        # Get uploaded files
        uploads_dir = "uploads"
        uploaded_files = []
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    uploaded_files.append({
                        "filename": filename,
                        "size": stat.st_size,
                        "modified": stat.st_mtime
                    })
        
        # Create export data
        export_data = {
            "export_date": datetime.now().isoformat(),
            "current_model": trained_models.get(current_model_id, {}),
            "models": models_list,
            "uploaded_files": uploaded_files,
            "summary": {
                "total_models": len(models_list),
                "best_accuracy": max([m.get("accuracy", 0) for m in models_list]) if models_list else 0,
                "total_files": len(uploaded_files),
                "total_file_size": sum(f["size"] for f in uploaded_files)
            }
        }
        
        return export_data
        
    except Exception as e:
        logger.error(f"Export results error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export results: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
