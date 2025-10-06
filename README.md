# Exoplanet Classification Backend

FastAPI backend for ML-powered exoplanet classification using Kepler data.

## Features

- **ML Model Integration**: Loads and uses a trained scikit-learn model for exoplanet classification
- **RESTful API**: FastAPI endpoints for prediction and health checks
- **CORS Support**: Configured for frontend communication
- **Data Validation**: Pydantic schemas for input validation
- **Logging**: Comprehensive logging for debugging and monitoring

## API Endpoints

### `GET /`
Root endpoint with basic API information.

### `GET /health`
Health check endpoint that returns service status and model loading status.

### `POST /predict`
Main prediction endpoint that accepts exoplanet data and returns classification.

**Request Body:**
```json
{
  "koi_period_err1": 0.1,
  "koi_period_err2": 0.1,
  "koi_time0bk": 1000.0,
  "koi_time0bk_err1": 0.5,
  "koi_time0bk_err2": 0.5,
  "koi_impact": 0.5,
  "koi_duration": 10.0,
  "koi_duration_err1": 0.1,
  "koi_duration_err2": 0.1,
  "koi_depth": 0.01,
  "koi_depth_err1": 0.001,
  "koi_model_snr": 20.0,
  "koi_steff_err1": 50.0,
  "koi_srad_err1": 0.1,
  "delivname_q1_q16_tce": "Q1-Q16",
  "delivname_q1_q17_dr24_tce": "Q1-Q17-DR24",
  "delivname_q1_q17_dr25_tce": "Q1-Q17-DR25"
}
```

**Response:**
```json
{
  "prediction": "CONFIRMED",
  "confidence": 0.92
}
```

### `GET /model/info`
Returns information about the loaded ML model.

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Place your trained ML model as `model/model.pkl`

3. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Model Format

The ML model should be a pickle file containing either:
- A scikit-learn model object directly
- A dictionary with keys:
  - `model`: The trained model object
  - `feature_names`: List of feature names in order

## Development

To create a sample model for testing:
```bash
python create_sample_model.py
```

This will generate synthetic training data and create a Random Forest classifier.

