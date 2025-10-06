from pydantic import BaseModel, Field
from typing import Optional


class ExoplanetData(BaseModel):
    """Schema for exoplanet data input to the ML model"""
    
    # Required fields for the ML model
    koi_period_err1: float = Field(..., description="KOI period error 1")
    koi_period_err2: float = Field(..., description="KOI period error 2")
    koi_time0bk: float = Field(..., description="KOI time of first transit")
    koi_time0bk_err1: float = Field(..., description="KOI time0bk error 1")
    koi_time0bk_err2: float = Field(..., description="KOI time0bk error 2")
    koi_impact: float = Field(..., description="KOI impact parameter")
    koi_duration: float = Field(..., description="KOI transit duration")
    koi_duration_err1: float = Field(..., description="KOI duration error 1")
    koi_duration_err2: float = Field(..., description="KOI duration error 2")
    koi_depth: float = Field(..., description="KOI transit depth")
    koi_depth_err1: float = Field(..., description="KOI depth error 1")
    koi_model_snr: float = Field(..., description="KOI model signal-to-noise ratio")
    koi_steff_err1: float = Field(..., description="KOI stellar effective temperature error 1")
    koi_srad_err1: float = Field(..., description="KOI stellar radius error 1")
    delivname_q1_q16_tce: str = Field(..., description="Delivery name Q1-Q16 TCE")
    delivname_q1_q17_dr24_tce: str = Field(..., description="Delivery name Q1-Q17 DR24 TCE")
    delivname_q1_q17_dr25_tce: str = Field(..., description="Delivery name Q1-Q17 DR25 TCE")
    
    # Optional fields for additional context
    koi_name: Optional[str] = Field(None, description="KOI name")
    ra: Optional[float] = Field(None, description="Right ascension")
    dec: Optional[float] = Field(None, description="Declination")
    koi_period: Optional[float] = Field(None, description="KOI orbital period")
    koi_steff: Optional[float] = Field(None, description="KOI stellar effective temperature")
    koi_srad: Optional[float] = Field(None, description="KOI stellar radius")


class PredictionResponse(BaseModel):
    """Schema for ML model prediction response"""
    prediction: str = Field(..., description="Prediction result: CONFIRMED or CANDIDATE")
    confidence: float = Field(..., description="Confidence score (0-1)")


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")

