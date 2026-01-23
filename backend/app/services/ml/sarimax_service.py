from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd

# Try to import statsmodels, fallback to simple average if not available
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def forecast_timeseries(
    historical_data: List[Dict],
    forecast_horizon: int = 24,
    model_path: Optional[Path] = None
) -> List[float]:
    """
    Forecast future risk using SARIMAX model.
    
    Args:
        historical_data: List of historical risk scores with timestamps
        forecast_horizon: Number of hours to forecast ahead
        model_path: Path to pre-trained model (optional)
    
    Returns:
        List of forecasted risk scores
    """
    if not historical_data:
        return [0.0] * forecast_horizon
    
    # If statsmodels not available or no model, use simple average
    if not STATSMODELS_AVAILABLE or not model_path or not model_path.exists():
        # Fallback: weighted moving average with trend
        risk_scores = [d.get('risk_score', 0.0) for d in historical_data]
        if len(risk_scores) < 2:
            return [float(np.mean(risk_scores))] * forecast_horizon
        
        # Simple trend-based forecast
        recent_scores = risk_scores[-24:] if len(risk_scores) >= 24 else risk_scores
        avg_recent = np.mean(recent_scores)
        
        # Calculate trend
        if len(risk_scores) >= 2:
            trend = (risk_scores[-1] - risk_scores[0]) / len(risk_scores)
        else:
            trend = 0.0
        
        # Forecast with trend
        forecast = []
        for i in range(forecast_horizon):
            forecast.append(max(0.0, min(1.0, avg_recent + trend * i)))
        
        return [float(f) for f in forecast]
    
    # Use trained SARIMAX model
    try:
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Make forecast
        forecast = model.forecast(steps=forecast_horizon)
        return [float(max(0.0, min(1.0, f))) for f in forecast]
    
    except Exception as e:
        # Fallback on error
        risk_scores = [d.get('risk_score', 0.0) for d in historical_data]
        avg_risk = np.mean(risk_scores)
        return [float(avg_risk)] * forecast_horizon
