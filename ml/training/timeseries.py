"""
SARIMAX model training script for time-series forecasting
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def prepare_timeseries_data(
    data: pd.DataFrame,
    time_col: str = 'event_time',
    value_col: str = 'risk_score'
) -> pd.Series:
    """
    Prepare time series data for SARIMAX model.
    
    Args:
        data: DataFrame with time and value columns
        time_col: Name of time column
        value_col: Name of value column
    
    Returns:
        Time series as pandas Series with DatetimeIndex
    """
    if data.empty:
        return pd.Series(dtype=float)

    df = data.copy()

    if time_col not in df.columns:
        for candidate in ('timestamp', 'event_time', 'time', 'date'):
            if candidate in df.columns:
                time_col = candidate
                break
        else:
            raise ValueError('Time column not found in data.')

    normalize_severity = False
    if value_col not in df.columns:
        if 'risk_score' in df.columns:
            value_col = 'risk_score'
        elif 'severity' in df.columns:
            value_col = 'severity'
            normalize_severity = True
        else:
            raise ValueError('Value column not found in data.')

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[time_col, value_col])
    if df.empty:
        return pd.Series(dtype=float)

    if normalize_severity:
        df[value_col] = df[value_col] / 5.0

    df[value_col] = df[value_col].clip(lower=0.0, upper=1.0)
    df = df.set_index(time_col)
    df = df.sort_index()
    
    # Resample to hourly if needed
    ts = df[value_col].resample('H').mean()
    ts = ts.ffill().fillna(0.0)
    
    return ts


def find_optimal_parameters(
    ts: pd.Series,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    seasonal_period: int = 24
) -> Tuple[int, int, int, int, int, int, int]:
    """
    Find optimal SARIMAX parameters using AIC.
    
    Args:
        ts: Time series data
        max_p, max_d, max_q: Maximum values for p, d, q parameters
    
    Returns:
        Tuple of (p, d, q, P, D, Q, s) parameters
    """
    seasonal_period = int(seasonal_period)
    if seasonal_period < 1:
        seasonal_period = 1

    best_aic = np.inf
    best_params = None
    
    # Try different parameter combinations
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(2):
                    for D in range(2):
                        for Q in range(2):
                            try:
                                model = SARIMAX(
                                    ts,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, seasonal_period),  # 24-hour seasonality
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                fitted_model = model.fit(disp=False, maxiter=50)
                                
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_params = (p, d, q, P, D, Q, seasonal_period)
                            except:
                                continue
    
    return best_params if best_params else (1, 1, 1, 0, 1, 1, seasonal_period)


def train_sarimax_model(
    data: pd.DataFrame,
    output_path: Optional[Path] = None,
    seasonal_period: int = 24
) -> SARIMAX:
    """
    Train SARIMAX model on time series data.
    
    Args:
        data: DataFrame with crime event data
        output_path: Path to save trained model
    
    Returns:
        Trained SARIMAX model
    """

    seasonal_period = int(seasonal_period)
    if seasonal_period < 1:
        seasonal_period = 1

    ts = prepare_timeseries_data(data)
    
    if len(ts) < 50:
        # Not enough data, use simple model
        order = (1, 1, 1)
        seasonal_order = (0, 1, 1, seasonal_period)
    else:
        # Find optimal parameters
        params = find_optimal_parameters(ts, seasonal_period=seasonal_period)
        order = params[:3]
        seasonal_order = params[3:]
    
    # Train model
    model = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_model = model.fit(disp=False, maxiter=100)
    
    # Save model
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(fitted_model, f)
    
    return fitted_model


def load_sarimax_model(model_path: Path) -> SARIMAX:
    """Load trained SARIMAX model from file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

