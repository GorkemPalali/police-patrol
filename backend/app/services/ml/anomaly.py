"""Anomaly detection using Isolation Forest"""
from typing import List
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(
    risk_scores: List[float],
    contamination: float = 0.1
) -> List[bool]:
    """
    Detect anomalous risk spikes using Isolation Forest.
    
    Args:
        risk_scores: List of risk scores
        contamination: Expected proportion of anomalies
    
    Returns:
        List of boolean flags indicating anomalies
    """
    if len(risk_scores) < 2:
        return [False] * len(risk_scores)
    
    # Reshape for sklearn
    X = np.array(risk_scores).reshape(-1, 1)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X)
    
    # Convert to boolean (anomaly = -1)
    return [pred == -1 for pred in predictions]



