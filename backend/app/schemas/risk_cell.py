from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Any, Optional


class RiskCellRead(BaseModel):
    id: UUID
    geom: Any  # GeoJSON
    risk_score: float
    confidence: float
    time_window_start: datetime
    time_window_end: datetime

    class Config:
        from_attributes = True


class RiskMapResponse(BaseModel):
    time_window: dict[str, str]
    risk_cells: list[dict[str, Any]]
    grid_size_m: float
    total_cells: int




