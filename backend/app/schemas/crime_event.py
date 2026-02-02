from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional


class CrimeEventBase(BaseModel):
    crime_type: str = Field(..., min_length=1, max_length=100)
    severity: int = Field(..., ge=1, le=5)
    event_time: datetime
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    street_name: Optional[str] = Field(None, max_length=255)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)


class CrimeEventCreate(CrimeEventBase):
    pass


class CrimeEventRead(CrimeEventBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class CrimeEventUpdate(BaseModel):
    crime_type: Optional[str] = Field(None, min_length=1, max_length=100)
    severity: Optional[int] = Field(None, ge=1, le=5)
    event_time: Optional[datetime] = None
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lng: Optional[float] = Field(None, ge=-180, le=180)
    street_name: Optional[str] = Field(None, max_length=255)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)



