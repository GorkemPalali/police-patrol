from pydantic import BaseModel, Field
from uuid import UUID
from typing import Optional, List, Literal


class GeoJSONPolygon(BaseModel):
    type: Literal["Polygon"]
    coordinates: List[List[List[float]]]


class PoliceStationBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=150)
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    capacity: int = Field(default=0, ge=0)
    active: bool = Field(default=True)


class PoliceStationCreate(PoliceStationBase):
    boundary: Optional[GeoJSONPolygon] = None


class PoliceStationRead(PoliceStationBase):
    id: UUID
    boundary: Optional[GeoJSONPolygon] = None

    class Config:
        from_attributes = True


class PoliceStationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=150)
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lng: Optional[float] = Field(None, ge=-180, le=180)
    capacity: Optional[int] = Field(None, ge=0)
    active: Optional[bool] = None
    boundary: Optional[GeoJSONPolygon] = None




