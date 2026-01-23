from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from geoalchemy2 import Geography
from sqlalchemy.sql import func
import uuid
from typing import List, Optional

from app.db.base import Base


class PoliceStation(Base):
    __tablename__ = "police_station"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(150), nullable=False)
    geom = Column(Geography(geometry_type="POINT", srid=4326), nullable=False)
    capacity = Column(Integer, nullable=False, default=0)
    active = Column(Boolean, nullable=False, default=True, index=True)
    neighborhoods = Column(ARRAY(String), nullable=True, comment="Array of neighborhood names this station is responsible for")
    created_at = Column(TIMESTAMP(timezone=False), server_default=func.now(), nullable=False)



