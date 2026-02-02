from sqlalchemy import Column, String, Integer, TIMESTAMP, Float, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geography
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class CrimeEvent(Base):
    __tablename__ = "crime_event"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    crime_type = Column(String(100), nullable=False, index=True)
    severity = Column(Integer, nullable=False, index=True)
    event_time = Column(TIMESTAMP(timezone=False), nullable=False, index=True)
    geom = Column(Geography(geometry_type="POINT", srid=4326), nullable=False)
    street_name = Column(String(255))
    confidence_score = Column(Float, nullable=False, default=1.0)
    created_at = Column(TIMESTAMP(timezone=False), server_default=func.now(), nullable=False)
    snapped_road_segment_id = Column(BigInteger, nullable=True, index=True)


