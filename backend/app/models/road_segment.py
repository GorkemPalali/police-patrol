from sqlalchemy import Column, String, Integer, Boolean, BigInteger, Float, TIMESTAMP
from geoalchemy2 import Geography
from sqlalchemy.sql import func

from app.db.base import Base


class RoadSegment(Base):
    __tablename__ = "road_segment"

    id = Column(BigInteger, primary_key=True)
    geom = Column(Geography(geometry_type="LINESTRING", srid=4326), nullable=False)
    road_type = Column(String(50))
    speed_limit = Column(Integer)
    one_way = Column(Boolean, nullable=False, default=False)
    source = Column(BigInteger)
    target = Column(BigInteger)
    cost = Column(Float)
    reverse_cost = Column(Float)
    risk_score = Column(Float, default=0.0)
    risk_confidence = Column(Float, default=0.0)
    risk_updated_at = Column(TIMESTAMP(timezone=False))
