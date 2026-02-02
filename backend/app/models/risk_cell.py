from sqlalchemy import Column, Float, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, TSRANGE
from geoalchemy2 import Geography
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class RiskCell(Base):
    __tablename__ = "risk_cell"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    geom = Column(Geography(geometry_type="POLYGON", srid=4326), nullable=False)
    time_window = Column(TSRANGE, nullable=False, index=True)
    risk_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(TIMESTAMP(timezone=False), server_default=func.now(), nullable=False)



