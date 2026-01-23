"""Administrative boundary model."""

from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geography
from datetime import datetime
import uuid

from app.db.base import Base


class AdministrativeBoundary(Base):
    """Administrative boundary model for storing polygon boundaries."""

    __tablename__ = "administrative_boundary"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    admin_level = Column(Integer)
    osm_id = Column(Integer, nullable=True, index=True)  # OSM relation ID
    geom = Column(Geography(geometry_type="POLYGON", srid=4326), nullable=False)
    created_at = Column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=False), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

