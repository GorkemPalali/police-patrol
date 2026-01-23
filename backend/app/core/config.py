from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Tuple


class Settings(BaseSettings):
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+psycopg2://police:policepwd@localhost:5432/policepatrol"
    redis_url: str = "redis://localhost:6379/0"

    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Risk forecasting parameters
    default_grid_size_m: float = 100.0
    kde_min_bandwidth_m: float = 50.0
    kde_max_bandwidth_m: float = 500.0
    default_risk_threshold: float = 0.7

    # Routing parameters
    default_route_time_minutes: int = 90
    max_route_time_minutes: int = 180

    # Küçükçekmece boundary settings
    # Boundary is now stored in database as polygon
    # These are fallback values for initial setup or when boundary is not loaded
    kucukcekmece_boundary_name: str = "Küçükçekmece"
    kucukcekmece_boundary_admin_level: int = 6  # İlçe seviyesi (district level) - OSM'de admin_level 6
    # Fallback bbox (used only if polygon is not available)
    kucukcekmece_fallback_bbox: Tuple[float, float, float, float] = (
        40.98,   # min_lat
        28.70,   # min_lng
        41.05,   # max_lat
        28.80    # max_lng
    )
    # Strict boundary validation (set to False to disable validation in development)
    strict_boundary_validation: bool = True

    # OSM Import settings
    # Try multiple Overpass API endpoints for reliability
    overpass_api_url: str = "https://overpass-api.de/api/interpreter"
    overpass_api_alternatives: List[str] = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]
    osm_import_on_startup: bool = True
    # OSM relation ID for Küçükçekmece administrative boundary
    osm_boundary_relation_id: int = 7786498  # Küçükçekmece İlçesi relation ID from OSM
    osm_highway_tags: List[str] = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "service",
        "unclassified",
    ]
    osm_topology_tolerance: float = 0.0001  # ~11 meters in degrees

    # Real-time risk updates settings
    realtime_enabled: bool = True  # Enable/disable real-time updates
    risk_cache_ttl_seconds: int = 3600  # Cache TTL (1 hour)
    websocket_heartbeat_interval: int = 30  # Heartbeat interval in seconds
    risk_update_broadcast_enabled: bool = True  # Enable/disable risk update broadcasts

    # Multi-station route coordination settings
    multi_station_coordination_enabled: bool = True  # Enable/disable multi-station coordination
    default_overlap_threshold: float = 0.2  # Maximum allowed overlap (0.0-1.0)
    capacity_weight: float = 0.3  # Weight for capacity in cell distribution (0-1)
    distance_weight: float = 0.4  # Weight for distance in cell distribution (0-1)
    risk_weight: float = 0.3  # Weight for risk score in cell distribution (0-1)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()

