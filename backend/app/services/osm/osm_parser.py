"""OSM XML/JSON parser and transformer."""

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RoadSegmentData:
    """Road segment data structure."""

    osm_id: int
    geom_coordinates: List[Tuple[float, float]]  # List of (lon, lat) tuples - standard GeoJSON/PostGIS order
    road_type: Optional[str] = None
    speed_limit: Optional[int] = None
    one_way: bool = False


class OSMParser:
    """Parser for OSM XML data."""

    # Highway type to road_type mapping
    # Only driveable roads - exclude pedestrian, footway, cycleway, path, etc.
    HIGHWAY_TYPE_MAPPING = {
        "motorway": "motorway",
        "trunk": "trunk",
        "primary": "primary",
        "secondary": "secondary",
        "tertiary": "tertiary",
        "residential": "residential",
        "service": "service",
        "unclassified": "unclassified",
        "living_street": "residential",
        # Exclude: pedestrian, footway, cycleway, path, steps, track (non-driveable)
    }

    # Common speed limit mappings (km/h)
    SPEED_LIMIT_MAPPING = {
        "motorway": 120,
        "trunk": 110,
        "primary": 90,
        "secondary": 70,
        "tertiary": 50,
        "residential": 30,
        "service": 20,
        "unclassified": 50,
    }

    def __init__(self):
        """
        Initialize OSM parser.
        No bbox filtering - Overpass API already filters the data.
        """
        pass

    def parse_xml(self, xml_data: str) -> List[RoadSegmentData]:
        """
        Parse OSM XML data and extract road segments.
        Supports both standard OSM format (with separate nodes) and 'out geom' format (with coordinates in nd elements).

        Args:
            xml_data: OSM XML data as string

        Returns:
            List of RoadSegmentData objects
        """
        try:
            root = ET.fromstring(xml_data)
            logger.info(f"Parsing OSM XML with {len(root)} elements")

            # Extract nodes first (for way coordinates) - standard OSM format
            nodes: Dict[int, Tuple[float, float]] = {}
            for node in root.findall("node"):
                node_id = int(node.get("id"))
                lat = float(node.get("lat"))
                lon = float(node.get("lon"))
                nodes[node_id] = (lat, lon)
            
            logger.info(f"Found {len(nodes)} nodes in XML")

            # Extract ways (road segments)
            road_segments: List[RoadSegmentData] = []
            ways_count = 0
            for way in root.findall("way"):
                ways_count += 1
                way_id = int(way.get("id"))
                road_segment = self._parse_way(way, way_id, nodes)
                if road_segment:
                    road_segments.append(road_segment)
            
            logger.info(f"Processed {ways_count} ways, found {len(road_segments)} valid road segments")
            logger.info(f"Parsed {len(road_segments)} road segments from OSM data")
            return road_segments

        except ET.ParseError as e:
            logger.error(f"Failed to parse OSM XML: {str(e)}")
            raise ValueError(f"Invalid OSM XML data: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error parsing OSM data: {str(e)}")
            raise

    def _parse_way(
        self, way: ET.Element, way_id: int, nodes: Dict[int, Tuple[float, float]]
    ) -> Optional[RoadSegmentData]:
        """
        Parse a single OSM way element.

        Args:
            way: OSM way element
            way_id: Way ID
            nodes: Dictionary of node_id -> (lat, lon) - OSM format

        Returns:
            RoadSegmentData or None if invalid
            Note: RoadSegmentData.geom_coordinates uses (lon, lat) order - standard GeoJSON/PostGIS format
        """
        # Extract tags
        tags = {}
        for tag in way.findall("tag"):
            key = tag.get("k")
            value = tag.get("v")
            if key and value:
                tags[key] = value

        # Check if it's a highway
        highway_type = tags.get("highway")
        if not highway_type:
            return None

        # Filter out non-driveable roads (exclude service, as it can be driveable)
        non_driveable = [
            "footway", "path", "cycleway", "steps", "pedestrian", 
            "track", "bridleway", "bus_guideway", "escape", 
            "raceway"
        ]
        if highway_type in non_driveable:
            # Check if it has motor_vehicle access
            access = tags.get("motor_vehicle", tags.get("access", "yes"))
            if access in ["no", "private", "permit"]:
                return None
        
        # Special handling for 'service' roads: only include if explicitly driveable
        if highway_type == "service":
            access = tags.get("motor_vehicle", tags.get("access", "yes"))
            if access in ["no", "private", "permit", "delivery", "agricultural", "forestry"]:
                return None

        # Map highway type (only driveable roads)
        road_type = self.HIGHWAY_TYPE_MAPPING.get(highway_type)
        if not road_type:
            # Unknown highway type, skip it
            return None

        # Extract coordinates from nd (node) references
        # Support both standard format (nd has ref attribute) and 'out geom' format (nd has lat/lon attributes)
        # Store as (lon, lat) - standard GeoJSON/PostGIS order
        coordinates: List[Tuple[float, float]] = []
        missing_nodes = []
        for nd in way.findall("nd"):
            # Check if nd has lat/lon attributes (out geom format)
            if nd.get("lat") is not None and nd.get("lon") is not None:
                lat = float(nd.get("lat"))
                lon = float(nd.get("lon"))
                # Store as (lon, lat) - standard GeoJSON/PostGIS order
                coordinates.append((lon, lat))
            else:
                # Standard format: use node reference
                # nodes dict stores (lat, lon), convert to (lon, lat)
                node_id = int(nd.get("ref"))
                if node_id in nodes:
                    lat, lon = nodes[node_id]
                    coordinates.append((lon, lat))
                else:
                    missing_nodes.append(node_id)
        
        # Log warning if many nodes are missing (might indicate incomplete data)
        if missing_nodes and len(missing_nodes) > len(coordinates):
            logger.debug(f"Way {way_id} has {len(missing_nodes)} missing nodes out of {len(missing_nodes) + len(coordinates)} total")

        # Validate coordinates
        if len(coordinates) < 2:
            logger.debug(f"Way {way_id} has less than 2 coordinates, skipping")
            return None

        # No bbox filtering in parser - Overpass API already filtered by relation/polygon/bbox
        # Parser just processes the data that Overpass API returned
        # bbox parameter kept for backward compatibility but not used for filtering

        # Extract speed limit
        speed_limit = self._extract_speed_limit(tags, road_type)

        # Extract one-way information
        one_way = self._extract_one_way(tags)

        return RoadSegmentData(
            osm_id=way_id,
            geom_coordinates=coordinates,
            road_type=road_type,
            speed_limit=speed_limit,
            one_way=one_way,
        )

    def _extract_speed_limit(self, tags: Dict[str, str], road_type: str) -> Optional[int]:
        """
        Extract speed limit from tags.

        Args:
            tags: OSM tags dictionary
            road_type: Road type

        Returns:
            Speed limit in km/h or None
        """
        # Try maxspeed tag first
        maxspeed = tags.get("maxspeed")
        if maxspeed:
            try:
                # Handle various formats: "50", "50 km/h", "50kph", etc.
                maxspeed_clean = maxspeed.lower().replace("km/h", "").replace("kph", "").strip()
                # Extract numbers
                import re

                numbers = re.findall(r"\d+", maxspeed_clean)
                if numbers:
                    return int(numbers[0])
            except (ValueError, AttributeError):
                pass

        # Fallback to default based on road type
        return self.SPEED_LIMIT_MAPPING.get(road_type)

    def _extract_one_way(self, tags: Dict[str, str]) -> bool:
        """
        Extract one-way information from tags.

        Args:
            tags: OSM tags dictionary

        Returns:
            True if one-way, False otherwise
        """
        oneway = tags.get("oneway", "no").lower()
        return oneway in ("yes", "true", "1", "-1")

    def validate_geometry(self, coordinates: List[Tuple[float, float]]) -> bool:
        """
        Validate geometry coordinates.

        Args:
            coordinates: List of (lat, lng) tuples

        Returns:
            True if valid, False otherwise
        """
        if len(coordinates) < 2:
            return False

        # Check coordinate ranges
        for lat, lng in coordinates:
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                return False

        # Check for duplicate consecutive points
        for i in range(len(coordinates) - 1):
            if coordinates[i] == coordinates[i + 1]:
                return False

        return True

