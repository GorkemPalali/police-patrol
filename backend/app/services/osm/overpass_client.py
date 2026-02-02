"""Overpass API client for fetching OSM data."""

import logging
import time
from typing import Dict, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class OverpassClient:
    """Client for interacting with Overpass API."""

    def __init__(
        self,
        api_url: str = "https://overpass-api.de/api/interpreter",
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize Overpass API client.

        Args:
            api_url: Overpass API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def build_bbox_query(
        self,
        bbox: Tuple[float, float, float, float],
        highway_tags: Optional[List[str]] = None,
    ) -> str:
        """
        Build Overpass QL query for bounding box.

        Args:
            bbox: Bounding box as (min_lat, min_lng, max_lat, max_lng)
            highway_tags: List of highway tag values to filter (e.g., ['motorway', 'trunk'])

        Returns:
            Overpass QL query string
        """
        min_lat, min_lng, max_lat, max_lng = bbox

        # Default highway tags - only driveable roads (exclude pedestrian, footway, cycleway, etc.)
        if highway_tags is None:
            highway_tags = [
                "motorway",
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "residential",
                "service",
                "unclassified",
            ]

        # Build highway filter
        highway_filter = "|".join(highway_tags)
        highway_condition = f'["highway"~"^({highway_filter})$"]'

        # Overpass QL query
        query = f"""
[out:xml][timeout:300];
(
  way{highway_condition}({min_lat},{min_lng},{max_lat},{max_lng});
);
out geom;
"""

        return query

    def build_relation_query(
        self,
        relation_id: int,
        highway_tags: Optional[List[str]] = None,
    ) -> str:
        """
        Build Overpass QL query for a specific administrative relation.
        This ensures we only get roads within the exact boundary.

        Args:
            relation_id: OSM relation ID for the administrative boundary
            highway_tags: List of highway tag values to filter (e.g., ['motorway', 'trunk'])

        Returns:
            Overpass QL query string
        """
        # Default highway tags - only driveable roads
        if highway_tags is None:
            highway_tags = [
                "motorway",
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "residential",
                "service",
                "unclassified",
            ]

        # Build highway filter - exclude non-driveable roads
        highway_filter = "|".join(highway_tags)
        highway_condition = f'["highway"~"^({highway_filter})$"]'

        # Overpass QL query using relation
        # First get relation, expand it, then convert to area and query ways
        query = f"""
[out:xml][timeout:300];
(
  relation({relation_id});
);
(._;>;);
map_to_area -> .searchArea;
(
  way{highway_condition}(area.searchArea);
);
out geom;
"""

        return query

    def build_polygon_query(
        self,
        polygon_coords: List[Tuple[float, float]],
        highway_tags: Optional[List[str]] = None,
    ) -> str:
        """
        Build Overpass QL query using polygon coordinates.
        This is more precise than bbox but requires polygon coordinates.

        Args:
            polygon_coords: List of (lon, lat) tuples forming the polygon - standard GeoJSON/PostGIS order
            highway_tags: List of highway tag values to filter

        Returns:
            Overpass QL query string
        Note: Overpass poly format uses "lat lng lat lng ..." order, so we convert from (lon, lat)
        """
        # Default highway tags - only driveable roads
        if highway_tags is None:
            highway_tags = [
                "motorway",
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "residential",
                "service",
                "unclassified",
            ]

        # Build highway filter
        highway_filter = "|".join(highway_tags)
        highway_condition = f'["highway"~"^({highway_filter})$"]'

        # Convert polygon coordinates to Overpass poly format: "lat1 lng1 lat2 lng2 ..."
        # Note: Overpass poly format uses (lat lng) order, but we receive (lon, lat)
        # Note: poly format requires closing the polygon (first point repeated at end)
        if len(polygon_coords) > 0 and polygon_coords[0] != polygon_coords[-1]:
            polygon_coords = polygon_coords + [polygon_coords[0]]
        
        # Convert from (lon, lat) to Overpass format (lat lng)
        poly_string = " ".join([f"{lat} {lon}" for lon, lat in polygon_coords])

        # Overpass QL query using polygon
        query = f"""
[out:xml][timeout:300];
(
  way{highway_condition}(poly:"{poly_string}");
);
out geom;
"""

        return query

    def fetch_osm_data(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        highway_tags: Optional[List[str]] = None,
        alternative_urls: Optional[List[str]] = None,
        relation_id: Optional[int] = None,
        polygon_coords: Optional[List[Tuple[float, float]]] = None,
    ) -> str:
        """
        Fetch OSM data from Overpass API, trying alternative endpoints if primary fails.
        Can use relation ID, polygon coordinates, or bounding box.

        Args:
            bbox: Bounding box as (min_lat, min_lng, max_lat, max_lng) - used if relation_id and polygon_coords are None
            highway_tags: List of highway tag values to filter (only driveable roads)
            alternative_urls: List of alternative Overpass API URLs to try
            relation_id: OSM relation ID for administrative boundary (preferred method)
            polygon_coords: List of (lat, lng) tuples forming polygon boundary

        Returns:
            OSM XML data as string

        Raises:
            requests.RequestException: If all endpoints fail after retries
            ValueError: If parameters are invalid
        """
        # Build query based on available parameters - only relation_id is used
        if relation_id is not None:
            # Use relation ID (most precise and only supported method)
            query = self.build_relation_query(relation_id, highway_tags)
            logger.info(f"Fetching OSM data for relation ID: {relation_id}")
        else:
            raise ValueError("Must provide relation_id. Boundary must be imported first.")

        # Try primary URL first, then alternatives
        urls_to_try = [self.api_url]
        if alternative_urls:
            urls_to_try.extend(alternative_urls)

        last_error = None
        for api_url in urls_to_try:
            try:
                logger.info(f"Trying Overpass API: {api_url}")
                response = self.session.post(
                    api_url,
                    data={"data": query},
                    timeout=self.timeout,
                    headers={"User-Agent": "PolicePatrol-OSM-Importer/1.0"},
                )
                response.raise_for_status()

                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", self.retry_delay))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    # Retry once more
                    response = self.session.post(
                        api_url,
                        data={"data": query},
                        timeout=self.timeout,
                        headers={"User-Agent": "PolicePatrol-OSM-Importer/1.0"},
                    )
                    response.raise_for_status()

                logger.info(f"Successfully fetched OSM data from {api_url} ({len(response.text)} bytes)")
                return response.text

            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout from {api_url}: {str(e)}")
                last_error = e
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to fetch from {api_url}: {str(e)}")
                last_error = e
                continue

        # All endpoints failed
        logger.error(f"All Overpass API endpoints failed. Last error: {str(last_error)}")
        raise requests.exceptions.RequestException(
            f"Failed to fetch OSM data from all endpoints. Last error: {str(last_error)}"
        )

    def check_api_status(self) -> Dict[str, any]:
        """
        Check Overpass API status.

        Returns:
            Dictionary with API status information
        """
        try:
            # Simple test query
            test_query = "[out:json][timeout:5];node(1);out;"
            response = self.session.post(
                self.api_url,
                data={"data": test_query},
                timeout=10,
                headers={"User-Agent": "PolicePatrol-OSM-Importer/1.0"},
            )
            response.raise_for_status()
            return {
                "status": "available",
                "url": self.api_url,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
            }
        except Exception as e:
            return {
                "status": "unavailable",
                "url": self.api_url,
                "error": str(e),
            }

