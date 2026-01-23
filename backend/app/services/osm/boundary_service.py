"""OSM boundary service for fetching administrative boundaries."""

import logging
import requests
from typing import Dict, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class BoundaryService:
    """Service for fetching administrative boundaries from OSM."""

    def __init__(
        self,
        api_url: str = "https://overpass-api.de/api/interpreter",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        """
        Initialize boundary service.

        Args:
            api_url: Overpass API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch_boundary_by_name(
        self,
        name: str,
        admin_level: int = 6,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> str:
        """
        Fetch administrative boundary from OSM by name and admin level.

        Args:
            name: Name of the administrative boundary (e.g., "Küçükçekmece")
            admin_level: Administrative level (6 for districts/ilçe in Turkey, 8 for neighborhoods/mahalle)
            bbox: Optional bounding box to limit search (min_lat, min_lng, max_lat, max_lng)

        Returns:
            OSM XML data as string

        Raises:
            requests.RequestException: If request fails
            ValueError: If parameters are invalid
        """
        # Build Overpass QL query
        # Use bbox to limit search area, then filter by name (case-insensitive regex)
        # Escape special regex characters in name
        escaped_name = name.replace('\\', '\\\\').replace('^', '\\^').replace('$', '\\$').replace('.', '\\.').replace('|', '\\|').replace('(', '\\(').replace(')', '\\)').replace('[', '\\[').replace(']', '\\]').replace('{', '\\{').replace('}', '\\}').replace('+', '\\+').replace('*', '\\*').replace('?', '\\?')
        
        if bbox:
            min_lat, min_lng, max_lat, max_lng = bbox
            # Use bbox in the query to limit search area
            query = f"""
[out:xml][timeout:300];
(
  relation["boundary"="administrative"]["admin_level"="{admin_level}"]({min_lat},{min_lng},{max_lat},{max_lng})["name"~"^{escaped_name}$",i];
);
(._;>;);
out geom;
"""
        else:
            # Without bbox, search globally (slower but works)
            query = f"""
[out:xml][timeout:300];
(
  relation["boundary"="administrative"]["admin_level"="{admin_level}"]["name"~"^{escaped_name}$",i];
);
(._;>;);
out geom;
"""

        logger.info(f"Fetching boundary for {name} (admin_level={admin_level})")

        try:
            response = self.session.post(
                self.api_url,
                data={"data": query},
                timeout=self.timeout,
                headers={"User-Agent": "PolicePatrol-Boundary-Importer/1.0"},
            )
            response.raise_for_status()

            logger.info(f"Successfully fetched boundary data ({len(response.text)} bytes)")
            return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch boundary: {str(e)}")
            raise

    def fetch_boundary_by_relation_id(self, relation_id: int) -> str:
        """
        Fetch administrative boundary from OSM by relation ID.

        Args:
            relation_id: OSM relation ID

        Returns:
            OSM XML data as string

        Raises:
            requests.RequestException: If request fails
        """
        query = f"""
[out:xml][timeout:300];
(
  relation({relation_id});
);
(._;>;);
out geom;
"""

        logger.info(f"Fetching boundary for relation ID {relation_id}")

        try:
            response = self.session.post(
                self.api_url,
                data={"data": query},
                timeout=self.timeout,
                headers={"User-Agent": "PolicePatrol-Boundary-Importer/1.0"},
            )
            response.raise_for_status()

            logger.info(f"Successfully fetched boundary data ({len(response.text)} bytes)")
            return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch boundary: {str(e)}")
            raise




