"""OSM boundary parser for extracting polygon from OSM relation data."""

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BoundaryParser:
    """Parser for OSM boundary/relation data."""

    def parse_boundary_xml(self, xml_data: str) -> Optional[List[List[Tuple[float, float]]]]:
        """
        Parse OSM XML and extract boundary polygon coordinates.

        Args:
            xml_data: OSM XML data as string

        Returns:
            List of polygon rings (outer ring + holes), each ring is a list of (lat, lng) tuples
            Returns None if parsing fails
        """
        try:
            root = ET.fromstring(xml_data)
            logger.info(f"Parsing OSM boundary XML with {len(root)} elements")

            # Extract nodes first
            nodes: Dict[int, Tuple[float, float]] = {}
            for node in root.findall("node"):
                node_id = int(node.get("id"))
                lat = float(node.get("lat"))
                lon = float(node.get("lon"))
                nodes[node_id] = (lat, lon)

            # Find relation with boundary=administrative
            relations = root.findall("relation")
            boundary_relation = None
            for relation in relations:
                tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
                if tags.get("boundary") == "administrative":
                    boundary_relation = relation
                    break

            if not boundary_relation:
                logger.warning("No administrative boundary relation found")
                return None

            # Extract outer way (main boundary)
            outer_ways = []
            inner_ways = []  # Holes in the polygon

            for member in boundary_relation.findall("member"):
                role = member.get("role")
                member_type = member.get("type")
                member_ref = int(member.get("ref"))

                if member_type == "way":
                    # Find the way element
                    way = root.find(f"way[@id='{member_ref}']")
                    if way is None:
                        continue

                    # Extract coordinates
                    coordinates = []
                    for nd in way.findall("nd"):
                        node_id = int(nd.get("ref"))
                        if node_id in nodes:
                            coordinates.append(nodes[node_id])

                    if len(coordinates) >= 3:  # At least 3 points for a polygon
                        if role == "outer" or role is None:
                            outer_ways.append(coordinates)
                        elif role == "inner":
                            inner_ways.append(coordinates)

            if not outer_ways:
                logger.warning("No outer ways found in boundary relation")
                return None

            # Combine outer ways into a single polygon
            # OSM relations can have multiple outer ways that need to be connected
            # Strategy: Find connected ways and merge them into rings
            merged_outer_ring = self._merge_outer_ways(outer_ways)
            
            if not merged_outer_ring or len(merged_outer_ring) < 3:
                logger.warning(f"Failed to merge outer ways, using first way only. Merged ring has {len(merged_outer_ring) if merged_outer_ring else 0} points")
                # Fallback: use first way if merge fails
                merged_outer_ring = outer_ways[0] if outer_ways else None
                if not merged_outer_ring or len(merged_outer_ring) < 3:
                    return None

            polygon_rings = [merged_outer_ring]

            # Add inner ways as holes
            polygon_rings.extend(inner_ways)

            logger.info(f"Parsed boundary with {len(polygon_rings)} rings ({len(outer_ways)} outer ways merged, {len(inner_ways)} inner)")
            return polygon_rings

        except ET.ParseError as e:
            logger.error(f"Failed to parse OSM XML: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing boundary: {str(e)}")
            return None

    def _merge_outer_ways(self, outer_ways: List[List[Tuple[float, float]]]) -> Optional[List[Tuple[float, float]]]:
        """
        Merge multiple outer ways into a single closed polygon ring.
        
        Args:
            outer_ways: List of way coordinate lists, each way is a list of (lat, lon) tuples
            
        Returns:
            Merged ring as list of (lat, lon) tuples, or None if merge fails
        """
        if not outer_ways:
            return None
        
        if len(outer_ways) == 1:
            # Single way - just ensure it's closed
            way = outer_ways[0]
            if len(way) < 3:
                return None
            # Close the ring if not already closed
            if way[0] != way[-1]:
                return way + [way[0]]
            return way
        
        # Multiple ways - need to connect them
        # Strategy: Build a graph of connected ways and traverse to form a ring
        # Ways are connected if the end of one matches the start/end of another
        
        def points_match(p1: Tuple[float, float], p2: Tuple[float, float], tolerance: float = 0.0001) -> bool:
            """Check if two points are close enough to be considered the same."""
            lat_diff = abs(p1[0] - p2[0])
            lon_diff = abs(p1[1] - p2[1])
            return lat_diff < tolerance and lon_diff < tolerance
        
        # Build list of way segments with their endpoints
        segments = []
        for way in outer_ways:
            if len(way) < 2:
                continue
            start = way[0]
            end = way[-1]
            segments.append({
                'coords': way,
                'start': start,
                'end': end,
                'used': False
            })
        
        if not segments:
            return None
        
        # Start with first segment
        merged = []
        current = segments[0]
        current['used'] = True
        merged.extend(current['coords'])
        last_point = current['end']
        
        # Try to connect remaining segments
        remaining = segments[1:]
        max_iterations = len(remaining) * 2  # Prevent infinite loops
        iterations = 0
        
        while remaining and iterations < max_iterations:
            iterations += 1
            found_match = False
            
            for i, seg in enumerate(remaining):
                if seg['used']:
                    continue
                
                # Check if segment connects to current end
                if points_match(last_point, seg['start']):
                    # Connect forward
                    merged.extend(seg['coords'][1:])  # Skip first point (already connected)
                    last_point = seg['end']
                    seg['used'] = True
                    remaining.pop(i)
                    found_match = True
                    break
                elif points_match(last_point, seg['end']):
                    # Connect backward (reverse the segment)
                    merged.extend(reversed(seg['coords'][:-1]))  # Skip last point, reverse order
                    last_point = seg['start']
                    seg['used'] = True
                    remaining.pop(i)
                    found_match = True
                    break
            
            if not found_match:
                # No more connections - check if we can close the ring
                if points_match(merged[0], last_point):
                    # Ring is closed
                    break
                else:
                    # Try to find any unused segment that might connect
                    # This handles disconnected parts
                    for i, seg in enumerate(remaining):
                        if not seg['used']:
                            # Start a new chain (will create a multipolygon, but we'll use first ring)
                            break
                    break
        
        # Close the ring if not already closed
        if len(merged) >= 3 and not points_match(merged[0], merged[-1]):
            merged.append(merged[0])
        
        if len(merged) < 3:
            logger.warning(f"Merged ring has only {len(merged)} points, insufficient for polygon")
            return None
        
        return merged

    def coordinates_to_wkt(
        self, coordinates: List[Tuple[float, float]], close_ring: bool = True
    ) -> str:
        """
        Convert coordinates to WKT format.

        Args:
            coordinates: List of (lon, lat) tuples - standard GeoJSON/PostGIS order
            close_ring: If True, ensure ring is closed (first point = last point)

        Returns:
            WKT string for POLYGON
        """
        if len(coordinates) < 3:
            raise ValueError("Polygon must have at least 3 points")

        # Close ring if needed
        if close_ring and coordinates[0] != coordinates[-1]:
            coordinates = coordinates + [coordinates[0]]

        # Convert to WKT format: (lng lat, lng lat, ...)
        # coordinates already in (lon, lat) order
        coords_str = ", ".join([f"{lon} {lat}" for lon, lat in coordinates])
        return f"POLYGON(({coords_str}))"

