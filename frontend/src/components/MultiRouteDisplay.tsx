import { useEffect, useMemo } from 'react';
import { Polyline, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import type { LatLngExpression } from 'leaflet';
import type { MultiStationRouteResponse, StationRoute } from '../types';

interface MultiRouteDisplayProps {
  routes: MultiStationRouteResponse | null;
  stationIcon?: L.Icon;
  showOverlap?: boolean;
}

// Color palette for different stations
const STATION_COLORS = [
  '#0066ff', // Blue
  '#ff6600', // Orange
  '#00cc66', // Green
  '#cc0066', // Magenta
  '#6600cc', // Purple
  '#ffcc00', // Yellow
  '#00cccc', // Cyan
  '#ff0066', // Pink
];

const MultiRouteDisplay = ({
  routes,
  stationIcon,
  showOverlap = true,
}: MultiRouteDisplayProps) => {
  const map = useMap();

  // Convert routes to display format
  const routeData = useMemo(() => {
    if (!routes || !routes.routes || routes.routes.length === 0) {
      return [];
    }

    return routes.routes.map((route, index) => {
      const color = STATION_COLORS[index % STATION_COLORS.length];
      const pathCoordinates: LatLngExpression[] = route.path?.coordinates
        ? route.path.coordinates.map((coord): [number, number] => {
            if (Array.isArray(coord) && coord.length >= 2) {
              const [lng, lat] = coord;
              if (typeof lng === 'number' && typeof lat === 'number') {
                return [lat, lng];
              }
            }
            return [0, 0];
          }).filter(([lat, lng]) => lat !== 0 || lng !== 0)
        : [];

      return {
        route,
        color,
        pathCoordinates,
      };
    });
  }, [routes]);

  // Fit map to all routes
  useEffect(() => {
    if (routeData.length > 0) {
      const allCoordinates: LatLngExpression[] = [];
      routeData.forEach(({ pathCoordinates }) => {
        allCoordinates.push(...pathCoordinates);
      });

      if (allCoordinates.length > 0) {
        const bounds = L.latLngBounds(allCoordinates);
        map.fitBounds(bounds, { padding: [50, 50] });
      }
    }
  }, [routeData, map]);

  if (!routes || !routes.routes || routes.routes.length === 0) {
    return null;
  }

  return (
    <>
      {/* Display each route with different color */}
      {routeData.map(({ route, color, pathCoordinates }, routeIndex) => (
        <div key={route.station_id}>
          {/* Route path */}
          {pathCoordinates.length > 0 && (
            <Polyline
              key={`route-${route.station_id}`}
              positions={pathCoordinates}
              color={color}
              weight={4}
              opacity={0.8}
            />
          )}

          {/* Waypoints */}
          {route.waypoints.map((waypoint, waypointIndex) => (
            <Marker
              key={`waypoint-${route.station_id}-${waypointIndex}`}
              position={[waypoint.lat, waypoint.lng]}
              icon={
                stationIcon ||
                L.icon({
                  iconUrl:
                    'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
                  shadowUrl:
                    'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
                  iconSize: [25, 41],
                  iconAnchor: [12, 41],
                  popupAnchor: [1, -34],
                })
              }
            >
              <Popup>
                <div>
                  <strong>{route.station_name}</strong>
                  <br />
                  <strong>Durak {waypointIndex + 1}</strong>
                  <br />
                  {waypoint.risk_score !== undefined && (
                    <>
                      Risk Skoru: {(waypoint.risk_score * 100).toFixed(1)}%
                      <br />
                    </>
                  )}
                  Mesafe: {(route.total_distance / 1000).toFixed(2)} km
                  <br />
                  Süre: {Math.round(route.total_time)} dakika
                  <br />
                  Risk Kapsamı: {(route.risk_coverage * 100).toFixed(1)}%
                  <br />
                  Koordinat: {waypoint.lat.toFixed(6)}, {waypoint.lng.toFixed(6)}
                </div>
              </Popup>
            </Marker>
          ))}
        </div>
      ))}

      {/* Overlap indicator (if enabled) */}
      {showOverlap && routes.overlap_percentage > 0 && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: 'rgba(255, 255, 255, 0.9)',
            padding: '10px',
            borderRadius: '5px',
            zIndex: 1000,
            fontSize: '12px',
          }}
        >
          <strong>Koordinasyon Bilgisi</strong>
          <br />
          Toplam Merkez: {routes.total_stations}
          <br />
          Risk Kapsamı: {(routes.total_risk_coverage * 100).toFixed(1)}%
          <br />
          Overlap: {(routes.overlap_percentage * 100).toFixed(1)}%
          <br />
          Koordinasyon Skoru: {(routes.coordination_score * 100).toFixed(1)}%
        </div>
      )}
    </>
  );
};

export default MultiRouteDisplay;




