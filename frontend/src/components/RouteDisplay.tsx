import { useEffect, useMemo } from 'react';
import { Polyline, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import type { LatLngExpression } from 'leaflet';
import type { RouteResponse } from '../types';

interface RouteDisplayProps {
  route: RouteResponse | null;
  stationIcon?: L.Icon;
}

const RouteDisplay = ({ route, stationIcon }: RouteDisplayProps) => {
  const map = useMap();

  // Convert GeoJSON coordinates [lng, lat] to Leaflet format [lat, lng]
  const pathCoordinates: LatLngExpression[] = useMemo(() => {
    if (!route?.path?.coordinates) {
      return [];
    }

    const coords = route.path.coordinates;
    // GeoJSON coordinates are [lng, lat], Leaflet expects [lat, lng]
    return coords.map((coord): [number, number] => {
      // Ensure we have at least 2 elements (lng, lat)
      if (Array.isArray(coord) && coord.length >= 2) {
        const [lng, lat] = coord;
        // Validate that both are numbers
        if (typeof lng === 'number' && typeof lat === 'number') {
          return [lat, lng];
        }
      }
      // Fallback (should not happen with valid GeoJSON)
      return [0, 0];
    }).filter(([lat, lng]) => lat !== 0 || lng !== 0); // Remove invalid coordinates
  }, [route]);

  useEffect(() => {
    if (pathCoordinates.length > 0) {
      // Fit map to route bounds
      const bounds = L.latLngBounds(pathCoordinates);
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [pathCoordinates, map]);

  if (!route) {
    return null;
  }

  const { waypoints, total_distance, total_time, risk_coverage } = route;

  // Default route color
  const routeColor = '#0066ff';
  const routeWeight = 4;

  return (
    <>
      {/* Route path */}
      {pathCoordinates.length > 0 && (
        <Polyline
          positions={pathCoordinates}
          color={routeColor}
          weight={routeWeight}
          opacity={0.8}
        />
      )}

      {/* Waypoints */}
      {waypoints.map((waypoint, index) => (
        <Marker
          key={`waypoint-${index}`}
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
              <strong>Durak {index + 1}</strong>
              <br />
              {waypoint.risk_score !== undefined && (
                <>
                  Risk Skoru: {(waypoint.risk_score * 100).toFixed(1)}%
                  <br />
                </>
              )}
              Koordinat: {waypoint.lat.toFixed(6)}, {waypoint.lng.toFixed(6)}
            </div>
          </Popup>
        </Marker>
      ))}

      {/* Route info popup (shown on first waypoint) */}
      {waypoints.length > 0 && (
        <Marker position={[waypoints[0].lat, waypoints[0].lng]}>
          <Popup>
            <div>
              <strong>Rota Bilgileri</strong>
              <br />
              Toplam Mesafe: {(total_distance / 1000).toFixed(2)} km
              <br />
              Tahmini Süre: {total_time.toFixed(0)} dakika
              <br />
              Risk Kapsamı: {(risk_coverage * 100).toFixed(1)}%
              <br />
              Durak Sayısı: {waypoints.length}
            </div>
          </Popup>
        </Marker>
      )}
    </>
  );
};

export default RouteDisplay;