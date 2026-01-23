export interface PoliceStation {
  id: string;
  name: string;
  lat: number;
  lng: number;
  capacity: number;
  active: boolean;
}

export interface CrimeEvent {
  id: string;
  crime_type: string;
  severity: number;
  event_time: string;
  lat: number;
  lng: number;
  street_name?: string;
  confidence_score: number;
  created_at: string;
}

export interface RiskCell {
  id: string;
  geom: GeoJSON.Polygon | GeoJSON.MultiPolygon;
  risk_score: number;
  confidence: number;
  time_window_start: string;
  time_window_end: string;
}

export interface RiskMapResponse {
  time_window: {
    start: string;
    end: string;
  };
  risk_cells: Array<{
    id: string;
    geom: GeoJSON.Polygon | GeoJSON.MultiPolygon;
    risk_score: number;
    confidence: number;
  }>;
  grid_size_m: number;
  total_cells: number;
}

export interface RouteWaypoint {
  lat: number;
  lng: number;
  risk_score?: number;
}

export interface RouteResponse {
  waypoints: RouteWaypoint[];
  total_distance: number;
  total_time: number;
  risk_coverage: number;
  path: GeoJSON.LineString;
}

export interface StationRoute {
  station_id: string;
  station_name: string;
  waypoints: RouteWaypoint[];
  total_distance: number;
  total_time: number;
  risk_coverage: number;
  path: GeoJSON.LineString;
}

export interface MultiStationRouteRequest {
  station_ids?: string[];
  risk_threshold?: number;
  max_minutes_per_station?: number;
  minimize_overlap?: boolean;
  distribute_by_capacity?: boolean;
}

export interface MultiStationRouteResponse {
  routes: StationRoute[];
  total_stations: number;
  total_risk_coverage: number;
  overlap_percentage: number;
  coordination_score: number;
}
