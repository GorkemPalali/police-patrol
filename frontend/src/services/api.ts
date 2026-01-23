import axios from 'axios';
import type {
  PoliceStation,
  CrimeEvent,
  RiskMapResponse,
  RouteResponse,
  MultiStationRouteRequest,
  MultiStationRouteResponse,
} from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Stations API
export const stationsApi = {
  list: (activeOnly: boolean = true) =>
    api.get<PoliceStation[]>('/stations', { params: { active_only: activeOnly } }),
  get: (id: string) => api.get<PoliceStation>(`/stations/${id}`),
  create: (data: Omit<PoliceStation, 'id'>) =>
    api.post<PoliceStation>('/stations', data),
  update: (id: string, data: Partial<PoliceStation>) =>
    api.patch<PoliceStation>(`/stations/${id}`, data),
  delete: (id: string) => api.delete(`/stations/${id}`),
};

// Crimes API
export const crimesApi = {
  list: (params?: {
    start_time?: string;
    end_time?: string;
    crime_type?: string;
    min_severity?: number;
    max_severity?: number;
    limit?: number;
  }) => api.get<CrimeEvent[]>('/crimes', { params }),
  get: (id: string) => api.get<CrimeEvent>(`/crimes/${id}`),
  create: (data: Omit<CrimeEvent, 'id' | 'created_at'>) =>
    api.post<CrimeEvent>('/crimes', data),
  update: (id: string, data: Partial<CrimeEvent>) =>
    api.patch<CrimeEvent>(`/crimes/${id}`, data),
  delete: (id: string) => api.delete(`/crimes/${id}`),
  snapToRoad: (lat: number, lng: number) =>
    api.get<{
      snapped_lat: number;
      snapped_lng: number;
      road_segment_id: number;
      original_lat: number;
      original_lng: number;
      distance_m: number;
    }>('/crimes/snap-to-road', { params: { lat, lng } }),
};

// Forecast API
export const forecastApi = {
  getRiskMap: (startTime: string, endTime: string) =>
    api.get<RiskMapResponse>('/forecast/risk-map', {
      params: {
        start_time: startTime,
        end_time: endTime,
      },
    }),
};

// Routing API
export const routingApi = {
  optimize: (data: {
    station_id: string;
    risk_threshold?: number;
    max_minutes?: number;
    end_station_id?: string;
    start_time?: string;
    end_time?: string;
  }) => api.post<RouteResponse>('/routing/optimize', data),
  optimizeMulti: (data: MultiStationRouteRequest) =>
    api.post<MultiStationRouteResponse>('/routing/optimize-multi', data),
};

// OSM API
export const osmApi = {
  getRoadNetwork: (params?: {
    min_lat?: number;
    min_lng?: number;
    max_lat?: number;
    max_lng?: number;
    limit?: number;
  }) => api.get<{
    type: 'FeatureCollection';
    features: Array<{
      type: 'Feature';
      properties: {
        id: number;
        road_type: string;
        speed_limit: number | null;
        one_way: boolean;
        risk_score: number;
      };
      geometry: any;
    }>;
    total: number;
  }>('/osm/road-network', { params }),
};

export default api;
