import { useEffect, useRef, useState } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet.heat";
import { forecastApi } from "../services/api";
import type { RiskMapResponse } from "../types";
import { RiskWebSocketClient } from "../services/websocket";
import type { RiskMapData } from "../types/websocket";

interface RiskHeatmapProps {
  startTime: string;
  endTime: string;
  threshold: number;
  onLoad?: (data: RiskMapResponse) => void;
  enableRealtime?: boolean; // Enable real-time updates via WebSocket
}

// Küçükçekmece Gölü sınırları (filtreleme için)
const LAKE_BOUNDS = {
  minLat: 41.0,
  maxLat: 41.02,
  minLng: 28.75,
  maxLng: 28.78,
};

const isInLake = (lat: number, lng: number): boolean => {
  return (
    lat >= LAKE_BOUNDS.minLat &&
    lat <= LAKE_BOUNDS.maxLat &&
    lng >= LAKE_BOUNDS.minLng &&
    lng <= LAKE_BOUNDS.maxLng
  );
};

// Extract center point from GeoJSON geometry
const getGeometryCenter = (geom: any): [number, number] | null => {
  if (!geom || !geom.coordinates) return null;

  try {
    if (geom.type === "Point") {
      return [geom.coordinates[1], geom.coordinates[0]]; // [lat, lng]
    }

    if (geom.type === "Polygon" && geom.coordinates[0]) {
      // Simple centroid approximation (average of ring coordinates)
      const coords = geom.coordinates[0];
      let sumLat = 0;
      let sumLng = 0;

      for (const coord of coords) {
        sumLng += coord[0];
        sumLat += coord[1];
      }

      const centerLat = sumLat / coords.length;
      const centerLng = sumLng / coords.length;
      return [centerLat, centerLng];
    }
  } catch (e) {
    console.error("Error extracting geometry center:", e);
  }

  return null;
};

const RiskHeatmap = ({ startTime, endTime, threshold, onLoad, enableRealtime = true }: RiskHeatmapProps) => {
  const map = useMap();
  const heatLayerRef = useRef<L.Layer | null>(null);
  const wsClientRef = useRef<RiskWebSocketClient | null>(null);
  const [isRealtimeConnected, setIsRealtimeConnected] = useState(false);

  // Helper function to update heatmap from data
  const updateHeatmapFromData = (cells: Array<{ lat: number; lng: number; risk_score: number }>, cancelled: boolean) => {
    if (cancelled) return;

    try {
      const heatLayerFactory = (L as any).heatLayer as
        | ((latlngs: Array<[number, number, number]>, options?: any) => L.Layer)
        | undefined;

      if (!heatLayerFactory) {
        throw new Error(
          "leaflet.heat is not initialized. Ensure `import 'leaflet.heat'` is executed and remove CDN scripts."
        );
      }

      // Remove existing heat layer
      if (heatLayerRef.current) {
        map.removeLayer(heatLayerRef.current);
        heatLayerRef.current = null;
      }

      const filteredCells = cells
        .filter((cell) => cell.risk_score >= threshold)
        .map((cell) => {
          // Exclude lake area
          if (isInLake(cell.lat, cell.lng)) return null;

          return {
            lat: cell.lat,
            lng: cell.lng,
            intensity: cell.risk_score,
          };
        })
        .filter((p): p is { lat: number; lng: number; intensity: number } => p !== null);

      if (filteredCells.length === 0) return;

      // heat expects: [lat, lng, intensity]
      const heatData: [number, number, number][] = filteredCells.map((c) => [
        c.lat,
        c.lng,
        c.intensity,
      ]);

      const maxIntensity = 1.0;

      const heatLayer = heatLayerFactory(heatData, {
        radius: 25,
        blur: 15,
        maxZoom: 17,
        max: maxIntensity,
        gradient: {
          0.0: "blue",
          0.25: "cyan",
          0.5: "lime",
          0.75: "yellow",
          1.0: "red",
        },
        minOpacity: 0.3,
      });

      if (cancelled) return;

      heatLayer.addTo(map);
      heatLayerRef.current = heatLayer;
    } catch (err: any) {
      console.error("Error updating heatmap:", err);
    }
  };

  useEffect(() => {
    let cancelled = false;

    const loadRiskMap = async () => {
      if (!startTime || !endTime) return;

      try {
        // leaflet.heat ESM import'u L'ye heatLayer ekler.
        const heatLayerFactory = (L as any).heatLayer as
          | ((latlngs: Array<[number, number, number]>, options?: any) => L.Layer)
          | undefined;

        if (!heatLayerFactory) {
          throw new Error(
            "leaflet.heat is not initialized. Ensure `import 'leaflet.heat'` is executed and remove CDN scripts."
          );
        }

        const startISO = new Date(startTime).toISOString();
        const endISO = new Date(endTime).toISOString();

        const response = await forecastApi.getRiskMap(startISO, endISO);
        const data: RiskMapResponse = response.data;

        if (cancelled) return;

        onLoad?.(data);

        // Remove existing heat layer
        if (heatLayerRef.current) {
          map.removeLayer(heatLayerRef.current);
          heatLayerRef.current = null;
        }

        // Convert GeoJSON cells to simple lat/lng format
        const cells = data.risk_cells.map((cell) => {
          const center = getGeometryCenter(cell.geom);
          if (!center) return null;
          return {
            lat: center[0],
            lng: center[1],
            risk_score: cell.risk_score,
          };
        }).filter((c): c is { lat: number; lng: number; risk_score: number } => c !== null);

        updateHeatmapFromData(cells, cancelled);
      } catch (err: any) {
        if (cancelled) return;

        const errorMsg =
          err?.response?.data?.detail ||
          err?.message ||
          "Risk haritası yüklenirken hata oluştu";

        console.error("Risk map error:", errorMsg, err);

        if (heatLayerRef.current) {
          map.removeLayer(heatLayerRef.current);
          heatLayerRef.current = null;
        }
      }
    };

    loadRiskMap();

    return () => {
      cancelled = true;
      if (heatLayerRef.current) {
        map.removeLayer(heatLayerRef.current);
        heatLayerRef.current = null;
      }
    };
  }, [startTime, endTime, threshold, map, onLoad]);

  // WebSocket real-time updates
  useEffect(() => {
    if (!enableRealtime || !startTime || !endTime) {
      return;
    }

    let cancelled = false;

    try {
      const wsClient = new RiskWebSocketClient({
        startTime: new Date(startTime),
        endTime: new Date(endTime),
        onRiskUpdate: (data: RiskMapData) => {
          if (cancelled) return;

          console.log("Received real-time risk update", data);

          // Convert WebSocket data format to heatmap format
          const cells = data.cells.map((cell) => ({
            lat: cell.lat,
            lng: cell.lng,
            risk_score: cell.risk_score,
          }));

          updateHeatmapFromData(cells, cancelled);
        },
        onConnect: () => {
          console.log("WebSocket connected for real-time updates");
          setIsRealtimeConnected(true);
        },
        onDisconnect: () => {
          console.log("WebSocket disconnected");
          setIsRealtimeConnected(false);
        },
        onError: (error) => {
          console.error("WebSocket error:", error);
          setIsRealtimeConnected(false);
        },
      });

      wsClientRef.current = wsClient;

      return () => {
        cancelled = true;
        if (wsClientRef.current) {
          try {
            wsClient.disconnect();
          } catch (error) {
            // Ignore errors during cleanup
            console.debug('WebSocket cleanup error (ignored):', error);
          }
          wsClientRef.current = null;
        }
        setIsRealtimeConnected(false);
      };
    } catch (error) {
      console.error("Error setting up WebSocket:", error);
    }
  }, [startTime, endTime, enableRealtime, threshold, map]);

  return null;
};

export default RiskHeatmap;
