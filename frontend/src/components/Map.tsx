import { useEffect, useState, useCallback, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap, useMapEvents, GeoJSON } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';
import { stationsApi, crimesApi, routingApi, osmApi } from '../services/api';
import type { PoliceStation, CrimeEvent, RouteResponse } from '../types';
import RiskHeatmap from './RiskHeatmap';
import RouteDisplay from './RouteDisplay';
import policeStationIcon from '../assets/icons/PoliceStation.png';

// Fix for default marker icons in React-Leaflet
const iconRetinaUrl = 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png';
const iconUrl = 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png';
const shadowUrl = 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  tooltipAnchor: [16, -28],
  shadowSize: [41, 41],
});

L.Marker.prototype.options.icon = DefaultIcon;

// Police station icon (custom)
const stationIcon = L.icon({
  iconUrl: policeStationIcon,
  shadowUrl,
  iconSize: [50, 75], // Scaled down from 1024x1536 (maintaining aspect ratio)
  iconAnchor: [25, 75], // Center horizontally, bottom vertically
  popupAnchor: [0, -75],
});

// Crime severity colors
const getCrimeColor = (severity: number): string => {
  if (severity >= 4) return 'red';
  if (severity >= 3) return 'orange';
  if (severity >= 2) return 'yellow';
  return 'green';
};

const createCrimeIcon = (severity: number) => {
  return L.icon({
    iconUrl: `https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-${getCrimeColor(severity)}.png`,
    shadowUrl,
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
  });
};

const MapContent = ({ 
  onMapClick, 
  mapClickMode,
  selectedLocationMarker,
  setSelectedLocationMarker
}: { 
  onMapClick?: (lat: number, lng: number) => void;
  mapClickMode?: boolean;
  selectedLocationMarker?: L.Marker | null;
  setSelectedLocationMarker?: (marker: L.Marker | null) => void;
}) => {
  const map = useMap();
  
  useEffect(() => {
    map.invalidateSize();
  }, [map]);

  // Add/remove marker when selectedLocationMarker changes
  useEffect(() => {
    if (selectedLocationMarker) {
      selectedLocationMarker.addTo(map);
      return () => {
        map.removeLayer(selectedLocationMarker);
      };
    }
  }, [selectedLocationMarker, map]);

  // Handle map click events
  useEffect(() => {
    if (!mapClickMode || !onMapClick) return;
    
    const handleClick = (e: L.LeafletMouseEvent) => {
      if (mapClickMode && onMapClick) {
        onMapClick(e.latlng.lat, e.latlng.lng);
      }
    };
    
    map.on('click', handleClick);
    
    return () => {
      map.off('click', handleClick);
    };
  }, [map, mapClickMode, onMapClick]);

  return null;
};

// Crime Marker Cluster Component
const CrimeMarkerCluster = ({ crimes }: { crimes: CrimeEvent[] }) => {
  const map = useMap();
  const clusterRef = useRef<any>(null);

  useEffect(() => {
    let isCancelled = false;

    if (!map || crimes.length === 0) {
      // Clean up if no crimes
      if (clusterRef.current) {
        map.removeLayer(clusterRef.current);
        clusterRef.current = null;
      }
      return;
    }

    // Load markercluster dynamically
    import('leaflet.markercluster').then((module) => {
      if (isCancelled) {
        return;
      }
      const MCG = (module as any).default?.MarkerClusterGroup || 
                  (module as any).MarkerClusterGroup ||
                  (L as any).markerClusterGroup;
      
      if (!MCG) {
        console.error('MarkerClusterGroup not available');
        return;
      }

      // Create marker cluster group
      const markerClusterGroup = new MCG({
        chunkedLoading: true,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        zoomToBoundsOnClick: true,
        maxClusterRadius: 50,
        iconCreateFunction: (cluster: any) => {
          const count = cluster.getChildCount();
          return L.divIcon({
            html: `<div style="background-color: rgba(220, 38, 38, 0.8); color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 3px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${count}</div>`,
            className: 'marker-cluster-custom',
            iconSize: L.point(40, 40),
          });
        },
      });

      // Add markers to cluster
      crimes
        .filter((crime) => crime.lat && crime.lng && crime.lat !== 0 && crime.lng !== 0)
        .forEach((crime) => {
          const circleMarker = L.circleMarker([crime.lat, crime.lng], {
            radius: 6,
            color: '#dc2626',
            fillColor: '#dc2626',
            fillOpacity: 0.8,
            weight: 2,
          });

          const popupContent = `
            <div>
              <strong>${crime.crime_type}</strong><br/>
              Şiddet: ${crime.severity}/5<br/>
              ${crime.street_name ? `Sokak: ${crime.street_name}<br/>` : ''}
              Tarih: ${new Date(crime.event_time).toLocaleString('tr-TR')}<br/>
            </div>
          `;
          circleMarker.bindPopup(popupContent);
          markerClusterGroup.addLayer(circleMarker);
        });

      if (!isCancelled) {
        map.addLayer(markerClusterGroup);
        clusterRef.current = markerClusterGroup;
      }
    }).catch((err) => {
      console.error('Failed to load leaflet.markercluster:', err);
    });

    return () => {
      isCancelled = true;
      if (clusterRef.current) {
        map.removeLayer(clusterRef.current);
        clusterRef.current = null;
      }
    };
  }, [map, crimes]);

  return null;
};

const Map = () => {
  const [stations, setStations] = useState<PoliceStation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Traffic network state (OSM drive roads)
  const [showTrafficMap, setShowTrafficMap] = useState(false);
  const [trafficRoads, setTrafficRoads] = useState<any>(null);
  const [loadingTraffic, setLoadingTraffic] = useState(false);
  
  // Risk map state
  const [showRiskMap, setShowRiskMap] = useState(false);
  const [riskStartTime, setRiskStartTime] = useState('');
  const [riskEndTime, setRiskEndTime] = useState('');
  const [riskThreshold, setRiskThreshold] = useState(0.3);
  
  // Crime events by date state
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [showCrimesByDate, setShowCrimesByDate] = useState(false);
  const [filteredCrimes, setFilteredCrimes] = useState<CrimeEvent[]>([]);
  const [loadingCrimes, setLoadingCrimes] = useState(false);
  
  // Route state
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [selectedStationId, setSelectedStationId] = useState<string>('');
  const [routeThreshold, setRouteThreshold] = useState(0.7);
  const [maxMinutes, setMaxMinutes] = useState(90);
  const [routeStartTime, setRouteStartTime] = useState('');
  const [routeEndTime, setRouteEndTime] = useState('');
  
  // Crime event creation state
  const [showAddCrimeForm, setShowAddCrimeForm] = useState(false);
  const [crimeFormData, setCrimeFormData] = useState({
    event_time: '',
    crime_type: '',
    severity: 1,
    lat: 0,
    lng: 0,
    snapped_lat: 0,
    snapped_lng: 0,
    isSnapped: false,
    snapping: false,
  });
  const [mapClickMode, setMapClickMode] = useState(false);
  const [selectedLocationMarker, setSelectedLocationMarker] = useState<L.Marker | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const stationsRes = await stationsApi.list(true);
        console.log('Stations API response:', stationsRes);
        console.log('Stations data:', stationsRes.data);
        const stationsData = stationsRes.data || [];
        console.log('Setting stations:', stationsData);
        setStations(stationsData);
        // Don't auto-select first station - let user choose
        // if (stationsRes.data.length > 0) {
        //   setSelectedStationId(stationsRes.data[0].id);
        // }
        setError(null);
      } catch (err) {
        setError('Veri yüklenirken hata oluştu');
        console.error('Error loading stations:', err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Load traffic network (OSM drive roads)
  const loadTrafficNetwork = useCallback(async () => {
    try {
      setLoadingTraffic(true);
      const response = await osmApi.getRoadNetwork({ limit: 10000 });
      setTrafficRoads(response.data);
      setError(null);
    } catch (err) {
      setError('Trafik ağı yüklenirken hata oluştu');
      console.error(err);
      setTrafficRoads(null);
    } finally {
      setLoadingTraffic(false);
    }
  }, []);

  // Load crimes by selected date
  const loadCrimesByDate = useCallback(async (date: string) => {
    if (!date) {
      setFilteredCrimes([]);
      return;
    }

    try {
      setLoadingCrimes(true);
      // Seçilen tarihin başlangıç ve bitiş zamanları (UTC)
      // date format: "YYYY-MM-DD"
      const startDate = new Date(date + 'T00:00:00Z');
      const endDate = new Date(date + 'T23:59:59Z');

      const response = await crimesApi.list({
        start_time: startDate.toISOString(),
        end_time: endDate.toISOString(),
        limit: 1000,
      });
      
      setFilteredCrimes(response.data);
      setError(null);
    } catch (err) {
      setError('Olaylar yüklenirken hata oluştu');
      console.error(err);
      setFilteredCrimes([]);
    } finally {
      setLoadingCrimes(false);
    }
  }, []);

  // Tarih değiştiğinde veya göster butonu aktif olduğunda crime'ları yükle
  useEffect(() => {
    if (showCrimesByDate && selectedDate) {
      loadCrimesByDate(selectedDate);
    } else {
      setFilteredCrimes([]);
    }
  }, [selectedDate, showCrimesByDate, loadCrimesByDate]);

  const handleOptimizeRoute = async () => {
    if (!selectedStationId) {
      alert('Lütfen bir karakol seçin');
      return;
    }

    if (!routeStartTime || !routeEndTime) {
      alert('Lütfen rota için başlangıç ve bitiş tarih/saat aralığını seçin');
      return;
    }

    try {
      setLoading(true);
      // Convert datetime-local format to ISO string
      const startTimeISO = new Date(routeStartTime).toISOString();
      const endTimeISO = new Date(routeEndTime).toISOString();
      
      const response = await routingApi.optimize({
        station_id: selectedStationId,
        risk_threshold: routeThreshold,
        max_minutes: maxMinutes,
        start_time: startTimeISO,
        end_time: endTimeISO,
      });
      setRoute(response.data);
      setError(null);
    } catch (err) {
      setError('Rota oluşturulurken hata oluştu');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Set default time range (next 24 hours)
  useEffect(() => {
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setHours(tomorrow.getHours() + 24);
    
    const nowStr = now.toISOString().slice(0, 16);
    const tomorrowStr = tomorrow.toISOString().slice(0, 16);
    
    setRiskStartTime(nowStr);
    setRiskEndTime(tomorrowStr);
    
    // Set default route time range (same as risk map)
    setRouteStartTime(nowStr);
    setRouteEndTime(tomorrowStr);
  }, []);

  // Küçükçekmece center coordinates
  const center: [number, number] = [41.015, 28.75];

  // Handle map click for crime location selection
  const handleMapClick = useCallback(async (lat: number, lng: number) => {
    if (!mapClickMode) return;
    
    setCrimeFormData(prev => ({
      ...prev,
      lat,
      lng,
      isSnapped: false,
      snapped_lat: 0,
      snapped_lng: 0,
    }));
    
    // Remove previous marker
    if (selectedLocationMarker) {
      selectedLocationMarker.remove();
      setSelectedLocationMarker(null);
    }
    
    // Add new marker
    const marker = L.marker([lat, lng], {
      icon: L.icon({
        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
        shadowUrl,
        iconSize: [25, 41],
        iconAnchor: [12, 41],
      }),
      draggable: true,
    });
    
    marker.on('dragend', async (e) => {
      const newPos = e.target.getLatLng();
      setCrimeFormData(prev => ({
        ...prev,
        lat: newPos.lat,
        lng: newPos.lng,
        isSnapped: false,
        snapped_lat: 0,
        snapped_lng: 0,
      }));
      
      // Try to snap new position
      try {
        setCrimeFormData(prev => ({ ...prev, snapping: true }));
        const snapResponse = await crimesApi.snapToRoad(newPos.lat, newPos.lng);
        setCrimeFormData(prev => ({
          ...prev,
          snapped_lat: snapResponse.data.snapped_lat,
          snapped_lng: snapResponse.data.snapped_lng,
          isSnapped: true,
          snapping: false,
        }));
      } catch (err: any) {
        console.error('Snap error:', err);
        setCrimeFormData(prev => ({ ...prev, snapping: false }));
      }
    });
    
    // Store marker
    setSelectedLocationMarker(marker);
    
    // Try to snap to road
    try {
      setCrimeFormData(prev => ({ ...prev, snapping: true }));
      const snapResponse = await crimesApi.snapToRoad(lat, lng);
      setCrimeFormData(prev => ({
        ...prev,
        snapped_lat: snapResponse.data.snapped_lat,
        snapped_lng: snapResponse.data.snapped_lng,
        isSnapped: true,
        snapping: false,
      }));
    } catch (err: any) {
      console.error('Snap error:', err);
      setCrimeFormData(prev => ({ ...prev, snapping: false }));
      if (err.response?.status !== 404) {
        alert('Yol segmentine snap edilirken hata oluştu');
      }
    }
  }, [mapClickMode, selectedLocationMarker, setSelectedLocationMarker]);

  // Handle crime form submission
  const handleCreateCrime = useCallback(async () => {
    if (!crimeFormData.event_time || !crimeFormData.crime_type || crimeFormData.lat === 0 || crimeFormData.lng === 0) {
      alert('Lütfen tüm alanları doldurun ve haritada bir konum seçin');
      return;
    }

    try {
      setLoading(true);
      
      // Use snapped coordinates if available, otherwise use original
      const finalLat = crimeFormData.isSnapped ? crimeFormData.snapped_lat : crimeFormData.lat;
      const finalLng = crimeFormData.isSnapped ? crimeFormData.snapped_lng : crimeFormData.lng;
      
      // Convert datetime-local to ISO string
      const eventTimeISO = new Date(crimeFormData.event_time).toISOString();
      
      const newCrime = await crimesApi.create({
        crime_type: crimeFormData.crime_type,
        severity: crimeFormData.severity,
        event_time: eventTimeISO,
        lat: finalLat,
        lng: finalLng,
        confidence_score: 1.0,
      });
      
      alert('Olay başarıyla eklendi!');
      
      // Reset form
      setCrimeFormData({
        event_time: '',
        crime_type: '',
        severity: 1,
        lat: 0,
        lng: 0,
        snapped_lat: 0,
        snapped_lng: 0,
        isSnapped: false,
        snapping: false,
      });
      setMapClickMode(false);
      setShowAddCrimeForm(false);
      
      // Remove marker
      if (selectedLocationMarker) {
        selectedLocationMarker.remove();
        setSelectedLocationMarker(null);
      }
      
      // Refresh crimes if showing
      if (showCrimesByDate && selectedDate) {
        await loadCrimesByDate(selectedDate);
      }
      
      setError(null);
    } catch (err: any) {
      console.error('Create crime error:', err);
      setError('Olay eklenirken hata oluştu: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  }, [crimeFormData, selectedLocationMarker, showCrimesByDate, selectedDate, loadCrimesByDate]);

  return (
    <div style={{ width: '100%', height: '100vh', position: 'relative' }}>
      {/* Control Panel */}
      <div
        style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          zIndex: 1000,
          background: '#1a1a1a',
          padding: '24px',
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.4), 0 2px 8px rgba(0,0,0,0.3)',
          maxWidth: '380px',
          maxHeight: 'calc(100vh - 20px)',
          overflowY: 'auto',
          border: '1px solid #333',
        }}
      >
        <h3 style={{ 
          marginTop: 0, 
          marginBottom: '20px', 
          fontSize: '20px',
          fontWeight: '600',
          color: '#ffffff',
          letterSpacing: '0.5px'
        }}>
          Kontrol Paneli
        </h3>
        
        {/* Add Crime Event Form */}
        <div style={{ marginBottom: '24px', paddingBottom: '24px', borderBottom: '1px solid #333' }}>
          <h4 style={{ 
            marginTop: 0, 
            marginBottom: '12px', 
            fontSize: '15px',
            fontWeight: '600',
            color: '#ffffff'
          }}>
            Olay Ekle
          </h4>
          {!showAddCrimeForm ? (
            <button
              onClick={() => {
                setShowAddCrimeForm(true);
                setMapClickMode(true);
                // Set default event time to now
                const now = new Date();
                const localDateTime = new Date(now.getTime() - now.getTimezoneOffset() * 60000)
                  .toISOString()
                  .slice(0, 16);
                setCrimeFormData(prev => ({ ...prev, event_time: localDateTime }));
              }}
              style={{
                width: '100%',
                padding: '10px',
                background: 'linear-gradient(135deg, #0066ff 0%, #0052cc 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                boxShadow: '0 2px 8px rgba(0, 102, 255, 0.3)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #0052cc 0%, #003d99 100%)';
                e.currentTarget.style.transform = 'translateY(-1px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 102, 255, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #0066ff 0%, #0052cc 100%)';
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 102, 255, 0.3)';
              }}
            >
              Yeni Olay Ekle
            </button>
          ) : (
            <div>
              <div style={{ marginBottom: '12px' }}>
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
                  Tarih ve Saat: *
                </label>
                <input
                  type="datetime-local"
                  value={crimeFormData.event_time}
                  onChange={(e) => setCrimeFormData(prev => ({ ...prev, event_time: e.target.value }))}
                  style={{ 
                    width: '100%', 
                    padding: '8px', 
                    fontSize: '12px', 
                    boxSizing: 'border-box',
                    background: '#2a2a2a',
                    border: '1px solid #444',
                    borderRadius: '6px',
                    color: '#ffffff',
                    outline: 'none',
                    transition: 'border-color 0.2s ease'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#0066ff'}
                  onBlur={(e) => e.target.style.borderColor = '#444'}
                />
              </div>
              
              <div style={{ marginBottom: '12px' }}>
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
                  Suç Türü: *
                </label>
                <select
                  value={crimeFormData.crime_type}
                  onChange={(e) => setCrimeFormData(prev => ({ ...prev, crime_type: e.target.value }))}
                  style={{ 
                    width: '100%', 
                    padding: '8px', 
                    fontSize: '12px', 
                    boxSizing: 'border-box',
                    background: '#2a2a2a',
                    border: '1px solid #444',
                    borderRadius: '6px',
                    color: '#ffffff',
                    outline: 'none',
                    cursor: 'pointer',
                    transition: 'border-color 0.2s ease'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#0066ff'}
                  onBlur={(e) => e.target.style.borderColor = '#444'}
                >
                  <option value="">-- Seçiniz --</option>
                  <option value="kavga">Kavga</option>
                  <option value="saldırı">Saldırı</option>
                  <option value="cinayet">Cinayet</option>
                  <option value="çete olayı">Çete Olayı</option>
                  <option value="gasp">Gasp</option>
                  <option value="hırsızlık">Hırsızlık</option>
                  <option value="kundaklama">Kundaklama</option>
                  <option value="uyuşturucu">Uyuşturucu</option>
                </select>
              </div>
              
              <div style={{ marginBottom: '12px' }}>
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
                  Severity (Önem): <span style={{ color: '#0066ff', fontWeight: '600' }}>{crimeFormData.severity}</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={crimeFormData.severity}
                  onChange={(e) => setCrimeFormData(prev => ({ ...prev, severity: parseInt(e.target.value) }))}
                  style={{ 
                    width: '100%',
                    height: '6px',
                    background: '#333',
                    borderRadius: '3px',
                    outline: 'none',
                    cursor: 'pointer',
                    WebkitAppearance: 'none',
                  }}
                />
                <div style={{ fontSize: '10px', color: '#999', marginTop: '4px', display: 'flex', justifyContent: 'space-between' }}>
                  <span>1 (Düşük)</span>
                  <span>5 (Kritik)</span>
                </div>
              </div>
              
              <div style={{ marginBottom: '12px' }}>
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
                  Konum:
                </label>
                {mapClickMode ? (
                  <div style={{ fontSize: '11px', color: '#ff8800', marginBottom: '8px', padding: '8px', background: '#2a1f00', borderRadius: '4px', border: '1px solid #664400' }}>
                    ⚠️ Haritada bir konum tıklayın
                  </div>
                ) : crimeFormData.lat !== 0 && crimeFormData.lng !== 0 ? (
                  <div style={{ fontSize: '11px', color: '#4CAF50', marginBottom: '8px', padding: '8px', background: '#1a2a1a', borderRadius: '4px', border: '1px solid #2a4a2a' }}>
                    ✓ Konum seçildi: {crimeFormData.lat.toFixed(6)}, {crimeFormData.lng.toFixed(6)}
                    {crimeFormData.isSnapped && (
                      <div style={{ fontSize: '10px', color: '#66b3ff', marginTop: '4px' }}>
                        → Yola snap edildi: {crimeFormData.snapped_lat.toFixed(6)}, {crimeFormData.snapped_lng.toFixed(6)}
                      </div>
                    )}
                    {crimeFormData.snapping && (
                      <div style={{ fontSize: '10px', color: '#999', marginTop: '4px' }}>
                        Yola snap ediliyor...
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: '#999', marginBottom: '8px', padding: '8px', background: '#2a2a2a', borderRadius: '4px', border: '1px solid #444' }}>
                    Konum seçilmedi
                  </div>
                )}
                <button
                  onClick={() => {
                    setMapClickMode(!mapClickMode);
                    if (mapClickMode) {
                      // Cancel mode - remove marker
                      if (selectedLocationMarker) {
                        selectedLocationMarker.remove();
                        setSelectedLocationMarker(null);
                      }
                      setCrimeFormData(prev => ({
                        ...prev,
                        lat: 0,
                        lng: 0,
                        snapped_lat: 0,
                        snapped_lng: 0,
                        isSnapped: false,
                      }));
                    }
                  }}
                  style={{
                    width: '100%',
                    padding: '8px',
                    background: mapClickMode ? 'linear-gradient(135deg, #ff4444 0%, #cc0000 100%)' : 'linear-gradient(135deg, #2196F3 0%, #1976D2 100%)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '12px',
                    fontWeight: '500',
                    transition: 'all 0.2s ease',
                    boxShadow: mapClickMode ? '0 2px 8px rgba(255, 68, 68, 0.3)' : '0 2px 8px rgba(33, 150, 243, 0.3)',
                  }}
                  onMouseEnter={(e) => {
                    if (!mapClickMode) {
                      e.currentTarget.style.background = 'linear-gradient(135deg, #1976D2 0%, #1565C0 100%)';
                      e.currentTarget.style.transform = 'translateY(-1px)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!mapClickMode) {
                      e.currentTarget.style.background = 'linear-gradient(135deg, #2196F3 0%, #1976D2 100%)';
                      e.currentTarget.style.transform = 'translateY(0)';
                    }
                  }}
                >
                  {mapClickMode ? 'Konum Seçimini İptal Et' : 'Konum Seç'}
                </button>
              </div>
              
              <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                <button
                  onClick={handleCreateCrime}
                  disabled={loading || !crimeFormData.event_time || !crimeFormData.crime_type || crimeFormData.lat === 0}
                  style={{
                    flex: 1,
                    padding: '10px',
                    background: (loading || !crimeFormData.event_time || !crimeFormData.crime_type || crimeFormData.lat === 0) 
                      ? '#444' 
                      : 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: (loading || !crimeFormData.event_time || !crimeFormData.crime_type || crimeFormData.lat === 0) ? 'not-allowed' : 'pointer',
                    fontSize: '13px',
                    fontWeight: '600',
                    transition: 'all 0.2s ease',
                    boxShadow: (loading || !crimeFormData.event_time || !crimeFormData.crime_type || crimeFormData.lat === 0) 
                      ? 'none' 
                      : '0 2px 8px rgba(76, 175, 80, 0.3)',
                  }}
                  onMouseEnter={(e) => {
                    if (!loading && crimeFormData.event_time && crimeFormData.crime_type && crimeFormData.lat !== 0) {
                      e.currentTarget.style.transform = 'translateY(-1px)';
                      e.currentTarget.style.boxShadow = '0 4px 12px rgba(76, 175, 80, 0.4)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!loading && crimeFormData.event_time && crimeFormData.crime_type && crimeFormData.lat !== 0) {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 2px 8px rgba(76, 175, 80, 0.3)';
                    }
                  }}
                >
                  {loading ? 'Kaydediliyor...' : 'Kaydet'}
                </button>
                <button
                  onClick={() => {
                    setShowAddCrimeForm(false);
                    setMapClickMode(false);
                    setCrimeFormData({
                      event_time: '',
                      crime_type: '',
                      severity: 1,
                      lat: 0,
                      lng: 0,
                      snapped_lat: 0,
                      snapped_lng: 0,
                      isSnapped: false,
                      snapping: false,
                    });
                    if (selectedLocationMarker) {
                      selectedLocationMarker.remove();
                      setSelectedLocationMarker(null);
                    }
                  }}
                  style={{
                    flex: 1,
                    padding: '10px',
                    background: 'linear-gradient(135deg, #666 0%, #555 100%)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '500',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'linear-gradient(135deg, #555 0%, #444 100%)';
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'linear-gradient(135deg, #666 0%, #555 100%)';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  İptal
                </button>
              </div>
            </div>
          )}
        </div>
        
        {/* Crime Events by Date Controls */}
        <div style={{ marginBottom: '24px', paddingBottom: '24px', borderBottom: '1px solid #333' }}>
          <h4 style={{ 
            marginTop: 0, 
            marginBottom: '12px', 
            fontSize: '15px',
            fontWeight: '600',
            color: '#ffffff'
          }}>
            Olayları Göster
          </h4>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
              Tarih Seç:
            </label>
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => {
                setSelectedDate(e.target.value);
              }}
              max={new Date().toISOString().split('T')[0]} // Bugünden önceki tarihler
              style={{ 
                width: '100%', 
                padding: '8px', 
                fontSize: '12px', 
                boxSizing: 'border-box',
                background: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#ffffff',
                outline: 'none',
                transition: 'border-color 0.2s ease'
              }}
              onFocus={(e) => e.target.style.borderColor = '#0066ff'}
              onBlur={(e) => e.target.style.borderColor = '#444'}
            />
          </div>
          <button
            onClick={async () => {
              if (selectedDate) {
                const newShowState = !showCrimesByDate;
                setShowCrimesByDate(newShowState);
                if (newShowState) {
                  // Butona basıldığında crime'ları yükle
                  await loadCrimesByDate(selectedDate);
                }
              } else {
                alert('Lütfen bir tarih seçin');
              }
            }}
            disabled={!selectedDate || loadingCrimes}
            style={{
              width: '100%',
              padding: '10px',
              background: (!selectedDate || loadingCrimes) 
                ? '#444' 
                : (showCrimesByDate 
                  ? 'linear-gradient(135deg, #ff4444 0%, #cc0000 100%)' 
                  : 'linear-gradient(135deg, #ff8800 0%, #ff6600 100%)'),
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: (!selectedDate || loadingCrimes) ? 'not-allowed' : 'pointer',
              fontSize: '13px',
              fontWeight: '600',
              transition: 'all 0.2s ease',
              boxShadow: (!selectedDate || loadingCrimes) 
                ? 'none' 
                : (showCrimesByDate 
                  ? '0 2px 8px rgba(255, 68, 68, 0.3)' 
                  : '0 2px 8px rgba(255, 136, 0, 0.3)'),
            }}
            onMouseEnter={(e) => {
              if (!(!selectedDate || loadingCrimes)) {
                e.currentTarget.style.transform = 'translateY(-1px)';
                if (showCrimesByDate) {
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(255, 68, 68, 0.4)';
                } else {
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(255, 136, 0, 0.4)';
                }
              }
            }}
            onMouseLeave={(e) => {
              if (!(!selectedDate || loadingCrimes)) {
                e.currentTarget.style.transform = 'translateY(0)';
                if (showCrimesByDate) {
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(255, 68, 68, 0.3)';
                } else {
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(255, 136, 0, 0.3)';
                }
              }
            }}
          >
            {loadingCrimes ? 'Yükleniyor...' : (showCrimesByDate ? 'Olayları Gizle' : 'Olayları Göster')}
          </button>
          {showCrimesByDate && !loadingCrimes && filteredCrimes.length > 0 && (
            <div style={{ marginTop: '12px', fontSize: '11px', color: '#999', textAlign: 'center', padding: '8px', background: '#2a2a2a', borderRadius: '4px' }}>
              {filteredCrimes.length} olay bulundu
            </div>
          )}
          {showCrimesByDate && !loadingCrimes && filteredCrimes.length === 0 && selectedDate && (
            <div style={{ marginTop: '12px', fontSize: '11px', color: '#ff6666', textAlign: 'center', padding: '8px', background: '#2a1a1a', borderRadius: '4px', border: '1px solid #442222' }}>
              Bu tarihte olay bulunamadı
            </div>
          )}
        </div>
        
        {/* Traffic Network Controls */}
        <div style={{ marginBottom: '24px', paddingBottom: '24px', borderBottom: '1px solid #333' }}>
          <h4 style={{ 
            marginTop: 0, 
            marginBottom: '12px', 
            fontSize: '15px',
            fontWeight: '600',
            color: '#ffffff'
          }}>
            Trafik Ağı
          </h4>
          <p style={{ margin: 0, marginBottom: '12px', fontSize: '11px', color: '#999' }}>
            OSM'den çekilen drive araç yollarını gösterir.
          </p>
          <button
            onClick={async () => {
              if (!showTrafficMap) {
                await loadTrafficNetwork();
              }
              setShowTrafficMap(!showTrafficMap);
            }}
            disabled={loadingTraffic}
            style={{
              width: '100%',
              padding: '10px',
              background: loadingTraffic
                ? '#444' 
                : (showTrafficMap 
                  ? 'linear-gradient(135deg, #ff4444 0%, #cc0000 100%)' 
                  : 'linear-gradient(135deg, #0066ff 0%, #0052cc 100%)'),
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: loadingTraffic ? 'not-allowed' : 'pointer',
              fontSize: '13px',
              fontWeight: '600',
              transition: 'all 0.2s ease',
              boxShadow: loadingTraffic
                ? 'none' 
                : (showTrafficMap 
                  ? '0 2px 8px rgba(255, 68, 68, 0.3)' 
                  : '0 2px 8px rgba(0, 102, 255, 0.3)'),
            }}
            onMouseEnter={(e) => {
              if (!loadingTraffic) {
                e.currentTarget.style.transform = 'translateY(-1px)';
                if (showTrafficMap) {
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(255, 68, 68, 0.4)';
                } else {
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 102, 255, 0.4)';
                }
              }
            }}
            onMouseLeave={(e) => {
              if (!loadingTraffic) {
                e.currentTarget.style.transform = 'translateY(0)';
                if (showTrafficMap) {
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(255, 68, 68, 0.3)';
                } else {
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 102, 255, 0.3)';
                }
              }
            }}
          >
            {loadingTraffic ? 'Yükleniyor...' : (showTrafficMap ? 'Trafiği Gizle' : 'Trafiği Göster')}
          </button>
          {showTrafficMap && trafficRoads && (
            <div style={{ marginTop: '12px', fontSize: '11px', color: '#999', textAlign: 'center', padding: '8px', background: '#2a2a2a', borderRadius: '4px' }}>
              {trafficRoads.total} yol segmenti gösteriliyor
            </div>
          )}
        </div>
        
        {/* Risk Map Controls */}
        <div style={{ marginBottom: '24px', paddingBottom: '24px', borderBottom: '1px solid #333' }}>
          <h4 style={{ 
            marginTop: 0, 
            marginBottom: '12px', 
            fontSize: '15px',
            fontWeight: '600',
            color: '#ffffff'
          }}>
            Risk Haritası
          </h4>
          <p style={{ margin: 0, marginBottom: '12px', fontSize: '11px', color: '#999' }}>
            OSM drive yolları üzerindeki anlık / öngörülen risk yoğunluğunu gösterir.
          </p>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
              Başlangıç:
            </label>
            <input
              type="datetime-local"
              value={riskStartTime}
              onChange={(e) => setRiskStartTime(e.target.value)}
              style={{ 
                width: '100%', 
                padding: '8px', 
                fontSize: '12px',
                boxSizing: 'border-box',
                background: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#ffffff',
                outline: 'none',
                transition: 'border-color 0.2s ease'
              }}
              onFocus={(e) => e.target.style.borderColor = '#0066ff'}
              onBlur={(e) => e.target.style.borderColor = '#444'}
            />
          </div>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
              Bitiş:
            </label>
            <input
              type="datetime-local"
              value={riskEndTime}
              onChange={(e) => setRiskEndTime(e.target.value)}
              style={{ 
                width: '100%', 
                padding: '8px', 
                fontSize: '12px',
                boxSizing: 'border-box',
                background: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#ffffff',
                outline: 'none',
                transition: 'border-color 0.2s ease'
              }}
              onFocus={(e) => e.target.style.borderColor = '#0066ff'}
              onBlur={(e) => e.target.style.borderColor = '#444'}
            />
          </div>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
              Eşik: <span style={{ color: '#0066ff', fontWeight: '600' }}>{(riskThreshold * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={riskThreshold}
              onChange={(e) => setRiskThreshold(parseFloat(e.target.value))}
              style={{ 
                width: '100%',
                height: '6px',
                background: '#333',
                borderRadius: '3px',
                outline: 'none',
                cursor: 'pointer',
                WebkitAppearance: 'none',
              }}
            />
          </div>
          <button
            onClick={() => {
              if (riskStartTime && riskEndTime) {
                setShowRiskMap(!showRiskMap);
              } else {
                alert('Lütfen başlangıç ve bitiş tarihlerini seçin');
              }
            }}
            disabled={!riskStartTime || !riskEndTime}
            style={{
              width: '100%',
              padding: '10px',
              background: (!riskStartTime || !riskEndTime) 
                ? '#444' 
                : (showRiskMap 
                  ? 'linear-gradient(135deg, #ff4444 0%, #cc0000 100%)' 
                  : 'linear-gradient(135deg, #0066ff 0%, #0052cc 100%)'),
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: (!riskStartTime || !riskEndTime) ? 'not-allowed' : 'pointer',
              fontSize: '13px',
              fontWeight: '600',
              transition: 'all 0.2s ease',
              boxShadow: (!riskStartTime || !riskEndTime) 
                ? 'none' 
                : (showRiskMap 
                  ? '0 2px 8px rgba(255, 68, 68, 0.3)' 
                  : '0 2px 8px rgba(0, 102, 255, 0.3)'),
            }}
            onMouseEnter={(e) => {
              if (!(!riskStartTime || !riskEndTime)) {
                e.currentTarget.style.transform = 'translateY(-1px)';
                if (showRiskMap) {
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(255, 68, 68, 0.4)';
                } else {
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 102, 255, 0.4)';
                }
              }
            }}
            onMouseLeave={(e) => {
              if (!(!riskStartTime || !riskEndTime)) {
                e.currentTarget.style.transform = 'translateY(0)';
                if (showRiskMap) {
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(255, 68, 68, 0.3)';
                } else {
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 102, 255, 0.3)';
                }
              }
            }}
          >
            {showRiskMap ? 'Risk Haritasını Gizle' : 'Risk Haritasını Göster'}
            {riskStartTime && riskEndTime && new Date(riskStartTime) > new Date() && ' (Forecast)'}
          </button>
        </div>

        {/* Route Optimization Controls */}
        <div>
          <h4 style={{ 
            marginTop: 0, 
            marginBottom: '12px', 
            fontSize: '15px',
            fontWeight: '600',
            color: '#ffffff'
          }}>
            Rota Optimizasyonu
          </h4>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
              Başlangıç Karakolu:
            </label>
            <select
              value={selectedStationId}
              onChange={(e) => setSelectedStationId(e.target.value)}
              style={{ 
                width: '100%', 
                padding: '8px', 
                fontSize: '12px', 
                boxSizing: 'border-box',
                background: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#ffffff',
                outline: 'none',
                cursor: 'pointer',
                transition: 'border-color 0.2s ease'
              }}
              onFocus={(e) => e.target.style.borderColor = '#0066ff'}
              onBlur={(e) => e.target.style.borderColor = '#444'}
            >
              <option value="">-- Karakol Seçin --</option>
              {stations.length > 0 ? (
                stations.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))
              ) : (
                <option value="" disabled>
                  {loading ? 'Yükleniyor...' : 'Karakol bulunamadı'}
                </option>
              )}
            </select>
            {stations.length === 0 && !loading && (
              <div style={{ marginTop: '6px', fontSize: '11px', color: '#ff6666', padding: '6px', background: '#2a1a1a', borderRadius: '4px', border: '1px solid #442222' }}>
                Karakollar yüklenemedi. Lütfen sayfayı yenileyin.
              </div>
            )}
          </div>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
              Başlangıç Tarih/Saat:
            </label>
            <input
              type="datetime-local"
              value={routeStartTime}
              onChange={(e) => setRouteStartTime(e.target.value)}
              style={{ 
                width: '100%', 
                padding: '8px', 
                fontSize: '12px', 
                boxSizing: 'border-box',
                background: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#ffffff',
                outline: 'none',
                transition: 'border-color 0.2s ease'
              }}
              onFocus={(e) => e.target.style.borderColor = '#0066ff'}
              onBlur={(e) => e.target.style.borderColor = '#444'}
            />
          </div>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
              Bitiş Tarih/Saat:
            </label>
            <input
              type="datetime-local"
              value={routeEndTime}
              onChange={(e) => setRouteEndTime(e.target.value)}
              style={{ 
                width: '100%', 
                padding: '8px', 
                fontSize: '12px', 
                boxSizing: 'border-box',
                background: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#ffffff',
                outline: 'none',
                transition: 'border-color 0.2s ease'
              }}
              onFocus={(e) => e.target.style.borderColor = '#0066ff'}
              onBlur={(e) => e.target.style.borderColor = '#444'}
            />
          </div>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
              Risk Eşiği: <span style={{ color: '#0066ff', fontWeight: '600' }}>{(routeThreshold * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={routeThreshold}
              onChange={(e) => setRouteThreshold(parseFloat(e.target.value))}
              style={{ 
                width: '100%',
                height: '6px',
                background: '#333',
                borderRadius: '3px',
                outline: 'none',
                cursor: 'pointer',
                WebkitAppearance: 'none',
              }}
            />
          </div>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '12px', color: '#cccccc', fontWeight: '500' }}>
              Maksimum Süre: <span style={{ color: '#0066ff', fontWeight: '600' }}>{maxMinutes} dk</span>
            </label>
            <input
              type="range"
              min="30"
              max="180"
              step="15"
              value={maxMinutes}
              onChange={(e) => setMaxMinutes(parseInt(e.target.value))}
              style={{ 
                width: '100%',
                height: '6px',
                background: '#333',
                borderRadius: '3px',
                outline: 'none',
                cursor: 'pointer',
                WebkitAppearance: 'none',
              }}
            />
          </div>
          <button
            onClick={handleOptimizeRoute}
            disabled={loading || !selectedStationId || !routeStartTime || !routeEndTime}
            style={{
              width: '100%',
              padding: '10px',
              background: (loading || !selectedStationId || !routeStartTime || !routeEndTime) 
                ? '#444' 
                : 'linear-gradient(135deg, #00aa00 0%, #008800 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: (loading || !selectedStationId || !routeStartTime || !routeEndTime) ? 'not-allowed' : 'pointer',
              fontSize: '13px',
              fontWeight: '600',
              transition: 'all 0.2s ease',
              boxShadow: (loading || !selectedStationId || !routeStartTime || !routeEndTime) 
                ? 'none' 
                : '0 2px 8px rgba(0, 170, 0, 0.3)',
            }}
            onMouseEnter={(e) => {
              if (!(loading || !selectedStationId || !routeStartTime || !routeEndTime)) {
                e.currentTarget.style.transform = 'translateY(-1px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 170, 0, 0.4)';
              }
            }}
            onMouseLeave={(e) => {
              if (!(loading || !selectedStationId || !routeStartTime || !routeEndTime)) {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 170, 0, 0.3)';
              }
            }}
          >
            {loading ? 'Yükleniyor...' : 'Rota Oluştur'}
          </button>
          {route && (
            <button
              onClick={() => setRoute(null)}
              style={{
                width: '100%',
                padding: '10px',
                marginTop: '10px',
                background: 'linear-gradient(135deg, #ff4444 0%, #cc0000 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                boxShadow: '0 2px 8px rgba(255, 68, 68, 0.3)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-1px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(255, 68, 68, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 2px 8px rgba(255, 68, 68, 0.3)';
              }}
            >
              Rotayı Temizle
            </button>
          )}
        </div>
      </div>

      {loading && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 1000,
            background: 'white',
            padding: '10px 20px',
            borderRadius: '5px',
            boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
          }}
        >
          Yükleniyor...
        </div>
      )}
      {error && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 1000,
            background: '#ff4444',
            color: 'white',
            padding: '10px 20px',
            borderRadius: '5px',
            boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
          }}
        >
          {error}
        </div>
      )}
      <MapContainer
        center={center}
        zoom={13}
        minZoom={11}
        maxZoom={18}
        style={{ width: '100%', height: '100%' }}
        scrollWheelZoom={true}
        bounds={[[40.98, 28.70], [41.05, 28.80]]}
        maxBounds={[[40.95, 28.65], [41.08, 28.85]]}
      >
        <MapContent 
          onMapClick={handleMapClick} 
          mapClickMode={mapClickMode}
          selectedLocationMarker={selectedLocationMarker}
          setSelectedLocationMarker={setSelectedLocationMarker}
        />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Traffic Network (OSM Drive Roads) */}
        {showTrafficMap && trafficRoads && (
          <GeoJSON
            data={trafficRoads}
            style={(feature) => {
              const roadType = feature?.properties?.road_type || 'unclassified';
              const colors: { [key: string]: string } = {
                motorway: '#ff0000',
                trunk: '#ff6600',
                primary: '#ffaa00',
                secondary: '#ffff00',
                tertiary: '#aaff00',
                residential: '#00ff00',
                service: '#00aaff',
                unclassified: '#666666',
              };
              return {
                color: colors[roadType] || '#666666',
                weight: roadType === 'motorway' ? 4 : roadType === 'trunk' ? 3 : roadType === 'primary' ? 3 : 2,
                opacity: 0.7,
              };
            }}
            onEachFeature={(feature, layer) => {
              if (feature.properties) {
                const props = feature.properties;
                const popupContent = `
                  <div>
                    <strong>Yol Tipi:</strong> ${props.road_type || 'Bilinmiyor'}<br/>
                    ${props.speed_limit ? `<strong>Hız Limiti:</strong> ${props.speed_limit} km/h<br/>` : ''}
                    <strong>Tek Yön:</strong> ${props.one_way ? 'Evet' : 'Hayır'}<br/>
                    ${props.risk_score > 0 ? `<strong>Risk Skoru:</strong> ${(props.risk_score * 100).toFixed(1)}%` : ''}
                  </div>
                `;
                layer.bindPopup(popupContent);
              }
            }}
          />
        )}

        {/* Risk Heatmap */}
        {showRiskMap && riskStartTime && riskEndTime && (
          <RiskHeatmap
            startTime={riskStartTime}
            endTime={riskEndTime}
            threshold={riskThreshold}
          />
        )}

        {/* Route Display */}
        {route && <RouteDisplay route={route} stationIcon={stationIcon} />}

        {/* Police Stations */}
        {stations.length > 0 && stations.map((station) => {
          // Validate coordinates before rendering
          if (!station.lat || !station.lng || station.lat === 0 || station.lng === 0) {
            console.warn(`Invalid coordinates for station ${station.name}:`, station);
            return null;
          }
          return (
            <Marker
              key={station.id}
              position={[station.lat, station.lng]}
              icon={stationIcon}
            >
              <Popup>
                <div>
                  <strong>{station.name}</strong>
                  <br />
                  Kapasite: {station.capacity}
                  <br />
                  Durum: {station.active ? 'Aktif' : 'Pasif'}
                </div>
              </Popup>
            </Marker>
          );
        })}

        {/* Crime Events - Show filtered crimes if date is selected, otherwise show none */}
        {showCrimesByDate && filteredCrimes.length > 0 && (
          <CrimeMarkerCluster crimes={filteredCrimes} />
        )}
      </MapContainer>
    </div>
  );
};

export default Map;
