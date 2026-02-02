# Küçükçekmece Sınırları Yapılandırması

## Genel Bakış

Sistem **Küçükçekmece, İstanbul** ilçesi sınırları içinde çalışacak şekilde yapılandırılmıştır. Sınırlar **polygon** olarak veritabanında saklanır ve OSM'den otomatik olarak import edilir.

## Polygon Boundary

Küçükçekmece ilçe sınırları OSM'den çekilip `administrative_boundary` tablosunda polygon olarak saklanır. Bu sayede daha hassas sınır kontrolleri yapılabilir.

### Fallback Bounding Box (Polygon Yoksa)

Polygon yüklenmemişse, fallback olarak şu bbox kullanılır:

```
Min Latitude:  40.98
Min Longitude: 28.70
Max Latitude:  41.05
Max Longitude: 28.80
```

## Uygulanan Sınırlamalar

### 1. Suç Olayları (Crime Events)
- Tüm API endpoint'leri sadece Küçükçekmece sınırları içindeki suç olaylarını döndürür
- `/api/v1/crimes` endpoint'i otomatik olarak filtreler
- Veritabanında 1505 kayıttan 1151'i Küçükçekmece sınırları içinde

### 2. Risk Haritası (Risk Map)
- Risk hücreleri sadece Küçükçekmece polygon sınırları içinde oluşturulur
- `/api/v1/forecast/risk-map` endpoint'i Küçükçekmece polygon'unu kullanır
- Grid oluşturma Küçükçekmece polygon sınırları ile sınırlandırılmıştır

### 3. Rota Optimizasyonu (Route Optimization)
- Risk hücreleri seçilirken Küçükçekmece sınırları kontrol edilir
- `/api/v1/routing/optimize` endpoint'i sadece Küçükçekmece içindeki risk hücrelerini kullanır

### 4. Frontend Harita
- Harita merkezi Küçükçekmece'ye ayarlanmıştır: `[41.015, 28.75]`
- Zoom seviyesi: 13 (Küçükçekmece için optimize)
- Min/Max bounds: Küçükçekmece sınırları

## Veritabanı Fonksiyonları

### `is_within_kucukcekmece(geom GEOGRAPHY)`
Küçükçekmece polygon sınırları içinde olup olmadığını kontrol eden fonksiyon. `administrative_boundary` tablosundan polygon'u okur.

**Kullanım:**
```sql
SELECT * FROM crime_event 
WHERE is_within_kucukcekmece(geom);
```

**Not:** Fonksiyon `administrative_boundary` tablosundan polygon'u okur. Eğer polygon yüklenmemişse `FALSE` döner.

### `crime_event_kucukcekmece` View
Küçükçekmece sınırları içindeki suç olaylarını gösteren view.

**Kullanım:**
```sql
SELECT * FROM crime_event_kucukcekmece;
```

## Yapılandırma

Boundary ayarları `backend/app/core/config.py` dosyasında:

```python
# Boundary name and admin level
kucukcekmece_boundary_name: str = "Küçükçekmece"
kucukcekmece_boundary_admin_level: int = 8

# Fallback bbox (used if polygon is not available)
kucukcekmece_fallback_bbox: Tuple[float, float, float, float] = (
    40.98,   # min_lat
    28.70,   # min_lng
    41.05,   # max_lat
    28.80    # max_lng
)
```

**Not:** Sistem artık polygon kullanır. Bbox sadece fallback olarak kullanılır.

## Boundary Import

Küçükçekmece boundary'si otomatik olarak import edilir. Manuel import için:

### Otomatik Import (Docker)

Docker container başlatıldığında otomatik olarak boundary import edilir.

### Manuel Import

**API Endpoint:**
```bash
# Boundary durumunu kontrol et
curl http://localhost:8000/api/v1/osm/boundary-status

# Boundary'yi import et
curl -X POST http://localhost:8000/api/v1/osm/import-boundary

# Force re-import
curl -X POST "http://localhost:8000/api/v1/osm/import-boundary?force=true"
```

**Script:**
```bash
python3 scripts/import_kucukcekmece_boundary.py
```

## OSM Verileri

OSM (OpenStreetMap) verilerini import ederken Küçükçekmece polygon sınırları içindeki yol ağı verilerini kullanın:

1. Overpass API ile Küçükçekmece polygon'u için veri çekin (otomatik)
2. `road_segment` tablosuna import edin (polygon içinde filtreleme yapılır)
3. `database/routing_setup.sql` script'ini çalıştırın

## Test

Küçükçekmece sınırlarının çalıştığını test etmek için:

```sql
-- Küçükçekmece içindeki suç olayları
SELECT COUNT(*) FROM crime_event 
WHERE is_within_kucukcekmece(geom);

-- API test
curl 'http://localhost:8000/api/v1/crimes?limit=10'
```

## Notlar
- ✅ Polygon desteği eklendi - Küçükçekmece sınırları artık polygon olarak saklanıyor
- Boundary OSM'den otomatik olarak import ediliyor
- Tüm filtreleme işlemleri polygon intersection kullanıyor
- Fallback bbox sadece polygon yüklenmemişse kullanılıyor