# Predictive Patrol Routing System (PPRS)

Suç risk tahmini, mekansal risk haritası üretimi ve devriye rotası optimizasyonu için uçtan uca platform.

## Öne Çıkanlar

- Spatio‑temporal risk haritası (Adaptive KDE + grid/hex hücreleri)
- Risk‑aware devriye rota optimizasyonu (tek merkez / çoklu merkez)
- Gerçek zamanlı risk güncellemesi (WebSocket + Redis cache)
- OSM tabanlı yol ağı ve otomatik pgRouting topology
- Küçükçekmece ilçe sınırları ile otomatik coğrafi validasyon
- Leaflet tabanlı interaktif harita arayüzü


## Teknoloji Stack

### Backend
- Python 3.11
- FastAPI
- SQLAlchemy

### Frontend
- React 18
- TypeScript
- Vite
- Leaflet / react-leaflet

### Database
- PostgreSQL 15+
- PostGIS 3.x
- pgRouting
- Redis

### Machine Learning
- scikit-learn

## Kurulum (Docker)

```bash
git clone https://github.com/GorkemPalali/PolicePatrol.git
cd PolicePatrol
cp env.example .env
```

`.env` içinde en azından aşağıdakileri güvenli şekilde güncelleyin:

```env
POSTGRES_PASSWORD=<güvenli_şifre_buraya>
DATABASE_URL=postgresql+psycopg2://police:<güvenli_şifre_buraya>@db:5432/policepatrol
```

Servisleri başlatın:

```bash
docker compose up -d
```

Sağlık kontrolü:

```bash
curl http://localhost:8000/api/v1/health
```

Servisler:
- Backend API: `http://localhost:8000`
- Frontend: `http://localhost:5173`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

## Lokal Geliştirme

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend, kök dizindeki `.env` dosyasını okur. Farklı bir konum kullanmak için `backend/app/core/config.py` dosyasını düzenleyin.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend için environment değişkenleri `VITE_` prefix'i ile başlamalıdır.

## Coğrafi Sınır ve Validasyon

Sistem Küçükçekmece ilçe sınırları içinde çalışır. Sınırlar OSM'den polygon olarak import edilir. Polygon bulunamazsa fallback bounding box kullanılır:

- Min Lat: 40.98
- Min Lng: 28.70
- Max Lat: 41.05
- Max Lng: 28.80

Sınır dışı koordinatlar API ve DB seviyesinde engellenir:
- API: CREATE/UPDATE çağrıları HTTP 400 döner
- DB: BEFORE INSERT/UPDATE trigger'ları ile koruma

Validasyon kapatma (development):

```env
STRICT_BOUNDARY_VALIDATION=false
```

Mevcut sınır dışı verileri temizleme:

```bash
python3 scripts/cleanup_out_of_boundary_data.py --dry-run
python3 scripts/cleanup_out_of_boundary_data.py --force
```

## OSM Verileri ve Routing

Docker başlangıcında otomatik olarak:
1. Küçükçekmece boundary import edilir
2. Overpass API üzerinden yol verileri alınır
3. `road_segment` tablosuna import edilir
4. pgRouting topology oluşturulur

Manuel import:

```bash
curl -X POST http://localhost:8000/api/v1/osm/import-boundary
curl -X POST http://localhost:8000/api/v1/osm/import
curl -X POST http://localhost:8000/api/v1/osm/refresh-topology?force=true
```

OSM ayarları `.env` veya `backend/app/core/config.py` üzerinden yönetilir:

```env
OVERPASS_API_URL=https://overpass-api.de/api/interpreter
OSM_IMPORT_ON_STARTUP=true
OSM_HIGHWAY_TAGS=motorway,trunk,primary,secondary,tertiary,residential,service,unclassified
OSM_TOPOLOGY_TOLERANCE=0.0001
```

## Veri Import

### Polis Karakolları

Küçükçekmece için 5 karakol başlangıç verisi mevcuttur ve veritabanı ilk kurulurken otomatik yüklenir.

Manuel yükleme:

```bash
# SQL script ile
docker compose exec db psql -U police -d policepatrol -f /docker-entrypoint-initdb.d/05-sample-data.sql
# veya host üzerinden
./scripts/load_police_stations.sh
```

### Suç Olayları

JSONL formatında import:

```bash
# Docker container içinde
docker compose cp /path/to/crime.jsonl backend:/tmp/crime.jsonl
docker compose cp scripts/import_crimes_jsonl.py backend:/app/import_crimes_jsonl.py
docker compose exec backend python /app/import_crimes_jsonl.py /tmp/crime.jsonl

# Local (venv aktifken)
cd backend
python ../scripts/import_crimes_jsonl.py /path/to/crime.jsonl
```

## API Özet

### Health
- `GET /api/v1/health`

### Police Stations
- `GET /api/v1/stations`
- `POST /api/v1/stations`
- `PATCH /api/v1/stations/{id}`
- `DELETE /api/v1/stations/{id}`

### Crime Events
- `GET /api/v1/crimes`
- `POST /api/v1/crimes`
- `PATCH /api/v1/crimes/{id}`
- `DELETE /api/v1/crimes/{id}`

### Risk Forecast
- `GET /api/v1/forecast/risk-map`
- `WS /api/v1/realtime/risk-updates`

### Route Optimization
- `POST /api/v1/routing/optimize`
- `POST /api/v1/routing/optimize-multi`

### ML Forecast
- `GET /api/v1/ml-forecast/timeseries`
- `GET /api/v1/ml-forecast/spatial-temporal`
- `GET /api/v1/ml-forecast/ensemble`

### OSM
- `GET /api/v1/osm/status`
- `POST /api/v1/osm/import`
- `POST /api/v1/osm/refresh-topology`
- `GET /api/v1/osm/topology-status`
- `POST /api/v1/osm/import-boundary`
- `GET /api/v1/osm/boundary-status`

## Test

```bash
cd backend
pytest
```

## Güvenlik Notları

- `.env` dosyası Git'e commit edilmez.
- Production ortamında şifreleri secret manager ile yönetin.
- Database portlarını dış dünyaya açmayın.
- SSL/TLS bağlantılarını tercih edin.

## Lisans

Bu proje UNDP & SAMSUNG INNOVATION AI CAMPUS kapsamında bitirme projesi olarak geliştirilmiştir.
