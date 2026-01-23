import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.core.config import get_settings
from app.api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

settings = get_settings()

app = FastAPI(
    title="Predictive Patrol Routing System",
    description="""
    ## Predictive Patrol Routing System API
    
    Suç risk tahmini ve devriye rotası optimizasyonu sistemi.
    
    ### Özellikler
    
    * **Risk Tahmini**: Geçmiş suç verilerine dayalı kernel density estimation ile risk haritası oluşturma
    * **Rota Optimizasyonu**: pgRouting kullanarak OSM yol ağı üzerinden optimize edilmiş devriye rotaları
    * **Gerçek Zamanlı Güncellemeler**: WebSocket üzerinden risk skorlarının gerçek zamanlı güncellenmesi
    * **OSM Entegrasyonu**: OpenStreetMap verilerini kullanarak yol ağı ve sınır bilgileri
    
    ### API Endpoint'leri
    
    * `/api/v1/health` - Sistem sağlık kontrolü
    * `/api/v1/stations` - Polis karakolu yönetimi
    * `/api/v1/crimes` - Suç olayı yönetimi
    * `/api/v1/forecast` - Risk tahmini ve harita
    * `/api/v1/routing` - Rota optimizasyonu
    * `/api/v1/osm` - OSM veri yönetimi
    * `/api/v1/realtime` - Gerçek zamanlı risk güncellemeleri (WebSocket)
    """,
    version="1.0.0",
    terms_of_service="https://example.com/terms/",
    contact={
        "name": "API Support",
        "email": "support@policepatrol.example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "Health",
            "description": "Sistem sağlık kontrolü ve durum bilgisi",
        },
        {
            "name": "Stations",
            "description": "Polis karakolu yönetimi - karakol ekleme, listeleme ve güncelleme",
        },
        {
            "name": "Crimes",
            "description": "Suç olayı yönetimi - olay ekleme, listeleme ve yol ağına snap işlemleri",
        },
        {
            "name": "Forecast",
            "description": "Risk tahmini ve harita oluşturma - KDE tabanlı risk hesaplama",
        },
        {
            "name": "Routing",
            "description": "Rota optimizasyonu - pgRouting kullanarak optimize edilmiş devriye rotaları",
        },
        {
            "name": "OSM",
            "description": "OpenStreetMap veri yönetimi - yol ağı import, boundary yönetimi",
        },
        {
            "name": "Realtime",
            "description": "Gerçek zamanlı risk güncellemeleri - WebSocket üzerinden canlı veri",
        },
        {
            "name": "ML Forecast",
            "description": "Makine öğrenmesi tabanlı risk tahmini",
        },
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get(
    "/",
    summary="API Kök Endpoint",
    description="API hakkında temel bilgileri döndürür",
    tags=["Health"]
)
def root():
    """
    API kök endpoint'i.
    
    Sistem adı, versiyon ve durum bilgisini döndürür.
    """
    return {
        "name": "Predictive Patrol Routing System",
        "version": "1.0.0",
        "status": "running",
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }


def custom_openapi():
    """Custom OpenAPI schema generator with better organization"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    
    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
