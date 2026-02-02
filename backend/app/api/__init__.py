from fastapi import APIRouter
from app.api.routes import health, stations, crimes, forecast, routing, ml_forecast, osm, realtime

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(stations.router)
api_router.include_router(crimes.router)
api_router.include_router(forecast.router)
api_router.include_router(routing.router)
api_router.include_router(ml_forecast.router)
api_router.include_router(osm.router)
api_router.include_router(realtime.router)
