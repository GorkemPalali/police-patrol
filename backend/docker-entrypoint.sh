#!/bin/bash
set -e

# Set Python path for scripts
export PYTHONPATH="/app:${PYTHONPATH}"

# Run boundary import script first (required for polygon-based filtering)
echo "Checking Küçükçekmece boundary import..."
if [ -f "/app/../scripts/import_kucukcekmece_boundary.py" ]; then
    cd /app && python3 /app/../scripts/import_kucukcekmece_boundary.py || echo "Boundary import failed or skipped, continuing..."
else
    echo "Boundary import script not found, skipping..."
fi

# Run OSM import script (blocking, but will skip if data exists)
echo "Checking OSM data import..."
if [ -f "/app/../scripts/import_osm_data.py" ]; then
    cd /app && python3 /app/../scripts/import_osm_data.py || echo "OSM import failed or skipped, continuing..."
else
    echo "OSM import script not found, skipping..."
fi

# Start the main application
echo "Starting FastAPI application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

