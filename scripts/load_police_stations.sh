#!/bin/bash
# Script to load police stations into the database
# Usage: ./scripts/load_police_stations.sh

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values
POSTGRES_USER=${POSTGRES_USER:-police}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-policepwd}
POSTGRES_DB=${POSTGRES_DB:-policepatrol}
POSTGRES_HOST=${POSTGRES_HOST:-localhost}
POSTGRES_PORT=${POSTGRES_PORT:-5432}

echo "Loading police stations into database..."
echo "Database: $POSTGRES_DB"
echo "Host: $POSTGRES_HOST:$POSTGRES_PORT"
echo "User: $POSTGRES_USER"

# Check if running in Docker
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    # Running inside Docker, use localhost
    PGHOST=localhost
else
    # Running locally, check if Docker container is running
    if docker ps | grep -q policepatrol-db; then
        PGHOST=localhost
    else
        PGHOST=$POSTGRES_HOST
    fi
fi

# Execute SQL script
PGPASSWORD=$POSTGRES_PASSWORD psql -h $PGHOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -f scripts/load_police_stations.sql

echo "Police stations loaded successfully!"




