#!/bin/bash
# Script to load police stations via API
# Usage: ./scripts/load_police_stations_api.sh

set -e

API_BASE_URL=${API_BASE_URL:-http://localhost:8000/api/v1}

echo "Loading police stations via API..."
echo "API Base URL: $API_BASE_URL"

# Police stations data
stations=(
  '{"name":"Halkalı Şehit Ahmet Zehir Polis Merkezi","lat":41.03586,"lng":28.78759,"capacity":20,"active":true}'
  '{"name":"İkitelli Şehit Zeki Kaya Polis Merkezi","lat":41.05721,"lng":28.80424,"capacity":15,"active":true}'
  '{"name":"Kanarya Polis Merkezi","lat":41.01108,"lng":28.78591,"capacity":18,"active":true}'
  '{"name":"Küçükçekmece Polis Merkezi","lat":40.98998,"lng":28.77028,"capacity":12,"active":true}'
  '{"name":"Sefaköy Polis Merkezi","lat":41.02197,"lng":28.79703,"capacity":12,"active":true}'
)

for station in "${stations[@]}"; do
  echo "Creating station: $(echo $station | jq -r '.name')"
  response=$(curl -s -X POST "$API_BASE_URL/stations" \
    -H "Content-Type: application/json" \
    -d "$station")
  
  if echo "$response" | jq -e '.id' > /dev/null 2>&1; then
    echo "  ✓ Success: $(echo $response | jq -r '.name')"
  else
    echo "  ✗ Failed: $response"
  fi
done

echo ""
echo "All police stations loaded!"




