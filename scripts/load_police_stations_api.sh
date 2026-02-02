#!/bin/bash
# Script to load police stations via API
# Usage: ./scripts/load_police_stations_api.sh

set -e

API_BASE_URL=${API_BASE_URL:-http://localhost:8000/api/v1}

echo "Loading police stations via API..."
echo "API Base URL: $API_BASE_URL"

# Police stations data
stations=(
  '{"name":"Halkalı Şehit Ahmet Zehir Polis Merkezi","lat":41.03586,"lng":28.78759,"capacity":20,"active":true,"boundary":{"type":"Polygon","coordinates":[[[28.7600,41.0600],[28.8100,41.0600],[28.8250,41.0400],[28.8050,41.0200],[28.7700,41.0250],[28.7450,41.0450],[28.7600,41.0600]]]}}'
  '{"name":"İkitelli Şehit Zeki Kaya Polis Merkezi","lat":41.05721,"lng":28.80424,"capacity":15,"active":true,"boundary":{"type":"Polygon","coordinates":[[[28.7900,41.0850],[28.8350,41.0800],[28.8450,41.0550],[28.8200,41.0400],[28.8000,41.0450],[28.7850,41.0700],[28.7900,41.0850]]]}}'
  '{"name":"Kanarya Polis Merkezi","lat":41.01108,"lng":28.78591,"capacity":18,"active":true,"boundary":{"type":"Polygon","coordinates":[[[28.7600,41.0200],[28.8050,41.0200],[28.8150,40.9950],[28.7900,40.9750],[28.7600,40.9850],[28.7450,41.0050],[28.7600,41.0200]]]}}'
  '{"name":"Küçükçekmece Polis Merkezi","lat":40.98998,"lng":28.77028,"capacity":12,"active":true,"boundary":{"type":"Polygon","coordinates":[[[28.7300,41.0000],[28.7650,41.0000],[28.7900,40.9750],[28.7700,40.9550],[28.7350,40.9600],[28.7200,40.9800],[28.7300,41.0000]]]}}'
  '{"name":"Sefaköy Polis Merkezi","lat":41.02197,"lng":28.79703,"capacity":12,"active":true,"boundary":{"type":"Polygon","coordinates":[[[28.8050,41.0450],[28.8450,41.0400],[28.8600,41.0150],[28.8300,40.9950],[28.8000,41.0000],[28.7950,41.0200],[28.8050,41.0450]]]}}'
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



