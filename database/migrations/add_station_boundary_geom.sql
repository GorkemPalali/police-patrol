-- Migration: Add boundary_geom to police_station
-- Stores station jurisdiction polygon boundaries

ALTER TABLE police_station
ADD COLUMN IF NOT EXISTS boundary_geom GEOGRAPHY(POLYGON, 4326);

CREATE INDEX IF NOT EXISTS idx_police_station_boundary_geom
ON police_station USING GIST (boundary_geom);

COMMENT ON COLUMN police_station.boundary_geom IS
'Karakol yetki alanı polygonu';
