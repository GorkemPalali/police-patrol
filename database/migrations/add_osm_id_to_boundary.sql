-- Add osm_id column to administrative_boundary table
-- This stores the OSM relation ID for the boundary, used for Overpass API queries

ALTER TABLE administrative_boundary 
ADD COLUMN IF NOT EXISTS osm_id INTEGER;

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_administrative_boundary_osm_id 
ON administrative_boundary(osm_id) 
WHERE osm_id IS NOT NULL;

-- Add comment
COMMENT ON COLUMN administrative_boundary.osm_id IS 
'OSM relation ID for this administrative boundary. Used for Overpass API queries to fetch OSM data within this boundary.';

