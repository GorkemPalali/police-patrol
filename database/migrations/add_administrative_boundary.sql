-- Migration: Add administrative_boundary table and update is_within_kucukcekmece function
-- This migration adds support for polygon-based boundary storage

-- Create administrative_boundary table if it doesn't exist
CREATE TABLE IF NOT EXISTS administrative_boundary (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    admin_level INTEGER,
    geom GEOGRAPHY(POLYGON, 4326) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(name, admin_level)
);

-- Create spatial index
CREATE INDEX IF NOT EXISTS idx_administrative_boundary_geom 
ON administrative_boundary USING GIST (geom);

-- Add comment
COMMENT ON TABLE administrative_boundary IS 'İdari sınırlar - polygon olarak saklanan ilçe/bölge sınırları';

-- Update is_within_kucukcekmece function to use polygon
CREATE OR REPLACE FUNCTION is_within_kucukcekmece(geom GEOGRAPHY)
RETURNS BOOLEAN AS $$
DECLARE
    boundary_geom GEOGRAPHY;
BEGIN
    -- Get Küçükçekmece boundary from database
    SELECT a.geom INTO boundary_geom
    FROM administrative_boundary a
    WHERE a.name = 'Küçükçekmece' AND a.admin_level = 8
    LIMIT 1;
    
    -- If no boundary found, return false (should not happen in production)
    IF boundary_geom IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Check if geometry is within boundary using ST_Within
    -- ST_Within works with geometry, so we cast geography to geometry
    RETURN ST_Within(geom::geometry, boundary_geom::geometry);
END;
$$ LANGUAGE plpgsql STABLE;

-- Update view
CREATE OR REPLACE VIEW crime_event_kucukcekmece AS
SELECT *
FROM crime_event
WHERE is_within_kucukcekmece(geom);

-- Update comment
COMMENT ON FUNCTION is_within_kucukcekmece IS 'Küçükçekmece polygon sınırları içinde olup olmadığını kontrol eder (administrative_boundary tablosundan okur)';
COMMENT ON VIEW crime_event_kucukcekmece IS 'Küçükçekmece polygon sınırları içindeki suç olayları';




