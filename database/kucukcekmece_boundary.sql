-- Küçükçekmece polygon boundary kontrolü için fonksiyon
-- administrative_boundary tablosundan polygon'u kullanır
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

-- Update view to use the new function
CREATE OR REPLACE VIEW crime_event_kucukcekmece AS
SELECT *
FROM crime_event
WHERE is_within_kucukcekmece(geom);

-- Index remains the same (partial index on filtered data)
CREATE INDEX IF NOT EXISTS idx_crime_event_kucukcekmece 
ON crime_event USING GIST (geom)
WHERE is_within_kucukcekmece(geom);

COMMENT ON FUNCTION is_within_kucukcekmece IS 'Küçükçekmece polygon sınırları içinde olup olmadığını kontrol eder (administrative_boundary tablosundan okur)';
COMMENT ON VIEW crime_event_kucukcekmece IS 'Küçükçekmece polygon sınırları içindeki suç olayları';




