-- Boundary Validation Triggers
-- Automatically reject INSERT/UPDATE operations for data outside Küçükçekmece boundary

-- Function to validate geometry before insert/update
CREATE OR REPLACE FUNCTION validate_boundary_before_modify()
RETURNS TRIGGER AS $$
DECLARE
    boundary_geom GEOGRAPHY;
    geom_to_check GEOGRAPHY;
BEGIN
    -- Get the geometry column (different tables have different column names)
    IF TG_TABLE_NAME = 'crime_event' OR TG_TABLE_NAME = 'police_station' THEN
        geom_to_check := NEW.geom;
    ELSIF TG_TABLE_NAME = 'risk_cell' THEN
        geom_to_check := NEW.geom;
    ELSE
        -- Unknown table, allow operation
        RETURN NEW;
    END IF;
    
    -- Get Küçükçekmece boundary
    SELECT a.geom INTO boundary_geom
    FROM administrative_boundary a
    WHERE a.name = 'Küçükçekmece' AND a.admin_level = 8
    LIMIT 1;
    
    -- If boundary exists, validate
    IF boundary_geom IS NOT NULL THEN
        -- For POINT geometries (crime_event, police_station), use ST_Within
        -- For POLYGON geometries (risk_cell), use ST_Intersects (polygon can cross boundary)
        IF TG_TABLE_NAME = 'risk_cell' THEN
            -- For polygons, check if it intersects with boundary
            IF NOT ST_Intersects(geom_to_check::geometry, boundary_geom::geometry) THEN
                RAISE EXCEPTION 'Geometri Küçükçekmece sınırları dışında (tablo: %, id: %)', 
                    TG_TABLE_NAME, COALESCE(NEW.id::text, 'new');
            END IF;
        ELSE
            -- For points, check if it's within boundary
            IF NOT ST_Within(geom_to_check::geometry, boundary_geom::geometry) THEN
                RAISE EXCEPTION 'Koordinatlar Küçükçekmece sınırları dışında (tablo: %, id: %)', 
                    TG_TABLE_NAME, COALESCE(NEW.id::text, 'new');
            END IF;
        END IF;
    END IF;
    
    -- If boundary doesn't exist, allow operation (fail open)
    -- This prevents system from breaking if boundary is not loaded yet
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for crime_event table
DROP TRIGGER IF EXISTS validate_crime_event_boundary ON crime_event;
CREATE TRIGGER validate_crime_event_boundary
    BEFORE INSERT OR UPDATE OF geom ON crime_event
    FOR EACH ROW
    EXECUTE FUNCTION validate_boundary_before_modify();

-- Trigger for police_station table
DROP TRIGGER IF EXISTS validate_police_station_boundary ON police_station;
CREATE TRIGGER validate_police_station_boundary
    BEFORE INSERT OR UPDATE OF geom ON police_station
    FOR EACH ROW
    EXECUTE FUNCTION validate_boundary_before_modify();

-- Trigger for risk_cell table (optional, but added for completeness)
DROP TRIGGER IF EXISTS validate_risk_cell_boundary ON risk_cell;
CREATE TRIGGER validate_risk_cell_boundary
    BEFORE INSERT OR UPDATE OF geom ON risk_cell
    FOR EACH ROW
    EXECUTE FUNCTION validate_boundary_before_modify();

-- Comments
COMMENT ON FUNCTION validate_boundary_before_modify() IS 
    'Küçükçekmece sınırları dışındaki verilerin eklenmesini/güncellenmesini engeller';
COMMENT ON TRIGGER validate_crime_event_boundary ON crime_event IS 
    'Crime event koordinatlarının Küçükçekmece sınırları içinde olmasını garanti eder';
COMMENT ON TRIGGER validate_police_station_boundary ON police_station IS 
    'Police station koordinatlarının Küçükçekmece sınırları içinde olmasını garanti eder';
COMMENT ON TRIGGER validate_risk_cell_boundary ON risk_cell IS 
    'Risk cell geometrisinin Küçükçekmece sınırları ile kesişmesini garanti eder';




