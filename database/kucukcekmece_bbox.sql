-- Küçükçekmece bounding box kontrolü için fonksiyon
-- ST_DWithin kullanarak geography tipi ile çalışır
CREATE OR REPLACE FUNCTION is_within_kucukcekmece(geom GEOGRAPHY)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN ST_DWithin(
        geom,
        ST_MakeEnvelope(28.70, 40.98, 28.80, 41.05, 4326)::geography,
        0
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE VIEW crime_event_kucukcekmece AS
SELECT *
FROM crime_event
WHERE is_within_kucukcekmece(geom);

CREATE INDEX IF NOT EXISTS idx_crime_event_kucukcekmece 
ON crime_event USING GIST (geom)
WHERE is_within_kucukcekmece(geom);

COMMENT ON FUNCTION is_within_kucukcekmece IS 'Küçükçekmece sınırları içinde olup olmadığını kontrol eder';
COMMENT ON VIEW crime_event_kucukcekmece IS 'Küçükçekmece sınırları içindeki suç olayları';