-- Load Police Stations into Database
-- This script can be run manually to load/update police stations

-- Delete existing stations (optional - comment out if you want to keep existing data)
-- DELETE FROM police_station;

-- Insert police stations
INSERT INTO police_station (name, geom, capacity, active) VALUES
('Halkalı Şehit Ahmet Zehir Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.78759 41.03586)'), 20, TRUE),
('İkitelli Şehit Zeki Kaya Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.80424 41.05721)'), 15, TRUE),
('Kanarya Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.78591 41.01108)'), 18, TRUE),
('Küçükçekmece Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.77028 40.98998)'), 12, TRUE),
('Sefaköy Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.79703 41.02197)'), 12, TRUE)
ON CONFLICT DO NOTHING;

-- Verify inserted stations
SELECT id, name, capacity, active, 
       ST_Y(geom::geometry) as lat, 
       ST_X(geom::geometry) as lng 
FROM police_station 
ORDER BY name;




