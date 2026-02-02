-- Load Police Stations into Database
-- This script can be run manually to load/update police stations

-- Delete existing stations (optional - comment out if you want to keep existing data)
-- DELETE FROM police_station;

-- Insert police stations
INSERT INTO police_station (name, geom, boundary_geom, capacity, active) VALUES
('Halkalı Şehit Ahmet Zehir Polis Merkezi',
 ST_GeogFromText('SRID=4326;POINT(28.78759 41.03586)'),
 ST_GeogFromText('SRID=4326;POLYGON((28.7600 41.0600, 28.8100 41.0600, 28.8250 41.0400, 28.8050 41.0200, 28.7700 41.0250, 28.7450 41.0450, 28.7600 41.0600))'),
 20, TRUE),
('İkitelli Şehit Zeki Kaya Polis Merkezi',
 ST_GeogFromText('SRID=4326;POINT(28.80424 41.05721)'),
 ST_GeogFromText('SRID=4326;POLYGON((28.7900 41.0850, 28.8350 41.0800, 28.8450 41.0550, 28.8200 41.0400, 28.8000 41.0450, 28.7850 41.0700, 28.7900 41.0850))'),
 15, TRUE),
('Kanarya Polis Merkezi',
 ST_GeogFromText('SRID=4326;POINT(28.78591 41.01108)'),
 ST_GeogFromText('SRID=4326;POLYGON((28.7600 41.0200, 28.8050 41.0200, 28.8150 40.9950, 28.7900 40.9750, 28.7600 40.9850, 28.7450 41.0050, 28.7600 41.0200))'),
 18, TRUE),
('Küçükçekmece Polis Merkezi',
 ST_GeogFromText('SRID=4326;POINT(28.77028 40.98998)'),
 ST_GeogFromText('SRID=4326;POLYGON((28.7300 41.0000, 28.7650 41.0000, 28.7900 40.9750, 28.7700 40.9550, 28.7350 40.9600, 28.7200 40.9800, 28.7300 41.0000))'),
 12, TRUE),
('Sefaköy Polis Merkezi',
 ST_GeogFromText('SRID=4326;POINT(28.79703 41.02197)'),
 ST_GeogFromText('SRID=4326;POLYGON((28.8050 41.0450, 28.8450 41.0400, 28.8600 41.0150, 28.8300 40.9950, 28.8000 41.0000, 28.7950 41.0200, 28.8050 41.0450))'),
 12, TRUE)
ON CONFLICT DO NOTHING;

-- Verify inserted stations
SELECT id, name, capacity, active, 
       ST_Y(geom::geometry) as lat, 
       ST_X(geom::geometry) as lng 
FROM police_station 
ORDER BY name;



