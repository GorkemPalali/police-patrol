-- Sample Police Stations
INSERT INTO police_station (name, geom, capacity, active) VALUES
('Halkalı Şehit Ahmet Zehir Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.78759 41.03586)'), 20, TRUE),
('İkitelli Şehit Zeki Kaya Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.80424 41.05721)'), 15, TRUE),
('Kanarya Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.78591 41.01108)'), 18, TRUE),
('Küçükçekmece Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.77028 40.98998)'), 12, TRUE),
('Sefaköy Polis Merkezi', ST_GeogFromText('SRID=4326;POINT(28.79703 41.02197)'), 12, TRUE)
ON CONFLICT DO NOTHING;

-- Sample Crime Events (last 30 days)
INSERT INTO crime_event (crime_type, severity, event_time, geom, street_name, confidence_score) VALUES
('Hırsızlık', 2, NOW() - INTERVAL '5 days', ST_GeogFromText('SRID=4326;POINT(28.9784 41.0082)'), 'İstiklal Caddesi', 0.9),
('Saldırı', 4, NOW() - INTERVAL '3 days', ST_GeogFromText('SRID=4326;POINT(29.0230 40.9819)'), 'Bağdat Caddesi', 0.95),
('Hırsızlık', 2, NOW() - INTERVAL '10 days', ST_GeogFromText('SRID=4326;POINT(29.0084 41.0422)'), 'Barbaros Bulvarı', 0.85),
('Vandalizm', 1, NOW() - INTERVAL '7 days', ST_GeogFromText('SRID=4326;POINT(28.9858 41.0608)'), 'Halaskargazi Caddesi', 0.8),
('Saldırı', 3, NOW() - INTERVAL '2 days', ST_GeogFromText('SRID=4326;POINT(28.9700 41.0150)'), 'Taksim Meydanı', 0.92),
('Hırsızlık', 2, NOW() - INTERVAL '15 days', ST_GeogFromText('SRID=4326;POINT(29.0100 40.9900)'), 'Moda Caddesi', 0.88),
('Saldırı', 5, NOW() - INTERVAL '1 day', ST_GeogFromText('SRID=4326;POINT(28.9950 41.0300)'), 'Nişantaşı', 0.98),
('Hırsızlık', 1, NOW() - INTERVAL '20 days', ST_GeogFromText('SRID=4326;POINT(29.0150 41.0500)'), 'Levent', 0.75)
ON CONFLICT DO NOTHING;


