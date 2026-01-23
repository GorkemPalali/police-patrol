-- Add neighborhoods field to police_station table
-- This stores the list of neighborhood names that each police station is responsible for

ALTER TABLE police_station 
ADD COLUMN IF NOT EXISTS neighborhoods TEXT[];

-- Add index for faster lookups
CREATE INDEX IF NOT EXISTS idx_police_station_neighborhoods 
ON police_station USING GIN(neighborhoods) 
WHERE neighborhoods IS NOT NULL;

-- Add comment
COMMENT ON COLUMN police_station.neighborhoods IS 
'Array of neighborhood names that this police station is responsible for. Used for filtering risk calculations and route generation.';

-- Update existing stations with their neighborhoods
UPDATE police_station 
SET neighborhoods = ARRAY['Mehmet Akif', 'Küçükçekmece İkitelli OSB', 'Atakent']
WHERE name LIKE '%İkitelli%';

UPDATE police_station 
SET neighborhoods = ARRAY['Halkalı Merkez', 'Atatürk', 'İstasyon', 'Yarımburgaz']
WHERE name LIKE '%Halkalı%';

UPDATE police_station 
SET neighborhoods = ARRAY['Fevzi Çakmak', 'Tevfikbey', 'Söğütlüçeşme', 'İnönü']
WHERE name LIKE '%Sefaköy%';

UPDATE police_station 
SET neighborhoods = ARRAY['Kanarya', 'Cumhuriyet', 'Fevzi Çakmak', 'Sultanmurat']
WHERE name LIKE '%Kanarya%';

UPDATE police_station 
SET neighborhoods = ARRAY['Kartaltepe', 'Beşyol', 'Gültepe', 'Kemalpaşa', 'Fatih', 'Yenimahalle', 'Yeşilova', 'Cennet']
WHERE name LIKE '%Küçükçekmece%' AND name NOT LIKE '%İkitelli%';

