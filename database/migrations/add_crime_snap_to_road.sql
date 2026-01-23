-- Add snapped_road_segment_id column to crime_event table for performance optimization
-- This allows caching which road segment each crime event is snapped to

ALTER TABLE crime_event 
ADD COLUMN IF NOT EXISTS snapped_road_segment_id BIGINT;

-- Add foreign key constraint (optional, for data integrity)
-- ALTER TABLE crime_event 
-- ADD CONSTRAINT fk_crime_event_road_segment 
-- FOREIGN KEY (snapped_road_segment_id) 
-- REFERENCES road_segment(id) 
-- ON DELETE SET NULL;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_crime_event_snapped_segment 
ON crime_event(snapped_road_segment_id) 
WHERE snapped_road_segment_id IS NOT NULL;

-- Add comment
COMMENT ON COLUMN crime_event.snapped_road_segment_id IS 
'Road segment ID that this crime event is snapped to (within 100m). Used for KDE-based risk calculation.';


