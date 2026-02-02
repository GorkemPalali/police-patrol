-- Add risk score and confidence columns to road_segment table
-- These will be populated by risk forecast service

ALTER TABLE road_segment 
ADD COLUMN IF NOT EXISTS risk_score DOUBLE PRECISION DEFAULT 0.0 CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
ADD COLUMN IF NOT EXISTS risk_confidence DOUBLE PRECISION DEFAULT 0.0 CHECK (risk_confidence >= 0.0 AND risk_confidence <= 1.0),
ADD COLUMN IF NOT EXISTS risk_updated_at TIMESTAMP WITHOUT TIME ZONE;

-- Create index for risk-based queries
CREATE INDEX IF NOT EXISTS idx_road_segment_risk_score ON road_segment (risk_score DESC) WHERE risk_score > 0;

COMMENT ON COLUMN road_segment.risk_score IS 'Risk score for this road segment (0-1), calculated from nearby crime events';
COMMENT ON COLUMN road_segment.risk_confidence IS 'Confidence level for risk score (0-1)';
COMMENT ON COLUMN road_segment.risk_updated_at IS 'Timestamp when risk score was last updated';




