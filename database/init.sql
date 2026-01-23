-- PostgreSQL + PostGIS Initialization Script
-- Predictive Patrol Routing System

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pgrouting;

-- Crime Event Table
CREATE TABLE IF NOT EXISTS crime_event (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    crime_type VARCHAR(100) NOT NULL,
    severity INT NOT NULL CHECK (severity BETWEEN 1 AND 5),
    event_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    geom GEOGRAPHY(POINT, 4326) NOT NULL,
    street_name VARCHAR(255),
    confidence_score DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

-- Police Station Table
CREATE TABLE IF NOT EXISTS police_station (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(150) NOT NULL,
    geom GEOGRAPHY(POINT, 4326) NOT NULL,
    capacity INT NOT NULL DEFAULT 0 CHECK (capacity >= 0),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

-- Road Segment Table (for routing)
CREATE TABLE IF NOT EXISTS road_segment (
    id BIGSERIAL PRIMARY KEY,
    geom GEOGRAPHY(LINESTRING, 4326) NOT NULL,
    road_type VARCHAR(50),
    speed_limit INT,
    one_way BOOLEAN NOT NULL DEFAULT FALSE,
    -- pgRouting fields (will be populated by routing_setup.sql)
    source BIGINT,
    target BIGINT,
    cost DOUBLE PRECISION,
    reverse_cost DOUBLE PRECISION
);

-- Risk Cell Table (grid-based risk system)
CREATE TABLE IF NOT EXISTS risk_cell (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    geom GEOGRAPHY(POLYGON, 4326) NOT NULL,
    time_window TSRANGE NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0.0),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

-- Administrative Boundary Table (for storing polygon boundaries)
CREATE TABLE IF NOT EXISTS administrative_boundary (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    admin_level INTEGER,
    geom GEOGRAPHY(POLYGON, 4326) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(name, admin_level)
);

-- Spatial Indexes (GiST for geography)
CREATE INDEX IF NOT EXISTS idx_crime_event_geom ON crime_event USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_police_station_geom ON police_station USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_road_segment_geom ON road_segment USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_risk_cell_geom ON risk_cell USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_administrative_boundary_geom ON administrative_boundary USING GIST (geom);

-- Temporal Indexes
CREATE INDEX IF NOT EXISTS idx_crime_event_time ON crime_event (event_time);
CREATE INDEX IF NOT EXISTS idx_risk_cell_time_window ON risk_cell USING GIST (time_window);

-- Composite Index for spatio-temporal queries
CREATE INDEX IF NOT EXISTS idx_crime_event_time_geom ON crime_event USING GIST (event_time, geom);

-- Additional useful indexes
CREATE INDEX IF NOT EXISTS idx_crime_event_type ON crime_event (crime_type);
CREATE INDEX IF NOT EXISTS idx_crime_event_severity ON crime_event (severity);
CREATE INDEX IF NOT EXISTS idx_police_station_active ON police_station (active) WHERE active = TRUE;

-- Comments for documentation
COMMENT ON TABLE crime_event IS 'Suç olayları - mekansal ve zamansal veriler';
COMMENT ON TABLE police_station IS 'Polis karakolları - devriye başlangıç noktaları';
COMMENT ON TABLE road_segment IS 'Yol segmentleri - OSM verilerinden routing için';
COMMENT ON TABLE risk_cell IS 'Grid-based risk hücreleri - forecast sonuçları';
COMMENT ON TABLE administrative_boundary IS 'İdari sınırlar - polygon olarak saklanan ilçe/bölge sınırları';
