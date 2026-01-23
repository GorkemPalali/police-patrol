-- pgRouting Kurulum Scripti
-- road_segment tablosundan yönlendirme ağını kurar

CREATE EXTENSION IF NOT EXISTS pgrouting;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='road_segment' AND column_name='source') THEN
        ALTER TABLE road_segment ADD COLUMN source BIGINT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='road_segment' AND column_name='target') THEN
        ALTER TABLE road_segment ADD COLUMN target BIGINT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='road_segment' AND column_name='cost') THEN
        ALTER TABLE road_segment ADD COLUMN cost DOUBLE PRECISION;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='road_segment' AND column_name='reverse_cost') THEN
        ALTER TABLE road_segment ADD COLUMN reverse_cost DOUBLE PRECISION;
    END IF;
END $$;

-- Geometri uzunluğuna dayalı maliyet hesapla Maliyet = mesafe metre cinsinden
UPDATE road_segment
SET cost = ST_Length(geom::geometry),
    reverse_cost = CASE 
        WHEN one_way THEN 1e9  -- Very high cost for reverse direction on one-way roads
        ELSE ST_Length(geom::geometry)
    END
WHERE cost IS NULL OR reverse_cost IS NULL;

-- pgRouting kullanarak topoloji oluştur
-- Not: Bu, road_segment tablosunun geçerli LineString geometrilerine sahip olması lazım
-- pgr_createTopology fonksiyonu, source ve target sütunlarını dolduracak
-- Kullanım: SELECT pgr_createTopology('road_segment', 0.0001, 'geom', 'id');

-- Yönlendirme topolojisini yenilemek için fonksiyon
CREATE OR REPLACE FUNCTION refresh_routing_topology()
RETURNS void AS $$
BEGIN
    -- Maliyetleri yeniden hesapla
    UPDATE road_segment
    SET cost = ST_Length(geom::geometry),
        reverse_cost = CASE 
            WHEN one_way THEN 1e9
            ELSE ST_Length(geom::geometry)
        END;
    
    -- Not: pgr_createTopology manuel olarak veya zamanlanmış bir görevle çağrılmalı
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_routing_topology() IS 'Yol segmentlerinin maliyetlerini günceller';

-- Otomatik topology oluşturma fonksiyonu
CREATE OR REPLACE FUNCTION create_routing_topology(
    tolerance DOUBLE PRECISION DEFAULT 0.0001,
    force_recreate BOOLEAN DEFAULT FALSE
)
RETURNS TABLE(
    success BOOLEAN,
    message TEXT,
    segment_count BIGINT,
    connected_segments BIGINT,
    connection_rate NUMERIC
) AS $$
DECLARE
    v_segment_count BIGINT;
    v_connected BIGINT;
    v_rate NUMERIC;
    v_result TEXT;
BEGIN
    -- Check if road_segment table has data
    SELECT COUNT(*) INTO v_segment_count FROM road_segment;
    
    IF v_segment_count = 0 THEN
        RETURN QUERY SELECT 
            FALSE::BOOLEAN,
            'No road segments found'::TEXT,
            0::BIGINT,
            0::BIGINT,
            0::NUMERIC;
        RETURN;
    END IF;
    
    -- Update costs first
    PERFORM refresh_routing_topology();
    
    -- Drop existing topology if force_recreate
    IF force_recreate THEN
        UPDATE road_segment SET source = NULL, target = NULL;
    END IF;
    
    -- Create topology
    BEGIN
        SELECT pgr_createTopology(
            'road_segment',
            tolerance,
            'geom',
            'id',
            'source',
            'target',
            rows_where := 'true'
        ) INTO v_result;
        
        -- Get statistics
        SELECT 
            COUNT(CASE WHEN source IS NOT NULL AND target IS NOT NULL THEN 1 END)
        INTO v_connected
        FROM road_segment;
        
        v_rate := (v_connected::NUMERIC / v_segment_count::NUMERIC * 100);
        
        RETURN QUERY SELECT 
            TRUE::BOOLEAN,
            ('Topology created successfully: ' || v_result)::TEXT,
            v_segment_count,
            v_connected,
            ROUND(v_rate, 2);
            
    EXCEPTION WHEN OTHERS THEN
        RETURN QUERY SELECT 
            FALSE::BOOLEAN,
            ('Topology creation failed: ' || SQLERRM)::TEXT,
            v_segment_count,
            0::BIGINT,
            0::NUMERIC;
    END;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION create_routing_topology(DOUBLE PRECISION, BOOLEAN) IS 
'pgRouting topology oluşturur. tolerance: topoloji toleransı (varsayılan 0.0001), force_recreate: mevcut topolojiyi silip yeniden oluştur (varsayılan false)';