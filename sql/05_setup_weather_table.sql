-- Indian weather documents table
-- Matches lakebase.py and weather_client.py

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_documents (
    id TEXT PRIMARY KEY,

    location TEXT NOT NULL,

    source_type TEXT NOT NULL,

    headline TEXT,

    narrative_text TEXT,

    issued_at TIMESTAMPTZ,

    payload JSONB NOT NULL,

    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);

CREATE INDEX IF NOT EXISTS idx_weather_documents_issued_at
    ON weather_documents (issued_at);

-- Verify
SELECT
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
