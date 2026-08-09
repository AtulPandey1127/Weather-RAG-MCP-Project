-- Indian weather documents table
-- Supports structured metadata filtering for RAG.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_documents (
    id TEXT PRIMARY KEY,

    location TEXT NOT NULL,

    state TEXT,

    district TEXT,

    latitude DOUBLE PRECISION,

    longitude DOUBLE PRECISION,

    source TEXT NOT NULL DEFAULT 'open-meteo',

    source_type TEXT NOT NULL,

    headline TEXT,

    narrative_text TEXT,

    forecast_date DATE,

    temperature_min_c DOUBLE PRECISION,

    temperature_max_c DOUBLE PRECISION,

    rainfall_mm DOUBLE PRECISION,

    precipitation_probability DOUBLE PRECISION,

    weather_code INTEGER,

    severity TEXT,

    issued_at TIMESTAMPTZ,

    payload JSONB NOT NULL,

    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Migration support for an existing weather_documents table.
ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS state TEXT;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS district TEXT;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'open-meteo';

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS forecast_date DATE;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS temperature_min_c DOUBLE PRECISION;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS temperature_max_c DOUBLE PRECISION;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS rainfall_mm DOUBLE PRECISION;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS precipitation_probability DOUBLE PRECISION;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS weather_code INTEGER;

ALTER TABLE weather_documents
    ADD COLUMN IF NOT EXISTS severity TEXT;

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_state
    ON weather_documents (state);

CREATE INDEX IF NOT EXISTS idx_weather_documents_district
    ON weather_documents (district);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source
    ON weather_documents (source);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);

CREATE INDEX IF NOT EXISTS idx_weather_documents_forecast_date
    ON weather_documents (forecast_date);

CREATE INDEX IF NOT EXISTS idx_weather_documents_precipitation_probability
    ON weather_documents (precipitation_probability);

CREATE INDEX IF NOT EXISTS idx_weather_documents_issued_at
    ON weather_documents (issued_at);
