-- Staging schema for Health Sentinel (Neon Postgres 17).
-- Apply before bulk load. Secondary indexes and expensive FKs are in 004.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS api;

COMMENT ON SCHEMA core IS 'Cleaned source records (DuckDB schema clean).';
COMMENT ON SCHEMA analytics IS 'Immutable score-run outputs from the JSON bundle.';
COMMENT ON SCHEMA api IS 'Read models for the Next.js server. Not for the browser.';
