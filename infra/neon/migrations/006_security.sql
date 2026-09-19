-- Server-only access. Do not enable Neon Data API on this project.
-- The Next.js runtime reads DATABASE_URL on the server; never NEXT_PUBLIC_*.

REVOKE ALL ON SCHEMA core, analytics FROM PUBLIC;
GRANT USAGE ON SCHEMA api TO CURRENT_USER;
GRANT SELECT ON ALL TABLES IN SCHEMA api TO CURRENT_USER;

DO $$
DECLARE
  t text;
BEGIN
  FOR t IN
    SELECT format('%I.%I', schemaname, tablename)
    FROM pg_tables
    WHERE schemaname IN ('core', 'analytics')
  LOOP
    EXECUTE format('ALTER TABLE %s ENABLE ROW LEVEL SECURITY', t);
  END LOOP;
END $$;
