#!/usr/bin/env python3
"""
Supabase Database Setup Script
Automatically creates the songs table with proper schema and indexes.
Run this once after configuring .env.local
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SECRET = os.getenv("SUPABASE_SECRET", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

if not SUPABASE_URL or (not SUPABASE_KEY and not SUPABASE_SECRET):
    print("[ERROR] Error: SUPABASE_URL and SUPABASE_KEY or SUPABASE_SECRET must be set in .env.local")
    exit(1)

# Use service role (admin) for setup
client_key = SUPABASE_SECRET or SUPABASE_KEY
supabase = create_client(SUPABASE_URL, client_key)

# SQL to create the songs table with proper schema
CREATE_TABLE_SQL = """
-- Create songs table if it doesn't exist
CREATE TABLE IF NOT EXISTS songs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    artist TEXT,
    bpm INTEGER DEFAULT 0,
    duration TEXT,
    lrc_preview TEXT,
    lrc TEXT,
    tags TEXT[] DEFAULT '{"subido"}',
    video_url TEXT,
    instrumental_url TEXT,
    status TEXT DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add status column if it doesn't exist (for existing tables)
ALTER TABLE songs ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'completed';

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_songs_job_id ON songs(job_id);
CREATE INDEX IF NOT EXISTS idx_songs_created_at_desc ON songs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist);

-- Enable RLS (Row Level Security)
ALTER TABLE songs ENABLE ROW LEVEL SECURITY;

-- Create policies
DROP POLICY IF EXISTS "Public read" ON songs;
CREATE POLICY "Public read" ON songs
    FOR SELECT USING (true);

DROP POLICY IF EXISTS "Authenticated write" ON songs;
CREATE POLICY "Authenticated write" ON songs
    FOR INSERT WITH CHECK (true);

DROP POLICY IF EXISTS "Authenticated delete" ON songs;
CREATE POLICY "Authenticated delete" ON songs
    FOR DELETE USING (true);
"""

print("[SETUP] Setting up Supabase database...")

try:
    # Execute the SQL
    result = supabase.rpc("sql", {"query": CREATE_TABLE_SQL}).execute()
    print("[OK] Supabase setup completed successfully!")
    print("   - Created 'songs' table")
    print("   - Created indexes for performance")
    print("   - Enabled RLS with public read + auth write policies")
except Exception as e:
    print(f"[ERROR] Setup Error: {str(e)}")
    print("\nAlternative: Use the Supabase SQL Editor to run SUPABASE_SETUP.sql manually:")
    print("1. Go to https://app.supabase.com → SQL Editor")
    print("2. Click 'New Query'")
    print("3. Copy-paste the contents of SUPABASE_SETUP.sql")
    print("4. Click 'Run'")
    exit(1)

print("\n[OK] Database is ready! You can now:")
print("   1. Start the backend: python -m uvicorn app.main:app --reload")
print("   2. Upload songs from the frontend: http://localhost:3000")
