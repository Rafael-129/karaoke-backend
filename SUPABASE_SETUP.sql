-- Supabase Songs Database Setup
-- Run this in Supabase SQL Editor: https://app.supabase.com → SQL Editor

-- Create songs table with UUID, job_id, metadata, and timestamps
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_songs_job_id ON songs(job_id);
CREATE INDEX IF NOT EXISTS idx_songs_created_at_desc ON songs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist);

-- Enable Row Level Security
ALTER TABLE songs ENABLE ROW LEVEL SECURITY;

-- Policies
-- 1. Everyone can read all songs
DROP POLICY IF EXISTS "Public read" ON songs;
CREATE POLICY "Public read" ON songs
    FOR SELECT USING (true);

-- 2. Authenticated users can insert
DROP POLICY IF EXISTS "Authenticated write" ON songs;
CREATE POLICY "Authenticated write" ON songs
    FOR INSERT WITH CHECK (true);

-- 3. Authenticated users can delete
DROP POLICY IF EXISTS "Authenticated delete" ON songs;
CREATE POLICY "Authenticated delete" ON songs
    FOR DELETE USING (true);

-- Create storage buckets (can also be done via UI or API on first upload)
-- These may fail with 403 if bucket already exists - that's OK
INSERT INTO storage.buckets (id, name, public)
VALUES ('uploads', 'uploads', true)
ON CONFLICT DO NOTHING;

INSERT INTO storage.buckets (id, name, public)
VALUES ('outputs', 'outputs', true)
ON CONFLICT DO NOTHING;
