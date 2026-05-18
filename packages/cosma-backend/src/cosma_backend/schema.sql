-- =============================================================================
-- File Processing and Organization Database Schema
-- =============================================================================
-- 

-- =============================================================================
-- Watched Directories Table
-- =============================================================================

-- Table for tracking directories that are being monitored for file changes
CREATE TABLE IF NOT EXISTS watched_directories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    is_active INTEGER DEFAULT 1 CHECK (is_active IN (0, 1)),
    recursive INTEGER DEFAULT 1 CHECK (recursive IN (0, 1)),
    file_pattern TEXT,  -- Optional glob pattern for filtering files (e.g., "*.pdf")
    last_scan INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now')) NOT NULL,
    updated_at INTEGER DEFAULT (strftime('%s', 'now')) NOT NULL
);

-- Index for watched directories
CREATE INDEX IF NOT EXISTS idx_watched_directories_is_active ON watched_directories(is_active);
CREATE INDEX IF NOT EXISTS idx_watched_directories_path ON watched_directories(path);

-- Trigger for updating watched_directories timestamp
CREATE TRIGGER IF NOT EXISTS update_watched_directories_timestamp 
    AFTER UPDATE ON watched_directories
    FOR EACH ROW
BEGIN
    UPDATE watched_directories SET updated_at = (strftime('%s', 'now')) WHERE id = NEW.id;
END;

-- =============================================================================

-- =============================================================================
-- Files Table
-- =============================================================================

-- Main files table with comprehensive metadata
CREATE TABLE IF NOT EXISTS files (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Stage 0: Discovery (file system metadata) - Required fields
    file_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    created INTEGER NOT NULL, 
    modified INTEGER NOT NULL,
    accessed INTEGER NOT NULL,
    
    -- Stage 1: Parsing (content extraction)
    content_type TEXT,
    content_hash TEXT,
    parsed_at INTEGER,  
    
    -- Stage 2: Summarization (AI processing)
    summary TEXT,  -- AI-generated summary
    title TEXT,
    summarized_at INTEGER, 
    
    -- Stage 3: Embedding (vector representation)
    embedded_at INTEGER,
    
    -- Meta. No CHECK on the status enum — Python's ProcessingStatus
    -- (models/status.py) is the source of truth, and a CHECK here just
    -- gets out of sync (the constraint missed INDEXED_PARTIAL when it
    -- was added, which silently rejected fresh-install writes; an
    -- enum living in two places is the kind of bug nobody finds until
    -- they wipe their DB). Valid values today:
    --   DISCOVERED | PARSED | SUMMARIZED | COMPLETE | FAILED
    --   INDEXED_PARTIAL | INDEXED_NAME_ONLY
    status TEXT DEFAULT 'DISCOVERED',
    processing_error TEXT,
    
    -- File owner and permissions (if available)
    owner TEXT,
    permissions TEXT,
    
    -- System timestamps
    created_at INTEGER DEFAULT (strftime('%s', 'now')) NOT NULL,
    updated_at INTEGER DEFAULT (strftime('%s', 'now')) NOT NULL
);

-- =============================================================================
-- Vector Embeddings Table (using sqlite-vec)
-- =============================================================================

-- Virtual table for storing file embeddings
-- Note: Adjust the dimension (e.g., float[384], float[768], float[1536]) 
-- based on your embedding model's output size
CREATE VIRTUAL TABLE IF NOT EXISTS file_embeddings USING vec0(
    file_id INTEGER PRIMARY KEY,  -- Links to files.id
    embedding_model TEXT,
    embedding_dimensions INTEGER,
    embedding float[1536]
);

CREATE TRIGGER IF NOT EXISTS delete_file_embeddings
AFTER DELETE ON files
BEGIN
    DELETE FROM file_embeddings WHERE file_id = OLD.id;
END;

-- =============================================================================
-- Keywords Table
-- =============================================================================

-- Keywords table (many-to-many relationship with files)
CREATE TABLE IF NOT EXISTS file_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    keyword TEXT NOT NULL,
    
    -- Indexes for performance
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    UNIQUE(file_id, keyword)  -- Prevent duplicate keywords per file
);

-- =============================================================================
-- Full-Text Search Table (using FTS5)
-- =============================================================================

-- Create a view that combines summary and keywords for each file
CREATE VIEW IF NOT EXISTS files_searchable AS
SELECT 
    f.id,
    f.summary,
    GROUP_CONCAT(fk.keyword, ' ') AS keywords
FROM files f
LEFT JOIN file_keywords fk ON f.id = fk.file_id
GROUP BY f.id;

-- Create the contentless FTS5 table
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    file_path,
    title,
    summary,
    keywords,
    content='',
    contentless_delete=1  -- Use this for UPDATE/DELETE support
);

-- Triggers to keep FTS index synchronized with your data
CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, file_path, title, summary, keywords)
    SELECT 
        new.id,
        new.file_path,
        new.title,
        new.summary,
        GROUP_CONCAT(fk.keyword, ' ')
    FROM file_keywords fk
    WHERE fk.file_id = new.id;
END;

CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    DELETE FROM files_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
    DELETE FROM files_fts WHERE rowid = old.id;
    INSERT INTO files_fts(rowid, file_path, title, summary, keywords)
    SELECT 
        new.id,
        new.file_path,
        new.title,
        new.summary,
        GROUP_CONCAT(fk.keyword, ' ')
    FROM file_keywords fk
    WHERE fk.file_id = new.id;
END;

-- Trigger for keyword changes
CREATE TRIGGER IF NOT EXISTS file_keywords_ai AFTER INSERT ON file_keywords BEGIN
    DELETE FROM files_fts WHERE rowid = new.file_id;
    INSERT INTO files_fts(rowid, file_path, title, summary, keywords)
    SELECT 
        f.id,
        f.file_path,
        f.title,
        f.summary,
        GROUP_CONCAT(fk.keyword, ' ')
    FROM files f
    LEFT JOIN file_keywords fk ON f.id = fk.file_id
    WHERE f.id = new.file_id
    GROUP BY f.id;
END;

CREATE TRIGGER IF NOT EXISTS file_keywords_ad AFTER DELETE ON file_keywords BEGIN
    DELETE FROM files_fts WHERE rowid = old.file_id;
    INSERT INTO files_fts(rowid, file_path, title, summary, keywords)
    SELECT 
        f.id,
        f.file_path,
        f.title,
        f.summary,
        GROUP_CONCAT(fk.keyword, ' ')
    FROM files f
    LEFT JOIN file_keywords fk ON f.id = fk.file_id
    WHERE f.id = old.file_id
    GROUP BY f.id;
END;

CREATE TRIGGER IF NOT EXISTS file_keywords_au AFTER UPDATE ON file_keywords BEGIN
    DELETE FROM files_fts WHERE rowid = old.file_id;
    INSERT INTO files_fts(rowid, file_path, title, summary, keywords)
    SELECT 
        f.id,
        f.file_path,
        f.title,
        f.summary,
        GROUP_CONCAT(fk.keyword, ' ')
    FROM files f
    LEFT JOIN file_keywords fk ON f.id = fk.file_id
    WHERE f.id = old.file_id
    GROUP BY f.id;
END;

-- =============================================================================
-- Applications Table
-- =============================================================================
--
-- Apps are a distinct source from files: the user's "where did I put
-- my Logic Pro?" question wants a totally different surface than a
-- doc search hit. Storing them in their own table (a) keeps the
-- schema honest — apps don't have file_size in any meaningful sense,
-- no content_hash, no parser status; and (b) lets the frontend
-- group app results separately with a visual divider when both
-- types match.
--
-- The same DB hosts both tables so search-time fusion is a UNION
-- ALL, no second connection or two-phase commit.

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- /Applications/Safari.app — the on-disk bundle path. Unique so
    -- a re-scan UPSERTs rather than duplicating.
    app_path TEXT NOT NULL UNIQUE,
    -- com.apple.Safari — from CFBundleIdentifier. Not all apps have
    -- one (old / unsigned builds), so this is nullable. We never use
    -- it as a foreign key.
    bundle_id TEXT,
    -- "Safari", from CFBundleDisplayName or CFBundleName, with the
    -- .app suffix trimmed. The single most search-relevant column.
    display_name TEXT NOT NULL,
    -- Marketing version, e.g. "17.4".
    short_version TEXT,
    -- LSApplicationCategoryType, e.g. "public.app-category.productivity".
    -- Useful for filtering ("show me my productivity apps").
    category TEXT,
    -- CFBundleGetInfoString or similar — short app description text
    -- if present. Most apps don't ship one.
    description TEXT,
    -- Optional LLM-generated "what does this app do" string. Filled
    -- by a future enrichment pass; nullable so first-pass indexing
    -- doesn't block on the LLM.
    use_cases TEXT,
    -- /Applications/Safari.app/Contents/Resources/AppIcon.icns —
    -- resolved during indexing so the UI can lazy-load.
    icon_path TEXT,
    indexed_at INTEGER DEFAULT (strftime('%s', 'now')) NOT NULL,
    updated_at INTEGER DEFAULT (strftime('%s', 'now')) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_applications_bundle_id ON applications(bundle_id);
CREATE INDEX IF NOT EXISTS idx_applications_display_name ON applications(display_name);
CREATE INDEX IF NOT EXISTS idx_applications_category ON applications(category);

CREATE TRIGGER IF NOT EXISTS update_applications_timestamp
    AFTER UPDATE ON applications
    FOR EACH ROW
BEGIN
    UPDATE applications SET updated_at = (strftime('%s', 'now')) WHERE id = NEW.id;
END;

-- Contentless FTS5 mirror, same shape as files_fts so the search
-- fusion code stays symmetrical. Tokenizing display_name plus
-- bundle_id plus category plus the two description fields makes the
-- "I forgot what it's called but it's the one that does X" query
-- work — a user typing "photo editor" hits an app whose category is
-- public.app-category.photography or whose use_cases mentions
-- editing.
CREATE VIRTUAL TABLE IF NOT EXISTS applications_fts USING fts5(
    display_name,
    bundle_id,
    category,
    description,
    use_cases,
    content='',
    contentless_delete=1
);

CREATE TRIGGER IF NOT EXISTS applications_ai AFTER INSERT ON applications BEGIN
    INSERT INTO applications_fts(rowid, display_name, bundle_id, category, description, use_cases)
    VALUES (new.id, new.display_name, new.bundle_id, new.category, new.description, new.use_cases);
END;

CREATE TRIGGER IF NOT EXISTS applications_ad AFTER DELETE ON applications BEGIN
    DELETE FROM applications_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS applications_au AFTER UPDATE ON applications BEGIN
    DELETE FROM applications_fts WHERE rowid = old.id;
    INSERT INTO applications_fts(rowid, display_name, bundle_id, category, description, use_cases)
    VALUES (new.id, new.display_name, new.bundle_id, new.category, new.description, new.use_cases);
END;

-- =============================================================================
-- Processing Statistics Table
-- =============================================================================

-- Processing statistics table
CREATE TABLE IF NOT EXISTS processing_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,  -- Processing session identifier
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    skipped_files INTEGER DEFAULT 0,
    processing_time_seconds REAL,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Main files table indexes
CREATE INDEX IF NOT EXISTS idx_files_extension ON files(extension);
CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_created_at ON files(created_at);
CREATE INDEX IF NOT EXISTS idx_files_file_path ON files(file_path);
CREATE INDEX IF NOT EXISTS idx_files_filename ON files(filename);

-- Keywords table indexes
CREATE INDEX IF NOT EXISTS idx_keywords_file_id ON file_keywords(file_id);
CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON file_keywords(keyword);

-- Processing stats indexes
CREATE INDEX IF NOT EXISTS idx_stats_session_id ON processing_stats(session_id);
CREATE INDEX IF NOT EXISTS idx_stats_started_at ON processing_stats(started_at);

-- =============================================================================
-- Triggers for Automatic Timestamp Updates
-- =============================================================================

-- Update the updated_at timestamp when files are modified
CREATE TRIGGER IF NOT EXISTS update_files_timestamp 
    AFTER UPDATE ON files
    FOR EACH ROW
BEGIN
    UPDATE files SET updated_at = (strftime('%s', 'now')) WHERE id = NEW.id;
END;

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- View for files with their keywords
CREATE VIEW IF NOT EXISTS files_with_keywords AS
SELECT 
    f.*,
    GROUP_CONCAT(fk.keyword, ', ') as keywords
FROM files f
LEFT JOIN file_keywords fk ON f.id = fk.file_id
GROUP BY f.id;

-- View for processing summary
CREATE VIEW IF NOT EXISTS processing_summary AS
SELECT 
    status,
    COUNT(*) as count,
    AVG(file_size) as avg_file_size,
    SUM(file_size) as total_size
FROM files 
WHERE status IS NOT NULL
GROUP BY status;

-- View for recent activity
CREATE VIEW IF NOT EXISTS recent_activity AS
SELECT 
    id,
    filename,
    extension,
    status,
    datetime(created_at, 'unixepoch') as created_date,
    datetime(parsed_at, 'unixepoch') as parsed_date,
    datetime(summarized_at, 'unixepoch') as summarized_date,
    datetime(embedded_at, 'unixepoch') as embedded_date
FROM files 
ORDER BY created_at DESC
LIMIT 100;

-- =============================================================================
-- Initial Data (Optional)
-- =============================================================================

-- Insert initial processing session if none exists
INSERT OR IGNORE INTO processing_stats (
    id,
    session_id,
    started_at,
    status
) VALUES (
    1,
    'initial_session',
    (strftime('%s', 'now')),
    'completed'
);

-- =============================================================================
-- Queue Items Table (persistence for IndexingQueue)
-- =============================================================================
-- Queue items are transient - they're regenerated when the watcher scans directories.
-- We drop and recreate on startup to ensure schema is current.

DROP TABLE IF EXISTS queue_items;
CREATE TABLE queue_items (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('index', 'delete', 'move', 'embed_fallback')),
    status TEXT NOT NULL CHECK (status IN ('cooling_down', 'waiting', 'processing')),
    enqueued_at REAL NOT NULL,
    cooldown_expires_at REAL NOT NULL,
    dest_path TEXT,
    retry_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_queue_items_status ON queue_items(status);
CREATE INDEX IF NOT EXISTS idx_queue_items_file_path ON queue_items(file_path);
