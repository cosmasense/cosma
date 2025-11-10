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
    
    -- Meta
    status TEXT DEFAULT 'DISCOVERED' CHECK (status IN ('DISCOVERED', 'PARSED', 'SUMMARIZED', 'COMPLETE', 'FAILED')),
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
