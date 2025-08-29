-- PostgreSQL initialization script
-- This file will be automatically executed when the container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    chat_id BIGINT PRIMARY KEY,
    api_key TEXT,
    endpoint TEXT DEFAULT 'https://openrouter.ai/api/v1/chat/completions',
    model TEXT DEFAULT 'mistralai/mistral-7b-instruct:free',
    reasoning_model TEXT DEFAULT 'deepseek/deepseek-r1-distill-llama-70b',
    router_model TEXT DEFAULT 'anthropic/claude-3.5-sonnet:beta',
    system_prompt TEXT DEFAULT 'You are a helpful assistant.',
    context_messages_count INTEGER DEFAULT 10,
    auto_thinking BOOLEAN DEFAULT true,
    search_enabled BOOLEAN DEFAULT true,
    global_context_size BIGINT DEFAULT 0,
    created_at BIGINT NOT NULL,
    updated_at BIGINT DEFAULT EXTRACT(epoch FROM NOW())
);

-- Create messages table with partitioning for better performance
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL REFERENCES users(chat_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at BIGINT NOT NULL DEFAULT EXTRACT(epoch FROM NOW()),
    message_hash VARCHAR(64), -- For deduplication
    token_count INTEGER DEFAULT 0
) PARTITION BY HASH (chat_id);

-- Create partitions for messages table (4 partitions)
CREATE TABLE IF NOT EXISTS messages_0 PARTITION OF messages FOR VALUES WITH (modulus 4, remainder 0);
CREATE TABLE IF NOT EXISTS messages_1 PARTITION OF messages FOR VALUES WITH (modulus 4, remainder 1);
CREATE TABLE IF NOT EXISTS messages_2 PARTITION OF messages FOR VALUES WITH (modulus 4, remainder 2);
CREATE TABLE IF NOT EXISTS messages_3 PARTITION OF messages FOR VALUES WITH (modulus 4, remainder 3);

-- Create file_contexts table for session-based context
CREATE TABLE IF NOT EXISTS file_contexts (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL REFERENCES users(chat_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    content TEXT NOT NULL,
    file_path TEXT,
    file_size INTEGER DEFAULT 0,
    file_hash VARCHAR(64), -- For deduplication
    created_at BIGINT NOT NULL DEFAULT EXTRACT(epoch FROM NOW()),
    expires_at BIGINT -- For automatic cleanup
);

-- Create global_context table for persistent user data
CREATE TABLE IF NOT EXISTS global_context (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL REFERENCES users(chat_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    content_hash VARCHAR(64), -- For deduplication
    created_at BIGINT NOT NULL DEFAULT EXTRACT(epoch FROM NOW()),
    updated_at BIGINT DEFAULT EXTRACT(epoch FROM NOW())
);

-- Create rate_limiting table
CREATE TABLE IF NOT EXISTS rate_limits (
    chat_id BIGINT NOT NULL,
    action_type TEXT NOT NULL,
    request_count INTEGER DEFAULT 1,
    window_start BIGINT NOT NULL,
    PRIMARY KEY (chat_id, action_type, window_start)
);

-- Create bot_metrics table for monitoring
CREATE TABLE IF NOT EXISTS bot_metrics (
    id SERIAL PRIMARY KEY,
    metric_name TEXT NOT NULL,
    metric_value NUMERIC,
    tags JSONB,
    created_at BIGINT NOT NULL DEFAULT EXTRACT(epoch FROM NOW())
);

-- Create error_logs table
CREATE TABLE IF NOT EXISTS error_logs (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    context_data JSONB,
    created_at BIGINT NOT NULL DEFAULT EXTRACT(epoch FROM NOW())
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id_created_at ON messages(chat_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_hash ON messages(message_hash) WHERE message_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_file_contexts_chat_id ON file_contexts(chat_id);
CREATE INDEX IF NOT EXISTS idx_file_contexts_created_at ON file_contexts(created_at);
CREATE INDEX IF NOT EXISTS idx_file_contexts_expires_at ON file_contexts(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_file_contexts_hash ON file_contexts(file_hash) WHERE file_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_global_context_chat_id ON global_context(chat_id);
CREATE INDEX IF NOT EXISTS idx_global_context_created_at ON global_context(created_at);
CREATE INDEX IF NOT EXISTS idx_global_context_hash ON global_context(content_hash) WHERE content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_rate_limits_chat_id_window ON rate_limits(chat_id, window_start);
CREATE INDEX IF NOT EXISTS idx_bot_metrics_name_created ON bot_metrics(metric_name, created_at);
CREATE INDEX IF NOT EXISTS idx_error_logs_chat_id_created ON error_logs(chat_id, created_at);

-- Create GIN index for JSONB columns
CREATE INDEX IF NOT EXISTS idx_bot_metrics_tags ON bot_metrics USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_error_logs_context ON error_logs USING GIN (context_data);

-- Create functions for automatic cleanup
CREATE OR REPLACE FUNCTION cleanup_old_data() RETURNS void AS $$
BEGIN
    -- Clean up old messages (keep last 30 days)
    DELETE FROM messages WHERE created_at < EXTRACT(epoch FROM NOW() - INTERVAL '30 days');
    
    -- Clean up expired file contexts
    DELETE FROM file_contexts WHERE expires_at IS NOT NULL AND expires_at < EXTRACT(epoch FROM NOW());
    
    -- Clean up old rate limits (keep last 24 hours)
    DELETE FROM rate_limits WHERE window_start < EXTRACT(epoch FROM NOW() - INTERVAL '24 hours');
    
    -- Clean up old metrics (keep last 7 days)
    DELETE FROM bot_metrics WHERE created_at < EXTRACT(epoch FROM NOW() - INTERVAL '7 days');
    
    -- Clean up old error logs (keep last 30 days)
    DELETE FROM error_logs WHERE created_at < EXTRACT(epoch FROM NOW() - INTERVAL '30 days');
    
    -- Update statistics
    ANALYZE;
END;
$$ LANGUAGE plpgsql;

-- Create function to update user's updated_at timestamp
CREATE OR REPLACE FUNCTION update_user_timestamp() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = EXTRACT(epoch FROM NOW());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_user_timestamp();

-- Create function for automatic global_context size calculation
CREATE OR REPLACE FUNCTION update_global_context_size() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE users SET global_context_size = global_context_size + NEW.size_bytes WHERE chat_id = NEW.chat_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE users SET global_context_size = global_context_size - OLD.size_bytes WHERE chat_id = OLD.chat_id;
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        UPDATE users SET global_context_size = global_context_size + (NEW.size_bytes - OLD.size_bytes) WHERE chat_id = NEW.chat_id;
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for global context size management
CREATE TRIGGER trigger_global_context_size
    AFTER INSERT OR UPDATE OR DELETE ON global_context
    FOR EACH ROW
    EXECUTE FUNCTION update_global_context_size();

-- Create a view for user statistics
CREATE OR REPLACE VIEW user_statistics AS
SELECT 
    u.chat_id,
    u.created_at as user_created_at,
    COUNT(DISTINCT m.id) as total_messages,
    COUNT(DISTINCT fc.id) as total_files,
    COUNT(DISTINCT gc.id) as total_global_contexts,
    u.global_context_size,
    MAX(m.created_at) as last_message_at
FROM users u
LEFT JOIN messages m ON u.chat_id = m.chat_id
LEFT JOIN file_contexts fc ON u.chat_id = fc.chat_id
LEFT JOIN global_context gc ON u.chat_id = gc.chat_id
GROUP BY u.chat_id, u.created_at, u.global_context_size;

-- Create scheduled job for cleanup (requires pg_cron extension, optional)
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data();');

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${POSTGRES_USER};
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${POSTGRES_USER};
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO ${POSTGRES_USER};

-- Insert initial admin user if needed (optional)
-- INSERT INTO users (chat_id, created_at) VALUES (123456789, EXTRACT(epoch FROM NOW())) ON CONFLICT DO NOTHING;

-- Optimize PostgreSQL settings for better performance
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Reload configuration
SELECT pg_reload_conf();