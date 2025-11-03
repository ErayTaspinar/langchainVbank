-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User salts table (NEW)
CREATE TABLE IF NOT EXISTS user_salts (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    salt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Unique IDs table
CREATE TABLE IF NOT EXISTS unique_ids (
    unique_id VARCHAR(255) PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE
);

-- Images table
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    unique_id VARCHAR(255) NOT NULL REFERENCES unique_ids(unique_id) ON DELETE CASCADE,
    image_url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chats table
CREATE TABLE IF NOT EXISTS chats (
    id SERIAL PRIMARY KEY,
    unique_id VARCHAR(255) NOT NULL REFERENCES unique_ids(unique_id) ON DELETE CASCADE,
    chat_text TEXT NOT NULL,
    message_type VARCHAR(50) NOT NULL DEFAULT 'human',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chat images table
CREATE TABLE IF NOT EXISTS chat_images (
    chat_id INT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    image_id INT REFERENCES images(id) ON DELETE CASCADE,
    image_token TEXT,
    start_pos INT NOT NULL,
    end_pos INT NOT NULL,
    PRIMARY KEY (chat_id, start_pos)
);

-- Refresh tokens table
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_user_salts_user_id ON user_salts(user_id);
CREATE INDEX IF NOT EXISTS idx_unique_ids_user_id ON unique_ids(user_id);
CREATE INDEX IF NOT EXISTS idx_images_unique_id ON images(unique_id);
CREATE INDEX IF NOT EXISTS idx_chats_unique_id ON chats(unique_id);
CREATE INDEX IF NOT EXISTS idx_chat_images_chat_id ON chat_images(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_images_image_id ON chat_images(image_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_token_hash ON refresh_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id);
