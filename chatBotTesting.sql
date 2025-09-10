CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE unique_ids (
    unique_id VARCHAR(255) PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    unique_id VARCHAR(255) NOT NULL REFERENCES unique_ids(unique_id) ON DELETE CASCADE,
    image_url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chats (
    id SERIAL PRIMARY KEY,
    unique_id VARCHAR(255) NOT NULL REFERENCES unique_ids(unique_id) ON DELETE CASCADE,
    chat_text TEXT NOT NULL,
    message_type VARCHAR(50) NOT NULL DEFAULT 'human',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE chat_images (
    chat_id INT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    image_id INT REFERENCES images(id) ON DELETE CASCADE,
    image_token TEXT,
    start_pos INT NOT NULL,
    end_pos INT NOT NULL,
    PRIMARY KEY (chat_id, start_pos)
);

CREATE TABLE refresh_tokens (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
