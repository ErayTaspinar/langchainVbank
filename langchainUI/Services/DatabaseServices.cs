using Npgsql;
using Dapper;
using Konscious.Security.Cryptography;
using System.Security.Cryptography;
using System.Text;

namespace langchainUI.Services
{
    public class UserDto
    {
        public int Id { get; set; }
        public string Email { get; set; } = string.Empty;
        public string Role { get; set; } = string.Empty;
    }

    public class DatabaseService
    {
        private readonly string _connectionString;
        private readonly string _pepper;

        public DatabaseService(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("DefaultConnection")
                ?? throw new InvalidOperationException("Connection string not found");
            _pepper = configuration["Security:Pepper"]
                ?? throw new InvalidOperationException("Pepper not found");
        }

        private string HashPassword(string password, byte[] salt)
        {
            var passwordWithPepper = password + _pepper;
            var passwordBytes = Encoding.UTF8.GetBytes(passwordWithPepper);
            using var argon2 = new Argon2id(passwordBytes);
            argon2.Salt = salt;
            argon2.DegreeOfParallelism = 8;
            argon2.MemorySize = 65536;
            argon2.Iterations = 4;
            var hash = argon2.GetBytes(32);
            return Convert.ToBase64String(hash);
        }

        private byte[] GenerateSalt()
        {
            var salt = new byte[16];
            using var rng = RandomNumberGenerator.Create();
            rng.GetBytes(salt);
            return salt;
        }

        private bool VerifyPassword(string password, string storedHash, byte[] salt)
        {
            var hash = HashPassword(password, salt);
            return hash == storedHash;
        }

        private string HashRefreshToken(string token)
        {
            using var sha256 = SHA256.Create();
            var tokenBytes = Encoding.UTF8.GetBytes(token);
            var hashBytes = sha256.ComputeHash(tokenBytes);
            return Convert.ToBase64String(hashBytes);
        }

        public async Task<UserDto?> ValidateAdmin(string email, string password)
        {
            var user = await ValidateUser(email, password);
            if (user == null || user.Role != "admin")
            {
                return null;
            }
            return user;
        }

        public async Task InitializeDatabase()
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();
            var createUserSaltsTable = @"
                CREATE TABLE IF NOT EXISTS user_salts (
                    id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )";
            await connection.ExecuteAsync(createUserSaltsTable);
            var createRefreshTokensTable = @"
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash VARCHAR(255) NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    is_revoked BOOLEAN NOT NULL DEFAULT FALSE
                )";
            await connection.ExecuteAsync(createRefreshTokensTable);
            await connection.ExecuteAsync("CREATE INDEX IF NOT EXISTS idx_user_salts_user_id ON user_salts(user_id)");
            await connection.ExecuteAsync("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_token_hash ON refresh_tokens(token_hash)");
            await connection.ExecuteAsync("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id)");
            await connection.ExecuteAsync("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)");
        }

        public async Task<bool> RegisterUser(string email, string password)
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();
            using var transaction = await connection.BeginTransactionAsync();
            try
            {
                var checkQuery = "SELECT COUNT(*) FROM users WHERE email = @Email";
                var exists = await connection.ExecuteScalarAsync<int>(checkQuery, new { Email = email }, transaction);
                if (exists > 0)
                {
                    await transaction.RollbackAsync();
                    return false;
                }
                var salt = GenerateSalt();
                var hashedPassword = HashPassword(password, salt);
                var insertUserQuery = "INSERT INTO users (email, password_hash, created_at) VALUES (@Email, @PasswordHash, @CreatedAt) RETURNING id";
                var userId = await connection.ExecuteScalarAsync<int>(insertUserQuery, new { Email = email, PasswordHash = hashedPassword, CreatedAt = DateTime.UtcNow }, transaction);
                var insertSaltQuery = "INSERT INTO user_salts (user_id, salt, created_at) VALUES (@UserId, @Salt, @CreatedAt)";
                await connection.ExecuteAsync(insertSaltQuery, new { UserId = userId, Salt = Convert.ToBase64String(salt), CreatedAt = DateTime.UtcNow }, transaction);
                await transaction.CommitAsync();
                return true;
            }
            catch
            {
                await transaction.RollbackAsync();
                throw;
            }
        }

        public async Task<UserDto?> ValidateUser(string email, string password)
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();
            var query = "SELECT u.id, u.email, u.role, u.password_hash, s.salt FROM users u INNER JOIN user_salts s ON u.id = s.user_id WHERE u.email = @Email";
            var user = await connection.QueryFirstOrDefaultAsync<dynamic>(query, new { Email = email });
            if (user == null) return null;
            var salt = Convert.FromBase64String((string)user.salt);
            bool isValidPassword = VerifyPassword(password, user.password_hash, salt);
            if (!isValidPassword) return null;
            return new UserDto { Id = user.id, Email = user.email, Role = user.role };
        }

        private string GenerateRefreshToken()
        {
            var randomBytes = new byte[64];
            using var rng = RandomNumberGenerator.Create();
            rng.GetBytes(randomBytes);
            return Convert.ToBase64String(randomBytes);
        }

        public async Task<string> RotateRefreshTokenAsync(int userId)
        {
            var newRefreshToken = GenerateRefreshToken();
            var newTokenHash = HashRefreshToken(newRefreshToken);
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();
            using var transaction = await connection.BeginTransactionAsync();
            try
            {
                var revokeQuery = "UPDATE refresh_tokens SET is_revoked = TRUE WHERE user_id = @UserId AND is_revoked = FALSE";
                await connection.ExecuteAsync(revokeQuery, new { UserId = userId }, transaction);
                var insertQuery = "INSERT INTO refresh_tokens (user_id, token_hash, expires_at, created_at, is_revoked) VALUES (@UserId, @TokenHash, @ExpiresAt, @CreatedAt, FALSE)";
                await connection.ExecuteAsync(insertQuery, new { UserId = userId, TokenHash = newTokenHash, ExpiresAt = DateTime.UtcNow.AddDays(7), CreatedAt = DateTime.UtcNow }, transaction);
                await transaction.CommitAsync();
                return newRefreshToken;
            }
            catch
            {
                await transaction.RollbackAsync();
                throw;
            }
        }

        public async Task RevokeUserRefreshTokens(string email)
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();
            var query = "UPDATE refresh_tokens SET is_revoked = TRUE WHERE user_id = (SELECT id FROM users WHERE email = @Email)";
            await connection.ExecuteAsync(query, new { Email = email });
        }

        public async Task CleanupExpiredTokens()
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();
            var query = @"DELETE FROM refresh_tokens WHERE expires_at < (NOW() AT TIME ZONE 'UTC') OR is_revoked = TRUE";
            await connection.ExecuteAsync(query);
        }
    }
}