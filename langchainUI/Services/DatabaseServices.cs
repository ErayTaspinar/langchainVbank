using Npgsql;
using Dapper;
using Konscious.Security.Cryptography;
using System.Security.Cryptography;
using System.Text;
using System.IdentityModel.Tokens.Jwt;
using Microsoft.IdentityModel.Tokens;
using System.Security.Claims;

namespace langchainUI.Services
{
    public class DatabaseService
    {
        private readonly string _connectionString;
        private readonly string _jwtSecret;
        private readonly string _pepper;

        public DatabaseService(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("DefaultConnection")
                ?? throw new InvalidOperationException("Connection string not found");
            _jwtSecret = configuration["Jwt:Secret"]
                ?? throw new InvalidOperationException("JWT secret not found");
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
                    return false;

                var salt = GenerateSalt();
                var hashedPassword = HashPassword(password, salt);

                var insertUserQuery = @"
                    INSERT INTO users (email, password_hash, created_at)
                    VALUES (@Email, @PasswordHash, @CreatedAt)
                    RETURNING id";

                var userId = await connection.ExecuteScalarAsync<int>(insertUserQuery, new
                {
                    Email = email,
                    PasswordHash = hashedPassword,
                    CreatedAt = DateTime.UtcNow
                }, transaction);

                var insertSaltQuery = @"
                    INSERT INTO user_salts (user_id, salt, created_at)
                    VALUES (@UserId, @Salt, @CreatedAt)";

                await connection.ExecuteAsync(insertSaltQuery, new
                {
                    UserId = userId,
                    Salt = Convert.ToBase64String(salt),
                    CreatedAt = DateTime.UtcNow
                }, transaction);

                await transaction.CommitAsync();
                return true;
            }
            catch
            {
                await transaction.RollbackAsync();
                throw;
            }
        }

        public async Task<bool> UserExists(string email)
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();

            var query = "SELECT COUNT(*) FROM users WHERE email = @Email";
            var count = await connection.ExecuteScalarAsync<int>(query, new { Email = email });
            
            return count > 0;
        }

        // Fast login: Check if user has valid refresh token
        public async Task<string?> QuickLoginWithRefreshToken(string email)
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();

            var query = @"
                SELECT rt.user_id, u.email, u.id
                FROM refresh_tokens rt
                INNER JOIN users u ON rt.user_id = u.id
                WHERE u.email = @Email
                AND rt.is_revoked = FALSE 
                AND rt.expires_at > NOW()
                ORDER BY rt.created_at DESC
                LIMIT 1";

            var tokenData = await connection.QueryFirstOrDefaultAsync<dynamic>(query, new { Email = email });

            if (tokenData == null)
                return null;

            // User has valid refresh token, generate new access token
            var accessToken = GenerateAccessToken(tokenData.id.ToString(), tokenData.email);
            return accessToken;
        }

        // Full login with password verification
        public async Task<string?> ValidateUserAndGetToken(string email, string password)
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();

            // First try quick login with existing refresh token
            var quickLogin = await QuickLoginWithRefreshToken(email);
            if (quickLogin != null)
            {
                Console.WriteLine("Quick login successful - using existing refresh token");
                return quickLogin;
            }

            // No valid refresh token, do full password verification
            Console.WriteLine("No valid refresh token found - performing full password verification");

            var query = @"
                SELECT u.id, u.email, u.password_hash, s.salt 
                FROM users u
                INNER JOIN user_salts s ON u.id = s.user_id
                WHERE u.email = @Email";
            
            var user = await connection.QueryFirstOrDefaultAsync<dynamic>(query, new { Email = email });

            if (user == null)
                return null;

            var salt = Convert.FromBase64String((string)user.salt);
            bool isValidPassword = VerifyPassword(password, user.password_hash, salt);
            
            if (!isValidPassword)
                return null;

            // Password verified, create new refresh token
            var refreshToken = GenerateRefreshToken();
            await StoreRefreshToken(user.id.ToString(), refreshToken);

            var accessToken = GenerateAccessToken(user.id.ToString(), user.email);
            return accessToken;
        }

        private string GenerateAccessToken(string userId, string email)
        {
            var securityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_jwtSecret));
            var credentials = new SigningCredentials(securityKey, SecurityAlgorithms.HmacSha256);

            var claims = new[]
            {
                new Claim(JwtRegisteredClaimNames.Sub, email),
                new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString())
            };

            var token = new JwtSecurityToken(
                issuer: "VBankLangChain",
                audience: "User",
                claims: claims,
                expires: DateTime.UtcNow.AddMinutes(15),
                signingCredentials: credentials
            );

            return new JwtSecurityTokenHandler().WriteToken(token);
        }

        private string GenerateRefreshToken()
        {
            var randomBytes = new byte[64];
            using var rng = RandomNumberGenerator.Create();
            rng.GetBytes(randomBytes);
            return Convert.ToBase64String(randomBytes);
        }

        private async Task StoreRefreshToken(string userId, string refreshToken)
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();

            var tokenHash = HashRefreshToken(refreshToken);

            var query = @"
                INSERT INTO refresh_tokens (token_hash, user_id, expires_at, created_at, is_revoked)
                VALUES (@TokenHash, @UserId, @ExpiresAt, @CreatedAt, FALSE)";

            await connection.ExecuteAsync(query, new
            {
                TokenHash = tokenHash,
                UserId = int.Parse(userId),
                ExpiresAt = DateTime.UtcNow.AddDays(7),
                CreatedAt = DateTime.UtcNow
            });
        }

        public async Task RevokeUserRefreshTokens(string email)
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();

            var query = @"
                UPDATE refresh_tokens 
                SET is_revoked = TRUE 
                WHERE user_id = (SELECT id FROM users WHERE email = @Email)";

            await connection.ExecuteAsync(query, new { Email = email });
        }

        public async Task CleanupExpiredTokens()
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();

            await connection.ExecuteAsync(
                "DELETE FROM refresh_tokens WHERE expires_at < NOW() OR is_revoked = TRUE"
            );
        }
    }
}