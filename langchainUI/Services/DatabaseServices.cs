using Npgsql;
using BCrypt.Net;
using Microsoft.Extensions.Configuration;

public class DatabaseService
{
    private readonly string _connectionString;

    public DatabaseService(IConfiguration configuration)
    {
        _connectionString = configuration.GetConnectionString("DefaultConnection");
    }

    public async Task<bool> UserExists(string email)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();
            
            var checkCmd = new NpgsqlCommand("SELECT COUNT(*) FROM users WHERE email = @email", conn);
            checkCmd.Parameters.AddWithValue("email", email);
            var userExists = (long)await checkCmd.ExecuteScalarAsync() > 0;

            Console.WriteLine($"User exists check for {email}: {userExists}");
            return userExists;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error checking if user exists: {ex.Message}");
            return false;
        }
    }

    public async Task<bool> RegisterUser(string email, string password)
    {
        try
        {
            Console.WriteLine($"Registration attempt for email: {email}");
            
            // Check if user already exists
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();
            
            var checkCmd = new NpgsqlCommand("SELECT COUNT(*) FROM users WHERE email = @email", conn);
            checkCmd.Parameters.AddWithValue("email", email);
            var userExists = (long)await checkCmd.ExecuteScalarAsync() > 0;

            if (userExists)
            {
                Console.WriteLine($"Registration failed - User already exists: {email}");
                return false; // User with this email already exists
            }

            string passwordHash = BCrypt.Net.BCrypt.HashPassword(password);
            Console.WriteLine($"Password hashed successfully for: {email}");

            // Insert new user into the database
            var insertCmd = new NpgsqlCommand("INSERT INTO users (email, password_hash) VALUES (@email, @password_hash)", conn);
            insertCmd.Parameters.AddWithValue("email", email);
            insertCmd.Parameters.AddWithValue("password_hash", passwordHash);
            
            await insertCmd.ExecuteNonQueryAsync();
            
            Console.WriteLine($"User registered successfully: {email}");
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Registration error for {email}: {ex.Message}");
            return false;
        }
    }

    public async Task<bool> ValidateUser(string email, string password)
    {
        try
        {
            Console.WriteLine($"Password validation attempt for: {email}");
            
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();
            
            var cmd = new NpgsqlCommand("SELECT password_hash FROM users WHERE email = @email", conn);
            cmd.Parameters.AddWithValue("email", email);
            
            var storedHash = (string)await cmd.ExecuteScalarAsync();

            if (storedHash == null)
            {
                Console.WriteLine($"No password hash found for user: {email}");
                return false; // User not found
            }
            
            bool isValid = BCrypt.Net.BCrypt.Verify(password, storedHash);
            Console.WriteLine($"Password validation for {email}: {isValid}");
            
            return isValid;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Password validation error for {email}: {ex.Message}");
            return false;
        }
    }

    // Helper method to get user details (useful for authentication)
    public async Task<(int userId, string role, bool emailVerified)?> GetUserDetails(string email)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();
            
            // Test connection first
            Console.WriteLine("Database connection successful");
            
            var cmd = new NpgsqlCommand("SELECT id, role, email_verified FROM users WHERE email = @email", conn);
            cmd.Parameters.AddWithValue("email", email);
            
            await using var reader = await cmd.ExecuteReaderAsync();
            
            if (await reader.ReadAsync())
            {
                var userId = reader.GetInt32(0); // Use index instead of column name
                var role = reader.IsDBNull(1) ? "user" : reader.GetString(1); // Handle potential null
                var emailVerified = reader.IsDBNull(2) ? false : reader.GetBoolean(2); // Handle potential null
                
                Console.WriteLine($"User details retrieved for {email}: ID={userId}, Role={role}, Verified={emailVerified}");
                return (userId, role, emailVerified);
            }
            
            Console.WriteLine($"No user details found for: {email}");
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error getting user details for {email}: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            return null;
        }
    }
}