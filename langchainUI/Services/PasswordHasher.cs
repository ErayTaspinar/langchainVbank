using Isopoh.Cryptography.Argon2;
using System.Security.Cryptography;
using System.Text;
using dotenv.net;

public class PasswordHasher
{
    private readonly string _pepper;

    public PasswordHasher()
    {
        DotEnv.Load();
        _pepper = Environment.GetEnvironmentVariable("PEPPER")
                  ?? throw new Exception("PEPPER not set in env!");
    }

    public string HashPassword(string password, byte[] salt)
    {
        string combined = password + _pepper;
        var config = new Argon2Config
        {
            Type = Argon2Type.DataIndependentAddressing,
            Version = Argon2Version.Nineteen,
            Salt = salt,
            Password = Encoding.UTF8.GetBytes(combined),
            TimeCost = 4,
            MemoryCost = 65536,
            Lanes = 4,
            Threads = 2,
            HashLength = 32
        };

        using var argon2 = new Argon2(config);
        return argon2.Hash().ToString();
    }

    public byte[] GenerateSalt()
    {
        byte[] salt = new byte[16];
        using var rng = RandomNumberGenerator.Create();
        rng.GetBytes(salt);
        return salt;
    }

    public bool VerifyPassword(string password, byte[] salt, string storedHash)
    {
        return HashPassword(password, salt) == storedHash;
    }
}