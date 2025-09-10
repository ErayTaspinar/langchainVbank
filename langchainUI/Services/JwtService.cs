using System;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Microsoft.IdentityModel.Tokens;
using dotenv.net;

public class JwtService
{
    private readonly string _jwtSecret;
    private readonly string _issuer;
    private readonly string _audience;

    public JwtService()
    { 
        DotEnv.Load();
        _jwtSecret = Environment.GetEnvironmentVariable("JWT_SECRET")
                     ?? throw new InvalidOperationException("JWT_SECRET not set in environment variables!");

        _issuer = "http://localhost:5162";
        _audience = "http://localhost:5162";
    }

    /// <summary>
    /// Generates a JWT for a given user ID and role.
    /// </summary>
    /// <param name="userId">The user's unique identifier.</param>
    /// <param name="role">The user's role (e.g., "Admin", "User").</param>
    /// <returns>A signed JWT string.</returns>
    public string GenerateToken(int userId, string role)
    {
        var claims = new[]
        {
            new Claim(JwtRegisteredClaimNames.Sub, userId.ToString()),
            new Claim(ClaimTypes.Role, role),
            new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString())
        };

        var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_jwtSecret));
        var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

        var token = new JwtSecurityToken(
            issuer: _issuer,
            audience: _audience,
            claims: claims,
            expires: DateTime.UtcNow.AddSeconds(60),
            signingCredentials: creds
        );

        return new JwtSecurityTokenHandler().WriteToken(token);
    }

    /// <summary>
    /// Validates an incoming JWT string.
    /// </summary>
    /// <param name="token">The JWT string to validate.</param>
    /// <returns>The ClaimsPrincipal from the token if valid; otherwise, null.</returns>
    public ClaimsPrincipal? ValidateToken(string token)
    {
        if (string.IsNullOrEmpty(token))
        {
            return null;
        }

        var tokenHandler = new JwtSecurityTokenHandler();
        var key = Encoding.UTF8.GetBytes(_jwtSecret);

        try
        {
            var principal = tokenHandler.ValidateToken(token, new TokenValidationParameters
            {
                ValidateIssuer = true,
                ValidIssuer = _issuer,

                ValidateAudience = true,
                ValidAudience = _audience,

                ValidateIssuerSigningKey = true,
                IssuerSigningKey = new SymmetricSecurityKey(key),

                ValidateLifetime = true,

                ClockSkew = TimeSpan.Zero
            }, out SecurityToken validatedToken);

            return principal;
        }
        catch (SecurityTokenException)
        {
            return null;
        }
        catch (Exception)
        {
            return null;
        }
    }
}