namespace langchainUI.Services
{
    public class TokenResponse
    {
        public string AccessToken { get; set; } = string.Empty;
        public string RefreshToken { get; set; } = string.Empty;
    }

    public class AuthService
    {
        private readonly DatabaseService _databaseService;
        private readonly JwtService _jwtService;

        public AuthService(DatabaseService databaseService, JwtService jwtService)
        {
            _databaseService = databaseService;
            _jwtService = jwtService;
        }

        public async Task<TokenResponse?> LoginUser(string email, string password)
        {
            var user = await _databaseService.ValidateUser(email, password);
            if (user == null) return null;
            
            var accessToken = _jwtService.GenerateAccessToken(user.Id.ToString(), user.Email, user.Role);
            var refreshToken = await _databaseService.RotateRefreshTokenAsync(user.Id);

            return new TokenResponse
            {
                AccessToken = accessToken,
                RefreshToken = refreshToken
            };
        }

        public async Task<TokenResponse?> LoginAdmin(string email, string password)
        {
            var adminUser = await _databaseService.ValidateAdmin(email, password);
            if (adminUser == null) return null;
            
            var accessToken = _jwtService.GenerateAccessToken(adminUser.Id.ToString(), adminUser.Email, adminUser.Role);
            var refreshToken = await _databaseService.RotateRefreshTokenAsync(adminUser.Id);

            return new TokenResponse
            {
                AccessToken = accessToken,
                RefreshToken = refreshToken
            };
        }

        public async Task<TokenResponse?> RegisterAndLoginUser(string email, string password)
        {
            var registrationSuccess = await _databaseService.RegisterUser(email, password);
            if (!registrationSuccess)
            {
                return null;
            }
            return await LoginUser(email, password);
        }
    }
}