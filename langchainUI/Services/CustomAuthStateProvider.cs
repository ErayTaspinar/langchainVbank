using System.Security.Claims;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.JSInterop;
using System.IdentityModel.Tokens.Jwt;

namespace langchainUI.Services
{
    public class CustomAuthStateProvider : AuthenticationStateProvider
    {
        private readonly IJSRuntime _jsRuntime;
        private ClaimsPrincipal _anonymous = new ClaimsPrincipal(new ClaimsIdentity());

        public CustomAuthStateProvider(IJSRuntime jsRuntime)
        {
            _jsRuntime = jsRuntime;
        }

        public override async Task<AuthenticationState> GetAuthenticationStateAsync()
        {
            try
            {
                var accessToken = await _jsRuntime.InvokeAsync<string>("localStorage.getItem", "accessToken");
                
                if (string.IsNullOrWhiteSpace(accessToken))
                    return new AuthenticationState(_anonymous);

                var handler = new JwtSecurityTokenHandler();
                var jwtToken = handler.ReadJwtToken(accessToken);

                // Check if token is expired
                if (jwtToken.ValidTo < DateTime.UtcNow)
                {
                    // Token expired, log out user
                    await NotifyUserLogoutAsync();
                    return new AuthenticationState(_anonymous);
                }

                var claims = new[]
                {
                    new Claim(ClaimTypes.Name, jwtToken.Subject ?? ""),
                    new Claim(ClaimTypes.Role, jwtToken.Claims.FirstOrDefault(c => c.Type == "role")?.Value ?? "User"),
                    new Claim("userId", jwtToken.Claims.FirstOrDefault(c => c.Type == "userId")?.Value ?? "")
                };

                var identity = new ClaimsIdentity(claims, "jwt");
                var user = new ClaimsPrincipal(identity);

                return new AuthenticationState(user);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GetAuthenticationStateAsync error: {ex.Message}");
                return new AuthenticationState(_anonymous);
            }
        }

        public async Task NotifyUserAuthenticationAsync(string accessToken)
        {
            // Store ONLY access token
            await _jsRuntime.InvokeVoidAsync("localStorage.setItem", "accessToken", accessToken);

            var handler = new JwtSecurityTokenHandler();
            var jwt = handler.ReadJwtToken(accessToken);

            var claims = new[]
            {
                new Claim(ClaimTypes.Name, jwt.Subject ?? ""),
                new Claim(ClaimTypes.Role, jwt.Claims.FirstOrDefault(c => c.Type == "role")?.Value ?? "User"),
                new Claim("userId", jwt.Claims.FirstOrDefault(c => c.Type == "userId")?.Value ?? "")
            };

            var identity = new ClaimsIdentity(claims, "jwt");
            var user = new ClaimsPrincipal(identity);

            NotifyAuthenticationStateChanged(Task.FromResult(new AuthenticationState(user)));
        }

        public async Task NotifyUserLogoutAsync()
        {
            await _jsRuntime.InvokeVoidAsync("localStorage.removeItem", "accessToken");
            NotifyAuthenticationStateChanged(Task.FromResult(new AuthenticationState(_anonymous)));
        }
    }
}