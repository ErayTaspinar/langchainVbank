using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.JSInterop;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;

namespace langchainUI.Services
{
    public class CustomAuthStateProvider : AuthenticationStateProvider
    {
        private readonly IJSRuntime _jsRuntime;
        private readonly ClaimsPrincipal _anonymous = new ClaimsPrincipal(new ClaimsIdentity());

        public CustomAuthStateProvider(IJSRuntime jsRuntime)
        {
            _jsRuntime = jsRuntime;
        }

        public override async Task<AuthenticationState> GetAuthenticationStateAsync()
        {
            try
            {
                var token = await _jsRuntime.InvokeAsync<string>("localStorage.getItem", "accessToken");
                if (string.IsNullOrWhiteSpace(token))
                {
                    return new AuthenticationState(_anonymous);
                }

                var handler = new JwtSecurityTokenHandler();
                var jwtToken = handler.ReadToken(token) as JwtSecurityToken;

                if (jwtToken == null || jwtToken.ValidTo < DateTime.UtcNow)
                {
                    await _jsRuntime.InvokeVoidAsync("localStorage.removeItem", "accessToken");
                    return new AuthenticationState(_anonymous);
                }

                return CreateAuthenticationStateFromToken(token);
            }
            catch
            {
                return new AuthenticationState(_anonymous);
            }
        }

        public async Task NotifyUserAuthenticationAsync(string token)
        {
            await _jsRuntime.InvokeVoidAsync("localStorage.setItem", "accessToken", token);
            var authState = CreateAuthenticationStateFromToken(token);
            NotifyAuthenticationStateChanged(Task.FromResult(authState));
        }

        public async Task NotifyAuthenticationStateChangedAsync()
        {
            var authState = await GetAuthenticationStateAsync();
            NotifyAuthenticationStateChanged(Task.FromResult(authState));
        }

        public async Task NotifyUserLogoutAsync()
        {
            await _jsRuntime.InvokeVoidAsync("localStorage.removeItem", "accessToken");
            NotifyAuthenticationStateChanged(Task.FromResult(new AuthenticationState(_anonymous)));
        }

        private AuthenticationState CreateAuthenticationStateFromToken(string token)
        {
            var claims = ParseClaimsFromJwt(token);
            if (!claims.Any()) return new AuthenticationState(_anonymous);

            var identity = new ClaimsIdentity(claims, "jwtAuthType", JwtRegisteredClaimNames.Email, ClaimTypes.Role);
            return new AuthenticationState(new ClaimsPrincipal(identity));
        }

        private IEnumerable<Claim> ParseClaimsFromJwt(string jwt)
        {
            if (string.IsNullOrWhiteSpace(jwt)) return Enumerable.Empty<Claim>();
            var handler = new JwtSecurityTokenHandler();
            var jsonToken = handler.ReadToken(jwt) as JwtSecurityToken;
            return jsonToken?.Claims ?? Enumerable.Empty<Claim>();
        }
    }
}