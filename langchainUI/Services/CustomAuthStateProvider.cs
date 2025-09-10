using System.Security.Claims;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Components.Authorization;

namespace langchainUI.Services 
{
    public class CustomAuthStateProvider : AuthenticationStateProvider
    {
        // This represents a user who is not logged in.
        private ClaimsPrincipal _anonymous = new ClaimsPrincipal(new ClaimsIdentity());

        // This is the main method that Blazor calls to check who is logged in.
        public override async Task<AuthenticationState> GetAuthenticationStateAsync()
        {
            return await Task.FromResult(new AuthenticationState(_anonymous));
        }

        // We will call this method from our Login component when the user logs in successfully.
        public void NotifyUserAuthentication(string email)
        {
            var identity = new ClaimsIdentity(new[]
            {
                new Claim(ClaimTypes.Name, email),
            }, "apiauth"); 

            var user = new ClaimsPrincipal(identity);
            NotifyAuthenticationStateChanged(Task.FromResult(new AuthenticationState(user)));
        }

        public void NotifyUserLogout()
        {
            NotifyAuthenticationStateChanged(Task.FromResult(new AuthenticationState(_anonymous)));
        }
    }
}