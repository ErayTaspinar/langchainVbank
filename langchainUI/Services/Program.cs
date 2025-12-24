using langchainUI.Components;
using langchainUI.Services;
using Microsoft.AspNetCore.Components.Authorization;
using DotNetEnv;
using System.IdentityModel.Tokens.Jwt;
using System.Text;

// Load .env file for local development (Rider)
Env.Load();

var builder = WebApplication.CreateBuilder(args);

var host = Environment.GetEnvironmentVariable("DB_HOST") ?? "localhost";
var port = Environment.GetEnvironmentVariable("DB_PORT") ?? "5432";
var database = Environment.GetEnvironmentVariable("DB_NAME");
var username = Environment.GetEnvironmentVariable("DB_USER");
var password = Environment.GetEnvironmentVariable("DB_PASSWORD");

if (string.IsNullOrEmpty(database) || string.IsNullOrEmpty(username))
{
    throw new InvalidOperationException("FATAL: Missing required database environment variables.");
}

var connectionString = $"Host={host};Port={port};Database={database};Username={username};Password={password};";
builder.Configuration["ConnectionStrings:DefaultConnection"] = connectionString;

var jwtSecret = Environment.GetEnvironmentVariable("JWT_SECRET");
var pepper = Environment.GetEnvironmentVariable("PEPPER");

if (string.IsNullOrEmpty(jwtSecret) || string.IsNullOrEmpty(pepper))
{
    throw new InvalidOperationException("FATAL: Missing security environment variables.");
}

if (Encoding.UTF8.GetByteCount(jwtSecret) < 16)
{
    throw new InvalidOperationException(
        "FATAL: JWT_SECRET is too short for HS256. Provide at least 16 bytes (128 bits), preferably 32+ bytes.");
}

builder.Configuration["Jwt:Secret"] = jwtSecret;
builder.Configuration["Security:Pepper"] = pepper;

JwtSecurityTokenHandler.DefaultInboundClaimTypeMap.Clear();
JwtSecurityTokenHandler.DefaultOutboundClaimTypeMap.Clear();

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents()
    .AddHubOptions(options =>
    {
        options.MaximumReceiveMessageSize = 10 * 1024 * 1024; 
        options.ClientTimeoutInterval = TimeSpan.FromMinutes(2);
        options.HandshakeTimeout = TimeSpan.FromMinutes(1);
    });

builder.Services.AddCascadingAuthenticationState();
builder.Services.AddScoped<AuthenticationStateProvider, CustomAuthStateProvider>();
builder.Services.AddAuthorizationCore();

var apiBaseUrl = Environment.GetEnvironmentVariable("API_BASE_URL") ?? "http://localhost:5001";

builder.Services.AddScoped(sp => new HttpClient
{
    BaseAddress = new Uri(apiBaseUrl),
    Timeout = TimeSpan.FromMinutes(5) 
});

builder.Services.AddScoped<AuthService>();
builder.Services.AddScoped<JwtService>();
builder.Services.AddScoped<DatabaseService>();

builder.Services.AddHostedService<TokenCleanupService>();

var app = builder.Build();

try
{
    using (var scope = app.Services.CreateScope())
    {
        var dbService = scope.ServiceProvider.GetRequiredService<DatabaseService>();
        await dbService.InitializeDatabase();
        Console.WriteLine("Database initialization check completed successfully.");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"FATAL: Database initialization failed. Error: {ex.Message}");
    throw;
}

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();