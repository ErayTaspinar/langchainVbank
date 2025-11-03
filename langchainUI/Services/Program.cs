using langchainUI.Components; 
using langchainUI.Services;
using Microsoft.AspNetCore.Components.Authorization;
using DotNetEnv;

var builder = WebApplication.CreateBuilder(args);


Env.Load();

// Database configuration
var host = Environment.GetEnvironmentVariable("DB_HOST") ?? "localhost";
var port = Environment.GetEnvironmentVariable("DB_PORT") ?? "5432";
var database = Environment.GetEnvironmentVariable("DB_NAME") ?? "";
var username = Environment.GetEnvironmentVariable("DB_USER") ?? "";
var password = Environment.GetEnvironmentVariable("DB_PASSWORD") ?? "";

if (string.IsNullOrEmpty(database) || string.IsNullOrEmpty(username))
{
    throw new InvalidOperationException("Missing required database environment variables. Check your .env file.");
}

var connectionString = $"Host={host};Port={port};Database={database};Username={username};Password={password};";

Console.WriteLine($"🔍 Built Connection String: Host={host};Port={port};Database={database};Username={username};Password=***;");

builder.Configuration["ConnectionStrings:DefaultConnection"] = connectionString;


// JWT and Security configuration
var jwtSecret = Environment.GetEnvironmentVariable("JWT_SECRET");
var pepper = Environment.GetEnvironmentVariable("PEPPER");

if (string.IsNullOrEmpty(jwtSecret))
{
    throw new InvalidOperationException("Missing JWT_SECRET environment variable. Check your .env file.");
}

if (string.IsNullOrEmpty(pepper))
{
    throw new InvalidOperationException("Missing SECURITY_PEPPER environment variable. Check your .env file.");
}

builder.Configuration["Jwt:Secret"] = jwtSecret;
builder.Configuration["Security:Pepper"] = pepper;

// Add services
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddCascadingAuthenticationState();
builder.Services.AddScoped<AuthenticationStateProvider, CustomAuthStateProvider>();
builder.Services.AddScoped<CustomAuthStateProvider>(); // Also register as concrete type

builder.Services.AddScoped<HttpClient>(s =>
    new HttpClient { BaseAddress = new Uri("https://localhost:5001") });

builder.Services.AddScoped<PasswordHasher>();
builder.Services.AddScoped<JwtService>();
builder.Services.AddScoped<DatabaseService>();
builder.Services.AddSingleton<ChatStateService>();

// Add authorization
builder.Services.AddAuthorizationCore();

// Add background service for token cleanup
builder.Services.AddHostedService<TokenCleanupService>();

var app = builder.Build();

// Initialize database on startup
try
{
    using (var scope = app.Services.CreateScope())
    {
        var dbService = scope.ServiceProvider.GetRequiredService<DatabaseService>();
        await dbService.InitializeDatabase();
        Console.WriteLine("Database initialized successfully");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Database initialization failed: {ex.Message}");
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