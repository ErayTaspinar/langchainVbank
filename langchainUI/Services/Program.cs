using langchainUI.Components; 
using langchainUI.Services;
using Microsoft.AspNetCore.Components.Authorization;
using DotNetEnv;

var builder = WebApplication.CreateBuilder(args);

Env.Load();

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

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddCascadingAuthenticationState();
builder.Services.AddScoped<AuthenticationStateProvider, CustomAuthStateProvider>();

builder.Services.AddScoped<HttpClient>(s =>
    new HttpClient { BaseAddress = new Uri("https://localhost:5001") });

builder.Services.AddScoped<PasswordHasher>();
builder.Services.AddScoped<JwtService>();
builder.Services.AddScoped<DatabaseService>();
builder.Services.AddSingleton<ChatStateService>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}

app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();