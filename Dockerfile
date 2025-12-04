# Use the official Python 3.12 slim image
# 'slim' is preferred over 'alpine' for Python as it has better wheel support
FROM python:3.12-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr (logs flow immediately)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (optional - uncomment if you need gcc, postgres-client, etc.)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m appuser

# Copy the rest of the application code
COPY . .

# Switch ownership of the application code to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port the app runs on (change 8000 to your port)
EXPOSE 5001

# The command to run the application
# Replace 'main.py' with your entry point script or Gunicorn command
CMD ["python", "langChain.py"]