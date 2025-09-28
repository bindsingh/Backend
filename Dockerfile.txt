# --- Stage 1: The Builder ---
# This stage installs all dependencies, including build tools.
FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies into the virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file first to leverage Docker's cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: The Final Image ---
# This stage is clean and minimal, only containing what's needed to run.
FROM python:3.11-slim

WORKDIR /app

# Copy the installed virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy all your application code
COPY . .

# Set the PATH to include the venv, so 'uvicorn' can be found
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 8000

# The command to run when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
